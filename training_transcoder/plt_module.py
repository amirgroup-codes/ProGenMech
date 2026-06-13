import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import numpy as np
from external.progen3.src.progen3.modeling import ProGen3ForCausalLM
from external.progen3.src.progen3.batch_preparer import ProGen3BatchPreparer
from training_transcoder.plt_model import PerLayerTranscoder
from training.clt_module import ProGen3ActivationCollector
from training.glm_helper import generate_glm_instance

np.random.seed(42)

# ──────────────────────────────────────────────────────────────────────────────
# Shape suffixes:
# B: Batch Size 
# L: Total number of LM layers
# T: Sequence length of protein (variable)
# D: CLT Latent dim (d_hidden)
# H: Embedding Dimension of LM (d_model)
# S: B * T
# ──────────────────────────────────────────────────────────────────────────────

class PLTLightningModule(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters()
        self.args = args
        self.num_layers = args.num_layers 
        
        # 1. Initialize PLT
        self.plt = PerLayerTranscoder(
            num_layers=self.num_layers,
            d_model=args.d_model,
            d_hidden=args.d_hidden,
            k=args.k,
            auxk=args.auxk,
            batch_size=args.batch_size,
            dead_steps_threshold=args.dead_steps_threshold,
        )
        
        # 2. Set up batch preparer and tokenizer
        self.batch_preparer = ProGen3BatchPreparer()
        self.tokenizer = self.batch_preparer.tokenizer

        # 3. Initialize & Load Progen3 Model
        self.progen3_model = ProGen3ForCausalLM.from_pretrained(self.args.model, torch_dtype=torch.bfloat16)

        # 4. Freeze Progen3 Model
        self.progen3_model = self.progen3_model.eval().to("cuda:0")
        for param in self.progen3_model.parameters():
            param.requires_grad = False

        # 5. Setup Collector
        self.collector = ProGen3ActivationCollector(self.progen3_model, self.tokenizer.get_vocab())
        self.collector.register_hooks()

    # Use infilling for 1/3 of the sequences (1:2 GLM to CLM ratio)
    def process_seqs(self, raw_seqs):
        processed_seqs = []
        for seq in raw_seqs:
            if np.random.rand() < 1/3:
                processed_seqs.append(generate_glm_instance(seq))
            else:
                processed_seqs.append(seq)
        return processed_seqs

    def training_step(self, batch, batch_idx):
        raw_seqs = batch["Sequence"]
        batch_size = len(raw_seqs)
        processed_seqs = self.process_seqs(raw_seqs)
        
        # 1. Collect Activations
        x_stack_trajectory_SLH, x_mlp_stack_trajectory_SLH, mask_S = self.collector.collect(self.batch_preparer, processed_seqs)
        
        # 2. Run PLT Forward
        recons_stack_SLH, auxk_stack_SLH, dead_mask_stack_LD = self.plt(x_stack_trajectory_SLH)
        
        total_loss = 0
        total_mse = 0
        total_aux = 0
        total_nmse = 0
        auxk_coef = 1.0 / 32.0 
        
        # 3. Identify Valid Tokens (Masking applied here)
        valid_indices = torch.nonzero(mask_S).squeeze()
        
        for l in range(self.num_layers):
            # Target: Layers 1 to L
            true_state_SH = x_mlp_stack_trajectory_SLH[:, l, :]
            pred_state_SH = recons_stack_SLH[:, l, :]
            
            true_masked = true_state_SH[valid_indices]
            pred_masked = pred_state_SH[valid_indices]
            
            mse = F.mse_loss(pred_masked, true_masked)
            target_var = torch.var(true_masked) + 1e-8
            nmse = mse / target_var
            
            total_loss += nmse
            total_mse += mse
            total_nmse += nmse
            
            if auxk_stack_SLH is not None:
                residual = (true_masked - pred_masked).detach()
                aux_out_masked = auxk_stack_SLH[:, l, :][valid_indices]
                
                aux_loss = F.mse_loss(aux_out_masked, residual)
                total_aux += aux_loss
                total_loss += (aux_loss * auxk_coef)
                
                self.log(f"train/aux_loss_layer_{l}", aux_loss, batch_size=batch_size)
                
            self.log(f"train/mse_layer_{l}", mse, batch_size=batch_size)
            self.log(f"train/nmse_layer_{l}", nmse, batch_size=batch_size)
            self.log(f"train/dead_neurons_{l}", dead_mask_stack_LD[l].sum().float(), batch_size=batch_size)

        avg_nmse = total_nmse / self.num_layers
        self.log("train/loss", total_loss, batch_size=batch_size)
        self.log("train/avg_nmse", avg_nmse, batch_size=batch_size)
        
        return total_loss

    def validation_step(self, batch, batch_idx):
        raw_seqs = batch["Sequence"]
        batch_size = len(raw_seqs)
        processed_seqs = self.process_seqs(raw_seqs)
        
        x_stack_trajectory_SLH, x_mlp_stack_trajectory_SLH, mask_S = self.collector.collect(self.batch_preparer, processed_seqs)
        
        recons_stack_SLH, _, _ = self.plt(x_stack_trajectory_SLH)
        
        total_loss = 0
        total_nmse = 0
        
        valid_indices = torch.nonzero(mask_S).squeeze()
        
        for l in range(self.num_layers):
            true_state_SH = x_mlp_stack_trajectory_SLH[:, l, :]
            pred_state_SH = recons_stack_SLH[:, l, :]
            
            true_masked = true_state_SH[valid_indices]
            pred_masked = pred_state_SH[valid_indices]
            
            mse = F.mse_loss(pred_masked, true_masked)
            target_var = torch.var(true_masked) + 1e-8
            nmse = mse / target_var
            
            total_loss += nmse
            total_nmse += nmse
            
            self.log(f"val/nmse_layer_{l}", nmse, batch_size=batch_size)
            
        avg_nmse = total_nmse / self.num_layers
        self.log("val/loss", total_loss, prog_bar=True, batch_size=batch_size)
        self.log("val/avg_nmse", avg_nmse, batch_size=batch_size)
        return total_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.args.lr, weight_decay=1e-5)