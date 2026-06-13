import os

import pytorch_lightning as pl
import torch
import numpy as np
import torch.nn.functional as F
from training.clt_model import CrossLayerTranscoder
from external.progen3.src.progen3.modeling import ProGen3ForCausalLM
from external.progen3.src.progen3.batch_preparer import ProGen3BatchPreparer
from training.glm_helper import generate_glm_instance

np.random.seed(42)


def resolve_progen3_dtype(device=None):
    """Pick a ProGen3 dtype. Override with PROGEN3_DTYPE if needed.

    Note: megablocks MoE kernels expect bfloat16 weights; float32/float16 are only
    honored when explicitly requested via PROGEN3_DTYPE.
    """
    override = os.environ.get("PROGEN3_DTYPE", "").lower().strip()
    if override in ("float32", "fp32", "float"):
        return torch.float32
    if override in ("bfloat16", "bf16"):
        return torch.bfloat16
    if override in ("float16", "fp16"):
        return torch.float16
    return torch.bfloat16


def needs_legacy_gpu_compat(device=None):
    """True on GPUs below Ampere (e.g. Colab T4), which lack bf16 Triton kernels."""
    if os.environ.get("PROGEN3_LEGACY_GPU", "").lower() in ("1", "true", "yes"):
        return True
    if not torch.cuda.is_available():
        return False
    dev = device if device is not None else torch.device("cuda", torch.cuda.current_device())
    major, _ = torch.cuda.get_device_capability(dev)
    return major < 8

# ──────────────────────────────────────────────────────────────────────────────
# Shape suffixes:
# B: Batch Size 
# L: Total number of LM layers
# T: Sequence length of protein (variable)
# D: CLT Latent dim (d_hidden)
# H: Embedding Dimension of LM (d_model)
# S: B * T
# ──────────────────────────────────────────────────────────────────────────────

class ProGen3ActivationCollector:
    def __init__(self, progen3_model : ProGen3ForCausalLM, vocabulary, target_layers=None):
        self.model = progen3_model
        self.vocabulary = vocabulary
        self.target_layers = target_layers if target_layers else list(range(len(progen3_model.model.layers)))
        self.activations = {} 
        self.hooks = []

    def _make_hook(self, key, scale=1.0, is_input=False, transpose=False):
        """
        Creates a hook that:
        1. Extracts tensor from tuple (if needed)
        2. Scales the tensor (for embeddings)
        3. Transposes (T, B, H) -> (B, T, H) (for layers)
        """
        def hook(module, input, output):
            if is_input:
                if isinstance(input, tuple):
                    data = input[0]
                else:
                    data = input
            else:
                if isinstance(output, tuple):
                    data = output[0]
                else:
                    data = output
            if scale != 1.0:
                data = data * scale                
            # (T, B, H) -> (B, T, H)
            if transpose:
                data = data.transpose(0, 1)                
            self.activations[key] = data.detach()
        return hook

    def register_hooks(self):
        # 1. Hook MLP Inputs (B, T, H)
        for layer_idx in self.target_layers:
            block = self.model.model.layers[layer_idx]
            hook_in = block.block_sparse_moe.register_forward_hook(
                self._make_hook(f"x_{layer_idx}", is_input=True, transpose=False)
            )
            self.hooks.append(hook_in)
        
        # 2. Hook MLP outputs
        for layer_idx in self.target_layers:
            block = self.model.model.layers[layer_idx]
            hook_out = block.block_sparse_moe.register_forward_hook(
                self._make_hook(f"y_{layer_idx}", is_input=False, transpose=False)
            )
            self.hooks.append(hook_out)

    def collect(self, batch_preparer, sequences, attention_mask=None):
        """
        input_ids: (B, T)
        Returns:
        x_stack_flat_SLH: (B*T, L, H)
        x_mlp_stack_flat_SLH: (B*T, L, H)
        mask_S: (B*T,)
        """
        self.activations = {} 
        with torch.no_grad():
            device = next(self.model.parameters()).device 
            inputs = batch_preparer.get_batch_kwargs(sequences, device=device, reverse=False)
            self.model(**inputs, return_dict=True)      
        if not self.activations:
            raise RuntimeError("Collector failed: No activations captured.")
        
        # x activations
        filtered_activations = {k: v for k, v in self.activations.items() 
                                if not (isinstance(k, str) and k.startswith("y_"))}
        sorted_keys = sorted(filtered_activations.keys(), key=lambda x: int(x.split("_")[1]))
        x_activations = [filtered_activations[k] for k in sorted_keys] # L tensors, each one is (B, T, H)

        # y activations
        mlp_keys = sorted(
            [k for k in self.activations.keys() if isinstance(k, str) and k.startswith("y_")],
            key=lambda x: int(x.split("_")[1])
        )
        y_activations = [self.activations[k] for k in mlp_keys] # L tensors, each one is (B, T, H)
        
        # (B, T, H) -> (B, L, T, H)
        x_stack_BLTH = torch.stack(x_activations, dim=1) 
        x_mlp_stack_BLTH = torch.stack(y_activations, dim=1)
        B, L, T, H = x_stack_BLTH.shape
        # Flatten: (B, L, T, H) -> (B, T, L, H) -> (B*T, L, H)
        x_stack_flat_SLH = x_stack_BLTH.permute(0, 2, 1, 3).reshape(B * T, L, H)
        x_mlp_stack_flat_SLH = x_mlp_stack_BLTH.permute(0, 2, 1, 3).reshape(B * T, L, H)

        special_token_names = [
            '<pad>', '<bos>', '<eos>', '<bos_glm>', '<eos_span>', '<mask>', '1', '2'
        ]

        for i in range(100):
            special_token_names.append(f'<span_{i}>')

        special_ids = {self.vocabulary[name] for name in special_token_names if name in self.vocabulary}

        mask_BT = torch.ones_like(inputs['input_ids'], dtype=torch.bool)
        for s_id in special_ids:
            mask_BT &= (inputs['input_ids'] != s_id)

        mask_S = mask_BT.view(-1)
        
        return x_stack_flat_SLH, x_mlp_stack_flat_SLH, mask_S
    
    def remove_hooks(self):
        for h in self.hooks: h.remove()
        self.hooks = []

class CLTLightningModule(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters()
        self.args = args
        self.num_layers = args.num_layers 
        
        # 1. Initialize CLT
        self.clt = CrossLayerTranscoder(
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
        progen3_dtype = resolve_progen3_dtype()
        self.progen3_model = ProGen3ForCausalLM.from_pretrained(self.args.model, torch_dtype=progen3_dtype)

        # 4. Freeze Progen3 Model
        self.progen3_model = self.progen3_model.eval()
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

        # 2. Run CLT Forward
        recons_stack_SLH, auxk_stack_SLH, dead_mask_stack_LD = self.clt(x_stack_trajectory_SLH)
        
        total_loss = 0
        total_mse = 0
        total_aux = 0
        total_nmse = 0
        auxk_coef = 1.0 / 32.0 
        
        # 3. Identify Valid Tokens (Masking applied here)
        valid_indices = torch.nonzero(mask_S).squeeze()
        
        # 4. Cumulative Reconstruction Loss
        for l in range(self.num_layers):
            #true_state_SH = x_stack_trajectory_SLH[:, l + 1, :]
            true_state_SH = x_mlp_stack_trajectory_SLH[:, l, :]
            pred_state_SH = recons_stack_SLH[:, l, :]
            
            # --- APPLY MASK ---
            true_masked = true_state_SH[valid_indices]
            pred_masked = pred_state_SH[valid_indices]
            # A. Main Reconstruction Loss
            mse = F.mse_loss(pred_masked, true_masked)
            # B. NMSE
            target_var = torch.var(true_masked) + 1e-8
            nmse = mse / target_var
            
            total_loss += nmse
            total_mse += mse
            total_nmse += nmse
            
            # C. AuxK Loss
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
    
    def configure_optimizers(self):
        return torch.optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=self.args.lr, weight_decay=1e-5)

    def validation_step(self, batch, batch_idx):
        raw_seqs = batch["Sequence"]
        batch_size = len(raw_seqs)
        processed_seqs = self.process_seqs(raw_seqs)
        
        x_stack_trajectory_SLH, x_mlp_stack_trajectory_SLH, mask_S = self.collector.collect(self.batch_preparer, processed_seqs)
        
        recons_stack_SLH, _, _ = self.clt(x_stack_trajectory_SLH)
        
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
