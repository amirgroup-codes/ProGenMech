import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import hashlib
from pathlib import Path
from typing import Dict, Tuple
try:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    progen3_src = repo_root / "external" / "progen3" / "src"
    if str(progen3_src) not in sys.path:
        sys.path.insert(0, str(progen3_src))
    from training.plt_module import PLTLightningModule
except ImportError:
    try:
        sys.path.append(str(repo_root / "training_transcoder"))
        from plt_module import PLTLightningModule
    except ImportError:
        PLTLightningModule = None
try:
    from .clt_circuit import CircuitDiscovererCLT
except ImportError:
    try:
        from clt_circuit import CircuitDiscovererCLT
    except ImportError:
        CircuitDiscovererCLT = None
try:
    from .progen3_activation import ProGen3ActivationCollector
except ImportError:
    try:
        from progen3_activation import ProGen3ActivationCollector
    except ImportError:
        ProGen3ActivationCollector = None
try:
    from progen3.batch_preparer import get_spans_to_mask, prepare_glm_string_from_spans
except ImportError:
    get_spans_to_mask = None
    prepare_glm_string_from_spans = None


# ──────────────────────────────────────────────────────────────────────────────
# Shape suffixes:
# B: Batch Size 
# L: Total number of LM layers
# T: Sequence length of protein (variable)
# D: PLT Latent dim (d_hidden)
# H: Embedding Dimension of LM (d_model)
# S: B * T
# ──────────────────────────────────────────────────────────────────────────────

class CircuitDiscovererPLT(CircuitDiscovererCLT):
    """
    PLT Circuit Discoverer (Sequential / Backprop Through Time).
    Subclass of CircuitDiscovererCLT with PLT-specific reconstruction logic.
    """
    def __init__(self, device, ckpt_path=None):
        if PLTLightningModule is None:
            raise ImportError("Could not import PLTLightningModule. Check sys.path.")
        self.device = device
        self.ckpt_path = ckpt_path or os.environ.get("PLT_CHECKPOINT")
        if not self.ckpt_path: 
            raise ValueError("PLT_CHECKPOINT env var not set")
        print(f"Loading PLT from {self.ckpt_path}...")
        try:
            self.pl_module = PLTLightningModule.load_from_checkpoint(self.ckpt_path, map_location=device) 
        except Exception as e:
            raise ValueError(f"Could not load PLT from {self.ckpt_path}")
        self.pl_module.to(device)
        self.pl_module.eval()
        self.plt = self.pl_module.plt
        self.num_layers = self.plt.num_layers

        # ProGen3 module handles
        self.progen = self.pl_module.progen3_model
        self.batch_preparer = self.pl_module.batch_preparer
        self.tokenizer = self.batch_preparer.tokenizer
        self.vocab = self.tokenizer.get_vocab()
        self.pad_id = self.batch_preparer.pad_token_id

        if ProGen3ActivationCollector is None:
            raise ImportError("Could not import ProGen3ActivationCollector.")
        self.collector = ProGen3ActivationCollector(self.progen, self.vocab)
        self.collector.register_hooks()

        special_token_names = [
            '<pad>', '<bos>', '<eos>', '<bos_glm>', '<eos_span>', '<mask>', '1', '2'
        ]
        for i in range(100):
            special_token_names.append(f'<span_{i}>')
        self.special_ids = {self.vocab[name] for name in special_token_names if name in self.vocab}

        # If these markers are present, treat the sequence as already structured (e.g., GLM/span format)
        # and do not force-wrap with boundary tokens 1...2.
        self._structured_markers = (
            "<bos_glm>",
            "<eos_span>",
            "<mask>",
            "<span_",
            "<bos>",
            "<eos>",
        )

    def _format_sequence_for_progen(self, seq: str) -> str:
        """Return a ProGen3-ready sequence string without adding CLM boundary tokens.

        Plain CLM strings stay raw and are wrapped by the batch preparer.
        GLM strings keep their span metadata, with optional leading 1/2 direction
        markers stripped or reversed into the canonical 1->2 orientation.
        """
        if not isinstance(seq, str):
            return seq
        s = seq.strip()
        if "[GLM]" in s:
            # ProGen3 examples use directed prompts for infill:
            #   1<AA...>[GLM]s-e-L   (forward)
            #   2<AA...>[GLM]s-e-L   (reverse)
            # BatchPreparer expects 1->2 oriented GLM strings without leading 1/2.
            if s.startswith("1"):
                return s[1:]
            if s.startswith("2"):
                if get_spans_to_mask is None or prepare_glm_string_from_spans is None:
                    raise ImportError("GLM reverse prompt conversion requires progen3.batch_preparer helpers")
                body = s[1:]
                seq_body, spans = get_spans_to_mask(body)
                seq_rev = seq_body[::-1]
                spans_rev: Dict[Tuple[int, int], int] = {
                    (len(seq_rev) - e, len(seq_rev) - a): v for (a, e), v in spans.items()
                }
                return seq_rev + prepare_glm_string_from_spans(spans_rev)
            return s

        if any(marker in s for marker in self._structured_markers):
            return s
        if s.startswith("1") and s.endswith("2"):
            return s[1:-1]
        return s

    def _prepare_inputs(self, batch_seqs):
        formatted_batch = [self._format_sequence_for_progen(s) for s in batch_seqs]
        return self.batch_preparer.get_batch_kwargs(formatted_batch, device=self.device, reverse=False)

    def _valid_token_mask(self, input_ids):
        mask_BT = torch.ones_like(input_ids, dtype=torch.bool)
        for s_id in self.special_ids:
            mask_BT &= (input_ids != s_id)
        return mask_BT
    
    def clear_cache(self):
        """Public method to clear the collector's cache."""
        self.collector.clear_cache()

    def _progen_attention(self, layer, x_curr_BTH, x_gt_BTH, position_ids, freeze_attention, model_dtype):
        """Run ProGen3 attention+norm branch and return (x_mlp_in_BTH, residual_pre_mlp_BTH)."""
        # residual0 = hidden_states
        # attn_in = input_layernorm(hidden_states)
        # attn_out = self_attn(attn_in)
        # residual_pre_mlp = residual0 + attn_out
        # x_mlp_in = post_attention_layernorm(residual_pre_mlp)
        if freeze_attention:
            # Use ground-truth stream for attention branch only,
            # but keep residual path from the drifting stream.
            x_attn_in_BTH = x_gt_BTH.to(dtype=model_dtype)
            residual0_BTH = x_curr_BTH.to(dtype=model_dtype)
            if layer.fused_attention_norm:
                attn_in_BTH = layer.norm_attn_norm.input_layernorm(x_attn_in_BTH)
                attn_out_BTH, _, _ = layer.norm_attn_norm.self_attn(
                    hidden_states=attn_in_BTH,
                    position_ids=position_ids,
                    past_key_value=None,
                    output_attentions=False,
                    use_cache=False,
                )
                residual_pre_mlp_BTH = residual0_BTH + attn_out_BTH
                x_mlp_in_BTH = layer.norm_attn_norm.post_attention_layernorm(residual_pre_mlp_BTH)
            else:
                hidden_states = layer.input_layernorm(x_attn_in_BTH)
                attn_out_BTH, _, _ = layer.self_attn(
                    hidden_states=hidden_states,
                    position_ids=position_ids,
                    past_key_value=None,
                    output_attentions=False,
                    use_cache=False,
                )
                residual_pre_mlp_BTH = residual0_BTH + attn_out_BTH
                x_mlp_in_BTH = layer.post_attention_layernorm(residual_pre_mlp_BTH)
        else:
            x_attn_in_BTH = x_curr_BTH.to(dtype=model_dtype)
            if layer.fused_attention_norm:
                x_mlp_in_BTH, residual_pre_mlp_BTH, _, _ = layer.norm_attn_norm(
                    hidden_states=x_attn_in_BTH,
                    position_ids=position_ids,
                    past_key_value=None,
                    output_attentions=False,
                    use_cache=False,
                )
            else:
                residual0_BTH = x_attn_in_BTH
                hidden_states = layer.input_layernorm(residual0_BTH)
                attn_out_BTH, _, _ = layer.self_attn(
                    hidden_states=hidden_states,
                    position_ids=position_ids,
                    past_key_value=None,
                    output_attentions=False,
                    use_cache=False,
                )
                residual_pre_mlp_BTH = residual0_BTH + attn_out_BTH
                x_mlp_in_BTH = layer.post_attention_layernorm(residual_pre_mlp_BTH)

        return x_mlp_in_BTH, residual_pre_mlp_BTH

    def _run_plt_forward(self, x_stack, position_ids, active_nodes=None, retain_grad=False, freeze_attention=True, sequential=True, return_all_layer_recons=False):
        """
        PLT Sequential Forward Pass.
        x_stack: (B, L+1, T, H) - Contains embeddings through all layers
        Returns: (B, T, H), List of Latents
        """
        # sequential = True is a dummy variable for compatibility with CLT
        self.plt.eval()
        latents_list_L = []
        recon_layers = []
        B = position_ids.shape[0]
        T = position_ids.shape[1]

        # Pre-calculate which latents influence each layer
        node_masks = None
        if active_nodes is not None:
            node_masks = []
            for l in range(self.num_layers):
                m = torch.zeros(self.plt.d_hidden, device=self.device)
                if l in active_nodes and len(active_nodes[l]) > 0:
                    m[list(active_nodes[l])] = 1.0
                node_masks.append(m.view(1, -1))
        # Accept either flattened collector output (B*T, L+1, H) or reshaped input (B, L+1, T, H)
        if x_stack.ndim == 3:
            x_stack = x_stack.view(B, T, x_stack.shape[1], x_stack.shape[2]).permute(0, 2, 1, 3)
        H = x_stack.shape[-1]
        S = B * T

        # 1. PLT starts with Layer 0 - initialize current stream (BTH)
        x_curr_BTH = x_stack[:, 0, :, :]  # (B, T, H)
        model_dtype = next(self.progen.parameters()).dtype
        x_curr_BTH = x_curr_BTH.to(dtype=model_dtype)
        pos_ids_BT = position_ids
        
        for l in range(self.num_layers):
            layer = self.progen.model.layers[l]

            # 2. Get ProGen attention
            x_mlp_in_BTH, residual_pre_mlp_BTH = self._progen_attention(
                layer=layer,
                x_curr_BTH=x_curr_BTH,
                x_gt_BTH=x_stack[:, l, :, :],
                position_ids=pos_ids_BT,
                freeze_attention=freeze_attention,
                model_dtype=model_dtype,
            )

            # 3. Encode PLT replacement for MLP (residual)
            x_mlp_in_SH = x_mlp_in_BTH.reshape(S, H).to(dtype=torch.float32)  # (B*T, H)
            x_norm_SH, mu, std = self.plt.LN(x_mlp_in_SH)
            x_norm_SH = x_norm_SH - self.plt.b_pre[l]
            enc_SD = self.plt.encoders[l](x_norm_SH)
            latents_SD = self.plt.topK_activation(enc_SD, k=self.plt.k)
            if retain_grad:
                latents_SD.retain_grad()

            # 4. Apply Sparse Mask (Ablation)
            if node_masks is not None:
                latents_SD = latents_SD * node_masks[l]

            latents_list_L.append(latents_SD)

            # 5. Decode and denormalize
            recon_SH = (latents_SD @ self.plt.decoders[l]) + self.plt.b_pre[l]
            recon_SH = recon_SH * std + mu
            recon_BTH = recon_SH.view(B, T, H)
            if return_all_layer_recons:
                recon_layers.append(recon_BTH.to(dtype=torch.float32))

            # 6. Update stream to next layer: residual_pre_mlp + reconstruction
            x_curr_BTH = residual_pre_mlp_BTH.to(dtype=torch.float32) + recon_BTH
            x_curr_BTH = x_curr_BTH.to(dtype=model_dtype)

        x_curr_BTH = x_curr_BTH.to(dtype=torch.float32)
        recon_mlp_BTH = recon_BTH.to(dtype=torch.float32)

        if return_all_layer_recons:
            recon_layers_BLTH = torch.stack(recon_layers, dim=1)
            return x_curr_BTH, latents_list_L, recon_mlp_BTH, recon_layers_BLTH

        return x_curr_BTH, latents_list_L, recon_mlp_BTH

    def _get_reconstruction_with_layer_embedding(self, model_inputs, active_nodes=None, retain_grad=False, freeze_attention=True, source="mlp_output", sequential=True):
        """
        Runs PLT to get reconstruction of layer embedding.
        Overrides parent CLT method with PLT-specific logic.

        Args:
            source: 
                "mlp_output": Returns ONLY the reconstructed MLP component. (Default)
                "layer_output": Returns the full stream with final layer norm.
        """
        # sequential = True is a dummy variable for compatibility with CLT
        input_ids = model_inputs["input_ids"]

        # 0. Generate cache key
        token_bytes = input_ids.detach().cpu().numpy().tobytes()
        cache_key = hashlib.md5(token_bytes).hexdigest()
        
        # 1. Collect activations
        x_stack_SLH, _, x_mlp_out_SLH, x_clt_in_SLH, _ = self.collector.collect(model_inputs, cache_key=cache_key)
        self.collector.remove_hooks()
        
        if retain_grad:
            x_stack_SLH = x_stack_SLH.detach()
            x_mlp_out_SLH = x_mlp_out_SLH.detach()
            x_clt_in_SLH = x_clt_in_SLH.detach()
        # x_stack_SLH: (B*T, L+1, H)
        # x_mlp_out_SLH: (B*T, L, H)
        # x_clt_in_SLH: (B*T, L, H)
        B, T = input_ids.shape
        depth, H = x_stack_SLH.shape[1:]

        # Reshape: (B*T, L+1, H) -> (B, T, L+1, H) -> (B, L+1, T, H)
        x_stack_reshaped_BLTH = x_stack_SLH.view(B, T, depth, H).permute(0, 2, 1, 3)
        # 2. Run PLT forward
        recon_stream_BTH, latents_list_L, recon_mlp_BTH = self._run_plt_forward(
            x_stack_reshaped_BLTH, 
            position_ids=model_inputs["position_ids"],
            active_nodes=active_nodes, 
            retain_grad=retain_grad,
            freeze_attention=freeze_attention,
        ) # each entry of latents_list_L is (T, B, D)
        self.collector.register_hooks()
        
        # 3. Determine Output
        if source == "mlp_output":
            return recon_mlp_BTH, latents_list_L
        elif source == "layer_output":
            return self.progen.model.norm(recon_stream_BTH), latents_list_L
        else:
            raise ValueError(f"Unknown source: {source}")
        
    def reconstruct_logits(self, batch_seqs=None, model_inputs=None, active_nodes=None, sequential=False, freeze_attention=True):
        """Return reconstructed logits for CLT.

        Provide either batch_seqs or pre-built model_inputs.
        """
        if model_inputs is None:
            if batch_seqs is None:
                raise ValueError("Provide either batch_seqs or model_inputs")
            model_inputs = self._prepare_inputs(batch_seqs)

        with torch.no_grad():
            lm_dtype = self.progen.lm_head.weight.dtype
            seq_hidden_BTH, _ = self._get_reconstruction_with_layer_embedding(
                model_inputs,
                active_nodes=active_nodes,
                retain_grad=False,
                sequential=sequential,
                freeze_attention=freeze_attention,
                source="layer_output",
            )
            return self.progen.lm_head(seq_hidden_BTH.to(dtype=lm_dtype)).float()

    def get_gradients(
        self,
        batch_seqs,
        freeze_attention=True,
        source="layer_output",
        sequential=True,
        generated_mask_BT=None,
        zero_shot=False
    ):
        """
        Compute Gradient * Activation scores using KL(ProGen3 || reconstruction)
        over generated tokens only.
        """
        # sequential = True is a dummy variable for compatibility with CLT
        self.plt.zero_grad()
        self.progen.zero_grad()

        model_inputs = self._prepare_inputs(batch_seqs)
        labels_BT = model_inputs["labels"]

        if zero_shot:
            with torch.no_grad():
                base_out = self.progen(**model_inputs, return_dict=True)
                base_logits_BTV = base_out.logits.float()

            modified_emb_BTH, latents_list_L = self._get_reconstruction_with_layer_embedding(
                model_inputs,
                active_nodes=None,
                retain_grad=True,
                sequential=sequential,
                freeze_attention=freeze_attention,
                source=source,
            )
            self.collector.remove_hooks()

            lm_dtype = self.progen.lm_head.weight.dtype
            if source == "layer_output":
                logits_source_BTH = modified_emb_BTH
            elif source == "mlp_output":
                logits_source_BTH = self.progen.model.norm(modified_emb_BTH.to(dtype=lm_dtype))
            else:
                raise ValueError(f"Unknown source: {source}")

            logits_BTV = self.progen.lm_head(logits_source_BTH.to(dtype=lm_dtype)).float()
            shift_logits_BTV = logits_BTV[:, :-1, :].contiguous()
            shift_labels_BT = labels_BT[:, 1:].contiguous()
            valid_BT = shift_labels_BT != self.pad_id

            input_ids = model_inputs["input_ids"]
            B, T, _ = logits_BTV.shape

            valid_mask_BT = self._valid_token_mask(input_ids)
            loss_pos = valid_mask_BT.float()

            base_log_probs_BTV = F.log_softmax(base_logits_BTV, dim=-1)
            clt_log_probs_BTV = F.log_softmax(logits_BTV, dim=-1)

            kl_per_pos_BT = (base_log_probs_BTV.exp() * (base_log_probs_BTV - clt_log_probs_BTV)).sum(dim=-1)

            masked_kl_BT = kl_per_pos_BT * loss_pos
            loss = masked_kl_BT.sum() / loss_pos.sum().clamp(min=1)

        else:
            if generated_mask_BT is None:
                raise ValueError("generated_mask_BT is required and must mark generated token positions only")

            if not torch.is_tensor(generated_mask_BT):
                generated_mask_BT = torch.as_tensor(generated_mask_BT, device=labels_BT.device)
            generated_mask_BT = generated_mask_BT.to(device=labels_BT.device, dtype=torch.bool)
            if generated_mask_BT.shape != labels_BT.shape:
                raise ValueError(
                    f"generated_mask_BT must have shape {tuple(labels_BT.shape)}, got {tuple(generated_mask_BT.shape)}"
                )

            with torch.no_grad():
                baseline_logits_BTV = self.progen(**model_inputs, return_dict=True).logits.float().detach()

            modified_emb_BTH, latents_list_L = self._get_reconstruction_with_layer_embedding(
                model_inputs, retain_grad=True, freeze_attention=freeze_attention, source=source
            )
            self.collector.remove_hooks()

            lm_dtype = self.progen.lm_head.weight.dtype
            if source == "layer_output":
                logits_source_BTH = modified_emb_BTH
            elif source == "mlp_output":
                logits_source_BTH = self.progen.model.norm(modified_emb_BTH.to(dtype=lm_dtype))
            else:
                raise ValueError(f"Unknown source: {source}")

            logits_BTV = self.progen.lm_head(logits_source_BTH.to(dtype=lm_dtype)).float()
            shift_logits_BTV = logits_BTV[:, :-1, :].contiguous()
            shift_baseline_BTV = baseline_logits_BTV[:, :-1, :].contiguous()
            shift_labels_BT = labels_BT[:, 1:].contiguous()
            valid_BT = shift_labels_BT != self.pad_id

            generated_shift_mask_BT = generated_mask_BT[:, 1:].contiguous()
            active_mask_BT = valid_BT & generated_shift_mask_BT
            if active_mask_BT.sum().item() == 0:
                raise ValueError("generated_mask_BT selects zero valid shifted positions")

            recon_selected_NV = shift_logits_BTV[active_mask_BT]
            baseline_selected_NV = shift_baseline_BTV[active_mask_BT]
            loss = F.kl_div(
                F.log_softmax(recon_selected_NV, dim=-1),
                F.softmax(baseline_selected_NV, dim=-1),
                reduction="batchmean",
            )

        loss.backward()

        results = {}
        for l, latents in enumerate(latents_list_L):
            if latents.grad is not None:
                attr = torch.abs(latents * latents.grad)
                if attr.ndim == 3:
                    score = attr.sum(dim=(0, 1))
                elif attr.ndim == 2:
                    score = attr.sum(dim=0)
                else:
                    score = attr.sum()
                results[l] = score.detach().cpu().numpy()
        del latents_list_L
        del modified_emb_BTH
        self.collector.register_hooks()

        return results

    def run_ablation(
        self,
        batch_seqs,
        active_nodes=None,
        freeze_attention=True,
        source="layer_output",
        sequential=True,
        generated_mask_BT=None,
    ):
        """
        Evaluate an ablation by KL(ProGen3 || reconstructed logits) over generated positions.
        """
        # sequential = True is a dummy variable for compatibility with CLT
        with torch.no_grad():
            model_inputs = self._prepare_inputs(batch_seqs)
            labels_BT = model_inputs["labels"]

            if generated_mask_BT is None:
                generated_mask_BT = labels_BT != self.pad_id
            elif not torch.is_tensor(generated_mask_BT):
                generated_mask_BT = torch.as_tensor(generated_mask_BT, device=labels_BT.device)
            generated_mask_BT = generated_mask_BT.to(device=labels_BT.device, dtype=torch.bool)
            if generated_mask_BT.shape != labels_BT.shape:
                raise ValueError(
                    f"generated_mask_BT must have shape {tuple(labels_BT.shape)}, got {tuple(generated_mask_BT.shape)}"
                )

            baseline_logits_BTV = self.progen(**model_inputs, return_dict=True).logits.float().detach()
            modified_emb_BTH, _ = self._get_reconstruction_with_layer_embedding(
                model_inputs,
                active_nodes=active_nodes,
                freeze_attention=freeze_attention,
                source=source,
            )
            lm_dtype = self.progen.lm_head.weight.dtype
            if source == "layer_output":
                logits_source_BTH = modified_emb_BTH
            elif source == "mlp_output":
                logits_source_BTH = self.progen.model.norm(modified_emb_BTH.to(dtype=lm_dtype))
            else:
                raise ValueError(f"Unknown source: {source}")
            recon_logits_BTV = self.progen.lm_head(logits_source_BTH.to(dtype=lm_dtype)).float()

            shift_recon_BTV = recon_logits_BTV[:, :-1, :].contiguous()
            shift_base_BTV = baseline_logits_BTV[:, :-1, :].contiguous()
            shift_labels_BT = labels_BT[:, 1:].contiguous()
            valid_BT = shift_labels_BT != self.pad_id
            generated_shift_mask_BT = generated_mask_BT[:, 1:].contiguous()
            active_mask_BT = valid_BT & generated_shift_mask_BT
            if active_mask_BT.sum().item() == 0:
                raise ValueError("generated_mask_BT selects zero valid shifted positions")

            recon_selected_NV = shift_recon_BTV[active_mask_BT]
            base_selected_NV = shift_base_BTV[active_mask_BT]

            kl = F.kl_div(
                F.log_softmax(recon_selected_NV, dim=-1),
                F.softmax(base_selected_NV, dim=-1),
                reduction="batchmean",
            )
            mse = F.mse_loss(recon_selected_NV, base_selected_NV)
            nmse = mse / (torch.var(base_selected_NV) + 1e-8)
            top1 = torch.argmax(recon_selected_NV, dim=-1)
            top1_base = torch.argmax(base_selected_NV, dim=-1)
            top1_match = (top1 == top1_base).float().mean()

            return {
                "kl": float(kl.item()),
                "nmse": float(nmse.item()),
                "top1": float(top1_match.item()),
                "num_positions": int(active_mask_BT.sum().item()),
            }



"""
Testing code
"""
def compare_reconstruction_logit_paths(discoverer, batch_seqs=None, model_inputs=None, sequential=True, freeze_attention=True):
    """Compare reconstructed logits from the layer-output and direct MLP paths."""
    if model_inputs is None:
        if batch_seqs is None:
            raise ValueError("Provide either batch_seqs or model_inputs")
        model_inputs = discoverer._prepare_inputs(batch_seqs)

    with torch.no_grad():
        x_stack_SLH, _, _, _, _ = discoverer.collector.collect(model_inputs)
        seq_hidden_BTH, _, recon_mlp_BTH = discoverer._run_plt_forward(
            x_stack_SLH,
            position_ids=model_inputs["position_ids"],
            active_nodes=None,
            retain_grad=False,
            freeze_attention=freeze_attention,
        )

        lm_dtype = discoverer.progen.lm_head.weight.dtype
        logits_layer_BTV = discoverer.progen.lm_head(discoverer.progen.model.norm(seq_hidden_BTH.to(dtype=lm_dtype))).float()
        logits_mlp_BTV = discoverer.progen.lm_head(recon_mlp_BTH.to(dtype=lm_dtype)).float()

        diff = (logits_layer_BTV - logits_mlp_BTV).float()
        kl_layer_vs_mlp = F.kl_div(
            F.log_softmax(logits_layer_BTV, dim=-1),
            F.softmax(logits_mlp_BTV, dim=-1),
            reduction="batchmean",
        )
        return {
            "max_abs_diff": float(diff.abs().max().item()),
            "mean_abs_diff": float(diff.abs().mean().item()),
            "kl_layer_vs_mlp": float(kl_layer_vs_mlp.item()),
        }


def _compute_autoregressive_metrics(discoverer, prompt, num_new_tokens=3):
    current_sequence = prompt
    rows = []

    for step in range(num_new_tokens):
        with torch.no_grad():
            model_inputs = discoverer._prepare_inputs([current_sequence])
            input_ids = model_inputs["input_ids"]
            B, T = input_ids.shape

            gt_out = discoverer.progen(**model_inputs, return_dict=True)
            gt_logits_BTV = gt_out.logits.float()

            seq_logits_BTV = discoverer.reconstruct_logits(model_inputs=model_inputs, sequential=True, freeze_attention=True)
            direct_logits_BTV = discoverer.reconstruct_logits(model_inputs=model_inputs, sequential=False, freeze_attention=True)
            recon_mlp_seq_BTH, _ = discoverer._get_reconstruction_with_layer_embedding(
                model_inputs,
                active_nodes=None,
                retain_grad=False,
                sequential=True,
                freeze_attention=True,
                source="mlp_output",
            )
            recon_mlp_direct_BTH, _ = discoverer._get_reconstruction_with_layer_embedding(
                model_inputs,
                active_nodes=None,
                retain_grad=False,
                sequential=False,
                freeze_attention=True,
                source="mlp_output",
            )

            _, _, x_mlp_out_SLH, _, _ = discoverer.collector.collect(model_inputs)

            valid_mask_BT = discoverer._valid_token_mask(input_ids)
            last_content_index = int(valid_mask_BT[0].nonzero(as_tuple=False)[-1].item())

            gt_layer_mlp_BTH = x_mlp_out_SLH[:, -1, :].view(B, T, -1).float()
            gt_layer_mlp_last_BH = gt_layer_mlp_BTH[:, last_content_index, :]
            seq_layer_mlp_last_BH = recon_mlp_seq_BTH[:, last_content_index, :]
            direct_layer_mlp_last_BH = recon_mlp_direct_BTH[:, last_content_index, :]
            seq_mlp_mse_last = F.mse_loss(seq_layer_mlp_last_BH, gt_layer_mlp_last_BH)
            direct_mlp_mse_last = F.mse_loss(direct_layer_mlp_last_BH, gt_layer_mlp_last_BH)
            seq_nmse_mlp_lasttok = float((seq_mlp_mse_last / (torch.var(gt_layer_mlp_last_BH) + 1e-8)).item())
            direct_nmse_mlp_lasttok = float((direct_mlp_mse_last / (torch.var(gt_layer_mlp_last_BH) + 1e-8)).item())

            gt_layer_logits_BV = gt_logits_BTV[0, last_content_index, :].unsqueeze(0)
            seq_layer_logits_BV = seq_logits_BTV[:, last_content_index, :]
            direct_layer_logits_BV = direct_logits_BTV[:, last_content_index, :]
            seq_layer_kl = F.kl_div(
                F.log_softmax(gt_layer_logits_BV, dim=-1),
                F.softmax(seq_layer_logits_BV, dim=-1),
                reduction="batchmean",
            )
            direct_layer_kl = F.kl_div(
                F.log_softmax(gt_layer_logits_BV, dim=-1),
                F.softmax(direct_layer_logits_BV, dim=-1),
                reduction="batchmean",
            )

            gt_top = _top_token_stats(gt_layer_logits_BV, discoverer.tokenizer)
            seq_top = _top_token_stats(seq_layer_logits_BV, discoverer.tokenizer)
            direct_top = _top_token_stats(direct_layer_logits_BV, discoverer.tokenizer)
            gt_top1_id = gt_top["top1_id"]
            seq_top1_id = seq_top["top1_id"]
            direct_top1_id = direct_top["top1_id"]
            gt_top1_tok = gt_top["top1_tok"]
            seq_top1_tok = seq_top["top1_tok"]

        rows.append(
            {
                "step": step + 1,
                "seq_nmse_mlp_lasttok": seq_nmse_mlp_lasttok,
                "seq_kl_logits": float(seq_layer_kl.item()),
                "seq_top1_match": int(gt_top1_id == seq_top1_id),
                "direct_nmse_mlp_lasttok": direct_nmse_mlp_lasttok,
                "direct_kl_logits": float(direct_layer_kl.item()),
                "direct_top1_match": int(gt_top1_id == direct_top1_id),
            }
        )

        # Keep raw CLM format by appending the generated token to the sequence body.
        current_sequence = current_sequence + seq_top1_tok

        if gt_top1_tok in {"<eos>", "<eos_span>"}:
            break

    return rows


def _compute_autoregressive_metrics_glm(discoverer, directed_prompt, num_new_tokens=5):
    if get_spans_to_mask is None or prepare_glm_string_from_spans is None:
        raise ImportError("GLM autoregressive test requires progen3.batch_preparer span helpers")

    if not directed_prompt:
        raise ValueError("GLM prompt must not be empty")

    if directed_prompt[0] in {"1", "2"}:
        is_fwd = directed_prompt.startswith("1")
        body = directed_prompt[1:]
    else:
        is_fwd = True
        body = directed_prompt
    is_glm = "[GLM]" in body
    if not is_glm:
        prompt_12 = body if is_fwd else body[::-1]
        direction = "fwd" if is_fwd else "rev"
    elif is_fwd:
        prompt_12 = body
        direction = "fwd"
    else:
        seq_body, spans = get_spans_to_mask(body)
        seq_rev = seq_body[::-1]
        spans_rev: Dict[Tuple[int, int], int] = {
            (len(seq_rev) - e, len(seq_rev) - a): v for (a, e), v in spans.items()
        }
        prompt_12 = seq_rev + prepare_glm_string_from_spans(spans_rev)
        direction = "rev"

    reverse_sequence = direction == "rev"
    generation_inputs = discoverer.batch_preparer.get_generation_kwargs(prompt_12, reverse_sequence)
    current_inputs = {k: v.to(discoverer.device) for k, v in generation_inputs.items()}

    rows = []
    eos_span_id = discoverer.tokenizer.token_to_id("<eos_span>")

    for step in range(num_new_tokens):
        with torch.no_grad():
            input_ids = current_inputs["input_ids"]
            B, T = input_ids.shape

            gt_out = discoverer.progen(**current_inputs, return_dict=True)
            gt_logits_BTV = gt_out.logits.float()

            seq_logits_BTV = discoverer.reconstruct_logits(model_inputs=current_inputs, sequential=True, freeze_attention=True)
            direct_logits_BTV = discoverer.reconstruct_logits(model_inputs=current_inputs, sequential=False, freeze_attention=True)
            recon_mlp_seq_BTH, _ = discoverer._get_reconstruction_with_layer_embedding(
                current_inputs,
                active_nodes=None,
                retain_grad=False,
                sequential=True,
                freeze_attention=True,
                source="mlp_output",
            )
            recon_mlp_direct_BTH, _ = discoverer._get_reconstruction_with_layer_embedding(
                current_inputs,
                active_nodes=None,
                retain_grad=False,
                sequential=False,
                freeze_attention=True,
                source="mlp_output",
            )

            _, _, x_mlp_out_SLH, _, _ = discoverer.collector.collect(current_inputs)

            last_index = T - 1
            valid_mask_BT = discoverer._valid_token_mask(input_ids)
            gt_layer_mlp_BTH = x_mlp_out_SLH[:, -1, :].view(B, T, -1).float()
            gt_layer_mlp_last_BH = gt_layer_mlp_BTH[:, last_index, :]
            seq_layer_mlp_last_BH = recon_mlp_seq_BTH[:, last_index, :]
            direct_layer_mlp_last_BH = recon_mlp_direct_BTH[:, last_index, :]
            seq_mlp_mse_last = F.mse_loss(seq_layer_mlp_last_BH, gt_layer_mlp_last_BH)
            direct_mlp_mse_last = F.mse_loss(direct_layer_mlp_last_BH, gt_layer_mlp_last_BH)
            seq_nmse_mlp_lasttok = float((seq_mlp_mse_last / (torch.var(gt_layer_mlp_last_BH) + 1e-8)).item())
            direct_nmse_mlp_lasttok = float((direct_mlp_mse_last / (torch.var(gt_layer_mlp_last_BH) + 1e-8)).item())

            gt_last_logits_BV = gt_logits_BTV[:, last_index, :]
            seq_last_logits_BV = seq_logits_BTV[:, last_index, :]
            direct_last_logits_BV = direct_logits_BTV[:, last_index, :]
            seq_layer_kl = F.kl_div(
                F.log_softmax(gt_last_logits_BV, dim=-1),
                F.softmax(seq_last_logits_BV, dim=-1),
                reduction="batchmean",
            )
            direct_layer_kl = F.kl_div(
                F.log_softmax(gt_last_logits_BV, dim=-1),
                F.softmax(direct_last_logits_BV, dim=-1),
                reduction="batchmean",
            )

            gt_top = _top_token_stats(gt_last_logits_BV, discoverer.tokenizer)
            seq_top = _top_token_stats(seq_last_logits_BV, discoverer.tokenizer)
            direct_top = _top_token_stats(direct_last_logits_BV, discoverer.tokenizer)
            gt_top1_id = gt_top["top1_id"]
            seq_top1_id = seq_top["top1_id"]
            direct_top1_id = direct_top["top1_id"]
            gt_top1_tok = gt_top["top1_tok"]
            seq_top1_tok = seq_top["top1_tok"]

        rows.append(
            {
                "step": step + 1,
                "seq_nmse_mlp_lasttok": seq_nmse_mlp_lasttok,
                "seq_kl_logits": float(seq_layer_kl.item()),
                "seq_top1_match": int(gt_top1_id == seq_top1_id),
                "direct_nmse_mlp_lasttok": direct_nmse_mlp_lasttok,
                "direct_kl_logits": float(direct_layer_kl.item()),
                "direct_top1_match": int(gt_top1_id == direct_top1_id),
            }
        )

        next_id = torch.tensor([[seq_top1_id]], device=current_inputs["input_ids"].device, dtype=current_inputs["input_ids"].dtype)
        current_inputs["input_ids"] = torch.cat([current_inputs["input_ids"], next_id], dim=1)

        next_pos = current_inputs["position_ids"][:, -1:] + 1
        current_inputs["position_ids"] = torch.cat([current_inputs["position_ids"], next_pos], dim=1)

        next_seq = current_inputs["sequence_ids"][:, -1:]
        current_inputs["sequence_ids"] = torch.cat([current_inputs["sequence_ids"], next_seq], dim=1)

        if eos_span_id is not None and seq_top1_id == eos_span_id:
            break

    return rows


def _glm_prompt_to_clm_seed(directed_prompt: str) -> str:
    """Convert a directed GLM prompt to a plain 1->2 CLM seed for left-to-right diagnostics."""
    if "[GLM]" not in directed_prompt:
        return directed_prompt
    if not directed_prompt:
        raise ValueError("Directed GLM prompt must not be empty")

    if directed_prompt[0] in {"1", "2"}:
        is_fwd = directed_prompt.startswith("1")
        body = directed_prompt[1:]
    else:
        is_fwd = True
        body = directed_prompt
    seq_body = body.split("[GLM]", 1)[0]
    seq_12 = seq_body if is_fwd else seq_body[::-1]
    return seq_12


def _top_token_stats(logits_BV: torch.Tensor, tokenizer, top_k: int = 2) -> dict:
    probs_BV = F.softmax(logits_BV, dim=-1)
    top_probs, top_ids = torch.topk(probs_BV, k=top_k, dim=-1)
    top1_id = int(top_ids[0, 0].item())
    top2_id = int(top_ids[0, 1].item()) if top_k > 1 else top1_id
    return {
        "top1_id": top1_id,
        "top1_tok": tokenizer.id_to_token(top1_id),
        "top1_prob": float(top_probs[0, 0].item()),
        "top2_id": top2_id,
        "top2_tok": tokenizer.id_to_token(top2_id),
        "top2_prob": float(top_probs[0, 1].item()) if top_k > 1 else float(top_probs[0, 0].item()),
        "margin": float((top_probs[0, 0] - top_probs[0, 1]).item()) if top_k > 1 else 0.0,
    }


import argparse
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt",
        type=str,
        default="/path/to/models/ProGen3_PLT_L10_D4608/checkpoints/last.ckpt",
        help="Path to CLT checkpoint",
    )
    parser.add_argument(
        "--seq",
        type=str,
        default="MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQER[GLM]64-74-11",
        help="Protein sequence",
    )
    parser.add_argument("--gen-tokens", type=int, default=10, help="Number of autoregressive tokens to generate")
    parser.add_argument("--compare-paths", action="store_true", help="Print reconstructed logit path comparison")
    parser.add_argument("--device", type=str, default=None, help="cuda or cpu")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[main] device: {device}")
    print(f"[main] checkpoint: {args.ckpt}")
    print(f"[main] sequence (raw): {args.seq}")

    discoverer = CircuitDiscovererPLT(device=device, ckpt_path=args.ckpt)
    seq = discoverer._format_sequence_for_progen(args.seq)
    print(f"[main] sequence (formatted): {seq}")

    if args.compare_paths:
        compare_stats = compare_reconstruction_logit_paths(discoverer, batch_seqs=[args.seq], sequential=True, freeze_attention=True)
        print(
            f"[main] compare_paths | max_abs_diff={compare_stats['max_abs_diff']:.8f} | "
            f"mean_abs_diff={compare_stats['mean_abs_diff']:.8f} | kl_layer_vs_mlp={compare_stats['kl_layer_vs_mlp']:.8f}"
        )

    if "[GLM]" in args.seq:
        print("[main] === GLM Infill Autoregressive Table ===")
        rows_glm = _compute_autoregressive_metrics_glm(discoverer, args.seq, num_new_tokens=args.gen_tokens)
        print("[main] step | seq_nmse | seq_kl | seq_top1 | direct_nmse | direct_kl | direct_top1")
        for row in rows_glm:
            print(
                f"[main] {row['step']:>4} | {row['seq_nmse_mlp_lasttok']:.8f} | {row['seq_kl_logits']:.8f} | {row['seq_top1_match']} | "
                f"{row['direct_nmse_mlp_lasttok']:.8f} | {row['direct_kl_logits']:.8f} | {row['direct_top1_match']}"
            )

        clm_seed = discoverer._format_sequence_for_progen(_glm_prompt_to_clm_seed(args.seq))
        print("\n[main] === Left-to-Right CLM Table (same base sequence) ===")
        print(f"[main] clm_seed: {clm_seed[:32]}...{clm_seed[-8:]}")
        rows_clm = _compute_autoregressive_metrics(discoverer, clm_seed, num_new_tokens=args.gen_tokens)
        print("[main] step | seq_nmse | seq_kl | seq_top1 | direct_nmse | direct_kl | direct_top1")
        for row in rows_clm:
            print(
                f"[main] {row['step']:>4} | {row['seq_nmse_mlp_lasttok']:.8f} | {row['seq_kl_logits']:.8f} | {row['seq_top1_match']} | "
                f"{row['direct_nmse_mlp_lasttok']:.8f} | {row['direct_kl_logits']:.8f} | {row['direct_top1_match']}"
            )
    else:
        rows = _compute_autoregressive_metrics(discoverer, seq, num_new_tokens=args.gen_tokens)
        print("[main] step | seq_nmse | seq_kl | seq_top1 | direct_nmse | direct_kl | direct_top1")
        for row in rows:
            print(
                f"[main] {row['step']:>4} | {row['seq_nmse_mlp_lasttok']:.8f} | {row['seq_kl_logits']:.8f} | {row['seq_top1_match']} | "
                f"{row['direct_nmse_mlp_lasttok']:.8f} | {row['direct_kl_logits']:.8f} | {row['direct_top1_match']}"
            )


if __name__ == "__main__":
    main()