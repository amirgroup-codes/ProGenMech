import argparse
import math
import os
import sys

import torch
import torch.nn.functional as F

_script_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_script_dir)
_progen3_src_dir = os.path.join(_parent_dir, 'external', 'progen3', 'src')
for path in (_progen3_src_dir,):
    if path not in sys.path:
        sys.path.append(path)

from progen3.model.attention import repeat_kv
from full_replacement_models import FullCLTReplacementModel, FullReplacementModel


def masked_nmse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    assert pred.shape == target.shape
    assert mask.shape == pred.shape[:2]
    selected_pred = pred[mask]
    selected_target = target[mask]
    if selected_target.numel() == 0:
        return float('nan')
    mse = F.mse_loss(selected_pred, selected_target)
    var = torch.var(selected_target)
    return float(mse / (var + 1e-8))


def compute_attention_pattern(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    # q, k shape: (B, T, A, d)
    B, T, A, d = q.shape
    q_flat = q.permute(0, 2, 1, 3).reshape(B * A, T, d)
    k_flat = k.permute(0, 2, 1, 3).reshape(B * A, T, d)
    scale = 1.0 / math.sqrt(d)
    scores = torch.bmm(q_flat, k_flat.transpose(1, 2)) * scale
    causal_mask = torch.triu(torch.full((T, T), float('-inf'), device=scores.device, dtype=scores.dtype), diagonal=1)
    scores = scores + causal_mask.unsqueeze(0)
    probs = F.softmax(scores, dim=-1)
    return probs.view(B, A, T, T)


def apply_attention_pattern(attn_pattern: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    # attn_pattern: (B, A, T, T), values: (B, T, A, d)
    B, T, A, d = values.shape
    pattern_flat = attn_pattern.reshape(B * A, T, T)
    values_flat = values.permute(0, 2, 1, 3).reshape(B * A, T, d)
    out_flat = torch.bmm(pattern_flat, values_flat)
    out = out_flat.reshape(B, A, T, d).permute(0, 2, 1, 3)
    return out.reshape(B, T, A * d)


class LocalReplacementModel(FullCLTReplacementModel):
    """Local replacement model adapted to ProGen3.

    This class stores base prompt attention patterns and CLT error vectors,
    then reuses those patterns for new inputs while reconstructing MLP with
    the CLT replacement model.
    """

    def __init__(self, pl_module, device, base_prompt: str):
        super().__init__(pl_module, device)
        self.base_prompt = base_prompt
        self.base_attn_patterns = {}
        self.base_variances = {}
        self.base_error_vectors = {}
        self.compute_base_values()

    def _attention_input_layernorm(self, layer, x_BTH: torch.Tensor) -> torch.Tensor:
        if layer.fused_attention_norm:
            return layer.norm_attn_norm.input_layernorm(x_BTH)
        return layer.input_layernorm(x_BTH)

    def _post_attention_layernorm(self, layer, x_BTH: torch.Tensor) -> torch.Tensor:
        if layer.fused_attention_norm:
            return layer.norm_attn_norm.post_attention_layernorm(x_BTH)
        return layer.post_attention_layernorm(x_BTH)

    def _fixed_rms_norm(self, ln, x_BTH: torch.Tensor, var_BT: torch.Tensor) -> torch.Tensor:
        target_dtype = ln.weight.dtype
        x_BTH = x_BTH.to(dtype=target_dtype)
        eps = ln.variance_epsilon
        scale = ln.weight.view(1, 1, -1)
        fixed_std = torch.sqrt(var_BT.to(dtype=target_dtype).unsqueeze(-1) + eps)
        return x_BTH * scale / fixed_std

    def _fixed_attention_layernorm(self, layer, x_BTH: torch.Tensor, l: int) -> torch.Tensor:
        if layer.fused_attention_norm:
            ln = layer.norm_attn_norm.input_layernorm
        else:
            ln = layer.input_layernorm
        return self._fixed_rms_norm(ln, x_BTH, self.base_variances[('attn', l)])

    def _fixed_post_attention_layernorm(self, layer, x_BTH: torch.Tensor, l: int) -> torch.Tensor:
        if layer.fused_attention_norm:
            ln = layer.norm_attn_norm.post_attention_layernorm
        else:
            ln = layer.post_attention_layernorm
        return self._fixed_rms_norm(ln, x_BTH, self.base_variances[('mlp', l)])

    def _progen_attention(self, layer, x_prev_TBH, x_gt_TBH, position_ids, model_dtype, layer_idx):
        x_curr_BTH = x_prev_TBH.transpose(0, 1).to(dtype=model_dtype)

        if x_gt_TBH is not None:
            x_attn_in_BTH = x_gt_TBH.to(dtype=model_dtype)
        else:
            x_attn_in_BTH = self._fixed_attention_layernorm(layer, x_curr_BTH, layer_idx)

        q, k, v = self._compute_qkv(layer, x_attn_in_BTH, position_ids)
        attn_pattern = self.base_attn_patterns[layer_idx].unsqueeze(0).expand(x_curr_BTH.shape[0], -1, -1, -1)
        attn_out = apply_attention_pattern(attn_pattern, v)
        attn_out = layer.self_attn.o_proj(attn_out)

        residual_pre_mlp_BTH = x_curr_BTH + attn_out
        x_mlp_in_BTH = self._fixed_post_attention_layernorm(layer, residual_pre_mlp_BTH, layer_idx)
        return x_mlp_in_BTH, residual_pre_mlp_BTH

    def _compute_qkv(self, layer, x_ln_BTH: torch.Tensor, position_ids: torch.Tensor):
        target_dtype = layer.self_attn.q_proj.weight.dtype
        x_ln_BTH = x_ln_BTH.to(dtype=target_dtype)
        q, k, v = layer.self_attn.prepare_qkv(x_ln_BTH, position_ids)
        k = repeat_kv(k, layer.self_attn.num_heads // layer.self_attn.num_kv_heads)
        v = repeat_kv(v, layer.self_attn.num_heads // layer.self_attn.num_kv_heads)
        return q, k, v

    def compute_base_values(self):
        self.base_attn_patterns = {}
        self.base_variances = {}
        self.base_error_vectors = {}

        model_inputs = self.prepare_inputs([self.base_prompt])
        tokens_BT = model_inputs['input_ids']
        sequence_ids = model_inputs['sequence_ids']
        position_ids = model_inputs['position_ids']

        x_BTH = self.progen.model.embed_tokens(tokens_BT) + self.progen.model.embed_seq_id(sequence_ids)
        q_proj = self.progen.model.layers[0].self_attn.q_proj if not self.progen.model.layers[0].fused_attention_norm else self.progen.model.layers[0].norm_attn_norm.self_attn.q_proj
        x_BTH = x_BTH.to(q_proj.weight.dtype)

        x_curr_BTH = x_BTH
        latents_list = []
        B = tokens_BT.shape[0]

        for l in range(self.num_layers):
            layer = self.progen.model.layers[l]
            self.base_variances[('attn', l)] = x_curr_BTH.detach().pow(2).mean(dim=-1)
            x_attn_in_BTH = self._attention_input_layernorm(layer, x_curr_BTH)
            q, k, v = self._compute_qkv(layer, x_attn_in_BTH, position_ids)
            attn_probs = compute_attention_pattern(q, k)
            self.base_attn_patterns[l] = attn_probs[0].detach()

            attn_out = apply_attention_pattern(attn_probs, v)
            attn_out = layer.self_attn.o_proj(attn_out)
            x_after_attn_BTH = x_curr_BTH + attn_out

            self.base_variances[('mlp', l)] = x_after_attn_BTH.detach().pow(2).mean(dim=-1)
            x_mlp_in_BTH = self._fixed_post_attention_layernorm(layer, x_after_attn_BTH, l)
            model_dtype = next(self.progen.parameters()).dtype
            x_mlp_in_BTH = x_mlp_in_BTH.to(dtype=model_dtype)
            with torch.no_grad():
                mlp_out = layer.block_sparse_moe(x_mlp_in_BTH)
                if isinstance(mlp_out, tuple):
                    mlp_out = mlp_out[0]

            x_curr_BTH = x_after_attn_BTH + mlp_out

            x_mlp_in_TBH = x_mlp_in_BTH.transpose(0, 1)
            latents_TBD, _, mu, std = self._encode_latents(l, x_mlp_in_TBH)
            latents_list.append(latents_TBD)
            clt_recon_TBH = self._decode_latents(l, latents_list)
            clt_recon_TBH = clt_recon_TBH + self.model.b_pre[l]
            clt_recon_TBH = clt_recon_TBH * std + mu
            clt_recon_BTH = clt_recon_TBH.transpose(0, 1)
            self.base_error_vectors[l] = (mlp_out - clt_recon_BTH).squeeze(0).detach()

    def layer_forward(self, l, x_prev_TBH, latents_list_L, x_gt=None, ablate_nodes=None, circuit=None, alphas=None, before=False, position_ids=None, padding_mask=None):
        layer = self.progen.model.layers[l]
        model_dtype = next(self.progen.parameters()).dtype

        x_mlp_in_BTH, residual_pre_mlp_BTH = self._progen_attention(
            layer=layer,
            x_prev_TBH=x_prev_TBH,
            x_gt_TBH=None,
            position_ids=position_ids,
            model_dtype=model_dtype,
            layer_idx=l,
        )
        x_after_attn_BTH = residual_pre_mlp_BTH

        with torch.no_grad():
            gt_mlp_BTH = layer.block_sparse_moe(x_mlp_in_BTH)
            if isinstance(gt_mlp_BTH, tuple):
                gt_mlp_BTH = gt_mlp_BTH[0]

        x_mlp_in_TBH = x_mlp_in_BTH.transpose(0, 1)
        latents_TBD, enc_TBD, mu, std = self._encode_latents(l, x_mlp_in_TBH)

        if circuit is not None and l in circuit:
            assert alphas is not None and alphas.shape[0] == latents_TBD.shape[1]
            node_indices = list(circuit[l])
            if len(node_indices) > 0:
                target_tensor = enc_TBD if before else latents_TBD
                current_max = target_tensor.amax(dim=(0, 2))
                injection_values = (current_max * alphas).view(1, -1, 1)
                target_tensor[:, :, node_indices] = injection_values
            if before:
                latents_TBD = self.model.topK_activation(target_tensor, k=self.model.k)

        if ablate_nodes is not None:
            node_indices = list(ablate_nodes)
            if len(node_indices) > 0:
                mask = torch.ones(latents_TBD.shape[-1], device=self.device)
                mask[node_indices] = 0.0
                latents_TBD = latents_TBD * mask.view(1, 1, -1)

        current_latents_list = latents_list_L + [latents_TBD]
        recon_TBH = self._decode_latents(l, current_latents_list)
        recon_TBH = recon_TBH + self.model.b_pre[l]
        recon_TBH = recon_TBH * std + mu

        x_curr_TBH = x_after_attn_BTH.transpose(0, 1) + recon_TBH
        gt_mlp_TBH = (x_after_attn_BTH + gt_mlp_BTH).transpose(0, 1)

        error_TBH = self.base_error_vectors[l].unsqueeze(1).expand(-1, x_curr_TBH.shape[1], -1)
        x_curr_TBH = x_curr_TBH + error_TBH

        return x_curr_TBH, latents_TBD, recon_TBH, gt_mlp_TBH

    def zero_error(self):
        self.base_error_vectors = {l: torch.zeros_like(v) for l, v in self.base_error_vectors.items()}


def main():
    parser = argparse.ArgumentParser(description="ProGen3 Local Replacement Model test")
    parser.add_argument("--seq", type=str, default="MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQER", help="Sequence to evaluate")
    parser.add_argument("--checkpoint", type=str, default="/path/to/models/ProGen3_CLT_L10_D4608/checkpoints/last.ckpt", help="CLT checkpoint path")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="PyTorch device")
    parser.add_argument("--base-prompt", type=str, default=None, help="Base prompt for local replacement (defaults to seq)")
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"CLT checkpoint not found: {args.checkpoint}")

    if args.base_prompt is None:
        args.base_prompt = args.seq

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    from training.clt_module import CLTLightningModule

    device = torch.device(args.device)
    print(f"Loading CLT checkpoint from {args.checkpoint} on device {device}")
    pl_module = CLTLightningModule.load_from_checkpoint(args.checkpoint, map_location=device)
    pl_module = pl_module.to(device)
    pl_module.eval()

    full_model = FullCLTReplacementModel(pl_module, device)
    local_model = LocalReplacementModel(pl_module, device, base_prompt=args.base_prompt)

    print(f"Testing sequence length {len(args.seq)} with base prompt length {len(args.base_prompt)}")

    model_inputs = local_model.prepare_inputs([args.seq])
    tokens_BT = model_inputs['input_ids']
    valid_mask = local_model._valid_token_mask(tokens_BT)

    with torch.no_grad():
        gt_out = local_model.progen(
            **model_inputs,
            return_dict=True,
            output_hidden_states=True,
        )
        if getattr(gt_out, 'hidden_states', None) is not None:
            actual_final = gt_out.hidden_states[-1]
        elif hasattr(gt_out, 'last_hidden_state') and gt_out.last_hidden_state is not None:
            actual_final = gt_out.last_hidden_state
        else:
            raise RuntimeError('Progen output did not include hidden_states or last_hidden_state')

    local_out, _, _, _, _ = local_model.forward([args.seq])
    local_norm = local_model.progen.model.norm(local_out)
    if actual_final.shape != local_norm.shape:
        if isinstance(actual_final, torch.Tensor) and actual_final.dim() == 3:
            if actual_final.permute(1, 0, 2).shape == local_norm.shape:
                actual_final = actual_final.permute(1, 0, 2)
            elif actual_final.transpose(0, 1).shape == local_norm.shape:
                actual_final = actual_final.transpose(0, 1)
            else:
                raise ValueError(
                    f"Shape mismatch between local_norm {local_norm.shape} and actual_final {actual_final.shape}"
                )
        else:
            raise ValueError(
                f"Shape mismatch between local_norm {local_norm.shape} and actual_final {getattr(actual_final, 'shape', None)}"
            )
    nmse_actual = masked_nmse(local_norm, actual_final, valid_mask)
    print(f"local vs actual final hidden nmse (amino acids only): {nmse_actual:.6e}")

    print("Testing zero-error equivalence to full replacement model with frozen attention...")
    local_zero = LocalReplacementModel(pl_module, device, base_prompt=args.base_prompt)
    local_zero.zero_error()
    local_zero_out, _, _, _, _ = local_zero.forward([args.seq])
    full_out, _, _, _, _ = full_model.forward([args.seq], freeze_attention=True)
    diff = (local_zero_out - full_out).abs()
    print(f"zero-error max abs diff: {diff.max().item():.6e}")
    print(f"zero-error nmse (amino acids only) on pre-norm stream: {masked_nmse(local_zero_out, full_out, valid_mask):.6e}")

    print("Done.")


if __name__ == '__main__':
    main()
