import argparse
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import ProGen3ActivationCollector for direct replacement model
# Use absolute path based on script location
_script_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_script_dir)
_circuit_utils_dir = os.path.join(_parent_dir, 'circuit_utils')
_progen3_src_dir = os.path.join(_parent_dir, 'external', 'progen3', 'src')

for path in (_circuit_utils_dir, _progen3_src_dir):
    if path not in sys.path:
        sys.path.append(path)

try:
    from progen3_activation import ProGen3ActivationCollector
except ImportError:
    try:
        if _parent_dir not in sys.path:
            sys.path.append(_parent_dir)
        from circuit_utils.progen3_activation import ProGen3ActivationCollector
    except ImportError:
        ProGen3ActivationCollector = None


# ──────────────────────────────────────────────────────────────────────────────
# Shape suffixes:
# B: Batch Size
# L: Total number of LM layers
# T: Sequence length of protein (variable)
# D: CLT/PLT Latent dim (d_hidden)
# H: Embedding Dimension of LM (d_model)
# V: LM vocabulary size
# S: B * T
# ──────────────────────────────────────────────────────────────────────────────


class FullReplacementModel(nn.Module):
    '''
    Base class for replacement models that use encoder/decoders to replace MLP, as
    described in:
    https://transformer-circuits.pub/2025/attribution-graphs/methods.html#building-replacement

    '''
    def __init__(self, pl_module, device):
        super().__init__()
        self.pl_module = pl_module
        self.device = device

        self.model = pl_module.clt if hasattr(pl_module, 'clt') else pl_module.plt
        self.progen = getattr(pl_module, 'progen3_model', getattr(pl_module, 'progen', None))
        self.batch_preparer = getattr(pl_module, 'batch_preparer', None)
        self.tokenizer = getattr(self.batch_preparer, 'tokenizer', None)

        if self.progen is None:
            raise ValueError('pl_module must expose progen3_model or progen')
        if self.batch_preparer is None:
            raise ValueError('pl_module must expose batch_preparer')
        if self.tokenizer is None:
            raise ValueError('batch_preparer must expose tokenizer')

        self.vocab = self.tokenizer.get_vocab()
        self.pad_id = self.batch_preparer.pad_token_id

        special_token_names = [
            '<pad>', '<bos>', '<eos>', '<bos_glm>', '<eos_span>', '<mask>', '1', '2'
        ]
        for i in range(100):
            special_token_names.append(f'<span_{i}>')
        self.special_ids = {self.tokenizer.token_to_id(name) for name in special_token_names if self.tokenizer.token_to_id(name) is not None}

        if ProGen3ActivationCollector is not None:
            self.collector = ProGen3ActivationCollector(self.progen, self.vocab)
            self.collector.register_hooks()
        else:
            self.collector = None

        self.num_layers = self.model.num_layers

    def prepare_inputs(self, seqs):
        return self.batch_preparer.get_batch_kwargs(seqs, device=self.device)

    def tokenize(self, seqs):
        """
        Tokenize a list of sequence strings using the ProGen3 batch preparer.

        Returns:
            input_ids: Tensor of shape (B, T)
        """
        return self.prepare_inputs(seqs)['input_ids']

    def _valid_token_mask(self, input_ids):
        mask = torch.ones_like(input_ids, dtype=torch.bool)
        for s_id in self.special_ids:
            mask &= input_ids != s_id
        return mask

    def _encode_latents(self, l, x_mlp_in_TBH):
        """
        Encode MLP input to latents.

        Args:
            l: Current layer index
            x_mlp_in_TBH: MLP input after layer norm (T, B, H)

        Returns:
            latents_TBD: Encoded latents after topK (T, B, D)
            enc_TBD: Raw encoder activations before topK (T, B, D)
            mu: Mean from LayerNorm
            std: Std from LayerNorm
        """
        x_norm_TBH, mu, std = self.model.LN(x_mlp_in_TBH)
        x_norm_TBH = x_norm_TBH - self.model.b_pre[l]
        enc_TBD = self.model.encoders[l](x_norm_TBH)
        latents_TBD = self.model.topK_activation(enc_TBD, k=self.model.k)
        return latents_TBD, enc_TBD, mu, std

    def _progen_attention(self, layer, x_prev_TBH, x_gt_TBH, position_ids, model_dtype):
        """Run ProGen3 attention plus normalization and return MLP input and residual."""
        x_curr_BTH = x_prev_TBH.transpose(0, 1).to(dtype=model_dtype)
        x_gt_BTH = x_gt_TBH.transpose(0, 1).to(dtype=model_dtype) if x_gt_TBH is not None else None

        if x_gt_BTH is not None:
            x_attn_in_BTH = x_gt_BTH
            residual0_BTH = x_curr_BTH
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
            x_attn_in_BTH = x_curr_BTH
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

    def _decode_latents(self, l, current_latents_list):
        """
        Decode latents to reconstruct MLP output. Override this in subclasses.

        Args:
            l: Current layer index
            current_latents_list: List of latents from layers 0 to l (inclusive)

        Returns:
            recon_TBH: Reconstructed MLP output (T, B, H)
        """
        raise NotImplementedError("Subclasses must implement _decode_latents")

    def forward_steered(self, seq, circuit, before=False, ablate_nodes=None, alphas=None, freeze_attention=False, add_correction=True):
        """
        Steered Forward Pass with Constrained Patching and Per-Layer Correction.
        Ensures steering doesn't create downstream feedback loops in the encoders.
        """
        self.model.eval()
        if isinstance(alphas, (float, int)):
            alphas = [alphas]
        alphas = torch.as_tensor(alphas, device=self.device, dtype=torch.float32)
        if alphas.ndim == 0: alphas = alphas.unsqueeze(0)
        B = alphas.shape[0]

        # 1. BASELINE RECORDING & CALIBRATION
        # We need the GT inputs for every transcoder to perform constrained patching
        model_inputs = self.prepare_inputs([seq])
        with torch.no_grad():
            # Collect the "Clean" inputs that the encoders SHOULD see
            _, _, _, x_clt_input_stack_flat, _ = self.collector.collect(model_inputs)
            T_len = model_inputs['input_ids'].shape[1]
            H_dim = x_clt_input_stack_flat.shape[-1]
            # Shape: (T, 1, L, H)
            x_baseline_enc_inputs = x_clt_input_stack_flat.view(1, T_len, self.num_layers, H_dim).transpose(0, 1)

            # Determine per-layer corrections (GT - Recon) to ensure alpha=0 identity
            layer_corrections = []
            if add_correction:
                calib_latents = []
                # Temporary stream for calibration
                x_calib_TBH = (self.progen.model.embed_tokens(model_inputs['input_ids']) + 
                               self.progen.model.embed_seq_id(model_inputs['sequence_ids'])).transpose(0, 1).to(dtype=torch.float32)
                
                for l in range(self.num_layers):
                    # We pass alphas=0 here to get the unsteered reconstruction
                    x_out, lats_l, _, gt_l = self.layer_forward(
                        l, x_calib_TBH, calib_latents, 
                        position_ids=model_inputs['position_ids'], 
                        alphas=torch.zeros(1, device=self.device)
                    )
                    layer_corrections.append((gt_l - x_out).detach())
                    calib_latents.append(lats_l)
                    x_calib_TBH = gt_l # Keep calibration on the GT manifold

        # 2. STEERED PASS
        steer_inputs = self.prepare_inputs([seq] * B)
        x_curr_BTH = self.progen.model.embed_tokens(steer_inputs['input_ids']) + \
                     self.progen.model.embed_seq_id(steer_inputs['sequence_ids'])
        x_curr_TBH = x_curr_BTH.transpose(0, 1).to(dtype=torch.float32)
        
        latents_list_L = []
        mask_BT = self._valid_token_mask(steer_inputs['input_ids'])
        
        # GT stack for frozen attention if requested
        gt_stack_B = None
        if freeze_attention:
            gt_stack_all, _, _, _, _ = self.collector.collect(steer_inputs)
            gt_stack_B = gt_stack_all.view(B, T_len, self.num_layers + 1, H_dim)

        last_recon_TBH = None
        last_gt_TBH = None

        for l in range(self.num_layers):
            x_gt_l = gt_stack_B[:, :, l, :].transpose(0, 1) if freeze_attention else None
            l_corr = layer_corrections[l].expand(-1, B, -1) if add_correction else None
            # Extract this layer's "Clean" baseline input for the encoder
            x_base_enc_l = x_baseline_enc_inputs[:, :, l, :].expand(-1, B, -1)

            x_curr_TBH, latents_TBD, recon_TBH, gt_mlp_TBH = self.layer_forward(
                l, x_curr_TBH, latents_list_L,
                circuit=circuit, alphas=alphas, before=before,
                ablate_nodes=ablate_nodes.get(l) if ablate_nodes else None,
                x_gt=x_gt_l, position_ids=steer_inputs['position_ids'],
                correction_TBH=l_corr,
                x_enc_baseline_TBH=x_base_enc_l # PASS BASELINE TO ENCODER
            )
            latents_list_L.append(latents_TBD)
            last_recon_TBH = recon_TBH
            last_gt_TBH = gt_mlp_TBH

        return x_curr_TBH.transpose(0, 1), latents_list_L, last_recon_TBH.transpose(0, 1), last_gt_TBH.transpose(0, 1), mask_BT

    def layer_forward(self, l, x_prev_TBH, latents_list_L, x_gt=None, ablate_nodes=None, 
                      circuit=None, alphas=None, before=False, position_ids=None, 
                      padding_mask=None, correction_TBH=None, x_enc_baseline_TBH=None):
        """
        Modified layer_forward supporting Constrained Patching.
        """
        layer = self.progen.model.layers[l]
        model_dtype = next(self.progen.parameters()).dtype

        # 1. Attention Branch (Uses the drifting/evolved stream)
        x_mlp_in_BTH, residual_pre_mlp_BTH = self._progen_attention(
            layer=layer, x_prev_TBH=x_prev_TBH, x_gt_TBH=x_gt,
            position_ids=position_ids, model_dtype=model_dtype
        )
        x_mlp_in_TBH = x_mlp_in_BTH.transpose(0, 1)
        residual_TBH = residual_pre_mlp_BTH.transpose(0, 1)

        # 2. Ground Truth MLP
        with torch.no_grad():
            gt_mlp_delta = layer.block_sparse_moe(x_mlp_in_TBH.to(dtype=model_dtype))
            if isinstance(gt_mlp_delta, tuple): gt_mlp_delta = gt_mlp_delta[0]
            gt_full_TBH = residual_TBH + gt_mlp_delta.to(dtype=torch.float32)

        # 3. Transcoder Encoding (CONSTRAINED PATCHING)
        # We use the baseline input if provided, otherwise the evolved stream
        x_for_encoder = x_enc_baseline_TBH if x_enc_baseline_TBH is not None else x_mlp_in_TBH
        latents_TBD, enc_TBD, mu, std = self._encode_latents(l, x_for_encoder.to(dtype=torch.float32))

        # 4. Steering Logic (Node-specific robust scaling + Additive)
        if circuit is not None and l in circuit:
            node_indices = list(circuit[l])
            if len(node_indices) > 0:
                target_tensor = enc_TBD if before else latents_TBD
                # Flatten T and D to get scale per batch item B
                flattened = target_tensor.detach().float().permute(1, 0, 2).reshape(target_tensor.shape[1], -1)
                # current_scale = torch.quantile(flattened, 0.99, dim=1) # Shape (B,)
                current_scale = flattened.abs().max(dim=1).values + 1e-6 # Shape (B,)
                
                injection = (current_scale * alphas).view(1, -1, 1)
                # target_tensor[:, :, node_indices] += injection # ADDITIV
                # target_tensor[616:, :, node_indices] = target_tensor[616:, :, node_indices] * (1 + injection) # MULTIPLICATIVE
                target_tensor[:, :, node_indices] *= (1 + alphas)
                if before:
                    latents_TBD = self.model.topK_activation(target_tensor, k=self.model.k)

        if ablate_nodes is not None:
            node_indices = list(ablate_nodes)
            if len(node_indices) > 0:
                mask = torch.ones(latents_TBD.shape[-1], device=self.device)
                mask[node_indices] = 0.0
                latents_TBD = latents_TBD * mask.view(1, 1, -1)

        # 5. Decoding and Correction
        current_latents_list = latents_list_L + [latents_TBD]
        recon_delta_TBH = self._decode_latents(l, current_latents_list)
        recon_delta_TBH = (recon_delta_TBH + self.model.b_pre[l]) * std + mu
        
        x_curr_TBH = residual_TBH + recon_delta_TBH.to(dtype=torch.float32)
        if correction_TBH is not None:
            x_curr_TBH = x_curr_TBH + correction_TBH.to(dtype=torch.float32)

        return x_curr_TBH, latents_TBD, recon_delta_TBH, gt_full_TBH
        # current_latents_list = latents_list_L + [latents_TBD]
        # recon_TBH = self._decode_latents(l, current_latents_list)
        # recon_TBH = recon_TBH + self.model.b_pre[l]
        # recon_TBH = recon_TBH * std + mu

        # x_curr_TBH = residual.to(dtype=torch.float32) + recon_TBH.to(dtype=torch.float32)
        # gt_mlp_TBH = residual.to(dtype=torch.float32) + gt_mlp_TBH.to(dtype=torch.float32)

        # return x_curr_TBH, latents_TBD, recon_TBH, gt_mlp_TBH

    def _run_clt_sequential(self, x_stack, position_ids, active_nodes=None, retain_grad=False, freeze_attention=True):
        """Run the exact CLT sequential reconstruction as in clt_circuit.py."""
        self.model.eval()
        latents_list_L = []
        B = position_ids.shape[0]
        T = position_ids.shape[1]

        if x_stack.ndim == 3:
            x_stack = x_stack.view(B, T, x_stack.shape[1], x_stack.shape[2]).permute(0, 2, 1, 3)
        H = x_stack.shape[-1]
        S = B * T

        x_curr_BTH = x_stack[:, 0, :, :]
        model_dtype = next(self.progen.parameters()).dtype
        x_curr_BTH = x_curr_BTH.to(dtype=model_dtype)
        pos_ids_BT = position_ids

        for l in range(self.num_layers):
            layer = self.progen.model.layers[l]
            x_mlp_in_BTH, residual_pre_mlp_BTH = self._progen_attention(
                layer=layer,
                x_prev_TBH=x_curr_BTH.transpose(0, 1),
                x_gt_TBH=x_stack[:, l, :, :].transpose(0, 1),
                position_ids=pos_ids_BT,
                model_dtype=model_dtype,
            )

            x_mlp_in_SH = x_mlp_in_BTH.reshape(S, H).to(dtype=torch.float32)
            x_norm_SH, mu, std = self.model.LN(x_mlp_in_SH)
            x_norm_SH = x_norm_SH - self.model.b_pre[l]
            enc_SD = self.model.encoders[l](x_norm_SH)
            latents_SD = self.model.topK_activation(enc_SD, k=self.model.k)
            if retain_grad:
                latents_SD.retain_grad()

            if active_nodes is not None:
                node_masks = []
                for layer_idx in range(self.num_layers):
                    m = torch.zeros(self.model.d_hidden, device=self.device)
                    if layer_idx in active_nodes and len(active_nodes[layer_idx]) > 0:
                        m[list(active_nodes[layer_idx])] = 1.0
                    node_masks.append(m.view(1, -1))
                latents_SD = latents_SD * node_masks[l]

            latents_list_L.append(latents_SD)

            recon_SH = torch.zeros_like(x_norm_SH)
            for src in range(l + 1):
                key = f"{src}_{l}"
                if key in self.model.decoders:
                    recon_SH = recon_SH + (latents_list_L[src] @ self.model.decoders[key])
            recon_SH = recon_SH + self.model.b_pre[l]
            recon_SH = recon_SH * std + mu
            recon_BTH = recon_SH.view(B, T, H)

            x_curr_BTH = residual_pre_mlp_BTH.to(dtype=torch.float32) + recon_BTH
            x_curr_BTH = x_curr_BTH.to(dtype=model_dtype)

        x_curr_BTH = x_curr_BTH.to(dtype=torch.float32)
        recon_mlp_BTH = recon_BTH.to(dtype=torch.float32)
        return x_curr_BTH, latents_list_L, recon_mlp_BTH

    def compare_reconstruction_logit_paths(self, batch_seqs=None, model_inputs=None, sequential=True, freeze_attention=True):
        """Run the same CLT reconstruction logit path comparison as clt_circuit.py."""
        if model_inputs is None:
            if batch_seqs is None:
                raise ValueError("Provide either batch_seqs or model_inputs")
            model_inputs = self.prepare_inputs(batch_seqs)

        with torch.no_grad():
            x_stack_SLH, _, _, _, _ = self.collector.collect(model_inputs)
            seq_hidden_BTH, _, recon_mlp_BTH = self._run_clt_sequential(
                x_stack_SLH,
                position_ids=model_inputs["position_ids"],
                active_nodes=None,
                retain_grad=False,
                freeze_attention=freeze_attention,
            )

            lm_dtype = self.progen.lm_head.weight.dtype
            logits_layer_BTV = self.progen.lm_head(self.progen.model.norm(seq_hidden_BTH.to(dtype=lm_dtype))).float()
            logits_mlp_BTV = self.progen.lm_head(recon_mlp_BTH.to(dtype=lm_dtype)).float()

            diff = logits_layer_BTV - logits_mlp_BTV
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

    def forward(self, batch_seqs, ablate_nodes=None, freeze_attention=False, stop=None):
        """
        Sequential Forward Pass.
        """
        self.model.eval()
        latents_list_L = []
        if ablate_nodes is not None and not all(isinstance(k, int) for k in ablate_nodes.keys()):
            ablate_nodes = {int(k): v for k, v in ablate_nodes.items()}

        model_inputs = self.prepare_inputs(batch_seqs)
        tokens_BT = model_inputs['input_ids']
        sequence_ids = model_inputs['sequence_ids']
        position_ids = model_inputs['position_ids']

        x_curr_BTH = self.progen.model.embed_tokens(tokens_BT) + self.progen.model.embed_seq_id(sequence_ids)
        model_dtype = next(self.progen.parameters()).dtype
        x_curr_BTH = x_curr_BTH.to(model_dtype)

        mask_BT = self._valid_token_mask(tokens_BT)
        padding_mask = tokens_BT == self.pad_id

        x_curr_TBH = x_curr_BTH.transpose(0, 1)
        last_recon_TBH = None
        B, T = tokens_BT.shape
        H = x_curr_BTH.shape[-1]

        if freeze_attention:
            if self.collector is None:
                raise RuntimeError("freeze_attention requires ProGen3ActivationCollector but it's not available")
            x_stack_flat_SLH, _, _, x_clt_input_stack_flat_SLH, _ = self.collector.collect(model_inputs)
        else:
            x_stack_flat_SLH = None

        for l in range(self.num_layers):
            layer_ablate_nodes = None
            if ablate_nodes is not None and l in ablate_nodes:
                layer_ablate_nodes = ablate_nodes[l]

            x_gt_BTH = None
            if x_stack_flat_SLH is not None:
                x_gt_BTH = x_stack_flat_SLH.view(B, T, self.num_layers + 1, H)[:, :, l, :].transpose(0, 1)

            x_curr_TBH, latents_TBD, recon_TBH, gt_mlp_TBH = self.layer_forward(
                l,
                x_curr_TBH,
                latents_list_L,
                ablate_nodes=layer_ablate_nodes,
                x_gt=x_gt_BTH,
                position_ids=position_ids,
                padding_mask=padding_mask,
            )

            latents_list_L.append(latents_TBD)
            last_recon_TBH = recon_TBH
            if stop is not None and l == stop:
                break

        x_curr_BTH = x_curr_TBH.transpose(0, 1)
        recon_mlp_BTH = last_recon_TBH.transpose(0, 1)
        gt_mlp_BTH = gt_mlp_TBH.transpose(0, 1)

        return x_curr_BTH, latents_list_L, recon_mlp_BTH, gt_mlp_BTH, mask_BT




class FullCLTReplacementModel(FullReplacementModel):
    '''
    Cross-Layer Transformer (CLT) replacement model.
    Uses latents from ALL previous layers (0...l) to reconstruct layer l's MLP output.
    '''
    def __init__(self, pl_module, device):
        print("Initializing FullCLTReplacementModel")
        super().__init__(pl_module, device)
        self.clt = self.model

    def _decode_latents(self, l, current_latents_list):
        T, B, D = current_latents_list[l].shape
        recon_TBH = torch.zeros(T, B, self.progen.config.hidden_size, device=self.device)
        for src in range(l + 1):
            key = f"{src}_{l}"
            if key in self.model.decoders:
                recon_TBH = recon_TBH + (current_latents_list[src] @ self.model.decoders[key])
        return recon_TBH


class FullPLTReplacementModel(FullReplacementModel):
    '''
    Per-Layer Transformer (PLT) replacement model.
    Uses only the current layer's latents to reconstruct layer l's MLP output.
    '''
    def __init__(self, pl_module, device):
        super().__init__(pl_module, device)
        self.plt = self.model

    def _decode_latents(self, l, current_latents_list):
        return current_latents_list[l] @ self.model.decoders[l]


class FullCLTDirectReplacementModel(FullReplacementModel):
    '''
    CLT Direct replacement model that uses ground-truth MLP inputs at each layer.
    Unlike sequential processing, this encodes all layers in parallel using ground-truth
    inputs, then reconstructs only the final layer.

    This requires ProGen3ActivationCollector to gather activations from all layers.
    '''
    def __init__(self, pl_module, device):
        super().__init__(pl_module, device)
        self.clt = self.model
        if self.collector is None:
            raise ImportError("ProGen3ActivationCollector required for FullCLTDirectReplacementModel")

    def _decode_latents(self, l, current_latents_list):
        T, B, D = current_latents_list[l].shape
        recon_TBH = torch.zeros(T, B, self.progen.config.hidden_size, device=self.device)
        for src in range(l + 1):
            key = f"{src}_{l}"
            if key in self.model.decoders:
                recon_TBH = recon_TBH + (current_latents_list[src] @ self.model.decoders[key])
        return recon_TBH

    def forward(self, batch_seqs, ablate_nodes=None):
        self.model.eval()
        if ablate_nodes is not None and not all(isinstance(k, int) for k in ablate_nodes.keys()):
            ablate_nodes = {int(k): v for k, v in ablate_nodes.items()}

        model_inputs = self.prepare_inputs(batch_seqs)
        tokens_BT = model_inputs['input_ids']
        sequence_ids = model_inputs['sequence_ids']
        B, T = tokens_BT.shape

        x_stack_flat_SLH, x_mlp_input_stack_flat_SLH, x_mlp_stack_flat_SLH, x_clt_input_stack_flat_SLH, _ = self.collector.collect(model_inputs)

        x_clt_input_BTLH = x_clt_input_stack_flat_SLH.view(B, T, self.num_layers, -1)
        x_mlp_input_BTLH = x_mlp_input_stack_flat_SLH.view(B, T, self.num_layers, -1)

        latents_list_L = []
        for l in range(self.num_layers):
            x_mlp_in_TBH = x_mlp_input_BTLH[:, :, l, :].transpose(0, 1)
            latents_TBD, _, mu, std = self._encode_latents(l, x_mlp_in_TBH)
            if ablate_nodes is not None and l in ablate_nodes:
                node_indices = list(ablate_nodes[l])
                if len(node_indices) > 0:
                    mask = torch.ones(latents_TBD.shape[-1], device=self.device)
                    mask[node_indices] = 0.0
                    latents_TBD = latents_TBD * mask.view(1, 1, -1)
            latents_list_L.append(latents_TBD)

        recon_SH = self._decode_latents(self.num_layers - 1, latents_list_L)
        recon_SH = recon_SH + self.model.b_pre[-1]
        recon_SH = recon_SH * std + mu

        orig_layer_out_BTH = x_stack_flat_SLH[:, -1, :].view(B, T, -1)
        orig_mlp_BTH = x_mlp_stack_flat_SLH[:, -1, :].view(B, T, -1)
        recon_mlp_BTH = recon_SH.transpose(0, 1)

        modified_stream_BTH = orig_layer_out_BTH - orig_mlp_BTH + recon_mlp_BTH

        return modified_stream_BTH, latents_list_L, recon_mlp_BTH, orig_mlp_BTH, self._valid_token_mask(tokens_BT)


def nmse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor = None) -> float:
    if mask is not None:
        # Flatten and filter by mask
        pred = pred.transpose(0, 1)[mask]
        target = target.transpose(0, 1)[mask]
    
    mse = F.mse_loss(pred, target)
    var = torch.var(target)
    return float(mse / (var + 1e-8))


def main():
    parser = argparse.ArgumentParser(description="ProGen3 Full Replacement Model NMSE comparison")
    parser.add_argument("--seq", type=str, required=False, default="MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQER", help="Sequence to evaluate")
    parser.add_argument("--checkpoint", type=str, default="/path/to/models/ProGen3_CLT_L10_D4608/checkpoints/last.ckpt", help="CLT checkpoint path")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="PyTorch device")
    parser.add_argument("--compare-paths", action="store_true", help="Run CLT logit-path comparison diagnostic")
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"CLT checkpoint not found: {args.checkpoint}")

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    from training.clt_module import CLTLightningModule

    device = torch.device(args.device)
    print(f"Loading CLT checkpoint from {args.checkpoint} on device {device}")
    pl_module = CLTLightningModule.load_from_checkpoint(args.checkpoint, map_location=device)
    pl_module = pl_module.to(device)
    pl_module.eval()

    model = FullCLTReplacementModel(pl_module, device)
    seq = args.seq
    print(f"Evaluating sequence length {len(seq)}")

    model_inputs = model.prepare_inputs([seq])
    print("Collecting ProGen3 activations...")
    x_stack_flat_SLH, x_mlp_input_stack_flat_SLH, x_mlp_stack_flat_SLH, x_clt_input_stack_flat_SLH, mask_S = model.collector.collect(model_inputs)

    if args.compare_paths:
        print("Running CLT logit-path comparison diagnostic...")
        compare_stats = model.compare_reconstruction_logit_paths(batch_seqs=[seq], sequential=True, freeze_attention=True)
        print(
            f"  compare_paths | max_abs_diff={compare_stats['max_abs_diff']:.8f} | "
            f"mean_abs_diff={compare_stats['mean_abs_diff']:.8f} | kl_layer_vs_mlp={compare_stats['kl_layer_vs_mlp']:.8f}"
        )

    B, T = model_inputs['input_ids'].shape
    H = x_mlp_stack_flat_SLH.shape[-1]
    x_mlp_BTLH = x_mlp_stack_flat_SLH.view(B, T, model.num_layers, H)

    def run_layerwise(freeze_attention: bool) -> list[float]:
        x_curr_BTH = model.progen.model.embed_tokens(tokens_BT) + model.progen.model.embed_seq_id(sequence_ids)
        x_curr_BTH = x_curr_BTH.to(model_dtype)
        x_curr_TBH = x_curr_BTH.transpose(0, 1)
        latents_list_L = []
        nmse_values = []
        valid_mask = model._valid_token_mask(tokens_BT)

        for l in range(model.num_layers):
            x_gt_BTH = None
            if freeze_attention:
                x_gt_BTH = x_stack_flat_SLH.view(B, T, model.num_layers + 1, H)[:, :, l, :].transpose(0, 1)

            x_curr_TBH, latents_TBD, recon_TBH, gt_mlp_TBH = model.layer_forward(
                l,
                x_curr_TBH,
                latents_list_L,
                x_gt=x_gt_BTH,
                position_ids=position_ids,
                padding_mask=padding_mask,
            )

            target_mlp_TBH = x_mlp_BTLH[:, :, l, :].transpose(0, 1)
            nmse_val = nmse(recon_TBH, target_mlp_TBH, mask=valid_mask)
            nmse_values.append(nmse_val)
            nmse_values.append(nmse_val)
            latents_list_L.append(latents_TBD)

        return nmse_values

    print("Running CLT full replacement forward pass layer by layer...")
    tokens_BT = model_inputs['input_ids']
    sequence_ids = model_inputs['sequence_ids']
    model_dtype = next(model.progen.parameters()).dtype
    position_ids = model_inputs['position_ids']
    padding_mask = tokens_BT == model.pad_id

    print("  Frozen attention (ground truth attention) NMSEs:")
    frozen_nmses = run_layerwise(freeze_attention=True)
    for l, nmse_val in enumerate(frozen_nmses):
        print(f"    Layer {l}: frozen-attn NMSE = {nmse_val:.6e}")

    print("  Unfrozen attention (drifting attention) NMSEs:")
    unfrozen_nmses = run_layerwise(freeze_attention=False)
    for l, nmse_val in enumerate(unfrozen_nmses):
        print(f"    Layer {l}: unfrozen-attn NMSE = {nmse_val:.6e}")

    print(f"Frozen attention avg NMSE: {sum(frozen_nmses)/len(frozen_nmses):.6e}")
    print(f"Unfrozen attention avg NMSE: {sum(unfrozen_nmses)/len(unfrozen_nmses):.6e}")
    print("Done.")


if __name__ == "__main__":
    main()
