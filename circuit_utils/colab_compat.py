"""Colab / legacy-GPU compatibility for ProGen3 without modifying external/progen3.

On GPUs below sm_80 (e.g. Colab T4), megablocks MoE and flash-attn Triton kernels
require bf16 paths that are unsupported. This module:
  1. Patches RMSNorm to a pure-PyTorch implementation (no Triton).
  2. Patches attention to avoid the FLASH_ATTENTION SDPA backend.
  3. Reloads ProGen3 with ``moe_implementation="eager"`` (no megablocks/Triton MoE).

Set PROGEN3_COLAB_COMPAT=0 to disable auto-detection on legacy GPUs.
"""

from __future__ import annotations

import os
import sys

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


def needs_colab_compat(device=None) -> bool:
    override = os.environ.get("PROGEN3_COLAB_COMPAT", "").lower().strip()
    if override in ("0", "false", "no"):
        return False
    if override in ("1", "true", "yes"):
        return True
    if not torch.cuda.is_available():
        return False
    dev = device if device is not None else torch.device("cuda", torch.cuda.current_device())
    return torch.cuda.get_device_capability(dev)[0] < 8


def _modeling_modules():
    """Return all loaded progen3.modeling modules (editable install vs repo path)."""
    seen = set()
    modules = []
    for name, mod in list(sys.modules.items()):
        if name == "progen3.modeling" or name.endswith(".progen3.modeling"):
            if id(mod) not in seen:
                seen.add(id(mod))
                modules.append(mod)
    if not modules:
        for import_path in (
            "external.progen3.src.progen3.modeling",
            "progen3.modeling",
        ):
            try:
                mod = __import__(import_path, fromlist=["RMSNorm"])
                if id(mod) not in seen:
                    seen.add(id(mod))
                    modules.append(mod)
            except ImportError:
                pass
    return modules


def _attention_modules():
    seen = set()
    modules = []
    for name, mod in list(sys.modules.items()):
        if name == "progen3.model.attention" or name.endswith(".progen3.model.attention"):
            if id(mod) not in seen:
                seen.add(id(mod))
                modules.append(mod)
    if not modules:
        for import_path in (
            "external.progen3.src.progen3.model.attention",
            "progen3.model.attention",
        ):
            try:
                mod = __import__(import_path, fromlist=["Attention"])
                if id(mod) not in seen:
                    seen.add(id(mod))
                    modules.append(mod)
            except ImportError:
                pass
    return modules


def _patch_rmsnorm() -> None:
    def pytorch_rmsnorm_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        x = hidden_states.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * x).to(input_dtype)

    for pg_modeling in _modeling_modules():
        if getattr(pg_modeling.RMSNorm, "_colab_compat_patched", False):
            continue
        pg_modeling.RMSNorm.forward = pytorch_rmsnorm_forward
        pg_modeling.RMSNorm._colab_compat_patched = True


def _patch_attention() -> None:
    causal_lower_right = repeat_kv = None
    for import_path in (
        "external.progen3.src.progen3.model.attention",
        "progen3.model.attention",
    ):
        try:
            mod = __import__(import_path, fromlist=["causal_lower_right", "repeat_kv"])
            causal_lower_right = mod.causal_lower_right
            repeat_kv = mod.repeat_kv
            break
        except ImportError:
            continue
    if causal_lower_right is None or repeat_kv is None:
        raise ImportError("Could not import progen3 attention utils for colab compat")

    def _sdpa_attn_compat(
        self, query_states: torch.Tensor, key_states: torch.Tensor, val_states: torch.Tensor
    ) -> tuple[torch.Tensor, None]:
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        val_states = val_states.transpose(1, 2)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        val_states = repeat_kv(val_states, self.num_key_value_groups)

        bsz, q_len = query_states.shape[0], query_states.shape[2]
        k_len = key_states.shape[2]

        causal_mask = None
        if k_len > q_len:
            causal_mask = causal_lower_right(q_len, k_len)
        elif k_len < q_len:
            raise ValueError("k_len must be greater than or equal to q_len")

        with sdpa_kernel(backends=[SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
            attn_output = F.scaled_dot_product_attention(
                query_states,
                key_states,
                val_states,
                is_causal=causal_mask is None,
                attn_mask=causal_mask,
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        return attn_output, None

    for pg_attention in _attention_modules():
        if getattr(pg_attention.Attention, "_colab_compat_patched", False):
            continue
        pg_attention.Attention._sdpa_attn = _sdpa_attn_compat
        pg_attention.Attention._colab_compat_patched = True


def load_clt_pl_module(ckpt_path, device):
    """Load CLTLightningModule and apply Colab/legacy-GPU compat when needed."""
    from training.clt_module import CLTLightningModule

    pl_module = CLTLightningModule.load_from_checkpoint(ckpt_path, map_location=device)
    apply_colab_compat(pl_module, device)
    return pl_module


def megablocks_to_eager_state_dict(state_dict: dict, config) -> dict:
    """Convert megablocks grouped MoE weights to eager per-expert ModuleList layout."""
    ffn_dim = config.intermediate_size
    n_experts = config.num_experts
    weight_map = {"w1": "w1.weight", "w2": "w2.weight", "v1": "w3.weight"}

    eager_sd = {}
    for key, tensor in state_dict.items():
        if ".block_sparse_moe.experts.mlp." in key:
            continue
        if key.endswith(".block_sparse_moe.router.layer.weight"):
            eager_sd[key.replace("router.layer.weight", "gate.weight")] = tensor
            continue
        eager_sd[key] = tensor

    for key, tensor in state_dict.items():
        if ".block_sparse_moe.experts.mlp." not in key:
            continue
        prefix, weight_name = key.rsplit(".", 1)
        if weight_name not in weight_map:
            continue
        layer_prefix = prefix.replace(".experts.mlp", "")
        for expert_idx in range(n_experts):
            chunk = tensor[expert_idx * ffn_dim : (expert_idx + 1) * ffn_dim]
            if weight_name == "w2":
                chunk = chunk.t().contiguous()
            eager_key = f"{layer_prefix}.experts.{expert_idx}.{weight_map[weight_name]}"
            eager_sd[eager_key] = chunk

    return eager_sd


def reload_progen3_eager(pl_module, device) -> None:
    """Replace megablocks ProGen3 with eager-MoE using converted checkpoint weights."""
    from external.progen3.src.progen3.config import ProGen3Config
    from external.progen3.src.progen3.modeling import ProGen3ForCausalLM
    from training.clt_module import ProGen3ActivationCollector

    megablocks_sd = pl_module.progen3_model.state_dict()
    config = ProGen3Config.from_pretrained(pl_module.args.model)
    config.moe_implementation = "eager"
    config.moe_grouped_gemm = False
    config.moe_memory_optimized = False

    eager_sd = megablocks_to_eager_state_dict(megablocks_sd, config)
    new_model = ProGen3ForCausalLM(config)
    missing, unexpected = new_model.load_state_dict(eager_sd, strict=False)
    moe_missing = [k for k in missing if "block_sparse_moe" in k]
    if moe_missing:
        raise RuntimeError(f"Failed to convert megablocks MoE weights for Colab compat: {moe_missing[:5]}")

    new_model = new_model.to(dtype=torch.bfloat16).eval()
    for param in new_model.parameters():
        param.requires_grad = False

    pl_module.progen3_model = new_model.to(device)

    if getattr(pl_module, "collector", None) is not None:
        pl_module.collector.remove_hooks()
    pl_module.collector = ProGen3ActivationCollector(
        pl_module.progen3_model,
        pl_module.tokenizer.get_vocab(),
    )
    pl_module.collector.register_hooks()


def apply_colab_compat(pl_module, device) -> bool:
    """Apply legacy-GPU compatibility patches to a loaded CLTLightningModule."""
    if not needs_colab_compat(device):
        return False

    _patch_rmsnorm()
    _patch_attention()
    reload_progen3_eager(pl_module, device)
    return True
