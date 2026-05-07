import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

# Ensure repository imports work from this script location.
_repo_root = Path(__file__).resolve().parents[1]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from function_circuit.prepare_data import sample_csv
from external.progen3.src.progen3.scorer import ProGen3Scorer
from steering_patching_correction.full_replacement_models import FullCLTReplacementModel, FullPLTReplacementModel


def infer_wildtype_from_dms(csv_path: str) -> str:
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"DMS CSV is empty: {csv_path}")

    row = df.iloc[0]
    mutant = str(row.get("mutant", ""))
    mutated_sequence = str(row.get("mutated_sequence", ""))
    if not mutated_sequence:
        raise ValueError(f"DMS CSV missing mutated_sequence in first row: {csv_path}")

    sequence = list(mutated_sequence)
    if mutant and mutant != "nan":
        for part in mutant.split(":"):
            part = part.strip()
            if len(part) < 2:
                continue
            aa = part[0]
            idx_str = "".join([c for c in part if c.isdigit()])
            if not idx_str:
                continue
            idx = int(idx_str) - 1
            if 0 <= idx < len(sequence):
                sequence[idx] = aa

    return "".join(sequence)


def load_dms_samples(csv_path: str, num_train_seq: int = 128, num_test_seq: int = 500, seed: int = 42, single_only: bool = False):
    train_json, test_json = sample_csv(csv_path, num_train_seq, num_test_seq, seed, single_only=single_only)
    train_sequences = [entry["mutated_sequence"] for entry in train_json["sequences"]]
    train_scores = [entry["DMS_score"] for entry in train_json["sequences"]]
    test_sequences = [entry["mutated_sequence"] for entry in test_json["sequences"]]
    test_scores = [entry["DMS_score"] for entry in test_json["sequences"]]
    return train_sequences, train_scores, test_sequences, test_scores


def load_circuit_json(json_path: str) -> dict:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "nodes" not in data:
        raise ValueError(f"Circuit JSON missing 'nodes': {json_path}")
    return {int(layer): latents for layer, latents in data["nodes"].items()}


def sample_latent_circuit_from_json(json_path: str, num_latents: int, seed: int | None = None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    nodes = load_circuit_json(json_path)
    candidates = []
    for layer, latents in nodes.items():
        for latent_idx in latents:
            candidates.append({"layer": layer, "idx": int(latent_idx)})

    if not candidates:
        return {}

    num_latents = min(num_latents, len(candidates))
    chosen_indices = np.random.choice(len(candidates), size=num_latents, replace=False)

    active_nodes = {}
    for idx in chosen_indices:
        item = candidates[int(idx)]
        active_nodes.setdefault(item["layer"], []).append(item["idx"])

    return active_nodes


def sample_latent_circuit(gradient_items, num_latents, top_k=1000, seed: int | None = None, replace: bool = False):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    if not gradient_items:
        return {}

    gradient_items = sorted(gradient_items, key=lambda x: x["score"], reverse=True)
    candidates = gradient_items[: min(len(gradient_items), top_k)]

    scores = np.array([max(0.0, item["score"]) + 1e-8 for item in candidates], dtype=np.float64)
    if scores.sum() <= 0.0:
        scores = np.ones_like(scores)
    probs = scores / scores.sum()

    if replace:
        chosen_indices = np.random.choice(len(candidates), size=num_latents, replace=True, p=probs)
    else:
        num_latents = min(num_latents, len(candidates))
        chosen_indices = np.random.choice(len(candidates), size=num_latents, replace=False, p=probs)

    active_nodes = {}
    for idx in chosen_indices:
        item = candidates[int(idx)]
        active_nodes.setdefault(item["layer"], []).append(item["idx"])

    return active_nodes


def top_p_sample(logits: torch.Tensor, p: float = 0.95, temperature: float = 1.0) -> int:
    logits = logits / temperature
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    sorted_probs = torch.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    cutoff = int((cumulative_probs >= p).nonzero(as_tuple=True)[0][0].item() + 1) if (cumulative_probs >= p).any() else len(sorted_logits)
    filtered_logits = logits.clone()
    if cutoff < len(sorted_indices):
        filtered_logits[sorted_indices[cutoff:]] = float("-inf")
    probs = torch.softmax(filtered_logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).item()


def make_special_token_mask(tokenizer):
    special_tokens = ["<pad>", "<bos>", "<eos>", "<mask>", "<eos_span>", "<bos_glm>"]
    token_ids = {tokenizer.token_to_id(tok) for tok in special_tokens if tokenizer.token_to_id(tok) is not None}
    return token_ids


def generate_baseline_suffix(prefix: str, suffix_len: int, progen_model, batch_preparer, tokenizer, special_ids, top_p: float, temperature: float, device: str):
    current = prefix
    generated = []
    for _ in range(suffix_len):
        model_inputs = batch_preparer.get_batch_kwargs([current], device=device, reverse=False)
        with torch.no_grad():
            output = progen_model(**model_inputs, return_dict=True)
        logits = output.logits[0, -1, :]
        logits = logits.clone()
        for token_id in special_ids:
            if 0 <= token_id < logits.shape[-1]:
                logits[token_id] = float("-inf")
        next_id = top_p_sample(logits, p=top_p, temperature=temperature)
        next_token = tokenizer.id_to_token(next_id)
        if next_token in ["<eos>", "<pad>", "<mask>"]:
            break
        current += next_token
        generated.append(next_token)
    return current


def generate_replacement_suffix(prefix: str, suffix_len: int, replacement_model, circuit, alpha: float, batch_preparer, tokenizer, special_ids, top_p: float, temperature: float, device: str, freeze_attention: bool, before: bool):
    current = prefix
    generated = []
    for _step in range(suffix_len):
        with torch.no_grad():
            emb_batch, _, _, _, _ = replacement_model.forward_steered(
                current,
                circuit,
                before=before,
                alphas=[alpha],
                freeze_attention=freeze_attention,
                add_correction=True,
            )
            lm_dtype = replacement_model.progen.lm_head.weight.dtype
            hidden = replacement_model.progen.model.norm(emb_batch.to(dtype=lm_dtype))
            logits_BTV = replacement_model.progen.lm_head(hidden)
        logits = logits_BTV[0, -1, :].clone()
        for token_id in special_ids:
            if 0 <= token_id < logits.shape[-1]:
                logits[token_id] = float("-inf")
        next_id = top_p_sample(logits, p=top_p, temperature=temperature)
        next_token = tokenizer.id_to_token(next_id)
        if next_token in ["<eos>", "<pad>", "<mask>"]:
            break
        current += next_token
        generated.append(next_token)
    return current


