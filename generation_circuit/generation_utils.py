import random
import hashlib
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


AA_ALPHABET = set("ACDEFGHIKLMNPQRSTVWY")


@dataclass
class TaskSpec:
    task_type: str
    prompt: str
    prompt_12: str
    reverse_sequence: bool
    target_sequence: str
    max_steps: int
    span_start: int | None = None
    span_end: int | None = None


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_clm_task(sequence: str, start_fraction: float = 0.10) -> TaskSpec:
    """
    Create a CLM continuation task from a prefix of the sequence.
    """
    seq_len = len(sequence)
    start_idx = int(seq_len * start_fraction)
    start_idx = max(1, min(seq_len - 1, start_idx))

    prompt = sequence[:start_idx]
    target = sequence[start_idx:]

    return TaskSpec(
        task_type="CLM",
        prompt=prompt,
        prompt_12=prompt,
        reverse_sequence=False,
        target_sequence=target,
        max_steps=len(target),
    )


def generate_two_non_overlapping_spans(sequence: str, gaussians: Sequence[Tuple[float, float]], rng: np.random.Generator):
    """Generate two non-overlapping GLM spans. Returns ((start1, end1), (start2, end2)) or None."""
    seq_len = len(sequence)
    if seq_len < 4:
        return None

    mu, sigma = gaussians[int(rng.integers(0, len(gaussians)))]
    while True:
        len1 = int(round(rng.normal(mu, sigma)))
        if 1 <= len1 < seq_len:
            break
    max_start = seq_len - len1
    start1 = int(rng.integers(0, max_start + 1))
    end1 = start1 + len1

    valid_intervals = []
    if start1 > 0:
        valid_intervals.append((0, start1))
    if end1 < seq_len:
        valid_intervals.append((end1, seq_len))
    if not valid_intervals:
        return None

    mu, sigma = gaussians[int(rng.integers(0, len(gaussians)))]
    while True:
        len2 = int(round(rng.normal(mu, sigma)))
        if 1 <= len2 < seq_len:
            break

    valid_placements = [(a, b) for a, b in valid_intervals if b - a >= len2]
    if not valid_placements:
        return None

    interval_a, interval_b = valid_placements[int(rng.integers(0, len(valid_placements)))]
    max_start2 = interval_b - len2
    start2 = int(rng.integers(interval_a, max_start2 + 1))
    end2 = start2 + len2
    return ((start1, end1), (start2, end2))


def build_glm_task_with_span(sequence: str, span_start: int, span_end: int) -> TaskSpec:
    """Create a GLM infill task with explicit span coordinates."""
    span_len = span_end - span_start
    prompt_directed = f"1{sequence}[GLM]{span_start}-{span_end}-{span_len}"
    prompt_12 = f"{sequence}[GLM]{span_start}-{span_end}-{span_len}"
    return TaskSpec(
        task_type="GLM",
        prompt=prompt_directed,
        prompt_12=prompt_12,
        reverse_sequence=False,
        target_sequence=sequence[span_start:span_end],
        max_steps=span_len,
        span_start=span_start,
        span_end=span_end,
    )


def token_is_amino_acid(token: str) -> bool:
    """
    Return True when a tokenizer token is a single amino-acid residue.
    """
    return len(token) == 1 and token in AA_ALPHABET


def token_ids_to_aa_mask_and_tokens(tokenizer, token_ids: Sequence[int]) -> Tuple[List[bool], List[str]]:
    """
    Convert token ids into amino-acid masks and tokenizer strings.
    """
    aa_mask = []
    tok_texts = []
    for tid in token_ids:
        tok = tokenizer.id_to_token(int(tid))
        tok_texts.append(tok)
        aa_mask.append(token_is_amino_acid(tok))
    return aa_mask, tok_texts


def compute_top1_vs_baseline(baseline_ids: Sequence[int], recon_ids: Sequence[int], aa_mask: Sequence[bool]) -> float:
    """
    Compute token-level top-1 accuracy between baseline and reconstruction.
    """
    sel_idx = [i for i, is_aa in enumerate(aa_mask) if is_aa]
    if not sel_idx:
        sel_idx = list(range(min(len(baseline_ids), len(recon_ids))))
    if not sel_idx:
        return 0.0

    correct = 0
    for i in sel_idx:
        if i < len(baseline_ids) and i < len(recon_ids) and int(baseline_ids[i]) == int(recon_ids[i]):
            correct += 1
    return float(correct) / float(len(sel_idx))


def compute_kl_and_nmse(baseline_logits: torch.Tensor, recon_logits: torch.Tensor, aa_mask: Sequence[bool]) -> Tuple[float, float]:
    """
    Compute KL divergence and normalized MSE for reconstructed logits.
    """
    # baseline_logits/recon_logits: (S, V)
    if baseline_logits.numel() == 0 or recon_logits.numel() == 0:
        return 0.0, 0.0

    sel_idx = [i for i, is_aa in enumerate(aa_mask) if is_aa]
    if not sel_idx:
        sel_idx = list(range(min(baseline_logits.shape[0], recon_logits.shape[0])))
    if not sel_idx:
        return 0.0, 0.0

    b = baseline_logits[sel_idx]
    r = recon_logits[sel_idx]

    # Compute KL token-by-token over the selected positions, then average.
    log_p = F.log_softmax(r, dim=-1)
    p = F.softmax(b, dim=-1)
    kl_per_token = F.kl_div(log_p, p, reduction="none").sum(dim=-1)
    kl = kl_per_token.mean()
    mse = F.mse_loss(r, b)
    nmse = mse / (torch.var(b) + 1e-8)
    return float(kl.item()), float(nmse.item())


def compile_output_sequence(task: TaskSpec, generated_tokens: Sequence[str], original_sequence: str) -> str:
    """
    Rebuild the final protein sequence from generated amino-acid tokens.
    """
    generated_aa = "".join([t for t in generated_tokens if token_is_amino_acid(t)])
    if task.task_type == "CLM":
        return task.prompt + generated_aa

    assert task.span_start is not None and task.span_end is not None
    return original_sequence[: task.span_start] + generated_aa + original_sequence[task.span_end :]


def deterministic_seed_from_prompt(task_type: str, prompt_12: str, reverse_sequence: bool) -> int:
    key = f"{task_type}|{int(reverse_sequence)}|{prompt_12}".encode("utf-8")
    digest = hashlib.sha256(key).digest()
    return int.from_bytes(digest[:8], "little") % (2**32 - 1)


def set_batch_preparer_seed_from_prompt(discoverer, task_type: str, prompt_12: str, reverse_sequence: bool) -> int:
    seed = deterministic_seed_from_prompt(task_type, prompt_12, reverse_sequence)
    discoverer.batch_preparer.rng = np.random.default_rng(seed)
    return seed


def set_runtime_seed_from_prompt(task_type: str, prompt_12: str, reverse_sequence: bool) -> int:
    seed = deterministic_seed_from_prompt(task_type, prompt_12, reverse_sequence)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed


def repetitive_content_fraction(
    sequence: str,
    min_repetitive_span_len: int = 6,
) -> float:
    """
    Return fraction of positions in repetitive spans, a proxy for low-quality generations.

    Only repetitive spans with total span length >= min_repetitive_span_len are counted.
    """
    n = len(sequence)
    if n == 0:
        return 0.0

    flagged = [False] * n

    # Check every motif length so homopolymers, alternating runs, and larger
    # tandem repeats are handled by the same logic.
    for period in range(1, n // 2 + 1):
        i = 0
        while i + period <= n:
            motif = sequence[i : i + period]
            copies = 1
            j = i + period
            while j + period <= n and sequence[j : j + period] == motif:
                copies += 1
                j += period

            span_len = copies * period
            if copies >= 2 and span_len >= min_repetitive_span_len:
                for k in range(i, i + span_len):
                    flagged[k] = True

            i += 1

    return float(sum(flagged)) / float(n)


def passes_generation_quality(
    sequence: str,
    max_repetitive_fraction: float = 0.25,
    min_repetitive_span_len: int = 6,
) -> bool:
    """Return True when repetitive content remains below the configured threshold."""
    frac = repetitive_content_fraction(
        sequence,
        min_repetitive_span_len=min_repetitive_span_len,
    )
    return frac <= max_repetitive_fraction


def append_generation_step(current_inputs: Dict[str, torch.Tensor], token_id: int) -> Dict[str, torch.Tensor]:
    device = current_inputs["input_ids"].device
    dtype = current_inputs["input_ids"].dtype

    next_id = torch.empty((1, 1), device=device, dtype=dtype)
    next_id.fill_(token_id)
    current_inputs["input_ids"] = torch.cat([current_inputs["input_ids"], next_id], dim=1)

    next_pos = current_inputs["position_ids"][:, -1:] + 1
    current_inputs["position_ids"] = torch.cat([current_inputs["position_ids"], next_pos], dim=1)

    next_seq = current_inputs["sequence_ids"][:, -1:]
    current_inputs["sequence_ids"] = torch.cat([current_inputs["sequence_ids"], next_seq], dim=1)
    return current_inputs


def cap_sequence_to_reference(task: TaskSpec, candidate_sequence: str | None, reference_sequence: str, original_sequence: str) -> str | None:
    if candidate_sequence is None:
        return None

    if task.task_type == "CLM":
        prompt = task.prompt
        ref_generated_len = max(0, len(reference_sequence) - len(prompt))
        if candidate_sequence.startswith(prompt):
            return prompt + candidate_sequence[len(prompt) : len(prompt) + ref_generated_len]
        return candidate_sequence[: len(reference_sequence)]

    if task.task_type == "GLM" and task.span_start is not None and task.span_end is not None:
        replaced_len = task.span_end - task.span_start
        ref_insert_len = max(0, len(reference_sequence) - (len(original_sequence) - replaced_len))
        cand_insert_len = max(0, len(candidate_sequence) - (len(original_sequence) - replaced_len))
        candidate_insert = candidate_sequence[task.span_start : task.span_start + cand_insert_len]
        return (
            original_sequence[: task.span_start]
            + candidate_insert[:ref_insert_len]
            + original_sequence[task.span_end :]
        )

    return candidate_sequence[: len(reference_sequence)]


def top_p_sample(logits: torch.Tensor, p: float = 0.95, temperature: float = 1.0) -> int:
    """
    Sample from logits using top-p sampling with temperature.
    """
    logits = logits / temperature
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    sorted_probs = torch.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    # Find the smallest k such that cumulative >= p
    cutoff_mask = cumulative_probs >= p
    if not cutoff_mask.any():
        # If no token reaches p, take all
        cutoff = len(sorted_logits)
    else:
        cutoff = torch.where(cutoff_mask)[0][0] + 1
    # Set logits beyond cutoff to -inf
    filtered_logits = logits.clone()
    if cutoff < len(sorted_indices):
        filtered_logits[sorted_indices[cutoff:]] = float('-inf')
    # Sample
    probs = torch.softmax(filtered_logits, dim=-1)
    return torch.multinomial(probs, 1).item()
