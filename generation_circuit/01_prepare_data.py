#!/usr/bin/env python3
"""
Prepare curated CLM/GLM generation dataset before circuit discovery.
"""
import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
from circuit_utils.clt_circuit import CircuitDiscovererCLT
from generation_circuit.generation_utils import (
    build_clm_task,
    build_glm_task_with_span,
    compile_output_sequence,
    generate_two_non_overlapping_spans,
    passes_generation_quality,
    set_global_seed,
    set_batch_preparer_seed_from_prompt,
    set_runtime_seed_from_prompt,
    token_ids_to_aa_mask_and_tokens,
    top_p_sample
)

def append_generation_step(current_inputs: Dict[str, torch.Tensor], token_id: int) -> Dict[str, torch.Tensor]:
    """
    Append one generated token to the model inputs and advance positions.
    """
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


def generate_progen_sequence(discoverer: CircuitDiscovererCLT, task, allow_eos: bool = True, p: float = 0.95, temperature: float = 0.85) -> str:
    """
    Run ProGen3 autoregressively for a single prepared CLM/GLM task.
    """
    tokenizer = discoverer.tokenizer
    discoverer.progen.eval()
    set_runtime_seed_from_prompt(task.task_type, task.prompt_12, task.reverse_sequence)
    set_batch_preparer_seed_from_prompt(discoverer, task.task_type, task.prompt_12, task.reverse_sequence)
    current_inputs = discoverer.batch_preparer.get_generation_kwargs(task.prompt_12, task.reverse_sequence)
    current_inputs = {
        k: (v if str(v.device) == str(discoverer.device) else v.to(discoverer.device, non_blocking=True))
        for k, v in current_inputs.items()
    }
    eos_token = "<eos_span>" if task.task_type == "GLM" else "<eos>"
    eos_token_id = tokenizer.token_to_id(eos_token)

    generated_ids: List[int] = []
    with torch.inference_mode():
        for _ in range(task.max_steps):
            out = discoverer.progen(**current_inputs, return_dict=True)
            logits_last = out.logits[:, -1, :].squeeze(0)
            top1_id = top_p_sample(logits_last, p=p, temperature=temperature)
            generated_ids.append(top1_id)
            if allow_eos and eos_token_id is not None and top1_id == eos_token_id:
                break

            current_inputs = append_generation_step(current_inputs, top1_id)

    _, gen_tokens = token_ids_to_aa_mask_and_tokens(tokenizer, generated_ids)
    return compile_output_sequence(task, gen_tokens, task.prompt_12.split("[GLM]", 1)[0] if task.task_type == "GLM" else task.prompt)


def extract_generated_portion(task, progen3_sequence: str) -> str:
    """Return only the model-generated segment for quality checks."""
    if task.task_type == "CLM":
        return progen3_sequence[len(task.prompt) :]

    if task.span_start is None or task.span_end is None:
        return ""

    original_sequence = task.prompt_12.split("[GLM]", 1)[0]
    replaced_len = task.span_end - task.span_start
    generated_len = len(progen3_sequence) - (len(original_sequence) - replaced_len)
    generated_len = max(0, generated_len)
    return progen3_sequence[task.span_start : task.span_start + generated_len]


def build_glm_pair_tasks(sequence: str, spans):
    """Return paired GLM tasks for a sequence and its two candidate spans."""
    (span1_start, span1_end), (span2_start, span2_end) = spans
    return (
        build_glm_task_with_span(sequence, span1_start, span1_end),
        build_glm_task_with_span(sequence, span2_start, span2_end),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--parquet_path",
        type=str,
        default="../data/swissprot_seqid30_75k_all_info_with_3di.parquet",
    )
    parser.add_argument("--output_csv", type=str, default="gen_circuit/gen_data.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_clm", type=int, default=1000)
    parser.add_argument("--n_glm", type=int, default=1000)
    parser.add_argument("--start_fraction", type=float, default=0.80)
    parser.add_argument("--max_seq_len", type=int, default=400)
    parser.add_argument("--max_repetitive_fraction", type=float, default=0.25)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--p", type=float, default=0.95, help="Top-p sampling value for generation")
    parser.add_argument("--T", type=float, default=0.85, help="Temperature for sampling during generation")
    args = parser.parse_args()

    set_global_seed(args.seed)
    rng = np.random.default_rng(args.seed + 11)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    # The CLT wrapper is used here only as a convenient way to access ProGen3.
    clt_ckpt = os.environ.get("CLT_CHECKPOINT")
    if not clt_ckpt:
        raise ValueError("CLT_CHECKPOINT must be set in environment")

    # Process candidate sequences from one deterministically shuffled pool.
    print(f"[main] device: {device}")
    print("[main] Loading baseline ProGen3 via CLT discoverer...")
    discoverer = CircuitDiscovererCLT(device=device, ckpt_path=clt_ckpt)
    print("[main] Loading candidate sequences...")
    source_df = pd.read_parquet(args.parquet_path)

    # Keep only rows with non-empty sequences.
    candidate_pool = source_df.dropna(subset=['Sequence']).copy()
    candidate_pool['Sequence'] = candidate_pool['Sequence'].astype(str).str.strip()
    candidate_pool = candidate_pool[candidate_pool['Sequence'] != ""]
    candidate_pool = candidate_pool[candidate_pool['Sequence'].str.len() <= args.max_seq_len]
    if len(candidate_pool) == 0:
        raise ValueError("No candidate sequences passed max_seq_len filter")

    # Deduplicate entire pool so we can keep drawing fresh, non-overlapping sets.
    candidate_pool = candidate_pool.drop_duplicates(subset=["Sequence"]).reset_index(drop=True)
    if len(candidate_pool) == 0:
        raise ValueError("No unique candidate sequences passed max_seq_len filter")

    candidate_order = rng.permutation(len(candidate_pool))
    shuffled_candidates = candidate_pool.iloc[candidate_order].reset_index(drop=True)

    # Build 1000 CLM + 1000 GLM pairs (each GLM pair contributes GLM_1 and GLM_2)
    target_clm_n = args.n_clm
    target_glm_n = args.n_glm
    kept_clm = 0
    kept_glm = 0
    discarded = 0
    scanned = 0
    rows = []
    used_for_clm = set()
    pbar = tqdm(total=target_clm_n + 2 * target_glm_n, desc="Preparing data")
    gaussians_glm = [(10.0, 5.0), (30.0, 10.0)]
    # Phase 1: fill CLM from the shuffled pool.
    for _, candidate_row in shuffled_candidates.iterrows():
        if kept_clm >= target_clm_n:
            break

        sequence = str(candidate_row.get("Sequence", "")).strip()
        if not sequence:
            continue

        scanned += 1
        clm_task = build_clm_task(sequence, start_fraction=args.start_fraction)
        try:
            progen3_sequence = generate_progen_sequence(discoverer, clm_task, allow_eos=True, p=args.p, temperature=args.T)
        except Exception:
            discarded += 1
            pbar.set_postfix({"clm": kept_clm, "glm": kept_glm, "discarded": discarded, "scanned": scanned})
            continue

        if not passes_generation_quality(
            extract_generated_portion(clm_task, progen3_sequence),
            max_repetitive_fraction=args.max_repetitive_fraction,
        ):
            discarded += 1
            pbar.set_postfix({"clm": kept_clm, "glm": kept_glm, "discarded": discarded, "scanned": scanned})
            continue

        rows.append(
            {
                "Entry Name": candidate_row.get("Entry Name", ""),
                "Protein names": candidate_row.get("Protein names", ""),
                "Entry": candidate_row.get("Entry", ""),
                "type": "CLM",
                "prompt": clm_task.prompt_12,
                "original_sequence": sequence,
            }
        )
        used_for_clm.add(sequence)
        kept_clm += 1
        pbar.update(1)
        pbar.set_postfix({"clm": kept_clm, "glm": kept_glm, "discarded": discarded, "scanned": scanned})

    # Phase 2: fill GLM from the remaining shuffled pool.
    for _, candidate_row in shuffled_candidates.iterrows():
        if kept_glm >= target_glm_n:
            break

        sequence = str(candidate_row.get("Sequence", "")).strip()
        if not sequence or sequence in used_for_clm:
            continue

        scanned += 1
        spans = generate_two_non_overlapping_spans(sequence, gaussians_glm, rng)
        if spans is None:
            discarded += 1
            pbar.set_postfix({"clm": kept_clm, "glm": kept_glm, "discarded": discarded, "scanned": scanned})
            continue

        glm_task_1, glm_task_2 = build_glm_pair_tasks(sequence, spans)

        try:
            progen3_sequence_1 = generate_progen_sequence(discoverer, glm_task_1, allow_eos=True, p=args.p, temperature=args.T)
            progen3_sequence_2 = generate_progen_sequence(discoverer, glm_task_2, allow_eos=True, p=args.p, temperature=args.T)
        except Exception:
            discarded += 1
            pbar.set_postfix({"clm": kept_clm, "glm": kept_glm, "discarded": discarded, "scanned": scanned})
            continue

        if not passes_generation_quality(
            extract_generated_portion(glm_task_1, progen3_sequence_1),
            max_repetitive_fraction=args.max_repetitive_fraction,
        ) or not passes_generation_quality(
            extract_generated_portion(glm_task_2, progen3_sequence_2),
            max_repetitive_fraction=args.max_repetitive_fraction,
        ):
            discarded += 1
            pbar.set_postfix({"clm": kept_clm, "glm": kept_glm, "discarded": discarded, "scanned": scanned})
            continue

        rows.append(
            {
                "Entry Name": candidate_row.get("Entry Name", ""),
                "Protein names": candidate_row.get("Protein names", ""),
                "Entry": candidate_row.get("Entry", ""),
                "type": "GLM_1",
                "prompt": glm_task_1.prompt_12,
                "original_sequence": sequence,
            }
        )
        rows.append(
            {
                "Entry Name": candidate_row.get("Entry Name", ""),
                "Protein names": candidate_row.get("Protein names", ""),
                "Entry": candidate_row.get("Entry", ""),
                "type": "GLM_2",
                "prompt": glm_task_2.prompt_12,
                "original_sequence": sequence,
            }
        )
        kept_glm += 1
        pbar.update(2)
        pbar.set_postfix({"clm": kept_clm, "glm": kept_glm, "discarded": discarded, "scanned": scanned})

    pbar.close()

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output_columns = ["Entry Name", "Protein names", "Entry", "type", "prompt", "original_sequence"]
    pd.DataFrame(rows, columns=output_columns).to_csv(out_path, index=False)

    print(f"[main] Kept CLM: {kept_clm}")
    print(f"[main] Kept GLM pairs: {kept_glm}")
    print(f"[main] Kept total rows: {len(rows)}")
    print(f"[main] Discarded: {discarded}")
    print(f"[main] Scanned: {scanned}")
    print(f"[main] Saved prepared dataset -> {out_path}")

    if kept_clm < target_clm_n or kept_glm < target_glm_n:
        print(
            f"[main] Warning: Did not reach target CLM={target_clm_n} and GLM pairs={target_glm_n} "
            "with current filters/candidate pool"
        )


if __name__ == "__main__":
    main()
