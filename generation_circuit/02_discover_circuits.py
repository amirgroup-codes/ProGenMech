#!/usr/bin/env python3
"""
Circuit discovery runner for prepared CLM/GLM generation data.

Loads gen_data.csv, generates a ProGen3 baseline sequence per row, then evaluates
multiple reconstruction methods with both active and ablated latent sets.
Results are written as one JSON file per entry under gen/<method_name>/.
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from circuit_utils.clt_circuit import CircuitDiscovererCLT
from circuit_utils.plt_circuit import CircuitDiscovererPLT
from circuit_utils.circuit_utils import compute_attribution, rank_nodes, circuit_search
from generation_circuit.generation_utils import (
    TaskSpec,
    build_clm_task,
    compile_output_sequence,
    compute_kl_and_nmse,
    compute_top1_vs_baseline,
    set_batch_preparer_seed_from_prompt,
    set_global_seed,
    set_runtime_seed_from_prompt,
    token_ids_to_aa_mask_and_tokens,
    append_generation_step,
    cap_sequence_to_reference,
    top_p_sample,
)
from progen3.scorer import ProGen3Scorer
from generation_circuit.scorer_circuit import ScorerCircuit


METHOD_CONFIGS = {
    "CLT_direct": {
        "discoverer_key": "clt",
        "sequential": False,
        "freeze_attention": True,
    },
    "CLT_sequential": {
        "discoverer_key": "clt",
        "sequential": True,
        "freeze_attention": True,
    },
    "CLT_sequential_no_frozen": {
        "discoverer_key": "clt",
        "sequential": True,
        "freeze_attention": False,
    },
    "PLT": {
        "discoverer_key": "plt",
        "sequential": True,
        "freeze_attention": True,
    },
    "PLT_no_frozen": {
        "discoverer_key": "plt",
        "sequential": True,
        "freeze_attention": False,
    },
}

GLM_PROMPT_PATTERN = re.compile(r"^(.*)\[GLM\](\d+)-(\d+)-(\d+)$")


def parse_glm_prompt(prompt: str) -> Tuple[int, int, int] | None:
    """Parse a GLM prompt suffix and return (span_start, span_end, span_len).

    Returns None when the prompt does not match the expected GLM format.
    """
    match = GLM_PROMPT_PATTERN.match(prompt)
    if match is None:
        return None
    return int(match.group(2)), int(match.group(3)), int(match.group(4))


def task_from_csv_row(row: pd.Series, start_fraction: float) -> Tuple[str, TaskSpec] | None:
    """Convert one CSV row into a TaskSpec for CLM or GLM.

    Returns (original_sequence, task) on success, or None for invalid rows.
    """
    task_type = str(row.get("type", "")).strip()
    original_sequence = str(row.get("original_sequence", "")).strip()
    prompt = str(row.get("prompt", "")).strip()

    if not original_sequence:
        return None

    if task_type == "CLM":
        if prompt and original_sequence.startswith(prompt):
            target = original_sequence[len(prompt) :]
            task = TaskSpec(
                task_type="CLM",
                prompt=prompt,
                prompt_12=prompt,
                reverse_sequence=False,
                target_sequence=target,
                max_steps=len(target),
            )
        else:
            task = build_clm_task(original_sequence, start_fraction=start_fraction)
        return original_sequence, task

    if task_type.startswith("GLM"):
        glm_info = parse_glm_prompt(prompt)
        if glm_info is None:
            return None
        span_start, span_end, span_len = glm_info
        if span_start < 0 or span_end > len(original_sequence) or span_start >= span_end:
            return None

        task = TaskSpec(
            task_type="GLM",
            prompt=prompt,
            prompt_12=prompt,
            reverse_sequence=False,
            target_sequence=original_sequence[span_start:span_end],
            max_steps=span_len,
            span_start=span_start,
            span_end=span_end,
        )
        return original_sequence, task

    return None


def load_tasks_from_csv(input_csv: str, start_fraction: float) -> List[Tuple[pd.Series, TaskSpec]]:
    """
    Load CLM/GLM generation tasks from a CSV file.

    The CSV file should contain columns with the following names:
    - "Entry": The unique identifier for each entry.
    - "Entry Name": The name of the entry.
    - "type": The type of the task, either "CLM" or starting with "GLM".
    - "prompt": The prompt string for the task.
    - "original_sequence": The original sequence of the task.
    """
    df = pd.read_csv(input_csv)
    required_columns = {"Entry", "Entry Name", "type", "prompt", "original_sequence"}
    missing = required_columns.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in input csv: {sorted(missing)}")

    task_rows: List[Tuple[pd.Series, TaskSpec]] = []
    for _, row in df.iterrows():
        parsed = task_from_csv_row(row, start_fraction=start_fraction)
        if parsed is not None:
            task_rows.append((row, parsed[1]))

    if not task_rows:
        raise ValueError("No valid CLM/GLM rows found in input csv")
    return task_rows


def get_generation_kwargs(discoverer, task: TaskSpec):
    """
    Build seeded generation inputs for a task on the discoverer device.

    Seeding is prompt-derived so repeated calls are deterministic for a given task.
    """
    set_runtime_seed_from_prompt(task.task_type, task.prompt_12, task.reverse_sequence)
    set_batch_preparer_seed_from_prompt(discoverer, task.task_type, task.prompt_12, task.reverse_sequence)
    current_inputs = discoverer.batch_preparer.get_generation_kwargs(task.prompt_12, task.reverse_sequence)
    return {
        k: (v if str(v.device) == str(discoverer.device) else v.to(discoverer.device, non_blocking=True))
        for k, v in current_inputs.items()
    }


def squeeze_last_logits(logits: torch.Tensor) -> torch.Tensor:
    """Return last-step logits as a detached float tensor of shape (V,)."""
    if logits.dim() == 3 and logits.shape[0] == 1:
        logits = logits.squeeze(0)
    return logits[-1, :].float().detach()


def build_generated_mask_from_inputs(discoverer, model_inputs, task: TaskSpec, generated_aa_count: int) -> torch.Tensor:
    """Create a token-position mask for generated amino-acid positions only.

    The mask shape matches labels (B, T) and is used by attribution methods.
    """
    input_ids = model_inputs["input_ids"]
    labels = model_inputs["labels"]
    mask = torch.zeros_like(labels, dtype=torch.bool)

    aa_positions = discoverer._valid_token_mask(input_ids)[0].nonzero(as_tuple=False).squeeze(-1)
    if aa_positions.numel() == 0 or generated_aa_count <= 0:
        return mask

    if task.task_type == "CLM":
        start_res = len(task.prompt)
    else:
        start_res = int(task.span_start or 0)

    start_res = max(0, min(start_res, int(aa_positions.numel())))
    end_res = max(start_res, min(start_res + int(generated_aa_count), int(aa_positions.numel())))
    selected_token_positions = aa_positions[start_res:end_res]
    if selected_token_positions.numel() > 0:
        mask[0, selected_token_positions] = True
    return mask


def evaluate_circuit(
    discoverer,
    cfg: Dict[str, object],
    task: TaskSpec,
    original_sequence: str,
    progen3_token_ids: List[int],
    progen3_logits_matrix: torch.Tensor,
    aa_mask: List[bool],
    active_nodes: Dict[int, set[int]] | None,
    p: int,
    T: int,
) -> Dict[str, object]:
    """
    Evaluate circuit in autoregressive mode against ProGen3.

    Replays the baseline trajectory, reconstructs step logits with active nodes,
    and returns sequence/top1/KL/NMSE in the same metric space used for JSON output.
    """
    tokenizer = discoverer.tokenizer

    # 1) Replay generation along the baseline trajectory with the selected circuit nodes.
    eos_token = "<eos_span>" if task.task_type == "GLM" else "<eos>"
    eos_token_id = tokenizer.token_to_id(eos_token)
    model_inputs = get_generation_kwargs(discoverer, task)
    recon_step_logits: List[torch.Tensor] = []

    with torch.inference_mode():
        for baseline_token_id in progen3_token_ids:
            logits = discoverer.reconstruct_logits(
                model_inputs=model_inputs,
                active_nodes=active_nodes,
                sequential=bool(cfg["sequential"]),
                freeze_attention=bool(cfg["freeze_attention"]),
            )
            recon_step_logits.append(squeeze_last_logits(logits))

            if eos_token_id is not None and int(baseline_token_id) == int(eos_token_id):
                break

            model_inputs = append_generation_step(model_inputs, int(baseline_token_id))

    # 2) Convert logits to ids/tokens and reconstruct the final output sequence.
    recon_logits_matrix = (
        torch.stack(recon_step_logits, dim=0)
        if recon_step_logits
        else torch.empty((0, 0), dtype=torch.float32)
    )
    recon_ids = [top_p_sample(logits, p=p, temperature=T) for logits in recon_step_logits] if recon_step_logits else []
    recon_tokens = [tokenizer.id_to_token(int(token_id)) for token_id in recon_ids]

    recon_sequence = compile_output_sequence(task, recon_tokens, original_sequence)
    progen3_tokens = [tokenizer.id_to_token(int(token_id)) for token_id in progen3_token_ids]
    baseline_sequence = compile_output_sequence(task, progen3_tokens, original_sequence)
    recon_sequence = cap_sequence_to_reference(task, recon_sequence, baseline_sequence, original_sequence)

    # 3) Compute evaluation metrics in the same autoregressive space used by reporting.
    recon_kl, recon_nmse = compute_kl_and_nmse(progen3_logits_matrix, recon_logits_matrix, aa_mask)
    return {
        "sequence": recon_sequence,
        "kl": float(recon_kl),
        "nmse": float(recon_nmse),
    }


def _make_generated_mask_fn(discoverer, task: TaskSpec, generated_aa_count: int):
    """Return a closure that builds generated-token masks from model inputs."""
    def generated_mask_fn(_batch, model_inputs):
        return build_generated_mask_from_inputs(discoverer, model_inputs, task, generated_aa_count)

    return generated_mask_fn


def evaluate_row(
    row: pd.Series,
    task: TaskSpec,
    clt_discoverer: CircuitDiscovererCLT,
    plt_discoverer: CircuitDiscovererPLT,
    scorer: ProGen3Scorer,
    recovery_ratio: float,
    step_size: int,
    max_nodes: int,
    method_names: List[str],
    p: int,
    T: int,
) -> Dict[str, object]:
    """Run full discovery/evaluation for one CSV row.

    1. Generate baseline ProGen3 trajectory and cache per-step logits.
    2. Compute max/base metrics for each method.
    3. Run circuit discovery toward target_kl.
    4. Store recovered metrics and selected nodes.
    """
    tokenizer = clt_discoverer.tokenizer
    clt_discoverer.progen.eval()
    plt_discoverer.progen.eval()

    # 0. Initialize storage for ProGen3 and replacement model logits per step
    progen3_token_ids: List[int] = []
    progen3_step_logits: List[torch.Tensor] = []
    method_step_logits: Dict[str, Dict[str, List[torch.Tensor]]] = {
        name: {"max": [], "base": []}
        for name in method_names
    }

    # 1. Run baseline ProGen3 generation once and cache per-step logits
    model_inputs = get_generation_kwargs(clt_discoverer, task)
    eos_token = "<eos_span>" if task.task_type == "GLM" else "<eos>"
    eos_token_id = tokenizer.token_to_id(eos_token)

    with torch.inference_mode():
        for _ in range(task.max_steps):
            baseline_out = clt_discoverer.progen(**model_inputs, return_dict=True)
            baseline_last_logits = baseline_out.logits[:, -1, :].squeeze(0).float().detach()
            baseline_next_id = top_p_sample(baseline_last_logits, p=p, temperature=T)

            progen3_token_ids.append(baseline_next_id)
            progen3_step_logits.append(baseline_last_logits)

            # Evaluate each method on the exact same generation state and grab logits
            for method_name in method_names:
                cfg = METHOD_CONFIGS[method_name]
                discoverer = clt_discoverer if cfg["discoverer_key"] == "clt" else plt_discoverer
                max_logits = discoverer.reconstruct_logits(
                    model_inputs=model_inputs,
                    active_nodes=None,
                    sequential=bool(cfg["sequential"]),
                    freeze_attention=bool(cfg["freeze_attention"]),
                )
                base_logits = discoverer.reconstruct_logits(
                    model_inputs=model_inputs,
                    active_nodes={},
                    sequential=bool(cfg["sequential"]),
                    freeze_attention=bool(cfg["freeze_attention"]),
                )
                method_step_logits[method_name]["max"].append(squeeze_last_logits(max_logits))
                method_step_logits[method_name]["base"].append(squeeze_last_logits(base_logits))

            if eos_token_id is not None and baseline_next_id == eos_token_id:
                break

            model_inputs = append_generation_step(model_inputs, baseline_next_id)

    # 2. Construct ProGen3 sequence and its respective logits matrix
    aa_mask, progen3_tokens = token_ids_to_aa_mask_and_tokens(tokenizer, progen3_token_ids)
    progen3_logits_matrix = (
        torch.stack(progen3_step_logits, dim=0)
        if progen3_step_logits
        else torch.empty((0, 0), dtype=torch.float32)
    )
    original_sequence = str(row.get("original_sequence", "")).strip()
    progen3_sequence = compile_output_sequence(task, progen3_tokens, original_sequence)
    record: Dict[str, object] = {
        "entry": str(row.get("Entry", "")).strip(),
        "entry_name": str(row.get("Entry Name", "")).strip(),
        "type": str(row.get("type", "")).strip(),
        "prompt": str(row.get("prompt", "")).strip(),
        "original_sequence": original_sequence,
        "progen3_sequence": progen3_sequence,
    }

    # 3. For each replacement model, run circuit discovery
    # Max = using all latents, Base = using no latents
    for method_name in method_names:
        cfg = METHOD_CONFIGS[method_name]
        discoverer = clt_discoverer if cfg["discoverer_key"] == "clt" else plt_discoverer
        max_step_logits = method_step_logits[method_name]["max"]
        base_step_logits = method_step_logits[method_name]["base"]

        max_logits_matrix = (
            torch.stack(max_step_logits, dim=0)
            if max_step_logits
            else torch.empty((0, 0), dtype=torch.float32)
        )
        base_logits_matrix = (
            torch.stack(base_step_logits, dim=0)
            if base_step_logits
            else torch.empty((0, 0), dtype=torch.float32)
        )
        max_ids = [top_p_sample(logits, p=p, temperature=T) for logits in max_step_logits] if max_step_logits else []
        base_ids = [top_p_sample(logits, p=p, temperature=T) for logits in base_step_logits] if base_step_logits else []
        max_tokens = [tokenizer.id_to_token(int(token_id)) for token_id in max_ids]
        base_tokens = [tokenizer.id_to_token(int(token_id)) for token_id in base_ids]

        # 3.1 Compute max/base metrics and get their sequence outputs
        max_sequence = compile_output_sequence(task, max_tokens, original_sequence)
        max_sequence = cap_sequence_to_reference(task, max_sequence, progen3_sequence, original_sequence)
        max_kl, max_nmse = compute_kl_and_nmse(progen3_logits_matrix, max_logits_matrix, aa_mask)
        base_sequence = compile_output_sequence(task, base_tokens, original_sequence)
        base_kl, base_nmse = compute_kl_and_nmse(progen3_logits_matrix, base_logits_matrix, aa_mask)
        record[f"{method_name}_max_sequence"] = max_sequence
        record[f"{method_name}_max_kl"] = float(max_kl)
        record[f"{method_name}_max_nmse"] = float(max_nmse)
        record[f"{method_name}_base_sequence"] = base_sequence
        record[f"{method_name}_base_kl"] = float(base_kl)
        record[f"{method_name}_base_nmse"] = float(base_nmse)

        # 3.2 Compute attribution scores on the generated positions
        generated_aa_count = int(sum(1 for is_aa in aa_mask if is_aa))
        generated_mask_fn = _make_generated_mask_fn(discoverer, task, generated_aa_count)
        attr_by_layer = compute_attribution(
            discoverer,
            [original_sequence],
            batch_size=1,
            sequential=bool(cfg["sequential"]),
            freeze_attention=bool(cfg["freeze_attention"]),
            source="layer_output",
            generated_mask_fn=generated_mask_fn,
        )
        ranking = rank_nodes(attr_by_layer)

        # 3.3 Circuit discover search that reaches the KL target.
        target_kl = float(recovery_ratio * float(max_kl))
        def _search_metric_fn(d, _probe, seqs, _y, active_nodes, _batch_size):
            metrics = evaluate_circuit(
                discoverer=d,
                cfg=cfg,
                task=task,
                original_sequence=seqs[0],
                progen3_token_ids=progen3_token_ids,
                progen3_logits_matrix=progen3_logits_matrix,
                aa_mask=aa_mask,
                active_nodes=active_nodes,
                p=p,
                T=T,
            )
            return -float(metrics["kl"])
        best_nodes, best_k, _ = circuit_search(
            discoverer,
            None,
            ranking,
            [original_sequence],
            None,
            target_metric=-target_kl,
            metric_fn=_search_metric_fn,
            batch_size=1,
            step_size=step_size,
            max_nodes=max_nodes,
            desc=f"{method_name}:{record['entry']}",
            metric_name="kl",
            display_transform=lambda x: -x,
        )

        # 3.4 Re-evaluate best node set and store recovered outputs.
        recovered = evaluate_circuit(
            discoverer,
            cfg,
            task,
            original_sequence,
            progen3_token_ids,
            progen3_logits_matrix,
            aa_mask,
            best_nodes,
            p=p,
            T=T
        )
        record[f"{method_name}_recovered_sequence"] = recovered["sequence"]
        record[f"{method_name}_recovered_kl"] = float(recovered["kl"])
        record[f"{method_name}_recovered_nmse"] = float(recovered["nmse"])
        record[f"{method_name}_recovered_k"] = int(best_k)
        record[f"{method_name}_target_kl"] = target_kl
        record[f"{method_name}_recovered_nodes"] = {
            str(layer): sorted(list(nodes))
            for layer, nodes in best_nodes.items()
        }

        # Compute LL for max and recovered sequences
        current_sequences = [record[f"{method_name}_max_sequence"], recovered["sequence"]]
        current_tasks = [task, task]
        current_scores = scorer.score_batch_with_tasks(current_sequences, current_tasks)
        max_ll = current_scores["log_likelihood"][0]
        max_perplexity = current_scores["perplexity"][0]
        recovered_ll = current_scores["log_likelihood"][1]
        recovered_perplexity = current_scores["perplexity"][1]
        record[f"{method_name}_max_ll"] = max_ll
        record[f"{method_name}_max_perplexity"] = max_perplexity
        record[f"{method_name}_recovered_ll"] = recovered_ll
        record[f"{method_name}_recovered_perplexity"] = recovered_perplexity

        tqdm.write(
            f"[{method_name} {record['entry']} {record['type']}] "
            f"max_kl={float(max_kl):.3f} | "
            f"max_ll={float(max_ll):.3f} | "
            f"recovered_kl={float(recovered['kl']):.3f} | "
            f"recovered_ll={float(recovered_ll):.3f} |  nodes={best_k}"
        )

    # Compute log-likelihood and perplexity for all sequences
    all_sequences = [progen3_sequence, original_sequence]
    method_sequences = []
    for method_name in method_names:
        method_sequences.extend([
            record[f"{method_name}_max_sequence"],
            record[f"{method_name}_base_sequence"],
            record[f"{method_name}_recovered_sequence"]
        ])
    all_sequences.extend(method_sequences)
    tasks_for_scoring = [task] * len(all_sequences)
    scores = scorer.score_batch_with_tasks(all_sequences, tasks_for_scoring)

    # Assign scores
    record["progen3_ll"] = scores["log_likelihood"][0]
    record["progen3_perplexity"] = scores["perplexity"][0]
    record["original_ll"] = scores["log_likelihood"][1]
    record["original_perplexity"] = scores["perplexity"][1]
    for i, method_name in enumerate(method_names):
        base_idx = 2 + i * 3
        record[f"{method_name}_max_ll"] = scores["log_likelihood"][base_idx]
        record[f"{method_name}_max_perplexity"] = scores["perplexity"][base_idx]
        record[f"{method_name}_base_ll"] = scores["log_likelihood"][base_idx + 1]
        record[f"{method_name}_base_perplexity"] = scores["perplexity"][base_idx + 1]
        record[f"{method_name}_recovered_ll"] = scores["log_likelihood"][base_idx + 2]
        record[f"{method_name}_recovered_perplexity"] = scores["perplexity"][base_idx + 2]

    return record


def task_type_matches_mode(row_type: str, mode: str) -> bool:
    """Return whether a row type should be processed under a selected mode."""
    if mode == "all":
        return True
    if mode == "clm":
        return row_type == "CLM"
    if mode == "glm":
        return row_type.startswith("GLM")
    return True


def limit_task_rows_for_debug(task_rows: List[Tuple[pd.Series, TaskSpec]], mode: str, limit: int = 20) -> List[Tuple[pd.Series, TaskSpec]]:
    """Keep only the first `limit` CLM rows and first `limit` GLM rows for quick debugging.

    The original row order is preserved within each type.
    """
    if mode == "clm":
        kept = 0
        limited: List[Tuple[pd.Series, TaskSpec]] = []
        for row, task in task_rows:
            if str(row.get("type", "")).strip() != "CLM":
                continue
            limited.append((row, task))
            kept += 1
            if kept >= limit:
                break
        return limited

    if mode == "glm":
        kept = 0
        limited = []
        for row, task in task_rows:
            if not str(row.get("type", "")).strip().startswith("GLM"):
                continue
            limited.append((row, task))
            kept += 1
            if kept >= limit:
                break
        return limited

    clm_kept = 0
    glm_kept = 0
    limited: List[Tuple[pd.Series, TaskSpec]] = []
    for row, task in task_rows:
        row_type = str(row.get("type", "")).strip()
        if row_type == "CLM":
            if clm_kept >= limit:
                continue
            clm_kept += 1
            limited.append((row, task))
        elif row_type.startswith("GLM"):
            if glm_kept >= limit:
                continue
            glm_kept += 1
            limited.append((row, task))
    return limited


def main() -> None:
    """CLI entrypoint for circuit discovery over prepared CLM/GLM rows."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=str, default="gen_circuit/gen_data.csv")
    parser.add_argument("--output_root", type=str, default="gen_circuit/gen")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start_fraction", type=float, default=0.80)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--recovery_ratio", type=float, default=1.2)
    parser.add_argument("--step_size", type=int, default=32)
    parser.add_argument("--max_nodes", type=int, default=1000)
    parser.add_argument("--mode", type=str, choices=["all", "clm", "glm"], default="all")
    parser.add_argument("--debug", action="store_true", help="Limit to the first 20 CLM rows and first 20 GLM rows.")
    parser.add_argument("--p", type=float, default=0.95, help="Top-p sampling value for generation")
    parser.add_argument("--T", type=float, default=0.85, help="Temperature for sampling during generation")
    args = parser.parse_args()

    set_global_seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Initialize
    clt_ckpt = os.environ.get("CLT_CHECKPOINT")
    plt_ckpt = os.environ.get("PLT_CHECKPOINT")
    if not clt_ckpt or not plt_ckpt:
        raise ValueError("CLT_CHECKPOINT and PLT_CHECKPOINT must be set in environment")
    clt_discoverer = CircuitDiscovererCLT(device=device, ckpt_path=clt_ckpt)
    plt_discoverer = CircuitDiscovererPLT(device=device, ckpt_path=plt_ckpt)
    scorer = ScorerCircuit(model=clt_discoverer.progen, tokenizer=clt_discoverer.tokenizer)

    # 2. Specify output directories
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    for method_name in METHOD_CONFIGS:
        (output_root / method_name).mkdir(parents=True, exist_ok=True)

    # 3. Load tasks and run discovery
    task_rows = load_tasks_from_csv(args.input_csv, start_fraction=args.start_fraction)
    if args.debug:
        task_rows = limit_task_rows_for_debug(task_rows, args.mode, limit=20)
    for row, task in tqdm(task_rows, total=len(task_rows), desc="Starting discovery"):
        entry = str(row.get("Entry", "")).strip()
        row_type = str(row.get("type", "")).strip()
        if not task_type_matches_mode(row_type, args.mode):
            continue

        pending_methods: List[str] = []
        for method_name in METHOD_CONFIGS:
            out_path = output_root / method_name / f"{entry}_{row_type}.json"
            if out_path.exists():
                continue
            pending_methods.append(method_name)

        if not pending_methods:
            continue

        row_record = evaluate_row(
            row,
            task,
            clt_discoverer,
            plt_discoverer,
            scorer,
            recovery_ratio=args.recovery_ratio,
            step_size=args.step_size,
            max_nodes=args.max_nodes,
            method_names=pending_methods,
            p=args.p,
            T=args.T,
        )

        for method_name in pending_methods:
            circuit_json = {
                "entry": row_record["entry"],
                "entry_name": row_record["entry_name"],
                "type": row_record["type"],
                "prompt": row_record["prompt"],
                "k": row_record[f"{method_name}_recovered_k"],
                "original_sequence": row_record["original_sequence"],
                "original_ll": row_record["original_ll"],
                "original_perplexity": row_record["original_perplexity"],
                "progen3_sequence": row_record["progen3_sequence"],
                "progen3_ll": row_record["progen3_ll"],
                "progen3_perplexity": row_record["progen3_perplexity"],
                "max_sequence": row_record[f"{method_name}_max_sequence"],
                "max_ll": row_record[f"{method_name}_max_ll"],
                "max_perplexity": row_record[f"{method_name}_max_perplexity"],
                "max_kl": row_record[f"{method_name}_max_kl"],
                "max_nmse": row_record[f"{method_name}_max_nmse"],
                "base_sequence": row_record[f"{method_name}_base_sequence"],
                "base_ll": row_record[f"{method_name}_base_ll"],
                "base_perplexity": row_record[f"{method_name}_base_perplexity"],
                "base_kl": row_record[f"{method_name}_base_kl"],
                "base_nmse": row_record[f"{method_name}_base_nmse"],
                "recovered_sequence": row_record[f"{method_name}_recovered_sequence"],
                "recovered_ll": row_record[f"{method_name}_recovered_ll"],
                "recovered_perplexity": row_record[f"{method_name}_recovered_perplexity"],
                "recovered_kl": row_record[f"{method_name}_recovered_kl"],
                "recovered_nmse": row_record[f"{method_name}_recovered_nmse"],
                "nodes": row_record[f"{method_name}_recovered_nodes"],
            }
            out_path = output_root / method_name / f"{entry}_{row_type}.json"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with out_path.open("w", encoding="utf-8") as handle:
                json.dump(circuit_json, handle, indent=2)
                handle.write("\n")

        clt_discoverer.clear_cache()
        plt_discoverer.clear_cache()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
