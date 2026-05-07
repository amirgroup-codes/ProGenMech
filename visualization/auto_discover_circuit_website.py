#!/usr/bin/env python3
"""
Auto-Discover Circuits for ProGen3: CLM, GLM, and Zero-Shot Function Tasks.

This script performs on-the-fly circuit discovery for:
- CLM/GLM: Generation tasks with KL divergence recovery.
- Zero-Shot: Regression tasks with Spearman correlation recovery.

Only CLT sequential is used for circuit discovery.
"""

import sys
import os
import argparse
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path

script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(script_dir)
sys.path.append(repo_root)
sys.path.append(os.path.join(repo_root, "function_circuit"))
sys.path.append(os.path.join(repo_root, "generation_circuit"))

try:
    from circuit_utils.clt_circuit import CircuitDiscovererCLT
    from circuit_utils.circuit_utils import compute_attribution, rank_nodes, circuit_search
    from function_utils import get_spearman_p
    from generation_circuit.generation_utils import (
        TaskSpec,
        compile_output_sequence,
        compute_kl_and_nmse,
        token_ids_to_aa_mask_and_tokens,
        top_p_sample,
        append_generation_step,
        cap_sequence_to_reference,
        set_batch_preparer_seed_from_prompt,
        set_runtime_seed_from_prompt,
    )
    from external.progen3.src.progen3.scorer import ProGen3Scorer
    from function_circuit.clt_plt_scorer import CLTPLTReconstructionScorer
    from generation_circuit.scorer_circuit import ScorerCircuit
except ImportError as e:
    print(f"❌ Error importing project modules: {e}")
    sys.exit(1)

np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

TARGET_PER_LAYER = 30
TARGET_DIFF_PER_LAYER = 30


def find_column(df, keyword):
    for c in df.columns:
        if keyword.lower() in c.lower():
            return c
    return None


def load_and_parse_csv_regression(csv_path):
    print(f"Loading regression data from {csv_path}...")
    df = pd.read_csv(csv_path)

    seq_col = find_column(df, "sequence")
    if not seq_col:
        if "mutated_sequence" in df.columns:
            seq_col = "mutated_sequence"
        else:
            raise ValueError(f"Could not find a column containing 'sequence' in {df.columns}")

    score_col = find_column(df, "score")
    if not score_col:
        score_col = find_column(df, "target")

    if not score_col:
        raise ValueError(f"Could not find a score/target column in {df.columns}")

    seqs = df[seq_col].astype(str).values
    scores = df[score_col].values
    if not np.issubdtype(scores.dtype, np.number):
        raise ValueError(f"Regression mode enabled, but column '{score_col}' is not numeric.")
    scores = scores.astype(float)
    return seqs, scores


def normalize_pos(pos):
    if pos is None:
        return None
    if pos < 1:
        raise ValueError("--pos must be 1 or greater")
    return pos - 1


def build_task_for_clm_glm(seq, pos, n_generated, task_type):
    if task_type == "CLM":
        if pos is None:
            prompt = seq
            target_sequence = ""
        else:
            prompt = seq[:pos]
            target_sequence = seq[pos:pos + n_generated]
        return TaskSpec(
            task_type="CLM",
            prompt=prompt,
            prompt_12=prompt,
            reverse_sequence=False,
            target_sequence=target_sequence,
            max_steps=n_generated,
        )

    if task_type == "GLM":
        if pos is None:
            raise ValueError("For GLM, --pos must be provided")
        prompt = f"{seq}[GLM]{pos}-{pos + n_generated}-{n_generated}"
        return TaskSpec(
            task_type="GLM",
            prompt=prompt,
            prompt_12=prompt,
            reverse_sequence=False,
            target_sequence=seq[pos:pos + n_generated],
            max_steps=n_generated,
            span_start=pos,
            span_end=pos + n_generated,
        )

    raise ValueError(f"Unknown task_type: {task_type}")


def get_generation_kwargs(discoverer, task: TaskSpec):
    set_runtime_seed_from_prompt(task.task_type, task.prompt_12, task.reverse_sequence)
    set_batch_preparer_seed_from_prompt(discoverer, task.task_type, task.prompt_12, task.reverse_sequence)
    kwargs = discoverer.batch_preparer.get_generation_kwargs(task.prompt_12, task.reverse_sequence)
    return {k: v.to(discoverer.device) for k, v in kwargs.items()}


def make_generated_mask(task: TaskSpec, discoverer, model_inputs):
    if task.task_type == "CLM" and len(task.target_sequence) == 0:
        return model_inputs["labels"] != discoverer.pad_id

    mask = torch.zeros_like(model_inputs["labels"], dtype=torch.bool)
    valid_mask = discoverer._valid_token_mask(model_inputs["input_ids"])[0]
    aa_positions = valid_mask.nonzero(as_tuple=True)[0]
    if task.task_type == "CLM":
        start_res = len(task.prompt)
    else:
        start_res = int(task.span_start or 0)
    end_res = start_res + len(task.target_sequence)
    selected = aa_positions[start_res:end_res]
    if selected.numel() > 0:
        mask[0, selected] = True
    return mask


def build_generated_mask_fn(discoverer, task: TaskSpec):
    def generated_mask_fn(_batch, model_inputs):
        return make_generated_mask(task, discoverer, model_inputs)
    return generated_mask_fn


def generate_progen3_sequence(task: TaskSpec, discoverer, tokenizer, original_sequence: str, p=0.95, temperature=0.85):
    model_inputs = get_generation_kwargs(discoverer, task)
    token_ids = []
    step_logits = []

    with torch.inference_mode():
        for _ in range(task.max_steps):
            outputs = discoverer.progen(**model_inputs, return_dict=True)
            logits = outputs.logits
            last_logits = logits[:, -1, :]
            token_id = top_p_sample(last_logits, p=p, temperature=temperature)
            token_ids.append(token_id)
            step_logits.append(last_logits.squeeze(0).detach().cpu())
            model_inputs = append_generation_step(model_inputs, token_id)

    aa_mask, tokens = token_ids_to_aa_mask_and_tokens(tokenizer, token_ids)
    sequence = compile_output_sequence(task, tokens, original_sequence)
    return token_ids, torch.stack(step_logits, dim=0), aa_mask, sequence


def load_dataset_reference(path):
    if not path or not os.path.exists(path):
        print(f"Warning: Reference file {path} not found.")
        return None
    print(f"Loading dataset reference from {path}...")
    try:
        data = torch.load(path, map_location="cpu", weights_only=True)
        return data.get("storage", [])
    except TypeError:
        data = torch.load(path, map_location="cpu")
        return data.get("storage", [])
    except Exception as e:
        msg = str(e)
        if "Weights only load failed" in msg or "Unsupported global" in msg:
            print(f"weights_only torch.load failed, retrying without weights_only: {e}")
            try:
                data = torch.load(path, map_location="cpu")
                return data.get("storage", [])
            except Exception as e2:
                print(f"Failed to load reference file without weights_only: {e2}")
                return None
        print(f"Failed to load reference file: {e}")
        return None


def get_adaptive_motif_window(sequence, trace, peak_trace_idx, min_radius=10, buffer=3, top_k=None):
    seq_idx = peak_trace_idx - 1
    seq_len = len(sequence)

    if seq_idx < 0:
        return "<CLS>", 0, 0
    if seq_idx >= seq_len:
        return "<EOS>", seq_len, seq_len

    scan_start_t = 0
    scan_end_t = len(trace)

    local_trace_slice = trace[scan_start_t:scan_end_t]
    if len(local_trace_slice) == 0:
        return "", 0, 0

    positive_indices_local = [i for i, val in enumerate(local_trace_slice) if val > 0]
    global_active_indices = [idx + scan_start_t for idx in positive_indices_local]

    if top_k is not None:
        global_active_indices.sort(key=lambda idx: trace[idx], reverse=True)
        important_indices = global_active_indices[:top_k]
    else:
        important_indices = global_active_indices

    important_indices.append(peak_trace_idx)
    min_important = min(important_indices)
    max_important = max(important_indices)

    start_trace_idx = min(peak_trace_idx - min_radius, min_important - buffer)
    end_trace_idx = max(peak_trace_idx + min_radius + 1, max_important + buffer + 1)

    start_seq = max(0, start_trace_idx - 1)
    end_seq = min(seq_len, end_trace_idx - 1)

    if end_seq - start_seq <= 0:
        return "", 0, 0

    window_data = []
    for pos in range(start_seq, end_seq):
        char = sequence[pos]
        val = float(trace[pos])
        window_data.append({"pos": pos, "char": char, "val": val})

    highlight_indices = set()
    if top_k is not None:
        window_data_sorted = sorted(window_data, key=lambda x: x["val"], reverse=True)
        for item in window_data_sorted[:top_k]:
            if item["val"] > 0:
                highlight_indices.add(item["pos"])
    else:
        for item in window_data:
            if item["val"] > 0:
                highlight_indices.add(item["pos"])

    motif_str = ""
    for item in window_data:
        if item["pos"] in highlight_indices:
            motif_str += f"[{item['char']}]"
        else:
            motif_str += item["char"]

    return motif_str, start_seq, end_seq


def format_global_hit(hit):
    entry = hit.get("Entry", "?")
    name = hit.get("Entry Name", "")
    pname = hit.get("Protein names", "")
    score = hit.get("Score", 0.0)

    seq = hit.get("Sequence") or hit.get("seq")
    if not seq:
        return f"{entry} ({name}) - Score: {score:.4f} (Sequence data unavailable)"

    trace = hit.get("Activations")
    if trace is not None and isinstance(trace, torch.Tensor):
        trace = trace.detach().cpu().numpy()

    peak_idx = hit.get("Peak_Index") or hit.get("peak_idx")
    if peak_idx is None:
        if trace is not None:
            peak_idx = np.argmax(trace)
        else:
            return f"{entry} ({name}) - Score: {score:.4f} (Peak location unavailable)"

    center_seq_idx = peak_idx - 1
    start = max(0, center_seq_idx - 10)
    end = min(len(seq), center_seq_idx + 11)

    snippet = ""
    bracket_indices = set()

    if trace is not None:
        t_start = max(0, start + 1)
        t_end = min(len(trace), end + 1)
        if t_end > t_start:
            local_trace = trace[t_start:t_end]
            for i, val in enumerate(local_trace):
                if val > 0:
                    bracket_indices.add(start + i)
    else:
        bracket_indices.add(center_seq_idx)

    for i in range(start, end):
        char = seq[i]
        if i in bracket_indices:
            snippet += f"[{char}]"
        else:
            snippet += char

    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(seq) else ""
    return f"{entry} ({name}) - Score: {score:.4f} - {prefix}{snippet}{suffix} - {pname}"


def print_result_block(f, rank_label, item, sequence, ref_storage):
    layer = item["layer"]
    latent = item["latent"]
    score = item["score"]
    trace = item["trace"]
    peak_idx = item["peak_idx"]

    motif_str, seq_start, seq_end = get_adaptive_motif_window(
        sequence, trace, peak_idx, min_radius=5, buffer=3, top_k=None
    )

    t_start = max(0, seq_start)
    t_end = min(len(trace), seq_end)
    trace_window = trace[t_start:t_end]

    activation = item.get("activation", score)
    attribution = item.get("attribution", score)

    f.write(f"--- {rank_label} ---\n")
    f.write(f"Node: Layer {layer + 1}, Latent {latent + 1}\n")
    f.write(f"Max Activation: {activation:.4f} (Attribution: {attribution:.4f})\n")
    f.write(f"Peak Location: Trace Idx {peak_idx} (AA #{peak_idx})\n")
    f.write(f"Motif Context: {motif_str}\n")
    f.write(f"Trace Window : {trace_window}\n")
    f.write("\n")

    f.write(f"   [Global Top 10 Reference Entries for L{layer + 1}-{latent + 1}]\n")
    if ref_storage and layer < len(ref_storage) and latent < len(ref_storage[layer]):
        top_hits = ref_storage[layer][latent]
        if top_hits:
            for i, hit in enumerate(top_hits[:10]):
                f.write(f"   {i+1}. {format_global_hit(hit)}\n")
        else:
            f.write("   (No activations recorded in dataset)\n")
    else:
        f.write("   (Reference not loaded or index out of bounds)\n")
    f.write("\n")


def print_differential_result_block(f, rank_label, item, seqA, seqB, ref_storage):
    layer = item["layer"]
    latent = item["latent"]
    diff = item["diff"]
    sA_data = item["seqA_data"]
    sB_data = item["seqB_data"]

    f.write(f"--- {rank_label} (Diff: {diff:.4f}) ---\n")
    f.write(f"Node: Layer {layer + 1}, Latent {latent + 1}\n")
    f.write(f"Seq1 Max: {sA_data['activation']:.4f} (Attribution: {sA_data['attribution']:.4f}) @ AA #{sA_data['peak_idx']}\n")
    f.write(f"Seq1 Context: {sA_data['motif']}\n")
    f.write(f"Seq1 Trace  : {sA_data['trace']}\n\n")

    f.write(f"Seq2 Max: {sB_data['activation']:.4f} (Attribution: {sB_data['attribution']:.4f}) @ AA #{sB_data['peak_idx']}\n")
    f.write(f"Seq2 Context: {sB_data['motif']}\n")
    f.write(f"Seq2 Trace  : {sB_data['trace']}\n\n")

    f.write(f"   [Global Top 10 Reference Entries for L{layer + 1}-{latent + 1}]\n")
    if ref_storage and layer < len(ref_storage) and latent < len(ref_storage[layer]):
        top_hits = ref_storage[layer][latent]
        if top_hits:
            for i, hit in enumerate(top_hits[:10]):
                f.write(f"   {i+1}. {format_global_hit(hit)}\n")
        else:
            f.write("   (No activations recorded in dataset)\n")
    else:
        f.write("   (Reference not loaded or index out of bounds)\n")
    f.write("\n")


def write_top_activation_json(output_path, nodes_set, ref_storage, family_name):
    top_acts = {"family": family_name, "layers": {}}
    nodes_by_layer = {}
    for l, latent in sorted(nodes_set):
        nodes_by_layer.setdefault(str(l), []).append(latent)

    for l_str, latents in nodes_by_layer.items():
        top_acts["layers"][l_str] = {}
        layer_idx = int(l_str)
        for latent in sorted(set(latents)):
            hits = []
            if ref_storage and layer_idx < len(ref_storage):
                if latent < len(ref_storage[layer_idx]):
                    hits = ref_storage[layer_idx][latent]
            clean_hits = []
            for h in hits:
                h_copy = dict(h)
                if "Peak_Index" in h_copy:
                    h_copy["Peak_Index"] = max(0, h_copy["Peak_Index"] - 1)
                if "peak_idx" in h_copy:
                    h_copy["peak_idx"] = max(0, h_copy["peak_idx"] - 1)
                if "Activations" in h_copy:
                    acts = h_copy["Activations"]
                    if isinstance(acts, (list, np.ndarray, torch.Tensor)) and len(acts) > 0:
                        h_copy["Activations"] = convert_to_json_serializable(acts[1:])
                clean_hits.append(h_copy)
            top_acts["layers"][l_str][str(latent)] = convert_to_json_serializable(clean_hits)

    with open(output_path, "w") as f:
        json.dump(top_acts, f, indent=2)


def write_activation_indices_json(output_path, nodes_set, layer_acts_cache, seq_len, start_pos=0):
    act_ind = []
    t_start = start_pos
    nodes_by_layer = {}
    for layer, latent in nodes_set:
        nodes_by_layer.setdefault(layer, []).append(latent)

    for layer_idx, layer_acts in layer_acts_cache.items():
        target_latents = nodes_by_layer.get(layer_idx, [])
        if not target_latents:
            continue
        for i in range(seq_len):
            t_idx = t_start + i
            if t_idx >= layer_acts.shape[0]:
                break
            for latent_id in target_latents:
                val = float(layer_acts[t_idx, latent_id])
                if abs(val) > 0.0:
                    act_ind.append([layer_idx, i, val, int(latent_id)])
    with open(output_path, "w") as f:
        json.dump(act_ind, f, indent=2)


def get_clt_latent_traces(discoverer, sequence: str):
    model_inputs = discoverer._prepare_inputs([sequence])
    valid_mask = discoverer._valid_token_mask(model_inputs["input_ids"])[0]
    aa_positions = valid_mask.nonzero(as_tuple=True)[0]
    _, _, _, x_clt_input_flat_SLH, _ = discoverer.collector.collect(model_inputs)
    B, T = 1, model_inputs["input_ids"].shape[1]
    L = discoverer.num_layers
    H = x_clt_input_flat_SLH.shape[-1]
    x_clt_input_BTLH = x_clt_input_flat_SLH.view(B, T, L, H)
    layer_traces = {}
    for l in range(L):
        x_layer_BTH = x_clt_input_BTLH[:, :, l, :]
        x_norm_SH, _, _ = discoverer.clt.LN(x_layer_BTH.reshape(-1, H))
        x_norm_SH = x_norm_SH - discoverer.clt.b_pre[l]
        pre_acts_SH = discoverer.clt.encoders[l](x_norm_SH)
        latents_SH = discoverer.clt.topK_activation(pre_acts_SH, k=discoverer.clt.k)
        layer_traces[l] = latents_SH.detach().view(B, T, -1).squeeze(0)[aa_positions].cpu().numpy()
    return layer_traces


def select_top_latents_per_layer(attr_by_layer, target_per_layer):
    top_nodes = []
    for layer, scores in attr_by_layer.items():
        order = np.argsort(scores)[::-1][:target_per_layer]
        for latent in order:
            top_nodes.append((layer, int(latent), float(scores[int(latent)])))
    return top_nodes


def build_analysis_items(attr_by_layer, traces_cache, target_per_layer):
    items = []
    for layer, scores in attr_by_layer.items():
        order = np.argsort(scores)[::-1][:target_per_layer]
        for latent in order:
            trace = traces_cache[layer][:, latent]
            peak_idx = int(np.argmax(trace)) + 1
            activation = float(np.max(trace))
            items.append({
                "layer": layer,
                "latent": int(latent),
                "score": float(scores[int(latent)]),
                "activation": activation,
                "trace": trace,
                "peak_idx": peak_idx,
            })
    return items


def build_items_for_nodes(nodes_set, attr_by_layer, traces_cache):
    items = []
    for layer, latent in sorted(nodes_set):
        scores = attr_by_layer.get(layer)
        score = float(scores[int(latent)]) if scores is not None and int(latent) < len(scores) else 0.0
        trace = traces_cache[layer][:, int(latent)]
        peak_idx = int(np.argmax(trace)) + 1
        activation = float(np.max(trace))
        items.append({
            "layer": layer,
            "latent": int(latent),
            "score": score,
            "activation": activation,
            "trace": trace,
            "peak_idx": peak_idx,
        })
    return items


def select_top_per_layer_items(items, target_per_layer):
    selected = []
    items_by_layer = {}
    for item in items:
        items_by_layer.setdefault(item["layer"], []).append(item)

    for layer, layer_items in items_by_layer.items():
        layer_items.sort(key=lambda x: x["score"], reverse=True)
        selected.extend(layer_items[:target_per_layer])
    return selected


def compute_sequence_attributions(discoverer, sequence, zero_shot=False):
    if zero_shot:
        return discoverer.get_gradients([sequence], sequential=True, freeze_attention=True, source="layer_output", zero_shot=True)
    return compute_attribution(
        discoverer,
        [sequence],
        batch_size=1,
        sequential=True,
        freeze_attention=True,
        source="layer_output",
    )


def build_diff_items(wt_attr, seq_attr, wt_traces, seq_traces, wt_seq, seq, circuit_nodes, target_per_layer):
    diff_items = []
    for layer, latents in circuit_nodes.items():
        for latent in sorted(latents):
            wt_score = float(wt_attr[layer][latent]) if layer in wt_attr and latent < len(wt_attr[layer]) else 0.0
            seq_score = float(seq_attr[layer][latent]) if layer in seq_attr and latent < len(seq_attr[layer]) else 0.0
            diff = abs(wt_score - seq_score)
            wt_trace = wt_traces[layer][:, latent]
            seq_trace = seq_traces[layer][:, latent]
            wt_peak = int(np.argmax(wt_trace)) + 1
            seq_peak = int(np.argmax(seq_trace)) + 1
            wt_motif, _, _ = get_adaptive_motif_window(wt_seq, wt_trace, wt_peak, min_radius=5, buffer=3)
            seq_motif, _, _ = get_adaptive_motif_window(seq, seq_trace, seq_peak, min_radius=5, buffer=3)
            diff_items.append({
                "layer": layer,
                "latent": latent,
                "diff": diff,
                "score": diff,
                "seqA_data": {
                    "activation": float(np.max(wt_trace)),
                    "attribution": wt_score,
                    "trace": wt_trace,
                    "peak_idx": wt_peak,
                    "motif": wt_motif,
                },
                "seqB_data": {
                    "activation": float(np.max(seq_trace)),
                    "attribution": seq_score,
                    "trace": seq_trace,
                    "peak_idx": seq_peak,
                    "motif": seq_motif,
                },
            })
    return select_top_per_layer_items(diff_items, target_per_layer)


def write_sequence_top_report(output_path, seq, items, ref_storage):
    with open(output_path, "w") as f:
        f.write(f"=== Top Activations ===\n")
        f.write(f"Sequence Length: {len(seq)}\n")
        f.write(f"Total Nodes: {len(items)}\n")
        f.write("-------------------------------------\n\n")
        for k, item in enumerate(sorted(items, key=lambda x: x["score"], reverse=True), start=1):
            rank_label = f"Rank #{k}"
            print_result_block(f, rank_label, item, seq, ref_storage)


def write_sequence_diff_report(output_path, seqA, seqB, diff_items, ref_storage):
    with open(output_path, "w") as f:
        f.write(f"=== Differential Analysis: Seq1 (WT) vs Seq2 ===\n")
        f.write(f"Total Diff Nodes Analyzed: {len(diff_items)}\n")
        f.write("-------------------------------------\n\n")
        for k, item in enumerate(diff_items, start=1):
            rank_label = f"Rank #{k}"
            print_differential_result_block(f, rank_label, item, seqA, seqB, ref_storage)


def write_sequence_outputs(output_dir, sequence_list, discoverer, circuit_nodes_map, ref_storage, clt_scorer, progen_scorer, task_config, zero_shot=False):
    circuit_set = {layer: set(latents) for layer, latents in circuit_nodes_map.items()}
    wt_seq = sequence_list[0]
    seq_traces = [get_clt_latent_traces(discoverer, seq) for seq in sequence_list]
    seq_attrs = [compute_sequence_attributions(discoverer, seq, zero_shot=zero_shot) for seq in sequence_list]

    active_nodes = {layer: list(latents) for layer, latents in circuit_nodes_map.items()}

    for idx, seq in enumerate(sequence_list):
        seq_dir = Path(output_dir) / f"seq{idx+1}"
        seq_dir.mkdir(parents=True, exist_ok=True)
        circuit_score = clt_scorer.score([seq], task=task_config, active_nodes=active_nodes)["log_likelihood"][0]
        progen_score = float(progen_scorer.evaluate([seq])["log_likelihood"][0])
        print(f"[seq{idx+1}] circuit score: {circuit_score:.4f}, progen3 score: {progen_score:.4f}")
        generation_path = seq_dir / "generation.fasta"
        generation_path.write_text(
            f">prompt\n{seq}\n>output\n{circuit_score}\n",
            encoding="utf-8",
        )

        top_items = build_items_for_nodes([(layer, latent) for layer, latents in circuit_set.items() for latent in sorted(latents)], seq_attrs[idx], seq_traces[idx])
        top_items = select_top_per_layer_items(top_items, TARGET_PER_LAYER)

        if zero_shot and idx > 0:
            diff_items = build_diff_items(
                seq_attrs[0],
                seq_attrs[idx],
                seq_traces[0],
                seq_traces[idx],
                sequence_list[0],
                seq,
                circuit_set,
                TARGET_DIFF_PER_LAYER,
            )
        else:
            diff_items = []

        nodes_set = {(item["layer"], item["latent"]) for item in top_items}
        if diff_items:
            nodes_set |= {(item["layer"], item["latent"]) for item in diff_items}

        write_top_activation_json(seq_dir / "top_activations.json", nodes_set, ref_storage, os.path.basename(output_dir))
        write_activation_indices_json(seq_dir / "activation_indices.json", nodes_set, seq_traces[idx], len(seq))

        write_sequence_top_report(Path(output_dir) / f"top_seq{idx+1}.txt", seq, top_items, ref_storage)

        if zero_shot and idx > 0:
            write_sequence_diff_report(Path(output_dir) / f"analysis_differential_1{idx+1}.txt", sequence_list[0], seq, diff_items, ref_storage)


def convert_to_json_serializable(obj):
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): convert_to_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [convert_to_json_serializable(x) for x in obj]
    return obj


def format_topk_logits(logits: torch.Tensor, tokenizer, k: int = 5) -> str:
    if logits.numel() == 0:
        return "(empty)"
    topk = torch.topk(logits, min(k, logits.shape[-1]), dim=-1)
    token_ids = topk.indices.tolist()
    values = topk.values.tolist()
    entries = []
    for tok_id, val in zip(token_ids, values):
        token = tokenizer.id_to_token(int(tok_id))
        entries.append(f"{token}:{val:.4f}")
    return ", ".join(entries)


def write_generated_logits_report(f, task, aa_mask, tokenizer, progen3_logits, max_logits, recovered_logits):
    aa_indices = [i for i, is_aa in enumerate(aa_mask) if is_aa]
    if not aa_indices:
        f.write("No generated amino-acid positions were produced by ProGen3.\n\n")
        return

    if task.task_type == "CLM":
        start_res = len(task.prompt)
    else:
        start_res = int(task.span_start or 0)

    f.write("Generated token top-5 logits per residue:\n")
    for aa_idx, step_idx in enumerate(aa_indices):
        residue = task.target_sequence[aa_idx] if aa_idx < len(task.target_sequence) else "?"
        pos = start_res + aa_idx + 1
        f.write(f"  Residue {pos} ({residue}):\n")
        f.write(f"    ProGen3: {format_topk_logits(progen3_logits[step_idx], tokenizer)}\n")
        f.write(f"    All Latents: {format_topk_logits(max_logits[step_idx], tokenizer)}\n")
        f.write(f"    Circuit: {format_topk_logits(recovered_logits[step_idx], tokenizer)}\n")
    f.write("\n")


def write_analysis_txt(output_path, task, sequence, nodes, attr_by_layer, ref_storage, tokenizer, progen3_logits, max_logits, recovered_logits, aa_mask):
    sorted_nodes = sorted(nodes, key=lambda x: x["score"], reverse=True)
    with open(output_path, "w") as f:
        f.write("=== CLT/GLM Top Activation Analysis ===\n")
        f.write(f"Task type: {task.task_type}\n")
        f.write(f"Prompt: {task.prompt}\n")
        f.write(f"Sequence Length: {len(sequence)}\n")
        if task.task_type == "CLM":
            start_res = len(task.prompt) + 1
            end_res = len(task.prompt) + len(task.target_sequence)
            f.write(f"Generated window: residues {start_res} to {end_res}\n")
        else:
            span_start = int(task.span_start or 0) + 1
            span_end = int(task.span_end or 0)
            f.write(f"Generated window: residues {span_start} to {span_end}\n")
        f.write(f"Total Gradient Nodes Selected: {len(sorted_nodes)}\n")
        f.write("-------------------------------------------\n\n")
        write_generated_logits_report(f, task, aa_mask, tokenizer, progen3_logits, max_logits, recovered_logits)
        for idx, item in enumerate(sorted_nodes, start=1):
            rank_label = f"Rank #{idx}"
            print_result_block(f, rank_label, item, sequence, ref_storage)


def generate_reconstructed_sequence(task: TaskSpec, discoverer, tokenizer, original_sequence: str, active_nodes, p=0.95, temperature=0.85):
    model_inputs = get_generation_kwargs(discoverer, task)
    token_ids = []
    step_logits = []

    with torch.inference_mode():
        for _ in range(task.max_steps):
            logits = discoverer.reconstruct_logits(
                model_inputs=model_inputs,
                active_nodes=active_nodes,
                sequential=True,
                freeze_attention=True,
            )
            last_logits = logits[:, -1, :]
            token_id = top_p_sample(last_logits, p=p, temperature=temperature)
            token_ids.append(token_id)
            step_logits.append(last_logits.squeeze(0).detach().cpu())
            model_inputs = append_generation_step(model_inputs, token_id)

    aa_mask, tokens = token_ids_to_aa_mask_and_tokens(tokenizer, token_ids)
    sequence = compile_output_sequence(task, tokens, original_sequence)
    return token_ids, torch.stack(step_logits, dim=0), aa_mask, sequence


def main():
    parser = argparse.ArgumentParser(description="Auto-Discover Circuits for ProGen3")
    parser.add_argument("--type", type=str, required=True, choices=["CLM", "GLM", "zero_shot"])
    parser.add_argument("--seq", type=str, help="Protein sequence for CLM/GLM")
    parser.add_argument("--sequences", type=str, nargs="+",
                        help="List of sequences for differential analysis. First sequence is treated as Wildtype/Reference.")
    parser.add_argument("--pos", type=int, help="1-indexed start position for CLM/GLM")
    parser.add_argument("--n_generated", type=int, default=5, help="Number of amino acids to generate")
    parser.add_argument("--generation_recovery_ratio", type=float, default=1.2, help="Recovery ratio for KL")
    parser.add_argument("--circuit_json", type=str, help="Optional existing circuit JSON path to skip circuit discovery and use precomputed nodes.")
    parser.add_argument("--csv_path", type=str, help="CSV path for zero_shot")
    parser.add_argument("--n_zero_shot_sequences", type=int, default=128, help="Number of sequences for zero_shot training")
    parser.add_argument("--n_test_sequences", type=int, default=1000, help="Number of test sequences")
    parser.add_argument("--zero_shot_recovery_ratio", type=float, default=0.7, help="Recovery ratio for Spearman")
    parser.add_argument("--output_dir", type=str, default="auto_circuits")
    parser.add_argument("--entry_name", type=str, default="experiment")
    parser.add_argument("--max_nodes", type=int, default=1000)
    parser.add_argument("--step_size", type=int, default=32)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    circuit_nodes_map = None
    circuit_family = None
    if args.circuit_json:
        if not os.path.exists(args.circuit_json):
            raise ValueError(f"Circuit JSON not found: {args.circuit_json}")
        with open(args.circuit_json, "r") as f:
            circuit_data = json.load(f)
        if "nodes" not in circuit_data:
            raise ValueError(f"Circuit JSON must contain a 'nodes' key: {args.circuit_json}")
        circuit_nodes_map = {str(k): set(v) for k, v in circuit_data["nodes"].items()}
        circuit_family = circuit_data.get("family", "Unknown")
        print(f"Loaded circuit JSON from {args.circuit_json}. Skipping circuit discovery.")

    default_clt_ckpt = os.path.join(repo_root, "models", "ProGen3_CLT_L10_D4608", "checkpoints", "last.ckpt")
    clt_ckpt = os.environ.get("CLT_CHECKPOINT", default_clt_ckpt)
    os.environ["CLT_CHECKPOINT"] = clt_ckpt

    if not os.path.exists(clt_ckpt):
        raise ValueError(
            f"CLT checkpoint not found: {clt_ckpt}.\n"
            "Set CLT_CHECKPOINT or place the checkpoint at the default path."
        )

    clt_discoverer = CircuitDiscovererCLT(device=device, ckpt_path=clt_ckpt)
    clt_scorer = CLTPLTReconstructionScorer(discoverer=clt_discoverer)
    progen_scorer = ProGen3Scorer(clt_discoverer.progen)
    circuit_scorer = ScorerCircuit(clt_discoverer.progen, clt_discoverer.tokenizer)
    tokenizer = clt_discoverer.tokenizer

    if args.type in ["CLM", "GLM"]:
        if not args.seq:
            raise ValueError("--seq required for CLM/GLM")
        pos = normalize_pos(args.pos)
        task = build_task_for_clm_glm(args.seq, pos, args.n_generated, args.type)

        print("Generating ProGen3 baseline...")
        progen3_ids, progen3_logits, aa_mask, progen3_sequence = generate_progen3_sequence(task, clt_discoverer, tokenizer, args.seq)
        print(f"ProGen3 sequence: {progen3_sequence}")

        reference_sequence = args.seq
        if task.task_type == "CLM" and pos is None and len(task.target_sequence) == 0:
            task.target_sequence = progen3_sequence[len(task.prompt):]
            reference_sequence = progen3_sequence
            if len(task.target_sequence) == 0:
                raise ValueError("ProGen3 generated no amino-acid output for CLM append-mode.")

        if task.task_type == "CLM":
            generated_output = progen3_sequence[len(task.prompt):]
            prompt_text = f"{task.prompt}<CLM>"
            output_label = "generated_output"
        else:
            generated_output = progen3_sequence[task.span_start:task.span_end]
            prompt_text = f"{args.seq[:task.span_start]}<GLM>{args.seq[task.span_end:]}"
            output_label = "output"
        generation_fasta = Path(args.output_dir) / "generation.fasta"
        generation_fasta.write_text(
            f">prompt\n{prompt_text}\n>{output_label}\n{generated_output}\n",
            encoding="utf-8",
        )

        print("Computing CLT max logits...")
        max_ids, max_logits, _, max_sequence = generate_reconstructed_sequence(task, clt_discoverer, tokenizer, args.seq, active_nodes=None)

        print("Computing CLT base logits...")
        base_ids, base_logits, _, base_sequence = generate_reconstructed_sequence(task, clt_discoverer, tokenizer, args.seq, active_nodes={})

        max_kl, max_nmse = compute_kl_and_nmse(progen3_logits, max_logits, aa_mask)
        base_kl, base_nmse = compute_kl_and_nmse(progen3_logits, base_logits, aa_mask)

        print(f"All Latents KL: {max_kl:.6f}")
        print(f"Base KL: {base_kl:.6f}")

        original_sequence = reference_sequence
        original_inputs = clt_discoverer._prepare_inputs([original_sequence])
        generated_mask_BT = make_generated_mask(task, clt_discoverer, original_inputs)
        global_attr = compute_attribution(
            clt_discoverer,
            [original_sequence],
            batch_size=1,
            sequential=True,
            freeze_attention=True,
            source="layer_output",
            generated_mask_fn=lambda batch, inputs: generated_mask_BT,
        )
        ranking = rank_nodes(global_attr)

        target_kl = args.generation_recovery_ratio * max_kl
        if args.circuit_json and circuit_nodes_map is not None:
            print(f"Skipping circuit discovery and using nodes from {args.circuit_json}.")
            best_nodes = {}
            for layer_str, latents in circuit_nodes_map.items():
                layer_idx = int(layer_str)
                best_nodes[layer_idx] = {int(latent) for latent in latents}
            best_k = sum(len(s) for s in best_nodes.values())
            best_metric = None
            print(f"Loaded {best_k} nodes from circuit JSON.")
        else:
            print(f"Searching for circuit with target KL <= {target_kl:.6f}... ")

            def search_metric(discoverer, probe, seqs, y, active_nodes, batch_size):
                ablation = discoverer.run_ablation(
                    batch_seqs=[original_sequence],
                    active_nodes=active_nodes,
                    sequential=True,
                    freeze_attention=True,
                    source="layer_output",
                    generated_mask_BT=generated_mask_BT,
                )
                return -float(ablation["kl"])

            best_nodes, best_k, best_metric = circuit_search(
                clt_discoverer,
                None,
                ranking,
                [task.prompt_12],
                [0.0],
                target_metric=-target_kl,
                metric_fn=search_metric,
                step_size=args.step_size,
                max_nodes=args.max_nodes,
                desc="Generation Search",
                metric_name="neg_kl",
            )
            print(f"Selected {best_k} nodes for the circuit")

        generated_mask_fn = build_generated_mask_fn(clt_discoverer, task)
        attr_by_layer = compute_attribution(
            clt_discoverer,
            [original_sequence],
            batch_size=1,
            sequential=True,
            freeze_attention=True,
            source="layer_output",
            generated_mask_fn=generated_mask_fn,
        )
        if args.sequences:
            sequence_list = args.sequences
            if len(sequence_list) < 1:
                raise ValueError("--sequences must contain at least one sequence")
            lengths = [len(s) for s in sequence_list]
            if len(set(lengths)) != 1:
                raise ValueError("All sequences in --sequences must have the same length.")

            print("Computing wildtype gradient ranking via zero-shot for sequence analysis...")
            attr_by_layer = clt_discoverer.get_gradients(
                [sequence_list[0]],
                sequential=True,
                freeze_attention=True,
                source="layer_output",
                zero_shot=True,
            )
            top_attr_nodes = select_top_latents_per_layer(attr_by_layer, TARGET_PER_LAYER)
            top_attr_nodes_by_layer = {}
            for layer, latent, score in top_attr_nodes:
                top_attr_nodes_by_layer.setdefault(str(layer), []).append((latent, score))

            seq_dirs = []
            traces_caches = []
            for idx, seq in enumerate(sequence_list):
                seq_dir = Path(args.output_dir) / f"seq{idx+1}"
                seq_dir.mkdir(parents=True, exist_ok=True)
                generation_path = seq_dir / "generation.fasta"
                generation_path.write_text(f">prompt\n{seq}\n", encoding="utf-8")
                seq_dirs.append(seq_dir)
                traces_caches.append(get_clt_latent_traces(clt_discoverer, seq))

            wt_traces = traces_caches[0]
            diff_items_by_seq = []
            for seq_idx in range(1, len(sequence_list)):
                seq_traces = traces_caches[seq_idx]
                seq_diff_scores = []
                for layer in range(clt_discoverer.num_layers):
                    wt_layer = wt_traces[layer]
                    other_layer = seq_traces[layer]
                    dim = min(wt_layer.shape[1], other_layer.shape[1])
                    for latent_id in range(dim):
                        wt_activation = float(np.max(wt_layer[:, latent_id]))
                        seq_activation = float(np.max(other_layer[:, latent_id]))
                        diff_val = abs(wt_activation - seq_activation)
                        wt_peak = int(np.argmax(wt_layer[:, latent_id])) + 1
                        seq_peak = int(np.argmax(other_layer[:, latent_id])) + 1
                        wt_motif, _, _ = get_adaptive_motif_window(sequence_list[0], wt_layer[:, latent_id], wt_peak, min_radius=5, buffer=3, top_k=None)
                        seq_motif, _, _ = get_adaptive_motif_window(sequence_list[seq_idx], other_layer[:, latent_id], seq_peak, min_radius=5, buffer=3, top_k=None)
                        seq_diff_scores.append({
                            "layer": layer,
                            "latent": int(latent_id),
                            "diff": diff_val,
                            "seqA_data": {
                                "activation": wt_activation,
                                "attribution": float(attr_by_layer[layer][latent_id]) if layer in attr_by_layer and latent_id < len(attr_by_layer[layer]) else 0.0,
                                "trace": wt_layer[:, latent_id],
                                "peak_idx": wt_peak,
                                "motif": wt_motif,
                            },
                            "seqB_data": {
                                "activation": seq_activation,
                                "attribution": float(attr_by_layer[layer][latent_id]) if layer in attr_by_layer and latent_id < len(attr_by_layer[layer]) else 0.0,
                                "trace": other_layer[:, latent_id],
                                "peak_idx": seq_peak,
                                "motif": seq_motif,
                            },
                        })
                selected_diff_nodes = select_top_nodes_generic(seq_diff_scores, "diff", TARGET_DIFF_PER_LAYER, top_k_global=20)
                diff_items_by_seq.append(selected_diff_nodes)

            global_union_nodes = set((layer, latent) for layer, latent, _ in top_attr_nodes)
            if args.circuit_json and best_nodes is not None:
                for layer, latents in best_nodes.items():
                    for latent in latents:
                        global_union_nodes.add((layer, latent))
            for diff_items in diff_items_by_seq:
                for item in diff_items:
                    global_union_nodes.add((item["layer"], item["latent"]))
        else:
            top_attr_nodes = select_top_latents_per_layer(attr_by_layer, TARGET_PER_LAYER)
            top_attr_nodes_by_layer = {}
            for layer, latent, score in top_attr_nodes:
                top_attr_nodes_by_layer.setdefault(str(layer), []).append((latent, score))
            global_union_nodes = set((layer, latent) for layer, latent, _ in top_attr_nodes)

        recovered_ids, recovered_logits, _, recovered_sequence = generate_reconstructed_sequence(
            task,
            clt_discoverer,
            tokenizer,
            args.seq,
            active_nodes=best_nodes,
        )
        recovered_kl, recovered_nmse = compute_kl_and_nmse(progen3_logits, recovered_logits, aa_mask)
        print(f"Circuit KL: {recovered_kl:.6f}")

        max_sequence = cap_sequence_to_reference(task, max_sequence, progen3_sequence, args.seq)
        base_sequence = cap_sequence_to_reference(task, base_sequence, progen3_sequence, args.seq)
        recovered_sequence = cap_sequence_to_reference(task, recovered_sequence, progen3_sequence, args.seq)

        all_sequences = [progen3_sequence, max_sequence, base_sequence, recovered_sequence]
        tasks_for_scoring = [task] * 4
        scores = circuit_scorer.score_batch_with_tasks(all_sequences, tasks_for_scoring)

        record = {
            "entry": args.entry_name,
            "type": args.type,
            "prompt": task.prompt,
            "k": int(best_k),
            "progen3_sequence": progen3_sequence,
            "progen3_ll": scores["log_likelihood"][0],
            "progen3_perplexity": scores["perplexity"][0],
            "max_sequence": max_sequence,
            "max_ll": scores["log_likelihood"][1],
            "max_perplexity": scores["perplexity"][1],
            "max_kl": float(max_kl),
            "max_nmse": float(max_nmse),
            "base_sequence": base_sequence,
            "base_ll": scores["log_likelihood"][2],
            "base_perplexity": scores["perplexity"][2],
            "base_kl": float(base_kl),
            "base_nmse": float(base_nmse),
            "recovered_sequence": recovered_sequence,
            "recovered_ll": scores["log_likelihood"][3],
            "recovered_perplexity": scores["perplexity"][3],
            "recovered_kl": float(recovered_kl),
            "recovered_nmse": float(recovered_nmse),
            "nodes": {str(layer): sorted(list(nodes)) for layer, nodes in best_nodes.items()},
            "top_gradient_nodes": {
                layer: [{"latent": int(latent), "score": float(score)} for latent, score in latents]
                for layer, latents in top_attr_nodes_by_layer.items()
            },
        }

        output_json = Path(args.output_dir) / f"{args.entry_name}.json"
        output_seq = Path(args.output_dir) / "seq.txt"
        output_json.write_text(json.dumps(record, indent=2), encoding="utf-8")
        output_seq.write_text(args.seq, encoding="utf-8")

        top10_ref_path = os.path.join(repo_root, "visualization", "top10_activations.pt")
        ref_storage = load_dataset_reference(top10_ref_path)
        if args.circuit_json and best_nodes is not None:
            circuit_nodes_set = set((layer, latent) for layer, latents in best_nodes.items() for latent in sorted(latents))
            nodes_set = sorted(circuit_nodes_set | global_union_nodes) if args.sequences else sorted(circuit_nodes_set)
        elif args.sequences:
            nodes_set = sorted(global_union_nodes)
        else:
            nodes_set = [(layer, latent) for layer, latent, _ in top_attr_nodes]
        write_top_activation_json(Path(args.output_dir) / "top_activations.json", nodes_set, ref_storage, args.entry_name)

        analysis_sequence = args.seq if task.task_type == "GLM" else task.prompt
        traces_cache = get_clt_latent_traces(clt_discoverer, analysis_sequence)
        write_activation_indices_json(Path(args.output_dir) / "activation_indices.json", nodes_set, traces_cache, len(analysis_sequence))

        analysis_items = build_analysis_items(attr_by_layer, traces_cache, TARGET_PER_LAYER)
        write_analysis_txt(
            Path(args.output_dir) / "analysis.txt",
            task,
            analysis_sequence,
            analysis_items,
            attr_by_layer,
            ref_storage,
            tokenizer,
            progen3_logits,
            max_logits,
            recovered_logits,
            aa_mask,
        )

        if args.sequences:
            print("Writing sequence-specific website files...")
            for idx, seq_dir in enumerate(seq_dirs):
                traces_cache_seq = traces_caches[idx]
                write_top_activation_json(seq_dir / "top_activations.json", nodes_set, ref_storage, args.entry_name)
                write_activation_indices_json(seq_dir / "activation_indices.json", nodes_set, traces_cache_seq, len(sequence_list[idx]))

                top_items = build_items_for_nodes(nodes_set, attr_by_layer, traces_cache_seq)
                with open(Path(args.output_dir) / f"top_seq{idx+1}.txt", "w") as f:
                    f.write(f"=== Top Activations: Seq{idx+1} ===\n")
                    f.write(f"Total Nodes: {len(top_items)}\n")
                    f.write("-------------------------------------\n\n")
                    for k, item in enumerate(sorted(top_items, key=lambda x: x["score"], reverse=True), start=1):
                        rank_label = f"Rank #{k}"
                        print_result_block(f, rank_label, item, sequence_list[idx], ref_storage)

            for j, diff_items in enumerate(diff_items_by_seq, start=1):
                path = Path(args.output_dir) / f"analysis_differential_1{j+1}.txt"
                title = f"Differential Analysis: Seq1 (WT) vs Seq{j+1}"
                with open(path, "w") as f:
                    f.write(f"=== {title} ===\n")
                    f.write(f"Total Diff Nodes Analyzed: {len(diff_items)}\n")
                    f.write("-------------------------------------\n\n")
                    for k, item in enumerate(diff_items, start=1):
                        rank_label = f"Rank #{k}"
                        print_differential_result_block(f, rank_label, item, sequence_list[0], sequence_list[j], ref_storage)

        np.save(Path(args.output_dir) / "logits.npy", recovered_logits.numpy())

    else:
        no_dataset_mode = False
        if args.circuit_json and args.sequences:
            all_seqs = np.array(args.sequences, dtype=object)
            all_scores = np.zeros(len(all_seqs), dtype=float)
            no_dataset_mode = True
        else:
            if not args.csv_path:
                raise ValueError("--csv_path required for zero_shot unless --circuit_json and --sequences are provided")
            all_seqs, all_scores = load_and_parse_csv_regression(args.csv_path)

        if no_dataset_mode:
            train_seqs = all_seqs
            train_scores = all_scores
            test_seqs = all_seqs
            test_scores = all_scores
            print(f"Using {len(all_seqs)} provided sequences with circuit JSON. Skipping zero-shot discovery.")
        else:
            if len(all_seqs) < args.n_zero_shot_sequences:
                n_train = max(1, int(np.ceil(0.1 * len(all_seqs))))
            else:
                n_train = min(args.n_zero_shot_sequences, len(all_seqs) - 1)
            n_test = min(args.n_test_sequences, len(all_seqs) - n_train)
            if n_test <= 0:
                raise ValueError("Not enough sequences for zero_shot evaluation after reserving training examples")

            rng = np.random.default_rng(42)
            perm = rng.permutation(len(all_seqs))
            train_idx = perm[:n_train]
            test_idx = perm[n_train:n_train + n_test]
            train_seqs = all_seqs[train_idx]
            train_scores = all_scores[train_idx]
            test_seqs = all_seqs[test_idx]
            test_scores = all_scores[test_idx]

            print(f"Zero-shot training on {len(train_seqs)} sequences and evaluating on {len(test_seqs)} sequences.")

        if args.circuit_json and circuit_nodes_map is not None:
            best_nodes = {}
            for layer_str, latents in circuit_nodes_map.items():
                layer_idx = int(layer_str)
                best_nodes[layer_idx] = {int(latent) for latent in latents}
            best_k = sum(len(s) for s in best_nodes.values())
            best_metric = None
            progen_spearman = None
            print(f"Loaded {best_k} nodes from circuit JSON. Skipping zero-shot discovery.")
        else:
            all_gradients = {}
            for seq in train_seqs:
                grads = clt_discoverer.get_gradients([seq], sequential=True, freeze_attention=True, zero_shot=True)
                for layer, values in grads.items():
                    if layer not in all_gradients:
                        all_gradients[layer] = np.array(values, copy=True)
                    else:
                        all_gradients[layer] += np.array(values, copy=True)

            ranking = rank_nodes(all_gradients)

            progen_scores = progen_scorer.evaluate(list(train_seqs))["log_likelihood"]
            progen_spearman, _ = get_spearman_p(progen_scores, train_scores)
            print(f"ProGen3 train Spearman: {progen_spearman:.4f}")

            def eval_fn(d, probe, seqs, y, active_nodes, batch_size):
                scores = clt_scorer.score(list(seqs), task={"sequential": True, "freeze_attention": True}, active_nodes=active_nodes)["log_likelihood"]
                spearman, _ = get_spearman_p(scores, y)
                return spearman

            best_nodes, best_k, best_metric = circuit_search(
                clt_discoverer,
                None,
                ranking,
                list(train_seqs),
                list(train_scores),
                target_metric=args.zero_shot_recovery_ratio * progen_spearman,
                metric_fn=eval_fn,
                step_size=args.step_size,
                max_nodes=args.max_nodes,
                desc="Zero-Shot Search",
                metric_name="spearman",
            )

            print(f"Selected {best_k} nodes for zero-shot circuit")
        if args.circuit_json and circuit_nodes_map is not None and no_dataset_mode:
            progen_test_scores = progen_scorer.evaluate(list(test_seqs))["log_likelihood"]
            max_test_scores = clt_scorer.score(list(test_seqs), task={"sequential": True, "freeze_attention": True}, active_nodes=None)["log_likelihood"]
            base_test_scores = clt_scorer.score(list(test_seqs), task={"sequential": True, "freeze_attention": True}, active_nodes={})["log_likelihood"]
            recovered_test_scores = clt_scorer.score(list(test_seqs), task={"sequential": True, "freeze_attention": True}, active_nodes=best_nodes)["log_likelihood"]

            progen_mean_ll = float(np.asarray(progen_test_scores).mean()) if len(progen_test_scores) > 0 else None
            max_mean_ll = float(np.asarray(max_test_scores).mean()) if len(max_test_scores) > 0 else None
            base_mean_ll = float(np.asarray(base_test_scores).mean()) if len(base_test_scores) > 0 else None
            recovered_mean_ll = float(np.asarray(recovered_test_scores).mean()) if len(recovered_test_scores) > 0 else None

            print(f"ProGen3 mean LL on provided sequences: {progen_mean_ll}")
            print(f"All Latents mean LL on provided sequences: {max_mean_ll}")
            print(f"Recovered circuit mean LL on provided sequences: {recovered_mean_ll}")

            record = {
                "entry": args.entry_name,
                "type": args.type,
                "k": int(best_k),
                "progen3_mean_ll": progen_mean_ll,
                "max_mean_ll": max_mean_ll,
                "base_mean_ll": base_mean_ll,
                "recovered_mean_ll": recovered_mean_ll,
                "nodes": {str(layer): sorted(list(nodes)) for layer, nodes in best_nodes.items()},
            }
        else:
            progen_test_scores = progen_scorer.evaluate(list(test_seqs))["log_likelihood"]
            progen_test_spearman, _ = get_spearman_p(progen_test_scores, test_scores)

            max_test_scores = clt_scorer.score(list(test_seqs), task={"sequential": True, "freeze_attention": True}, active_nodes=None)["log_likelihood"]
            max_test_spearman, _ = get_spearman_p(max_test_scores, test_scores)

            base_test_scores = clt_scorer.score(list(test_seqs), task={"sequential": True, "freeze_attention": True}, active_nodes={})["log_likelihood"]
            base_test_spearman, _ = get_spearman_p(base_test_scores, test_scores)

            recovered_test_scores = clt_scorer.score(list(test_seqs), task={"sequential": True, "freeze_attention": True}, active_nodes=best_nodes)["log_likelihood"]
            recovered_test_spearman, _ = get_spearman_p(recovered_test_scores, test_scores)

            print(f"ProGen3 test Spearman: {progen_test_spearman:.4f}")
            print(f"All Latents test Spearman: {max_test_spearman:.4f}")
            print(f"Recovered circuit test Spearman: {recovered_test_spearman:.4f}")

            record = {
                "entry": args.entry_name,
                "type": args.type,
                "k": int(best_k),
                "progen3_spearman": float(progen_test_spearman),
                "max_spearman": float(max_test_spearman),
                "base_spearman": float(base_test_spearman),
                "recovered_spearman": float(recovered_test_spearman),
                "nodes": {str(layer): sorted(list(nodes)) for layer, nodes in best_nodes.items()},
            }

        output_json = Path(args.output_dir) / f"{args.entry_name}.json"
        output_json.write_text(json.dumps(record, indent=2), encoding="utf-8")

        if args.circuit_json and args.sequences:
            print("Writing zero-shot sequence-specific website files...")
            top10_ref_path = os.path.join(repo_root, "visualization", "top10_activations.pt")
            ref_storage = load_dataset_reference(top10_ref_path)
            write_sequence_outputs(
                Path(args.output_dir),
                args.sequences,
                clt_discoverer,
                best_nodes,
                ref_storage,
                clt_scorer,
                progen_scorer,
                task_config={"sequential": True, "freeze_attention": True},
                zero_shot=True,
            )

    print(f"Saved output to {args.output_dir}")


if __name__ == "__main__":
    main()
