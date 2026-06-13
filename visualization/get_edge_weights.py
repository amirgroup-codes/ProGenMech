#!/usr/bin/env python3
import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "external" / "progen3" / "src"))

from training.clt_module import CLTLightningModule
from circuit_utils.colab_compat import apply_colab_compat
from local_replacement_models import LocalReplacementModel

BATCH_SIZE = 32
EPSILON = 1e-10
DEFAULT_CLT_CKPT = str(REPO_ROOT / "models" / "ProGen3_CLT_L10_D4608" / "checkpoints" / "last.ckpt")


def detect_input_files(base_folder: Path) -> tuple[str, str, str]:
    fasta_path = base_folder / "generation.fasta"
    seq_path = base_folder / "seq.txt"
    if fasta_path.exists():
        return "fasta", str(fasta_path), str(seq_path) if seq_path.exists() else None
    if seq_path.exists():
        return "txt", None, str(seq_path)
    raise FileNotFoundError(
        f"Could not find generation.fasta or seq.txt in {base_folder.resolve()}"
    )


def parse_generation_fasta(fasta_path: Path):
    prompt = None
    output = None
    output_header = None
    with fasta_path.open("r", encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f if line.strip()]

    current_header = None
    current_text = []
    for line in lines:
        if line.startswith(">"):
            if current_header is not None:
                if current_header == "prompt":
                    prompt = "\n".join(current_text).strip()
                elif current_header in ("output", "generated_output"):
                    output = "\n".join(current_text).strip()
                    output_header = current_header
            current_header = line[1:].strip()
            current_text = []
        else:
            current_text.append(line)
    if current_header is not None:
        if current_header == "prompt":
            prompt = "\n".join(current_text).strip()
        elif current_header in ("output", "generated_output"):
            output = "\n".join(current_text).strip()
            output_header = current_header

    if prompt is None:
        raise ValueError(f"generation.fasta missing >prompt block: {fasta_path}")
    return prompt, output, output_header


def print_edge_stats(edges_list, label):
    print(f"\n{'='*50}")
    print(f"{label}: {len(edges_list)} edges with |weight| > {EPSILON}")
    if edges_list:
        mags = [abs(row[6]) for row in edges_list]
        print(f"  Avg |weight|: {sum(mags) / len(mags):.6f}")
        print(f"  Min |weight|: {min(mags):.6f}")
        print(f"  Max |weight|: {max(mags):.6f}")
    print(f"{'='*50}\n")


def load_activation_indices(path: Path):
    data = json.loads(path.read_text(encoding='utf-8'))
    return [(int(item[0]), int(item[1]), int(item[3])) for item in data]


def main():
    parser = argparse.ArgumentParser(description="Compute ProGen3 edge weights for CLT and logits")
    parser.add_argument("--base_folder", type=str, required=True, help="Base folder path")
    parser.add_argument("--clt_ckpt", type=str, default=DEFAULT_CLT_CKPT, help="Path to CLT checkpoint")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="PyTorch device")
    args = parser.parse_args()

    base_folder = Path(args.base_folder)
    if not base_folder.exists():
        raise FileNotFoundError(f"Base folder not found: {base_folder}")

    input_type, fasta_path, seq_path = detect_input_files(base_folder)
    if input_type == "fasta":
        prompt, output, output_header = parse_generation_fasta(Path(fasta_path))
        if "<CLM>" in prompt and output is not None:
            mode = "CLM"
            prompt_text = prompt.replace("<CLM>", "")
            full_sequence = prompt_text + output
            generated_output = output
        elif "<GLM>" in prompt and output is not None:
            raise NotImplementedError("GLM prompt conversion is not implemented yet.")
        else:
            mode = "zero_shot"
            if seq_path:
                full_sequence = Path(seq_path).read_text(encoding='utf-8').strip()
            else:
                full_sequence = prompt.replace("<CLM>", "").replace("<GLM>", "")
            generated_output = None
    else:
        mode = "zero_shot"
        full_sequence = Path(seq_path).read_text(encoding='utf-8').strip()
        generated_output = None

    activation_path = base_folder / "activation_indices.json"
    if not activation_path.exists():
        raise FileNotFoundError(f"activation_indices.json not found in {base_folder}")
    activation_indices = load_activation_indices(activation_path)
    seq_len = len(full_sequence)
    print(f"Loaded {len(activation_indices)} activation indices, sequence length: {seq_len}")
    print(f"Mode: {mode}")

    device = torch.device(args.device)
    print(f"Using device: {device}")

    if not Path(args.clt_ckpt).exists():
        raise FileNotFoundError(f"CLT checkpoint not found: {args.clt_ckpt}")

    clt_pl = CLTLightningModule.load_from_checkpoint(args.clt_ckpt, map_location=device)
    if apply_colab_compat(clt_pl, device):
        print("Legacy GPU: using eager MoE + PyTorch fallbacks for edge-weight computation.")
    clt_pl.to(device).eval()
    edge_model = LocalReplacementModel(clt_pl, device, base_prompt=full_sequence)

    activations = [(layer, token, feature) for layer, token, feature in activation_indices]
    activations_by_layer = defaultdict(list)
    for layer, token, feature in activations:
        activations_by_layer[layer].append((layer, token, feature))

    sorted_layers = sorted(activations_by_layer.keys(), reverse=True)
    print(f"\nNodes per layer:")
    for layer in sorted(activations_by_layer.keys()):
        print(f"  Layer {layer}: {len(activations_by_layer[layer])} nodes")
    print(f"  Total: {len(activations)} nodes\n")

    edges = []
    print(f"Computing edge weights (epsilon={EPSILON}, batch_size={BATCH_SIZE})")

    all_sources = []
    for src_layer in range(max(sorted_layers) + 1 if sorted_layers else 0):
        all_sources.extend(activations_by_layer.get(src_layer, []))

    for tgt_layer in tqdm(sorted_layers, desc="Target layers"):
        sources = []
        for src_layer in range(tgt_layer):
            sources.extend(activations_by_layer[src_layer])
        if not sources:
            continue
        for tgt in tqdm(activations_by_layer[tgt_layer], desc=f"Layer {tgt_layer}", leave=False):
            weights = edge_model.compute_target_batched(sources, tgt, rerun_base=False)
            for (src_layer, src_token, src_feature), weight in zip(sources, weights):
                if abs(weight) > EPSILON:
                    edges.append([
                        int(src_token),
                        int(src_layer),
                        int(src_feature),
                        int(tgt[1]),
                        int(tgt[0]),
                        int(tgt[2]),
                        float(weight),
                    ])

    if mode == "CLM" and generated_output:
        vocab_size = edge_model.tokenizer.get_vocab_size()
        generated_len = len(generated_output)
        prompt_len = len(prompt_text)
        for step in range(generated_len):
            tgt_token = prompt_len + step
            for token_id in range(vocab_size):
                weights = edge_model.compute_target_batched(all_sources, (edge_model.num_layers, tgt_token, token_id), rerun_base=False)
                for (src_layer, src_token, src_feature), weight in zip(all_sources, weights):
                    if abs(weight) > EPSILON:
                        edges.append([
                            int(src_token),
                            int(src_layer),
                            int(src_feature),
                            int(tgt_token),
                            int(edge_model.num_layers),
                            int(token_id),
                            float(weight),
                        ])

    print_edge_stats(edges, "All edges")
    output_path = base_folder / "virtual_weights.json"
    output_path.write_text(json.dumps(edges), encoding="utf-8")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    from tqdm import tqdm
    main()
