import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# Make repo imports available
_repo_root = Path(__file__).resolve().parents[1]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from circuit_utils.clt_circuit import CircuitDiscovererCLT
from circuit_utils.plt_circuit import CircuitDiscovererPLT
from steering_patching_correction.full_replacement_models import FullCLTReplacementModel, FullPLTReplacementModel
from steering_patching_correction.steering_utils import (
    infer_wildtype_from_dms,
    load_dms_samples,
    sample_latent_circuit,
    sample_latent_circuit_from_json,
    generate_baseline_suffix,
    generate_replacement_suffix,
    make_special_token_mask,
)
from external.progen3.src.progen3.scorer import ProGen3Scorer
from function_circuit.function_utils import set_global_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Run CLT/PLT steering generation experiments")
    parser.add_argument("--dms_csv", type=str, required=True)
    parser.add_argument("--clt_ckpt", type=str, required=True)
    parser.add_argument("--plt_ckpt", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, default="steering_results")
    parser.add_argument("--num_train_seq", type=int, default=128)
    parser.add_argument("--num_test_seq", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prefix_frac", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--alpha_min", type=float, default=0.1)
    parser.add_argument("--alpha_max", type=float, default=2.0)
    parser.add_argument("--alpha_steps", type=int, default=5)
    parser.add_argument("--num_latents", type=int, default=32)
    parser.add_argument("--top_k", type=int, default=1000)
    parser.add_argument("--clt_json", type=str, default=None,
                        help="Optional CLT circuit JSON file to sample steering nodes from")
    parser.add_argument("--plt_json", type=str, default=None,
                        help="Optional PLT circuit JSON file to sample steering nodes from")
    parser.add_argument("--freeze_attention", action="store_true")
    parser.add_argument("--before", action="store_true")
    return parser.parse_args()


def build_alpha_list(alpha_min, alpha_max, alpha_steps):
    if alpha_steps <= 1:
        return [alpha_min]
    return np.linspace(alpha_min, alpha_max, alpha_steps).tolist()


def sample_gradient_circuit(discoverer, sequences, num_latents, top_k, seed, sequential, freeze_attention, json_path=None):
    if json_path is not None:
        print(f"  Sampling steering circuit from JSON: {json_path}")
        return sample_latent_circuit_from_json(json_path, num_latents=num_latents, seed=seed)

    print(f"  Gathering gradients for {discoverer.__class__.__name__} (...)" )
    batch_grads = discoverer.get_gradients(
        sequences,
        sequential=sequential,
        freeze_attention=freeze_attention,
        zero_shot=True,
    )
    grad_items = []
    for layer, values in batch_grads.items():
        arr = np.asarray(values)
        for idx, score in enumerate(arr.flatten()):
            grad_items.append({"layer": int(layer), "idx": int(idx), "score": float(score)})
    return sample_latent_circuit(grad_items, num_latents=num_latents, top_k=top_k, seed=seed)


def score_sequences(progen_scorer, sequences):
    if not sequences:
        return []
    with torch.no_grad():
        scores = progen_scorer.evaluate(sequences)["log_likelihood"].tolist()
    return [float(v) for v in scores]


def main():
    args = parse_args()
    set_global_seed(args.seed)

    if not os.path.exists(args.dms_csv):
        raise FileNotFoundError(f"DMS CSV not found: {args.dms_csv}")

    print(f"Loading DMS samples from {args.dms_csv}")
    train_seqs, train_scores, test_seqs, test_scores = load_dms_samples(
        args.dms_csv,
        num_train_seq=args.num_train_seq,
        num_test_seq=args.num_test_seq,
        seed=args.seed,
    )
    wildtype = infer_wildtype_from_dms(args.dms_csv)
    print(f"Inferred wildtype length {len(wildtype)}")

    device = torch.device(args.device)
    print("Loading discoverers and replacement models...")
    clt_discoverer = CircuitDiscovererCLT(device, ckpt_path=args.clt_ckpt)
    plt_discoverer = CircuitDiscovererPLT(device, ckpt_path=args.plt_ckpt)
    clt_model = FullCLTReplacementModel(clt_discoverer.pl_module, device)
    plt_model = FullPLTReplacementModel(plt_discoverer.pl_module, device)
    progen_scorer = ProGen3Scorer(clt_discoverer.progen)
    progen_scorer.model = progen_scorer.model.to(device)

    tokenizer = clt_discoverer.batch_preparer.tokenizer
    special_ids = make_special_token_mask(tokenizer)

    prefix_len = max(1, int(len(wildtype) * args.prefix_frac))
    prefix = wildtype[:prefix_len]
    suffix_len = len(wildtype) - prefix_len
    print(f"Using prefix length {prefix_len}, suffix length {suffix_len}")

    clt_circuit = sample_gradient_circuit(
        clt_discoverer,
        train_seqs,
        num_latents=args.num_latents,
        top_k=args.top_k,
        seed=args.seed,
        sequential=True,
        freeze_attention=args.freeze_attention,
        json_path=args.clt_json,
    )
    plt_circuit = sample_gradient_circuit(
        plt_discoverer,
        train_seqs,
        num_latents=args.num_latents,
        top_k=args.top_k,
        seed=args.seed + 1,
        sequential=True,
        freeze_attention=args.freeze_attention,
        json_path=args.plt_json,
    )
    print(f"CLT circuit nodes: {sum(len(v) for v in clt_circuit.values())}")
    print(f"PLT circuit nodes: {sum(len(v) for v in plt_circuit.values())}")

    alphas = build_alpha_list(args.alpha_min, args.alpha_max, args.alpha_steps)
    print(f"Alpha values: {alphas}")

    os.makedirs(args.output_dir, exist_ok=True)
    result_rows = []

    wt_ll = score_sequences(progen_scorer, [wildtype])[0]
    print(f"Wildtype LL: {wt_ll:.4f}")
    result_rows.append({
        "method": "wildtype",
        "alpha": 0.0,
        "sequence": wildtype,
        "log_likelihood": wt_ll,
        "wildtype_likelihood": wt_ll,
        "generated_length": len(wildtype),
    })

    for alpha in alphas:
        progen3_seq = generate_baseline_suffix(
            prefix=prefix,
            suffix_len=suffix_len,
            progen_model=clt_discoverer.progen,
            batch_preparer=clt_discoverer.batch_preparer,
            tokenizer=tokenizer,
            special_ids=special_ids,
            top_p=args.top_p,
            temperature=args.temperature,
            device=device,
        )
        progen3_ll = score_sequences(progen_scorer, [progen3_seq])[0]
        print(f"alpha={alpha:.3f} progen3 generated sequence length {len(progen3_seq)}")
        print(f"alpha={alpha:.3f} progen3 LL: {progen3_ll:.4f}")

        clt_seq = generate_replacement_suffix(
            prefix=prefix,
            suffix_len=suffix_len,
            replacement_model=clt_model,
            circuit=clt_circuit,
            alpha=alpha,
            batch_preparer=clt_discoverer.batch_preparer,
            tokenizer=tokenizer,
            special_ids=special_ids,
            top_p=args.top_p,
            temperature=args.temperature,
            device=device,
            freeze_attention=args.freeze_attention,
            before=args.before,
        )
        plt_seq = generate_replacement_suffix(
            prefix=prefix,
            suffix_len=suffix_len,
            replacement_model=plt_model,
            circuit=plt_circuit,
            alpha=alpha,
            batch_preparer=plt_discoverer.batch_preparer,
            tokenizer=tokenizer,
            special_ids=special_ids,
            top_p=args.top_p,
            temperature=args.temperature,
            device=device,
            freeze_attention=args.freeze_attention,
            before=args.before,
        )
        scores = score_sequences(progen_scorer, [clt_seq, plt_seq])
        result_rows.append({
            "method": "progen3",
            "alpha": alpha,
            "sequence": progen3_seq,
            "log_likelihood": progen3_ll,
            "generated_length": len(progen3_seq),
        })
        result_rows.append({
            "method": "clt",
            "alpha": alpha,
            "sequence": clt_seq,
            "log_likelihood": scores[0],
            "generated_length": len(clt_seq),
        })
        result_rows.append({
            "method": "plt",
            "alpha": alpha,
            "sequence": plt_seq,
            "log_likelihood": scores[1],
            "generated_length": len(plt_seq),
        })
        print(f"alpha={alpha:.3f} progen3_ll={progen3_ll:.4f} clt_ll={scores[0]:.4f} plt_ll={scores[1]:.4f}")

    out_path = Path(args.output_dir) / f"steering_{Path(args.dms_csv).stem}.csv"
    df = pd.DataFrame(result_rows)
    df.to_csv(out_path, index=False)
    print(f"Saved steering results to {out_path}")


if __name__ == "__main__":
    main()
