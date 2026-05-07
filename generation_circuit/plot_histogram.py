#!/usr/bin/env python3
"""
Plot histograms of selected metrics.

Scans a gen output directory, collects values for:
- CLT_sequential progen3_ll
- CLT_sequential max_ll
- CLT_sequential recovered_ll
- PLT max_ll
- PLT recovered_ll

Then renders 5 subplots with histograms for each.
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


METRICS = ["max_ll", "recovered_ll", "progen3_ll"]
GROUPS = [
    ("CLT_direct", "progen3_ll"),
    ("CLT_direct", "max_ll"),
    ("CLT_direct", "recovered_ll"),
    ("CLT_sequential", "max_ll"),
    ("CLT_sequential", "recovered_ll"),
    ("PLT", "max_ll"),
    ("PLT", "recovered_ll"),
]
CLM_PATTERN = re.compile(r"_CLM\.json$")
GLM_PATTERN = re.compile(r"_GLM_(1|2)\.json$")


def filename_matches_mode(json_path: Path, mode: str) -> bool:
    """Return True when a JSON filename should be included for the selected mode."""
    if mode == "all":
        return True
    if mode == "clm":
        return bool(CLM_PATTERN.search(json_path.name))
    if mode == "glm":
        return bool(GLM_PATTERN.search(json_path.name))
    return True


def read_method_metric_values(method_dir: Path, mode: str) -> Dict[str, List[float]]:
    """Collect metric values from JSON files in one method directory for a mode."""
    values: Dict[str, List[float]] = {m: [] for m in METRICS}

    for json_path in sorted(method_dir.glob("*.json")):
        if not filename_matches_mode(json_path, mode):
            continue

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                record = json.load(f)
        except Exception:
            continue

        for metric in METRICS:
            v = record.get(metric)
            if isinstance(v, (int, float)):
                values[metric].append(float(v))

    return values


def collect_method_values(gen_dir: Path, mode: str) -> Dict[str, Dict[str, List[float]]]:
    """Collect raw metric values from JSON files for each method and metric under a mode."""
    method_values: Dict[str, Dict[str, List[float]]] = {}

    method_dirs = sorted([p for p in gen_dir.iterdir() if p.is_dir()])
    if not method_dirs:
        raise ValueError(f"No method directories found in {gen_dir}")

    for method_dir in method_dirs:
        values = read_method_metric_values(method_dir, mode)
        method_values[method_dir.name] = values

    return method_values


def plot_overlapping_histograms(method_values: Dict[str, Dict[str, List[float]]], output_png: Path) -> None:
    """Create and save 5 subplots with histograms for selected method-metric combinations."""
    n_groups = len(GROUPS)

    fig, axes = plt.subplots(1, n_groups, figsize=(6*n_groups, 6), sharey=True)
    if n_groups == 1:
        axes = [axes]

    for i, (method, metric) in enumerate(GROUPS):
        ax = axes[i]
        values = method_values[method][metric]
        if not values:
            continue
        ax.hist(
            values,
            bins=30,
            alpha=0.7,
            color='skyblue',
            edgecolor='black',
            linewidth=0.5,
        )
        ax.set_xlabel(f"{method} {metric}")
        ax.set_ylabel("Frequency" if i == 0 else "")
        ax.set_title(f"{method} {metric}")
        ax.grid(axis="y", linestyle="--", alpha=0.3)

    fig.suptitle("Histograms of Selected Metrics")
    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=200)
    plt.close(fig)


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gen_dir",
        type=str,
        default="results_p0.95_T0.5",
        help="Directory with method subdirectories containing JSON results.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="clm",
        choices=["all", "clm", "glm"],
        help="Which JSON rows to plot: all, only *_CLM.json, or only *_GLM_1/_GLM_2.json.",
    )
    parser.add_argument(
        "--output_png",
        type=str,
        default="metrics_grouped_bar.png",
        help="Path to save the grouped bar chart PNG.",
    )
    args = parser.parse_args()

    gen_dir = Path(args.gen_dir)
    output_png = Path(args.output_png)

    if not gen_dir.exists():
        raise ValueError(f"gen_dir does not exist: {gen_dir}")

    method_values = collect_method_values(gen_dir, args.mode)

    print(f"Mode: {args.mode}")
    for method, metric in GROUPS:
        values = method_values[method][metric]
        if values:
            mean = np.mean(values)
            std = np.std(values, ddof=0)
            print(f"  {method} {metric}: n={len(values)} mean={mean:.6f} std={std:.6f}")
        else:
            print(f"  {method} {metric}: no data")

    plot_overlapping_histograms(method_values, output_png)
    print(f"Saved plot -> {output_png}")


if __name__ == "__main__":
    main()
