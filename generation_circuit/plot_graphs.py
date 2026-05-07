#!/usr/bin/env python3
"""
Plot grouped bar charts of discovery metrics across methods.

Scans a gen output directory where each method has a subdirectory containing
JSON result files, computes mean +/- std for:
- max_kl
- recovered_kl
- max_top1
- recovered_top1

Then renders a grouped bar chart with 4 x-axis groups (one per metric) and one
bar per method within each group.
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


METRICS = ["max_kl", "recovered_kl"]
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


def aggregate_method_stats(gen_dir: Path, mode: str) -> Dict[str, Dict[str, float]]:
    """Aggregate mean/std per method for all configured metrics under a mode."""
    method_stats: Dict[str, Dict[str, float]] = {}

    method_dirs = sorted([p for p in gen_dir.iterdir() if p.is_dir()])
    if not method_dirs:
        raise ValueError(f"No method directories found in {gen_dir}")

    for method_dir in method_dirs:
        values = read_method_metric_values(method_dir, mode)
        stats: Dict[str, float] = {}

        for metric in METRICS:
            arr = np.array(values[metric], dtype=np.float64)
            if arr.size == 0:
                stats[f"{metric}_mean"] = np.nan
                stats[f"{metric}_std"] = np.nan
            else:
                stats[f"{metric}_mean"] = float(arr.mean())
                stats[f"{metric}_std"] = float(arr.std(ddof=0))

        method_stats[method_dir.name] = stats

    return method_stats


def plot_grouped_bars(method_stats: Dict[str, Dict[str, float]], output_png: Path) -> None:
    """Create and save grouped bar plot (mean +/- std)."""
    methods = sorted(method_stats.keys())
    n_methods = len(methods)
    n_groups = len(METRICS)

    x = np.arange(n_groups, dtype=np.float64)
    total_width = 0.82
    bar_width = total_width / max(1, n_methods)

    fig, ax = plt.subplots(figsize=(14, 7))

    for i, method in enumerate(methods):
        means = [method_stats[method][f"{m}_mean"] for m in METRICS]
        stds = [method_stats[method][f"{m}_std"] for m in METRICS]

        offset = (i - (n_methods - 1) / 2.0) * bar_width
        ax.bar(
            x + offset,
            means,
            width=bar_width,
            yerr=stds,
            capsize=3,
            label=method,
            alpha=0.9,
            edgecolor="black",
            linewidth=0.4,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(METRICS)
    ax.set_ylabel("Metric Value")
    ax.set_title("Discovery Metrics by Method (mean +/- std)")
    ax.set_ylim(0.0, 0.8)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(title="Method", frameon=False, ncol=min(3, n_methods))

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
        default="glm",
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

    method_stats = aggregate_method_stats(gen_dir, args.mode)

    print(f"Mode: {args.mode}")
    for method in sorted(method_stats.keys()):
        print(f"[{method}]")
        for metric in METRICS:
            mean = method_stats[method][f"{metric}_mean"]
            std = method_stats[method][f"{metric}_std"]
            print(f"  {metric}: mean={mean:.6f} std={std:.6f}")

    plot_grouped_bars(method_stats, output_png)
    print(f"Saved plot -> {output_png}")


if __name__ == "__main__":
    main()
