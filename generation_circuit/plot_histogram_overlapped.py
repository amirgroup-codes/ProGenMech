#!/usr/bin/env python3
"""
plot_histogram_overlapping.py

Creates a 1x2 visualization comparing the distributions of LL scores.
- Left: Max-latent performance vs Ground Truth.
- Right: Recovered-circuit performance vs Ground Truth.
Uses identical x and y limits for precise visual comparison.
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

# Configuration for the two panels
LEFT_PLOT_GROUP = [
    ("CLT_direct", "progen3_ll", "#333333", "ProGen3 (GT)"),
    ("CLT_sequential", "max_ll", "#1f77b4", "CLT Max"),
    ("PLT", "max_ll", "#d62728", "PLT Max"),
]

RIGHT_PLOT_GROUP = [
    ("CLT_direct", "progen3_ll", "#333333", "ProGen3 (GT)"),
    ("CLT_sequential", "recovered_ll", "#ff7f0e", "CLT Recovered"),
    ("PLT", "recovered_ll", "#2ca02c", "PLT Recovered"),
]

def read_data(gen_dir: Path, mode: str) -> Dict[str, Dict[str, List[float]]]:
    """Reads JSON results and filters by mode (clm/glm/all)."""
    data = {}
    clm_pat = re.compile(r"_CLM\.json$")
    glm_pat = re.compile(r"_GLM_(1|2)\.json$")

    for method_dir in gen_dir.iterdir():
        if not method_dir.is_dir(): continue
        
        m_name = method_dir.name
        data[m_name] = {"progen3_ll": [], "max_ll": [], "recovered_ll": []}
        
        for json_p in method_dir.glob("*.json"):
            if mode == "clm" and not clm_pat.search(json_p.name): continue
            if mode == "glm" and not glm_pat.search(json_p.name): continue
            
            try:
                with open(json_p, "r") as f:
                    rec = json.load(f)
                    for k in data[m_name]:
                        if k in rec: data[m_name][k].append(rec[k])
            except: continue
    return data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen_dir", type=str, default="results_p0.95_T0.7", help="Directory with method subdirectories containing JSON results.")
    parser.add_argument("--mode", type=str, default="clm")
    parser.add_argument("--output", type=str, default="dist_raw_hist_T0.7"
    ".png")
    parser.add_argument("--bins", type=int, default=30)
    args = parser.parse_args()

    data = read_data(Path(args.gen_dir), args.mode)

    # 1. Determine Global Axis Limits
    all_vals = []
    for m in data:
        for met in data[m]:
            all_vals.extend(data[m][met])
    
    if not all_vals:
        print("Error: No data found."); return

    x_min, x_max = np.min(all_vals), np.max(all_vals)
    x_range = (x_min - 0.1, x_max + 0.1)

    # To find global Y-limit, we need to pre-calculate histogram heights
    max_freq = 0
    for group in [LEFT_PLOT_GROUP, RIGHT_PLOT_GROUP]:
        for method, metric, _, _ in group:
            vals = data.get(method, {}).get(metric, [])
            if not vals: continue
            counts, _ = np.histogram(vals, bins=args.bins, range=x_range)
            max_freq = max(max_freq, np.max(counts))

    y_limit = max_freq * 1.1  # Add 10% headroom

    # 2. Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left Plot: Max comparison
    for method, metric, color, label in LEFT_PLOT_GROUP:
        vals = data.get(method, {}).get(metric, [])
        ax1.hist(vals, bins=args.bins, range=x_range, color=color, 
                 alpha=0.4, label=label, edgecolor=color, linewidth=1)

    # Right Plot: Recovered comparison
    for method, metric, color, label in RIGHT_PLOT_GROUP:
        vals = data.get(method, {}).get(metric, [])
        ax2.hist(vals, bins=args.bins, range=x_range, color=color, 
                 alpha=0.4, label=label, edgecolor=color, linewidth=1)

    # 3. Formatting
    for ax, title in zip([ax1, ax2], ["Upper Bound (Max)", "Circuit Recovery (Recovered)"]):
        ax.set_xlim(x_range)
        # ax.set_xlim(0, 100)
        ax.set_ylim(0, y_limit)
        ax.set_title(f"{title}\nMode: {args.mode}", fontsize=13, fontweight='bold')
        ax.set_xlabel("Log Likelihood", fontsize=11)
        ax.set_ylabel("Frequency", fontsize=11)
        ax.legend(loc="upper left", frameon=True, fontsize=9)
        ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(args.output, dpi=300)
    print(f"Comparison plot saved to {args.output}")

if __name__ == "__main__":
    main()