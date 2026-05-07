#!/usr/bin/env python3
"""
Combined Plotting Script
Generates:
performance_summary.pdf (1x2): Family F1 scores | Function Spearman correlations
"""

import os
import json
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.ticker import MultipleLocator
from scipy import stats
from pathlib import Path

# Configuration
FAMILY_DIR = "../generation_circuit/results_p0.95_T0.5"
FUNCTION_DIR = "../function_circuit/circuits"
OUTPUT_DIR = "plots"

# Models and Colors
CLT_MODEL = "CLT"
PLT_MODEL = "PLT"
METHOD_MAP = {"CLT_sequential": "CLT", "PLT": "PLT", "clt_sequential_freeze": "CLT", "plt_sequential_freeze": "PLT"}
COLOR_MAP = {CLT_MODEL: '#1b75bb', PLT_MODEL: '#f6921e'}
clean_color = 'gray'

# Thresholds
MIN_CLEAN_SPEARMAN = 0.01

# Sizing & Styling 
font_path = '../circuit_utils/Helvetica.ttf'
try:
    font_prop = fm.FontProperties(fname=font_path)
    fm.fontManager.addfont(font_path)
    font_name = font_prop.get_name()
    plt.rcParams['font.family'] = font_name
except:
    plt.rcParams['font.family'] = 'sans-serif'
font_size = 8
plt.rcParams['font.size'] = font_size
mm = 1/72  
linewidth = 0.25 
plt.rcParams['axes.linewidth'] = linewidth
plt.rcParams['lines.linewidth'] = linewidth
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
REF_FIG_WIDTH = 280 * mm
REF_FIG_HEIGHT = 115  * mm

# Data loading
def load_clm_data(data_dir=FAMILY_DIR):
    """
    Load CLM data for NLL from generation_circuit.
    """
    records = []
    base_path = Path(data_dir)
    if not base_path.exists():
        print(f"Warning: {data_dir} not found.")
        return pd.DataFrame()

    for method_dir in base_path.iterdir():
        if not method_dir.is_dir():
            continue
        method_name = method_dir.name
        mapped_method = METHOD_MAP.get(method_name, method_name)
        for json_p in method_dir.glob("*.json"):
            if "_CLM.json" not in json_p.name:
                continue
            try:
                with open(json_p, "r") as f:
                    data = json.load(f)
                row = {
                    "method": mapped_method,
                    "progen3_ll": data.get("progen3_ll", 0),
                    "max_ll": data.get("max_ll", 0),
                    "recovered_ll": data.get("recovered_ll", 0),
                }
                records.append(row)
            except:
                pass
    return pd.DataFrame(records)


def load_function_circuits(base_dir=FUNCTION_DIR, min_clean=None):
    """
    Parses results from the structure: 
    {base_dir}/{method_name}/{dataset}/fold{idx}.json
    Automatically maps raw method directory names to readable ProtoMech/PLT names.
    """
    records = []
    base_path = Path(base_dir)
    
    if not base_path.exists():
        print(f"Warning: {base_dir} not found.")
        return pd.DataFrame()

    # Search for all fold JSONs
    files = glob.glob(str(base_path / "**" / "seq256_fold*.json"), recursive=True)
    
    for fpath in files:
        try:
            path_obj = Path(fpath)
            # parts will be (..., 'raw_method_name', 'dataset_name', 'foldX.json')

            # # If dataset_name is GFP, continue
            # if path_obj.parts[-2] == "F7YBW8_MESOW_Ding_2023":
            #     continue
            parts = path_obj.parts
            
            with open(fpath, "r") as f:
                data = json.load(f)
            
            # Translate the raw directory name using your METHOD_MAP
            raw_method = parts[-3]
            mapped_method = METHOD_MAP.get(raw_method, raw_method)
            
            # Extract basic metadata
            row = {
                "method": mapped_method,
                "dataset": parts[-2],
                "fold": path_obj.stem,  # 'fold0', 'fold1', etc.
                "n_train": data.get("n_train", 0),
                "k": data.get("k", 0),
                "sequential": data.get("sequential", False),
                "freeze_attention": data.get("freeze_attention", False),
                "clean": data.get("clean_spearman", 0.0),
                "max": data.get("max_spearman", 0.0),
                "ablated": data.get("no_latents_spearman", 0.0),
                "recovered": data.get("recovered_spearman", 0.0),
            }
            
            # Dynamically count nodes for every layer present in the JSON
            nodes_dict = data.get("nodes", {})
            for layer_idx, node_list in nodes_dict.items():
                row[f"layer_{layer_idx}_count"] = len(node_list)
            
            records.append(row)
        except Exception as e:
            print(f"Error parsing {fpath}: {e}")
            continue
            
    df = pd.DataFrame(records)
    if df.empty:
        return df

    # Fill NaNs for layers (if Fold 0 found nodes in Layer 9 but Fold 1 didn't)
    layer_cols = [c for c in df.columns if c.startswith("layer_")]
    df[layer_cols] = df[layer_cols].fillna(0).astype(int)

    # Optional clean-performance filter
    if min_clean is not None:
        df = df[df["clean"] >= min_clean]
    print(df.columns)
    return df


# Helper functions
def draw_significance(ax, x1, x2, y_max, p_val):
    if p_val >= 0.05: return
    h = 0.02 * (ax.get_ylim()[1] - ax.get_ylim()[0])
    if p_val < 0.001: sig_symbol = "***"
    elif p_val < 0.01: sig_symbol = "**"
    else: sig_symbol = "*"
    ax.text((x1+x2)*.5, y_max+h, sig_symbol, ha='center', va='bottom', color='k', fontsize=font_size)

def plot_family_performance(ax, df):
    """Bar chart: ProGen3 vs All Latents vs Circuit NLL (CLM)"""
    # ProGen3: -1 * CLT_direct progen3_ll
    progen3_vals = df[df["method"] == "CLT_direct"]["progen3_ll"] * -1
    clean_mean = progen3_vals.mean()
    clean_std = progen3_vals.std()
    print(f"ProGen3 NLL: {clean_mean:.3f} ± {clean_std:.3f}")

    # Max and Recovered for CLT_sequential and PLT
    grouped_max = df[df["method"].isin([CLT_MODEL, PLT_MODEL])].groupby("method")["max_ll"]
    stats_max = grouped_max.mean() * -1
    stats_max_std = grouped_max.std()
    grouped_rec = df[df["method"].isin([CLT_MODEL, PLT_MODEL])].groupby("method")["recovered_ll"]
    stats_rec = grouped_rec.mean() * -1
    stats_rec_std = grouped_rec.std()

    bar_width = 0.25
    pos_orig, pos_all, pos_circ = 0, 1.0, 2.0
    offsets = [-bar_width/2, bar_width/2]

    # ProGen3
    ax.bar(pos_orig, clean_mean, yerr=clean_std, width=bar_width,
           color=clean_color, edgecolor='black', linewidth=linewidth,
           capsize=2, error_kw={'linewidth': linewidth}, label='ProGen3')

    # All Latents & Circuit
    max_y = clean_mean + clean_std
    for i, model_key in enumerate([CLT_MODEL, PLT_MODEL]):
        if model_key not in stats_max.index: continue
        
        # Max
        m = stats_max.loc[model_key]
        s = stats_max_std.loc[model_key]
        print(f"All Latents ({model_key}): {m:.3f} ± {s:.3f}")
        ax.bar(pos_all + offsets[i], m, yerr=s, width=bar_width,
               color=COLOR_MAP[model_key], edgecolor='black', linewidth=linewidth, 
               capsize=2, error_kw={'linewidth': linewidth})
        max_y = max(max_y, m + s)

        # Recovered
        m = stats_rec.loc[model_key]
        s = stats_rec_std.loc[model_key]
        print(f"Circuit ({model_key}): {m:.3f} ± {s:.3f}")
        ax.bar(pos_circ + offsets[i], m, yerr=s, width=bar_width,
               color=COLOR_MAP[model_key], edgecolor='black', linewidth=linewidth, 
               capsize=2, error_kw={'linewidth': linewidth}, label=model_key)
        max_y = max(max_y, m + s)

    # Significance for All Latents and Circuit
    if CLT_MODEL in stats_max.index and PLT_MODEL in stats_max.index:
        max_clt = df[df["method"] == CLT_MODEL]["max_ll"] * -1
        max_plt = df[df["method"] == PLT_MODEL]["max_ll"] * -1
        if len(max_clt) > 1 and len(max_plt) > 1:
            _, p_max = stats.ttest_rel(max_clt, max_plt)
            draw_significance(ax, pos_all + offsets[0], pos_all + offsets[1], max_y, p_max)
        
        rec_clt = df[df["method"] == CLT_MODEL]["recovered_ll"] * -1
        rec_plt = df[df["method"] == PLT_MODEL]["recovered_ll"] * -1
        if len(rec_clt) > 1 and len(rec_plt) > 1:
            _, p_rec = stats.ttest_rel(rec_clt, rec_plt)
            draw_significance(ax, pos_circ + offsets[0], pos_circ + offsets[1], max_y, p_rec)

    ax.set_ylabel(r"NLL (CLM) $\downarrow$")
    ax.set_xticks([pos_orig, pos_all, pos_circ])
    ax.set_xticklabels(["ProGen3", "All latents", "Circuit"])
    ax.set_ylim(0, max_y * 1.1)

def plot_function_performance(ax, df):
    """Grouped Bar chart for Function: ProGen3 vs All Latents vs Circuit Spearman."""
    # ProGen3: mean of per-dataset clean means
    base_perf = df.groupby("dataset")["clean"].mean()
    clean_mean = base_perf.mean()
    clean_std = base_perf.std()
    print(f"ProGen3: {clean_mean:.3f} ± {clean_std:.3f}")

    # Max and Recovered for CLT_sequential and PLT
    method_stats = df.groupby("method")[["max", "recovered"]].agg(['mean', 'std'])
    stats_max = method_stats[("max", "mean")]
    stats_max_std = method_stats[("max", "std")]
    stats_rec = method_stats[("recovered", "mean")]
    stats_rec_std = method_stats[("recovered", "std")]

    bar_width = 0.25
    pos_orig, pos_all, pos_circ = 0, 1.0, 2.0
    offsets = [-bar_width/2, bar_width/2]
    
    # ProGen3
    ax.bar(pos_orig, clean_mean, yerr=clean_std, width=bar_width,
           color=clean_color, edgecolor='black', linewidth=linewidth,
           capsize=2, error_kw={'linewidth': linewidth}, label='ProGen3')

    # All Latents & Circuit
    max_y_global = clean_mean + clean_std
    for i, model_key in enumerate([CLT_MODEL, PLT_MODEL]):
        if model_key not in stats_max.index: continue
        
        # Max
        m = stats_max.loc[model_key]
        s = stats_max_std.loc[model_key]
        print(f"All Latents ({model_key}): {m:.3f} ± {s:.3f}")
        ax.bar(pos_all + offsets[i], m, yerr=s, width=bar_width,
               color=COLOR_MAP[model_key], edgecolor='black', linewidth=linewidth, 
               capsize=2, error_kw={'linewidth': linewidth})
        max_y_global = max(max_y_global, m + s)

        # Recovered
        m = stats_rec.loc[model_key]
        s = stats_rec_std.loc[model_key]
        print(f"Circuit ({model_key}): {m:.3f} ± {s:.3f}")
        ax.bar(pos_circ + offsets[i], m, yerr=s, width=bar_width,
               color=COLOR_MAP[model_key], edgecolor='black', linewidth=linewidth, 
               capsize=2, error_kw={'linewidth': linewidth}, label=model_key)
        max_y_global = max(max_y_global, m + s)

    # Significance for All Latents and Circuit
    if CLT_MODEL in method_stats.index and PLT_MODEL in method_stats.index:
        max_clt = df[df["method"] == CLT_MODEL]["max"]
        max_plt = df[df["method"] == PLT_MODEL]["max"]
        if len(max_clt) > 1 and len(max_plt) > 1:
            _, p_max = stats.ttest_rel(max_clt, max_plt)
            draw_significance(ax, pos_all + offsets[0], pos_all + offsets[1], max_y_global, p_max)
        
        rec_clt = df[df["method"] == CLT_MODEL]["recovered"]
        rec_plt = df[df["method"] == PLT_MODEL]["recovered"]
        if len(rec_clt) > 1 and len(rec_plt) > 1:
            _, p_rec = stats.ttest_rel(rec_clt, rec_plt)
            draw_significance(ax, pos_circ + offsets[0], pos_circ + offsets[1], max_y_global, p_rec)

    ax.set_ylabel(r"Function Spearman ρ $\uparrow$")
    ax.set_xticks([pos_orig, pos_all, pos_circ])
    ax.set_xticklabels(["ProGen3", "All latents", "Circuit"])
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.set_ylim(0, 0.5) 
    # ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=3, frameon=False, fontsize=font_size-1)

def main():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    print("Loading data...")
    df_fam = load_clm_data()
    df_func = load_function_circuits(min_clean=MIN_CLEAN_SPEARMAN)
    
    print(f"Loaded {len(df_fam)} CLM records and {len(df_func)} function records.")

    print("Generating performance summary...")
    fig1, axes1 = plt.subplots(1, 2, figsize=(REF_FIG_WIDTH, REF_FIG_HEIGHT))
    
    if not df_fam.empty:
        print('Plotting NLL performance...')
        plot_family_performance(axes1[0], df_fam)
    
    if not df_func.empty:
        print('Plotting function performance...')
        plot_function_performance(axes1[1], df_func)
    
    plt.tight_layout()
    fig1.savefig(f"{OUTPUT_DIR}/performance_summary.pdf", dpi=300)
    plt.close(fig1)

    print(f"Done. Plots saved to {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()