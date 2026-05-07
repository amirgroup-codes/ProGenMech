"""
Combined Circuit Plots
Generates 4 multi-panel figures comparing ProtoMech/PLT performance.
Fig 1: Performance (F1 / Spearman)
Fig 2: Node Distribution per Layer [Fixed Data Loading]
Fig 3: Function Splits Performance
Fig 4: Low F1 Families Performance
"""

import os
import json
import glob
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from pathlib import Path

# --- Configuration & Style ---
PLOTS_DIR = "plots"
GENERATION_DIR = "../generation_circuit/results_p0.95_T0.5"

# Thresholds
MIN_CLEAN_SPEARMAN = 0.01 

AXIS_LINEWIDTH = 0.5
BAR_EDGE_WIDTH = 0.25
ERROR_LINEWIDTH = 0.5

font_path = "../circuit_utils/Helvetica.ttf"
fm.fontManager.addfont(font_path)
font_name = fm.FontProperties(fname=font_path).get_name()

mpl.rcParams.update({
    "font.family": font_name,
    "mathtext.fontset": "custom",
    "mathtext.rm": font_name,
    "mathtext.it": font_name,
    "mathtext.bf": font_name,
})

font_size = 8
plt.rcParams['font.size'] = font_size

mm = 1/25.4

plt.rcParams['axes.linewidth'] = AXIS_LINEWIDTH
plt.rcParams['lines.linewidth'] = 0.5

# Methods and ordering
METHOD_MAP = {
    "clt_direct": "CLT (direct)",
    "clt_sequential_freeze": "CLT (sequential)",
    "plt_sequential_freeze": "PLT (sequential)",
    "clt_sequential_unfreeze": "CLT (full replacement)",
    "plt_sequential_unfreeze": "PLT (full replacement)"
}

GENERATION_METHOD_MAP = {
    "CLT_direct": "CLT (direct)",
    "CLT_sequential": "CLT (sequential)",
    "CLT_sequential_no_frozen": "CLT (full replacement)",
    "PLT": "PLT (sequential)",
    "PLT_no_frozen": "PLT (full replacement)"
}

ORDERED_METHODS = [
    "CLT (direct)", 
    "CLT (sequential)", 
    "PLT (sequential)", 
    "CLT (full replacement)", 
    "PLT (full replacement)"
]
COLORS = ['#1b75bb', '#af588a', '#f6921e', '#00A087', '#DC0000']
COLOR_MAP = dict(zip(ORDERED_METHODS, COLORS))

# Data loading
def load_generation_data(data_dir=GENERATION_DIR, clm=True):
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
        mapped_method = GENERATION_METHOD_MAP.get(method_name, method_name)
        for json_p in method_dir.glob("*.json"):
            if clm and "_CLM.json" not in json_p.name:
                continue
            elif not clm and "GLM" not in json_p.name:
                continue
            try:
                with open(json_p, "r") as f:
                    data = json.load(f)
                row = {
                    "method": mapped_method,
                    "dataset": json_p.stem,
                    "k": data.get("k", 0),
                    "clean": -data.get("progen3_ll", 0),
                    "max": -data.get("max_ll", 0),
                    "recovered": -data.get("recovered_ll", 0)
                }
                nodes_dict = data.get("nodes", {})
                for layer_idx, node_list in nodes_dict.items():
                    row[f"layer_{layer_idx}_count"] = len(node_list)
                records.append(row)
            except:
                pass
    df = pd.DataFrame(records)

    if df.empty:
        return df

    layer_cols = [c for c in df.columns if c.startswith("layer_")]
    df[layer_cols] = df[layer_cols].fillna(0).astype(int)

    return df

def load_circuit_results(base_dir="../function_circuit/circuits", min_clean=None):
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

def draw_grouped_bars(ax, df, show_legend=False, ylabel="Spearman ρ", 
                      show_xticks=True, ylim=None, yticks=None, font_size=10, linewidth=1.5, lower_is_better=False):
    """
    Standard grouped bar plot for performance metrics.
    Labels adapted to match the reference style: [Base Model], [All latents], [Circuit].
    """
    if df.empty:
        return

    # =========================================================
    # 1. BASELINE PERFORMANCE
    # =========================================================
    base_perf = df.groupby("dataset")["clean"].mean()
    clean_mean = base_perf.mean()
    clean_std = base_perf.std()

    print(f"\n--- {ylabel} Statistics ---")
    print(f"ProGen3 baseline: {clean_mean:.3f} ± {clean_std:.3f}")

    # =========================================================
    # 2. METHOD STATS
    # =========================================================
    method_stats = df.groupby("method")[["max", "recovered"]].agg(['mean', 'std'])

    # Optional additional diagnostics
    for method in ORDERED_METHODS:
        if method not in method_stats.index:
            continue

        max_mean = method_stats.loc[method, ('max', 'mean')]
        max_std = method_stats.loc[method, ('max', 'std')]

        rec_mean = method_stats.loc[method, ('recovered', 'mean')]
        rec_std = method_stats.loc[method, ('recovered', 'std')]

        print(f"{method} All latents: {max_mean:.3f} ± {max_std:.3f}")
        print(f"{method} Circuit:     {rec_mean:.3f} ± {rec_std:.3f}")

    # =========================================================
    # PLOTTING
    # =========================================================

    xticklabels = ["ProGen3", "All latents", "Circuit"]

    bar_width = 0.1
    pos_base = 0
    group_1_center = 0.85
    group_2_center = 1.85
    centers = [pos_base, group_1_center, group_2_center]
    x_lims = (-0.5, 2.5)

    indices = np.arange(len(ORDERED_METHODS))
    offsets = (indices - (len(ORDERED_METHODS)-1)/2) * (bar_width * 1.1)

    # A. ProGen3 Baseline Bar
    clean_lower_err = min(clean_mean, clean_std)
    clean_yerr = [[clean_lower_err], [clean_std]]

    ax.bar(
        pos_base,
        clean_mean,
        yerr=clean_yerr,
        width=bar_width,
        color='gray',
        edgecolor='black',
        linewidth=BAR_EDGE_WIDTH,
        capsize=2,
        error_kw={'linewidth': ERROR_LINEWIDTH},
        label='ProGen3'
    )

    # B. All Latents
    for i, method in enumerate(ORDERED_METHODS):
        if method not in method_stats.index:
            continue

        mean_val = method_stats.loc[method, ('max', 'mean')]
        std_val = method_stats.loc[method, ('max', 'std')]

        lower_err = min(mean_val, std_val)
        yerr_asym = [[lower_err], [std_val]]

        ax.bar(
            group_1_center + offsets[i],
            mean_val,
            yerr=yerr_asym,
            width=bar_width,
            color=COLOR_MAP[method],
            edgecolor='black',
            linewidth=BAR_EDGE_WIDTH,
            capsize=2,
            error_kw={'linewidth': ERROR_LINEWIDTH}
        )

    # C. Circuit
    for i, method in enumerate(ORDERED_METHODS):
        if method not in method_stats.index:
            continue

        mean_val = method_stats.loc[method, ('recovered', 'mean')]
        std_val = method_stats.loc[method, ('recovered', 'std')]

        lower_err = min(mean_val, std_val)
        yerr_asym = [[lower_err], [std_val]]

        ax.bar(
            group_2_center + offsets[i],
            mean_val,
            yerr=yerr_asym,
            width=bar_width,
            color=COLOR_MAP[method],
            edgecolor='black',
            linewidth=BAR_EDGE_WIDTH,
            capsize=2,
            error_kw={'linewidth': ERROR_LINEWIDTH},
            label=method
        )

    # =========================================================
    # STYLING
    # =========================================================

    ax.set_ylabel(ylabel, fontsize=font_size)
    ax.set_xlim(x_lims)

    if ylim:
        ax.set_ylim(ylim)

    if yticks is not None:
        ax.set_yticks(yticks)

    if show_xticks:
        ax.set_xticks(centers)
        ax.set_xticklabels(xticklabels, fontsize=font_size)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if show_legend:
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))

        ordered_labels = ["ProGen3"] + [
            m for m in ORDERED_METHODS if m in by_label
        ]

        ordered_handles = [
            by_label[l] for l in ordered_labels if l in by_label
        ]

        ax.legend(
            ordered_handles,
            ordered_labels,
            loc='lower center',
            bbox_to_anchor=(0.5, 1.05),
            ncol=3,
            frameon=False,
            fontsize=font_size,
            handletextpad=0.5,
            columnspacing=1.0
        )

def plot_nodes_per_layer(ax, df, title_label="", show_legend=False, xaxis="Layer Index", 
                         ordered_methods=None, color_map=None, linewidth=1.5, font_size=10):
    """
    Plots a grouped bar chart of Average Node Count for ProGen3's 10 layers.
    Legend is formatted to appear at the top, matching the reference style.
    """
    print(f"\n--- {title_label} Node Statistics ---")
    
    layer_indices = range(10) 
    x = np.arange(len(layer_indices))
    methods = ordered_methods if ordered_methods else df['method'].unique()
    num_methods = len(methods)
    total_width = 0.8
    bar_width = total_width / num_methods
    offsets = (np.arange(num_methods) - (num_methods - 1) / 2) * bar_width
    
    for i, method in enumerate(methods):
        subset = df[df['method'] == method]
        if subset.empty:
            continue
            
        means, stds = [], []
        for l in layer_indices:
            col = f"layer_{l}_count"
            if col in subset.columns:
                means.append(subset[col].mean())
                stds.append(subset[col].std())
            else:
                means.append(0)
                stds.append(0)
        
        total_nodes = subset['k']
        print(f"[{method}] Avg k: {total_nodes.mean():.2f} ± {total_nodes.std():.2f}")

        lower_errs = [min(m, s) for m, s in zip(means, stds)]
        yerr_asym = [lower_errs, stds]
        method_color = color_map.get(method, 'gray') if color_map else None

        ax.bar(x + offsets[i], means, yerr=yerr_asym, width=bar_width,
               color=method_color, edgecolor='black', linewidth=BAR_EDGE_WIDTH,
               capsize=2, error_kw={'linewidth': ERROR_LINEWIDTH},
               label=method)

    ax.set_ylabel("Average Latent Count")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{l+1}" for l in layer_indices]) 
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if xaxis:
        ax.set_xlabel(xaxis, fontsize=font_size)
    
    if show_legend:
        # Match the reference image: top-centered, 3 columns, no frame
        ax.legend(
            loc='lower center', 
            bbox_to_anchor=(0.5, 1.05), # Moves it just above the axes
            ncol=3, 
            fontsize=font_size, 
            frameon=False,
            handletextpad=0.5,
            columnspacing=1.0
        )

def make_fig1_performance(df, task_name="zero_shot"):
    """
    Performance Summary: ProGen3 vs. All Latents vs. Circuit.
    Calibrated for full-column width (183mm).
    """
    mm = 1 / 25.4
    fig_width = 183 * mm
    fig_height = 80 * mm 
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    print('--- Running Performance Plot ---')

    if task_name == "zero_shot":
        y_label = "Spearman ρ"
        ylim = (0, 0.6)
        yticks = [0, 0.2, 0.4, 0.6]
    else:
        y_label = f"NLL ({task_name})"
        ylim = (0, 4)
        yticks = [0, 1, 2, 3, 4]
    
    draw_grouped_bars(
        ax, 
        df, 
        show_legend=True, 
        ylabel=y_label,
        ylim=ylim,
        yticks=yticks,
        lower_is_better=(task_name != "zero_shot")
    )
    
    plt.tight_layout()
    
    output_path = f"{PLOTS_DIR}/circuit_performance_summary_{task_name}.pdf"
    plt.savefig(output_path, bbox_inches='tight')
    print(f"Success: Performance plot saved to {output_path}")
    plt.close()

def make_fig2_node_distribution_plot(df, task_name="zero_shot"):
    mm = 1 / 25.4
    fig_width = 183 * mm
    fig_height = 80 * mm
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    plot_nodes_per_layer(
        ax, 
        df, 
        title_label=f"Circuit Latent Distribution ({task_name})", 
        show_legend=True, 
        xaxis='Layer Index',
        ordered_methods=ORDERED_METHODS,
        color_map=COLOR_MAP
    )
    
    plt.tight_layout()
    
    output_path = f"{PLOTS_DIR}/node_distribution_{task_name}.pdf"
    plt.savefig(output_path, bbox_inches='tight')
    print(f"Success: Node distribution plot saved to {output_path}")
    plt.close()

def main():
    Path(PLOTS_DIR).mkdir(exist_ok=True)
    
    print("Loading Function Data...")
    df = load_circuit_results(min_clean=MIN_CLEAN_SPEARMAN)
    print(f"Loaded {len(df)} function records.")

    print("Loading Generation Data...")
    clm_df = load_generation_data()
    print(f"Loaded {len(clm_df)} generation records for CLM.")
    glm_df = load_generation_data(clm=False)
    print(f"Loaded {len(glm_df)} generation records for GLM.")
    
    if df.empty and clm_df.empty and glm_df.empty:
        print("Error: Missing all data. Check directories.")
        return
    make_fig1_performance(df)
    make_fig2_node_distribution_plot(df)  
    make_fig1_performance(clm_df, task_name="CLM")  # CLM NLL performance from generation_circuit
    make_fig2_node_distribution_plot(clm_df, task_name="CLM")
    make_fig1_performance(glm_df, task_name="GLM")  # GLM NLL performance from generation_circuit
    make_fig2_node_distribution_plot(glm_df, task_name="GLM")

    print("All plots generated.")

if __name__ == "__main__":
    main()