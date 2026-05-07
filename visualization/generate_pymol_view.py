import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from Bio import PDB

# =============================================================================
# USER CONFIGURATION
# =============================================================================

# Kinase
PDB_NAME = 'P83104_5_281_4o91.1.A.cif'
# HRD motif
SEQUENCE = "MVKQVDFAEVKLSEKFLGAGSGGAVRKATFQNQEIAVKIFDFLEETIKKNAEREITHLSEIDHENVIRVIGRASNGKKDYLLMEYLEEGSLHNYLYGDDKWEYTVEQAVRWALQCAKALAYLHSLDRPIVHR"
# Targets: {Layer: [Latents]}
TARGET_NODES = {
    8: [897], # HRD detector 
    5: [1090], # Catalytic loop before/at HRD motif
    1: [3183], # R amino acids
    2: [1754], # R amino acids
    7: [2070], # ATP binding site and HRD region
    # 0: [248],
    # 2: [3026],  # Completely killed on W in low fitness
    # 5: [3028, 1609],  # High fitness activates now!
}
# DFG motif
SEQUENCE = "MVKQVDFAEVKLSEKFLGAGSGGAVRKATFQNQEIAVKIFDFLEETIKKNAEREITHLSEIDHENVIRVIGRASNGKKDYLLMEYLEEGSLHNYLYGDDKWEYTVEQAVRWALQCAKALAYLHSLDRPIVHRDIKPQNMLLYNQHEDLKICDF"
TARGET_NODES = {
    4: [2366], # DFG detector 
    3: [1068], # F amino acids
    1: [934], # F amino acids
    7: [2070], # ATP binding site and HRD region
    8: [1710], # N-lobe and C-helix
}
# 4. Analysis Window (1-indexed PDB numbering)
START_POS = 3
END_POS = 393
PDB_OFFSET = 1


# Layer 8, Latent 2748
# GRB2
PDB_NAME = '2VWF.cif'
# Wildtype
SEQUENCE = "TYVQALFDFDPQEDGELGFRRGDFIHVMDNSDPNWWKGACHGQTGMFPRNYVTPVNRNV"
TARGET_NODES = {
    1: [2693, 1522], # D amino acids; W amino acids
    10: [3297, 225], # binding interface site; binding/stability sites
    9: [2113], # Clasp region
}

# H26D, higher fitness outside residue 
SEQUENCE = "TYVQALFDFDPQEDGELGFRRGDFIEVMDNSDPNWWKGACHGQTGMFPRNYVTPVNRNV"
TARGET_NODES = {
    1: [2693, 1522], # D amino acids; W amino acids
    10: [3297, 225], # binding interface site; binding/stability sites, gets stronger by 58.33%
    9: [2113], # Clasp region
}

# # Y51D, lower fitness
# SEQUENCE = "TYVQALFDFDPQEDGELGFRRGDFIHVMDNSDPNWWKGACHGQTGMFPRNDVTPVNRNV"
# TARGET_NODES = {
#     1: [2693, 1522], # D amino acids increased by 1
#     10: [3297, 225], # binding interface site, gets weaker by 57.14%, binding/stability sites
#     9: [2113], # Clasp region gets weaker by 20%
# }

# =============================================================================
# IMPORTS & MODEL SETUP
# =============================================================================

_script_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_script_dir)
_progen3_src_dir = os.path.join(_parent_dir, 'external', 'progen3', 'src')
for path in (
    _parent_dir,
    os.path.join(_parent_dir, 'training'),
    os.path.join(_parent_dir, 'steering'),
    os.path.join(_parent_dir, 'circuit_utils'),
    _progen3_src_dir,
):
    if path not in sys.path:
        sys.path.append(path)

try:
    from training.clt_module import CLTLightningModule
except ImportError as e:
    raise ImportError(f"Could not import CLTLightningModule: {e}")

try:
    from steering.full_replacement_models import FullCLTReplacementModel
except ImportError as e:
    raise ImportError(f"Could not import FullCLTReplacementModel from steering: {e}")


def parse_pdb_residues_local(filepath, chain_id="A"):
    if filepath.endswith(".cif"):
        parser = PDB.MMCIFParser(QUIET=True)
    else:
        parser = PDB.PDBParser(QUIET=True)

    try:
        structure = parser.get_structure("model", filepath)
        model = next(iter(structure))
    except Exception as e:
        print(f"Error parsing PDB: {e}")
        return [], None

    if chain_id not in model:
        chains = list(model.get_chains())
        if not chains:
            return [], None
        chain = chains[0]
        chain_id = chain.id
        print(f"Note: Using chain {chain_id}")
    else:
        chain = model[chain_id]

    residue_ids = []
    for res in chain:
        if res.id[0] == " ":
            residue_ids.append(res.id[1])

    return residue_ids, chain_id


def get_activation_map(full_trace, pdb_ids, start_pos, end_pos):
    """
    Maps aa-only ProGen3 CLT trace indices to PDB residue IDs.

    The trace passed here is filtered to valid amino-acid positions only,
    so the first sequence residue corresponds to trace index 0.
    """
    mapping = {}

    for res_num in pdb_ids:
        trace_idx = res_num - PDB_OFFSET
        if trace_idx < 0 or trace_idx >= len(full_trace):
            continue

        val = float(full_trace[trace_idx])

        if start_pos is not None and res_num < start_pos:
            val = 0.0
        if end_pos is not None and res_num > end_pos:
            val = 0.0

        mapping[res_num] = val
    return mapping


def generate_combined_py(pdb_path, chain_id, all_activations, output_dir="pymol_viz"):
    filename = "circuit_visualization.py"
    filepath = os.path.join(output_dir, filename)

    lines = [
        "from pymol import cmd",
        "",
        "cmd.reinitialize()",
        "cmd.bg_color('white')",
        "cmd.set('ray_trace_mode', 1)",
        "cmd.set('ray_shadows', 0)",
        "cmd.set('antialias', 2)",
        "",
        "# Custom Colors",
        "cmd.set_color('base_blue', [91/255, 150/255, 210/255])",
        "",
        "# Load Structure",
        f"cmd.load('{pdb_path}', 'base_struct')",
        "cmd.hide('everything', 'base_struct')",
        "cmd.color('base_blue', 'base_struct')",
        "cmd.show('cartoon', 'base_struct')",
        "",
        "# Coloring Helper (Normalized)",
        "def apply_spectrum_norm(obj_name, raw_max):",
        "    cmd.color('base_blue', obj_name)",
        "    print(f'Object {obj_name}: Raw Max = {raw_max:.4f}')",
        "    if raw_max < 0.0001: return",
        "    selection = f'{obj_name} and b > 0.1'",
        "    cmd.spectrum('b', 'white_red', selection=selection, minimum=0.1, maximum=1.0)",
        "",
    ]

    for (layer, latent), act_map in all_activations.items():
        obj_name = f"L{layer}_{latent}"
        vals = list(act_map.values())
        raw_max = max(vals) if vals else 0.0

        lines.append(f"# --- {obj_name} (Raw Max: {raw_max:.4f}) ---")
        lines.append(f"cmd.create('{obj_name}', 'base_struct')")
        lines.append(f"cmd.alter('{obj_name}', 'b=0.0')")

        for r, v in act_map.items():
            if v > 0.001:
                norm_val = v * (1.0 / raw_max if raw_max > 0 else 0.0)
                lines.append(f"cmd.alter('{obj_name} and chain {chain_id} and resi {r}', 'b={norm_val:.4f}')")

        lines.append(f"apply_spectrum_norm('{obj_name}', {raw_max})")
        lines.append(f"cmd.group('Circuit_Analysis', '{obj_name}')")
        lines.append("")

    lines += [
        "cmd.disable('base_struct')",
        "cmd.disable('Circuit_Analysis')",
        "cmd.zoom('base_struct')",
        "print('Done! Enable specific objects in Circuit_Analysis to view.')",
    ]

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        f.write("\n".join(lines))
    return filepath


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--clt_ckpt",
        type=str,
        default="../models/ProGen3_CLT_L10_D4608/checkpoints/last.ckpt",
        help="CLT checkpoint path",
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_dir", type=str, default="pymol_viz")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device}")

    print(f"Parsing PDB: {PDB_NAME}...")
    pdb_ids, chain_id = parse_pdb_residues_local(PDB_NAME)
    if not pdb_ids:
        print("Error: PDB parsing failed.")
        return

    print("Loading CLT checkpoint...")
    pl_module = CLTLightningModule.load_from_checkpoint(args.clt_ckpt, map_location=device)
    pl_module.to(device).eval()

    model = FullCLTReplacementModel(pl_module, device)

    print(f"Running inference on Sequence ({len(SEQUENCE)} AA)...")
    inputs = model.prepare_inputs([SEQUENCE])
    valid_mask = model._valid_token_mask(inputs["input_ids"])[0]

    with torch.no_grad():
        _, latents_list, _, _, _ = model.forward([SEQUENCE], freeze_attention=True)

    all_activations = {}
    for layer, latents in TARGET_NODES.items():
        layer_idx = layer - 1
        if layer_idx < 0 or layer_idx >= len(latents_list):
            raise IndexError(f"Layer {layer} out of range for model with {len(latents_list)} layers")
        for latent in latents:
            latent_idx = latent - 1
            print(f"Processing L{layer} (layer idx {layer_idx}) - latent {latent} (1-indexed) -> latent idx {latent_idx}...")
            tensor = latents_list[layer_idx]
            if tensor.ndim != 3:
                raise RuntimeError(f"Unexpected latents shape {tensor.shape} for layer {layer}")
            if latent_idx < 0 or latent_idx >= tensor.shape[-1]:
                raise IndexError(f"Latent {latent} out of range for layer {layer} with dim {tensor.shape[-1]}")

            trace = tensor[valid_mask, 0, latent_idx].cpu().numpy()
            act_map = get_activation_map(trace, pdb_ids, START_POS, END_POS)
            all_activations[(layer, latent)] = act_map

    out_path = generate_combined_py(PDB_NAME, chain_id, all_activations, args.output_dir)

    print("-" * 40)
    print(f"✅ Generated Python script: {out_path}")
    print("Run in PyMOL via 'File -> Run Script...'")
    print("-" * 40)


if __name__ == "__main__":
    main()
