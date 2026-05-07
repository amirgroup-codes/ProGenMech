#!/bin/bash
set -e 

# Usage:
#   sh main_probe_steering.sh

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ROOT_DIR="/path/to/project"
PYTHON_BIN="/path/to/python"

REPO_ROOT="$(dirname "$(pwd)")"
CIRCUIT_BASE="../function_circuit/circuits"

DATASETS=(
    "${ROOT_DIR}/data/DMS_ProteinGym_substitutions/A4_HUMAN_Seuma_2022.csv"
    "${ROOT_DIR}/data/DMS_ProteinGym_substitutions/CAPSD_AAV2S_Sinai_2021.csv"
    "${ROOT_DIR}/data/DMS_ProteinGym_substitutions/F7YBW8_MESOW_Ding_2023.csv"
    "${ROOT_DIR}/data/DMS_ProteinGym_substitutions/GFP_AEQVI_Sarkisyan_2016.csv"
    "${ROOT_DIR}/data/DMS_ProteinGym_substitutions/GRB2_HUMAN_Faure_2021.csv"
    "${ROOT_DIR}/data/DMS_ProteinGym_substitutions/RASK_HUMAN_Weng_2022_abundance.csv"
    "${ROOT_DIR}/data/DMS_ProteinGym_substitutions/SPG1_STRSG_Olson_2014.csv"
    "${ROOT_DIR}/data/DMS_ProteinGym_substitutions/YAP1_HUMAN_Araya_2012.csv"
)

CLT_PATH="../models/CLT_L12_D4800/checkpoints/last.ckpt"
PLT_PATH="../models/PLT_L12_D4800/checkpoints/last.ckpt" 
ESM_PATH="../models/esm2_t12_35M_UR50D.pt"

SUPP_VALUES="4 8 16"
FOLDS="0,1,2,3,4"
OUTPUT_BASE="probe_results_35M"

echo "========================================"
echo " [Setup] Configuration - PROBE STEERING"
echo "========================================"
echo "  > DMS Data Directory:   $DMS_DATA_DIR"
echo "  > Circuit Base Dir:     $CIRCUIT_BASE"
echo "  > CLT Checkpoint:       $CLT_PATH"
echo "  > PLT Checkpoint:       $PLT_PATH"
echo "  > ESM Weights:          $ESM_PATH"
echo "========================================"

for SUPP in $SUPP_VALUES; do
    for MUT in $MAX_MUTATIONS; do
        echo " [Running] Probe Steering with SUPP=$SUPP, MAX_MUTATIONS=${MUT}"
        python run_probe_steering.py \
            --dms_dir "$DMS_DATA_DIR" \
            --output_dir "${OUTPUT_BASE}/supp${SUPP}" \
            --clt_ckpt "$CLT_PATH" \
            --plt_ckpt "$PLT_PATH" \
            --esm_weights "$ESM_PATH" \
            --circuit_base "$CIRCUIT_BASE" \
            --supp "$SUPP" \
            --configs "CLT_sequential,CLT_sequential_no_frozen,PLT_no_frozen" \
            --folds "$FOLDS" \
            --max_mutations "$MUT" \
            --alpha_steps 25

        echo ""
    done
done

echo "Pipeline Complete."
