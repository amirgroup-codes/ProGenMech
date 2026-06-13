# =================================================================
# CIRCUIT DISCOVERY: Script for circuit discovery
# =================================================================

#!/bin/bash
set -e 
# Usage:
#   sh circuit_discovery.sh

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ROOT_DIR="/path/to/project"
PYTHON_BIN="/path/to/python"

export CLT_CHECKPOINT="${CLT_CHECKPOINT:-${ROOT_DIR}/models/ProGen3_CLT_L10_D4608/checkpoints/last.ckpt}"
export PLT_CHECKPOINT="${PLT_CHECKPOINT:-${ROOT_DIR}/models/ProGen3_PLT_L10_D4608/checkpoints/last.ckpt}"

cd "${ROOT_DIR}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=$PYTHONPATH:external/progen3/src

echo "Checking for scipy..."
"${PYTHON_BIN}" -m pip install --user scipy --no-deps

# 1. Define Paths
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

NUM_THREADS=4
BATCH_SIZE=64
NUM_TEST_SEQ=500
NUM_TRAIN_SEQ=128

"${PYTHON_BIN}" -m function_circuit.discover_circuits \
    --datasets "${DATASETS[@]}" \
    --torch_num_threads $NUM_THREADS \
    --batch_size $BATCH_SIZE \
    --num_test_seq $NUM_TEST_SEQ \
    --num_train_seq $NUM_TRAIN_SEQ