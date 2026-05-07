#!/bin/bash
set -e 
# Usage:
#   sh main.sh

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ROOT_DIR="/path/to/project"
PYTHON_BIN="/path/to/python"
SEED="42"

# Generation parameters
p="0.95"
T="0.85"

# Data
PARQUET_PATH="${ROOT_DIR}/data/swissprot_seqid30_75k_all_info_with_3di.parquet"
PREP_CSV="${ROOT_DIR}/generation_circuit/generation_data_p${p}_T${T}.csv"
N_CLM_SAMPLES="500"
N_GLM_SAMPLES="250" # (we're collecting twice the amount of sequences because we're collecting two sequences with different spans)

export CLT_CHECKPOINT="${CLT_CHECKPOINT:-${ROOT_DIR}/models/ProGen3_CLT_L10_D4608/checkpoints/last.ckpt}"
export PLT_CHECKPOINT="${PLT_CHECKPOINT:-${ROOT_DIR}/models/ProGen3_PLT_L10_D4608/checkpoints/last.ckpt}"

cd "${ROOT_DIR}"

# "${PYTHON_BIN}" generation_circuit/01_prepare_data.py \
#   --parquet_path "${PARQUET_PATH}" \
#   --output_csv "${PREP_CSV}" \
#   --seed "${SEED}" \
#   --n_clm "${N_CLM_SAMPLES}" \
#   --n_glm "${N_GLM_SAMPLES}" \
#   --p "${p}" \
#   --T "${T}" \

"${PYTHON_BIN}" generation_circuit/02_discover_circuits.py \
  --input_csv "${PREP_CSV}" \
  --output_root "generation_circuit/results_p${p}_T${T}" \
  --seed "${SEED}" \
  --p "${p}" \
  --T "${T}" \
  # --debug