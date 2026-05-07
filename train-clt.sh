#!/bin/bash

# =================================================================
# MAIN.SH - Training Script for CLT
# =================================================================

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=$PYTHONPATH:external/progen3/src

# 1. Define Paths
REPO_ROOT="$(dirname "$(pwd)")"
DATA_FILE="./data/training_sequences_5m.parquet"
OUTPUT_DIR="./models"

if [ ! -f "$DATA_FILE" ]; then
    echo "ERROR: Data file not found at $DATA_FILE"
    exit 1
fi

MODEL="Profluent-Bio/progen3-112m"
NUM_LAYERS=10
D_MODEL=384
D_HIDDEN=4608
BATCH_SIZE=16
EPOCHS=1
LR=2e-4
K=64
AUXK=$((K * 2))

echo "Starting training..."
echo "Data: $DATA_FILE"
echo "Output: $OUTPUT_DIR"

python -m training.run_clt \
    --data-dir "$DATA_FILE" \
    --model "$MODEL" \
    --output-dir "$OUTPUT_DIR" \
    --num-layers $NUM_LAYERS \
    --d-model $D_MODEL \
    --d-hidden $D_HIDDEN \
    --batch-size $BATCH_SIZE \
    --lr $LR \
    --k $K \
    --auxk $AUXK \
    --max-epochs $EPOCHS \
    --num-devices 2 \
    --wandb-project "ProGen3-CLT-small"

echo "Training complete."