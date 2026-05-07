#!/bin/bash
set -e

ROOT_DIR="/path/to/project"
PYTHON_BIN="/path/to/python"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$PYTHONPATH:${ROOT_DIR}/external/progen3/src"

CLT_CHECKPOINT="${CLT_CHECKPOINT:-${ROOT_DIR}/models/ProGen3_CLT_L10_D4608/checkpoints/last.ckpt}"
PLT_CHECKPOINT="${PLT_CHECKPOINT:-${ROOT_DIR}/models/ProGen3_PLT_L10_D4608/checkpoints/last.ckpt}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/steering_results_patching_correction}"

DATASETS=(
    "${ROOT_DIR}/data/DMS_ProteinGym_substitutions/A4_HUMAN_Seuma_2022.csv"
    # "${ROOT_DIR}/data/DMS_ProteinGym_substitutions/CAPSD_AAV2S_Sinai_2021.csv"
    "${ROOT_DIR}/data/DMS_ProteinGym_substitutions/F7YBW8_MESOW_Ding_2023.csv"
    # "${ROOT_DIR}/data/DMS_ProteinGym_substitutions/GFP_AEQVI_Sarkisyan_2016.csv"
    "${ROOT_DIR}/data/DMS_ProteinGym_substitutions/GRB2_HUMAN_Faure_2021.csv"
    # "${ROOT_DIR}/data/DMS_ProteinGym_substitutions/RASK_HUMAN_Weng_2022_abundance.csv"
    # "${ROOT_DIR}/data/DMS_ProteinGym_substitutions/SPG1_STRSG_Olson_2014.csv"
    "${ROOT_DIR}/data/DMS_ProteinGym_substitutions/YAP1_HUMAN_Araya_2012.csv"
)
SUPP_VALUES=(4 8 16 32 64)
FOLDS=(0)

for DMS_CSV in "${DATASETS[@]}"; do
    dataset_name="$(basename "$DMS_CSV" .csv)"
    for fold in "${FOLDS[@]}"; do
        clt_json="$ROOT_DIR/function_circuit/circuits/clt_sequential_freeze/${dataset_name}/seq256_fold${fold}.json"
        plt_json="$ROOT_DIR/function_circuit/circuits/plt_sequential_freeze/${dataset_name}/seq256_fold${fold}.json"

        if [[ ! -f "$clt_json" || ! -f "$plt_json" ]]; then
            echo "Skipping $dataset_name fold $fold because JSON missing:"
            [[ ! -f "$clt_json" ]] && echo "  missing $clt_json"
            [[ ! -f "$plt_json" ]] && echo "  missing $plt_json"
            continue
        fi

        for supp in "${SUPP_VALUES[@]}"; do
            output_dir="$OUTPUT_DIR/${dataset_name}/fold${fold}/supp${supp}"
            mkdir -p "$output_dir"

            echo "Running dataset=$dataset_name fold=$fold supp=$supp"
            ${PYTHON_BIN} run_steering.py \
                --dms_csv "$DMS_CSV" \
                --clt_ckpt "$CLT_CHECKPOINT" \
                --plt_ckpt "$PLT_CHECKPOINT" \
                --clt_json "$clt_json" \
                --plt_json "$plt_json" \
                --device cuda \
                --output_dir "$output_dir" \
                --num_train_seq 128 \
                --num_test_seq 500 \
                --prefix_frac 0.9 \
                --alpha_min 0.1 \
                --alpha_max 5.0 \
                --alpha_steps 10 \
                --num_latents "$supp" \
                --top_p 0.95 \
                --temperature 1 \
                --freeze_attention \
                --before
        done
    done
    echo "Finished dataset $dataset_name"
    echo
done
