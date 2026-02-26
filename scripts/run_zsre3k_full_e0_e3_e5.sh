#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate ftedit

DATA_PATH="./data/zsre/zsre_3k.json"
SEED=0
NUM_SAMPLES=3000

run_train () {
  local cfg="$1"
  echo "[TRAIN] $cfg"
  python -m train.train_patch --config_path "$cfg"
}

run_eval () {
  local model_path="$1"
  local out_path="$2"
  echo "[EVAL] $model_path -> $out_path"
  CUDA_VISIBLE_DEVICES=0 python eval_edit_metric.py \
    --data_path "$DATA_PATH" \
    --model_path "$model_path" \
    --tp_size 1 \
    --num_samples "$NUM_SAMPLES" \
    --seed "$SEED" \
    --quiet \
    --output_path "$out_path"
}

mkdir -p runs

run_train "configs/zsre3k_full_e0.yaml"
run_eval "saves/zsre3k_full_e0_off_sft" "runs/zsre3k_full_e0_eval${NUM_SAMPLES}_seed${SEED}.json"

run_train "configs/zsre3k_full_e3.yaml"
run_eval "saves/zsre3k_full_e3_mix_dpo" "runs/zsre3k_full_e3_eval${NUM_SAMPLES}_seed${SEED}.json"

run_train "configs/zsre3k_full_e5.yaml"
run_eval "saves/zsre3k_full_e5_dynamic_gate" "runs/zsre3k_full_e5_eval${NUM_SAMPLES}_seed${SEED}.json"

echo "[DONE] Results in runs/"

