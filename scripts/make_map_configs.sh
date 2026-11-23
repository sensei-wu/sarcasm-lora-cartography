#!/usr/bin/env bash
set -euo pipefail

SNAPS="epoch1 epoch2_q1 epoch2_q2 epoch2_q3 epoch2 epoch3_q1 epoch3_q2 epoch3_q3 epoch3"
mkdir -p configs/map_train

for s in $SNAPS; do
  cat > configs/map_train/$s.yaml <<EOF
run_name: map_all_seed1_train_$s
model_id: Qwen/Qwen3.5-2B
adapter: /workspace/adapters/map_all_seed1/$s
results_root: /workspace/results
batch_size: 32
splits:
  train: /workspace/data/processed/train.jsonl
EOF
done
ls configs/map_train