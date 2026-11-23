#!/usr/bin/env bash
set -euo pipefail
MAP=/workspace/results/cartography_map_all_seed1
mkdir -p configs/phase2

write_train() {   # $1 = run_name, $2 = seed, $3 = subset
  cat > configs/phase2/$1.yaml <<EOF
run_name: $1
model_id: Qwen/Qwen3.5-2B
adapters_root: /workspace/adapters
train: /workspace/data/processed/train.jsonl
val: /workspace/data/processed/val.jsonl
batch_size: 32
seed: $2
epochs: 2
lr: 2.0e-4
train_batch_size: 16
lora: {r: 16, alpha: 32, dropout: 0.05, target_modules: all-linear}
subset: $3
EOF
}

write_eval() {    # $1 = run_name
  cat > configs/phase2/eval_$1.yaml <<EOF
run_name: $1
model_id: Qwen/Qwen3.5-2B
adapter: /workspace/adapters/$1/epoch2
results_root: /workspace/results
batch_size: 32
splits:
  val: /workspace/data/processed/val.jsonl
  test: /workspace/data/processed/test.jsonl
  isarcasm_test: /workspace/data/processed/isarcasm_test.jsonl
EOF
}

for seed in 1 2; do
  for pct in 33 10; do
    write_train amb${pct}_seed$seed  $seed "{kind: ids, ids_path: $MAP/frac$pct/ambiguous_ids.txt}"
    write_train lm${pct}_seed$seed   $seed "{kind: ids, ids_path: $MAP/frac$pct/label_matched_seed${seed}_ids.txt}"
    write_train rand${pct}_seed$seed $seed "{kind: random, fraction: 0.$pct}"
    for arm in amb lm rand; do write_eval ${arm}${pct}_seed$seed; done
  done
done
ls configs/phase2