#!/usr/bin/env bash
source /workspace/env.sh
set -euo pipefail

for seed in 1 2; do
  for pct in 33 10; do
    for arm in amb lm rand; do
      n=${arm}${pct}_seed$seed
      echo "=== $n  $(date +%H:%M:%S)"
      python scripts/train.py configs/phase2/$n.yaml 2>&1 | tee /workspace/logs/$n.log
      python scripts/evaluate.py configs/phase2/eval_$n.yaml 2>&1 | tee /workspace/logs/eval_$n.log
    done
  done
done
echo "=== all done $(date +%H:%M:%S)"