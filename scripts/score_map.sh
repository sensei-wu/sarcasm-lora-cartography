#!/usr/bin/env bash
source /workspace/env.sh
set -euo pipefail

for s in epoch1 epoch2_q1 epoch2_q2 epoch2_q3 epoch2 epoch3_q1 epoch3_q2 epoch3_q3 epoch3; do
  echo "=== $s  $(date +%H:%M:%S)"
  python scripts/evaluate.py configs/map_train/$s.yaml 2>&1 | tee /workspace/logs/map_train_$s.log
done
echo "=== all done $(date +%H:%M:%S)"