#!/usr/bin/env bash
# Continue Stage 1b (v2) from 30k to 60k steps: lower learning rate, memory (skip-ahead) loss weight 1.0.
set -u
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
L=runs/wm/logs
M=methods/08_world_model
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
$PY $M/dynamics.py --run v2_long --init v2 --tokens tokens_v2 --clock --skip --unroll 0.25 --train_split fit \
    --steps 30000 --lr 1e-4 --skip_w 1.0 --keep_checkpoints > $L/dyn_v2_long.log 2>&1 || { echo "training failed"; exit 1; }
echo "$(date -Is) v2_long trained"
$PY $M/dynamics.py --run v2_long --refit > $L/refit_v2_long.log 2>&1
echo "$(date -Is) v2_long re-fit"
$PY $M/evaluate.py v2_long:model_refit.pt --n 200 --videos 8 --out stage1b_long_eval.json > $L/eval_long.log 2>&1
echo "$(date -Is) v2_long evaluated"
