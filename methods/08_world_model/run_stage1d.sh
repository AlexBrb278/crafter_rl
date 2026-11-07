#!/usr/bin/env bash
# Stage 1d: tile tokens + position/odometry + clock, with and without the top-down map memory.
set -u
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
L=runs/wm/logs
M=methods/08_world_model
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
COMMON="--tokens tokens_tile --pos --clock --train_split fit --steps 30000 --keep_checkpoints"
$PY $M/dynamics.py --run v3 $COMMON --map > $L/dyn_v3.log 2>&1 &
sleep 60
$PY $M/dynamics.py --run v3_nomap $COMMON > $L/dyn_v3_nomap.log 2>&1 &
wait
echo "$(date -Is) stage 1d trained"
for r in v3 v3_nomap; do $PY $M/dynamics.py --run $r --refit > $L/refit_$r.log 2>&1; done
echo "$(date -Is) stage 1d re-fit"
$PY $M/evaluate.py v2_long:model_refit.pt v3:model_refit.pt v3_nomap:model_refit.pt --n 200 --videos 8 --out stage1d_eval.json > $L/eval_1d.log 2>&1
echo "$(date -Is) stage 1d evaluated"
