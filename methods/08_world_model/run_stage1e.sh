#!/usr/bin/env bash
# Stage 1e: the map model with positions from pixels only (pixel_odometry.py, maps_tile_pix), with and without
# unrolled training on its own predictions. Starts when Stage 1d has been evaluated.
set -u
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
L=runs/wm/logs
M=methods/08_world_model
until grep -q "stage 1d evaluated" $L/stage1d.log 2>/dev/null; do sleep 60; done
echo "$(date -Is) stage 1d finished, starting 1e"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
COMMON="--tokens tokens_tile --pos --clock --map --maps maps_tile_pix --pos_dir pos_pix --train_split fit --steps 30000 --keep_checkpoints"
$PY $M/dynamics.py --run v4 $COMMON --unroll 0.25 > $L/dyn_v4.log 2>&1 &
sleep 60
$PY $M/dynamics.py --run v4_nou $COMMON > $L/dyn_v4_nou.log 2>&1 &
wait
echo "$(date -Is) stage 1e trained"
for r in v4 v4_nou; do $PY $M/dynamics.py --run $r --refit > $L/refit_$r.log 2>&1; done
echo "$(date -Is) stage 1e re-fit"
$PY $M/evaluate.py v3:model_refit.pt --pos_dir pos_pix --n 200 --videos 0 --out stage1e_v3pix.json > $L/eval_1e_v3pix.log 2>&1
$PY $M/evaluate.py v3:model_refit.pt v4:model_refit.pt v4_nou:model_refit.pt --n 200 --videos 8 --out stage1e_eval.json > $L/eval_1e.log 2>&1
echo "$(date -Is) stage 1e evaluated"
