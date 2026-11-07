#!/usr/bin/env bash
# Stage 1c (diagnostic): Stage 1b + the agent's position (from derive_positions.py) as input and an odometry
# output. Initialised from v2; 20k steps alongside v2_long. Then head re-fit and evaluation.
set -u
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
L=runs/wm/logs
M=methods/08_world_model
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
$PY $M/dynamics.py --run v2_pos --init v2 --pos --tokens tokens_v2 --clock --skip --unroll 0.25 --train_split fit \
    --steps 20000 --lr 2e-4 --skip_w 1.0 --keep_checkpoints > $L/dyn_v2_pos.log 2>&1 || { echo "training failed"; exit 1; }
echo "$(date -Is) v2_pos trained"
$PY $M/dynamics.py --run v2_pos --refit > $L/refit_v2_pos.log 2>&1
echo "$(date -Is) v2_pos re-fit"
# wait for v2_long's own evaluation to finish, then evaluate both side by side with Stage 1b
while ! grep -q "v2_long evaluated" $L/stage1b_long.log 2>/dev/null; do sleep 120; done
$PY $M/evaluate.py v2:model_refit.pt v2_long:model_refit.pt v2_pos:model_refit.pt --n 200 --videos 8 --out stage1c_eval.json > $L/eval_1c.log 2>&1
echo "$(date -Is) stage 1c evaluated"
