#!/usr/bin/env bash
# Semantic (tile-class) evaluation of the final Stage 1d and 1e models, each as soon as its stage is evaluated.
set -u
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
L=runs/wm/logs
M=methods/08_world_model
until grep -q "stage 1d evaluated" $L/stage1d.log 2>/dev/null; do sleep 60; done
$PY $M/semantic_eval.py v2_long:model_refit.pt v3:model_refit.pt v3_nomap:model_refit.pt --out stage1d_semantic.json > $L/sem_1d.log 2>&1
echo "$(date -Is) stage 1d semantic done"
until grep -q "stage 1e evaluated" $L/stage1e.log 2>/dev/null; do sleep 60; done
$PY $M/semantic_eval.py v4:model_refit.pt v4_nou:model_refit.pt --out stage1e_semantic.json > $L/sem_1e.log 2>&1
echo "$(date -Is) stage 1e semantic done"
