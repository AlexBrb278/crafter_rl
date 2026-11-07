#!/usr/bin/env bash
# Stage 1 pipeline (paper/designs/world_model_stage1.md): wait for record.py, train the tokenizer,
# cache tokens, then train the uniform and rarity-weighted world models side by side.
set -u
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
L=runs/wm/logs
mkdir -p $L
# anchored to the interpreter: a shell whose command line merely mentions record.py must not match
while pgrep -f "^\.venv/bin/python [^ ]*record\.py" > /dev/null; do sleep 60; done
echo "$(date -Is) recording done: $(ls runs/wm/data/*/ | grep -c npz) episodes"
$PY methods/08_world_model/tokenizer.py --steps 20000 > $L/tokenizer.log 2>&1 || { echo "tokenizer failed"; exit 1; }
echo "$(date -Is) tokenizer done: $(tail -1 runs/wm/tokenizer/train_log.jsonl)"
rm -rf runs/wm/tokens
$PY methods/08_world_model/cache_tokens.py > $L/cache.log 2>&1 || { echo "cache failed"; exit 1; }
echo "$(date -Is) tokens cached"
$PY methods/08_world_model/dynamics.py --run uniform --steps 30000 > $L/dyn_uniform.log 2>&1 &
sleep 30
$PY methods/08_world_model/dynamics.py --run rarity --steps 30000 --rare_weight 5 > $L/dyn_rarity.log 2>&1 &
wait
echo "$(date -Is) dynamics done"
