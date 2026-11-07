#!/usr/bin/env bash
# Stage 1b pipeline (paper/designs/world_model_stage1b.md). Waits for record.py --plan b, then:
# tokenizer v2 (hard-pixel loss) + diagnostic creature-weighted tokenizer, token cache, dynamics v2 (all
# fixes) + v2 without the memory signal, head re-fit, and evaluation on the fixed Stage 1 test set.
set -u
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
L=runs/wm/logs
M=methods/08_world_model
mkdir -p $L
# anchored to the interpreter, so a shell that merely mentions record.py cannot match
while pgrep -f "^\.venv/bin/python [^ ]*record\.py" > /dev/null; do sleep 60; done
echo "$(date -Is) recording done: $(ls runs/wm/data/*/ | grep -c npz) episodes"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
$PY $M/tokenizer.py --loss hard --out tokenizer_v2 --steps 25000 --max_frames 1200000 > $L/tokenizer_v2.log 2>&1 &
T1=$!
sleep 120
$PY $M/tokenizer.py --loss sem --out tokenizer_sem --steps 12000 --max_frames 1000000 > $L/tokenizer_sem.log 2>&1 &
wait $T1 || { echo "tokenizer_v2 failed"; exit 1; }
echo "$(date -Is) tokenizer_v2 done: $(tail -1 runs/wm/tokenizer_v2/train_log.jsonl)"
$PY $M/cache_tokens.py --tokenizer tokenizer_v2 --out tokens_v2 > $L/cache_v2.log 2>&1 || { echo "cache failed"; exit 1; }
echo "$(date -Is) tokens_v2 cached"
wait  # the diagnostic tokenizer
echo "$(date -Is) tokenizer_sem done: $(tail -1 runs/wm/tokenizer_sem/train_log.jsonl 2>/dev/null)"
COMMON="--tokens tokens_v2 --clock --unroll 0.25 --train_split fit --steps 30000 --keep_checkpoints"
$PY $M/dynamics.py --run v2 $COMMON --skip > $L/dyn_v2.log 2>&1 &
sleep 60
$PY $M/dynamics.py --run v2_nomem $COMMON > $L/dyn_v2_nomem.log 2>&1 &
wait
echo "$(date -Is) dynamics done"
for r in v2 v2_nomem; do $PY $M/dynamics.py --run $r --refit > $L/refit_$r.log 2>&1; done
echo "$(date -Is) heads re-fit"
$PY $M/evaluate.py uniform v2 v2:model_refit.pt v2_nomem:model_refit.pt --n 200 --videos 8 --out stage1b_eval.json > $L/eval_1b.log 2>&1
echo "$(date -Is) evaluation done"
