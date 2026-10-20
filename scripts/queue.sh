#!/usr/bin/env bash
# Linux counterpart of queue.ps1: run one configuration for several seeds, one after another
# (finished runs are skipped, interrupted ones resume). Run names are <prefix>-s<seed>.
#   scripts/queue.sh --config ppo --bonus 2.0 --prefix ppo-rarity2 --seeds 0,1,2
#   scripts/queue.sh --config ppo_ad --bonus 2.0 --prefix ad-rarity2-5090 --seeds 0,1,2
# Any other argument goes to train.py. Unlike queue.ps1 there is no micro-batching by default
# (it was only needed on the 4 GB laptop); add --micro_batch 256 --aux_micro_batch 128 to match those runs.
set -uo pipefail
CONFIG=ppo BONUS=0 PREFIX=ppo SEEDS=0
EXTRA=()
while [ $# -gt 0 ]; do
  case "$1" in
    --config) CONFIG=$2; shift 2 ;;
    --bonus)  BONUS=$2;  shift 2 ;;
    --prefix) PREFIX=$2; shift 2 ;;
    --seeds)  SEEDS=$2;  shift 2 ;;
    *)        EXTRA+=("$1"); shift ;;
  esac
done
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY=.venv/bin/python
[ -x "$PY" ] || PY=.venv/Scripts/python.exe   # Git Bash on Windows
IFS=',' read -ra SEED_LIST <<< "$SEEDS"
for s in "${SEED_LIST[@]}"; do
  s=${s// /}
  name="$PREFIX-s$s"
  [ -f "runs/$name/final.json" ] && continue
  "$PY" -u methods/05_achievement_distillation/train.py --config "$CONFIG" --seed "$s" \
    --run_name "$name" --rarity_bonus "$BONUS" --resume "${EXTRA[@]}" \
    || echo "queue: $name exited with an error, moving on" >&2
done
