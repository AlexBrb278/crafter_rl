#!/usr/bin/env bash
# Overnight queue: PPO-RNN (Craftax_Baselines defaults) at 1B steps.
# Runs sequentially on the one GPU; a failed run does not stop the queue.
# Progress: tail -f runs/ppo_rnn/queue.log   |   per run: runs/ppo_rnn/<name>/stdout.log
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
OUT=runs/ppo_rnn
mkdir -p "$OUT"
QLOG="$OUT/queue.log"

# name | env | seed   (ordered by priority: most important first)
RUNS=(
  "classic_1b_s0|Craftax-Classic-Symbolic-v1|0"   # ~3 h  sanity check vs paper (~90% of max reward)
  "craftax_1b_s0|Craftax-Symbolic-v1|0"           # ~5 h  main benchmark (paper: 15.3% of max)
  "classic_1b_s1|Craftax-Classic-Symbolic-v1|1"   # ~3 h  second seed, if time allows
)

echo "$(date -Is) queue start ($(git rev-parse --short HEAD 2>/dev/null))" >> "$QLOG"
for spec in "${RUNS[@]}"; do
  IFS='|' read -r name env seed <<< "$spec"
  dir="$OUT/$name"
  if [ -f "$dir/DONE" ]; then echo "$(date -Is) skip $name (done)" >> "$QLOG"; continue; fi
  rm -rf "$dir"; mkdir -p "$dir"
  echo "$(date -Is) start $name env=$env seed=$seed" >> "$QLOG"
  .venv/bin/python methods/01_ppo_rnn/train.py --run_dir "$dir" \
      --env_name "$env" --total_timesteps 1e9 --seed "$seed" --save_policy \
      > "$dir/stdout.log" 2>&1
  rc=$?
  [ $rc -eq 0 ] && touch "$dir/DONE"
  echo "$(date -Is) end $name rc=$rc $(grep -h '^SPS' "$dir/stdout.log")" >> "$QLOG"
done
echo "$(date -Is) queue finished" >> "$QLOG"
