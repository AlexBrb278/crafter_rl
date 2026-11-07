#!/usr/bin/env bash
# World model Stage 2a (paper/designs/world_model_stage2.md): AD + rarity β=2, plus PPO on dreamed rollouts
# from the frozen v4 world model (256 dreams x 16 steps per epoch, sampled, half from iron-chain states).
# Diagnostic: the world model was trained on other agents' play, so the score is not 1M-comparable.
# Seeds 0-2 in parallel, same flags as the AD + rarity β=2 baseline, each with a spectator.
set -u
PREFIX=${1:-ad-rarity2-dream}
shift || true
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p runs/logs
echo "$(date -Is) start: $PREFIX seeds 0-2 (extra: $*)"
for s in 0 1 2; do
  scripts/queue.sh --config ppo_ad --bonus 2.0 --prefix "$PREFIX" --seeds "$s" --dream v4:model_refit.pt "$@" \
      --micro_batch 256 --aux_micro_batch 128 --vram_frac 0.30 > "runs/logs/$PREFIX-s$s.out" 2>&1 &
  sleep 20
  .venv/bin/python scripts/spectator.py "$PREFIX-s$s" --every 15 --steps 600 \
      > "runs/logs/$PREFIX-s$s.spectator.out" 2>&1 &
done
wait
echo "$(date -Is) all done"
