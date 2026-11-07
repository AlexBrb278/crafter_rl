#!/usr/bin/env bash
# Map-memory experiment: AD + rarity β=2, plus the agent's own top-down map built from its frames and
# actions only (common/pixel_map.py, --pixel_map 1). Seeds 0-2 in parallel, same flags as the 3-seed
# AD + rarity β=2 baseline (micro-batch 256 / aux 128), each with a spectator filming checkpoints.
set -u
PREFIX=${1:-ad-rarity2-map}
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p runs/logs
echo "$(date -Is) start: $PREFIX seeds 0-2"
for s in 0 1 2; do
  scripts/queue.sh --config ppo_ad --bonus 2.0 --prefix "$PREFIX" --seeds "$s" --pixel_map 1 \
      --micro_batch 256 --aux_micro_batch 128 --vram_frac 0.30 > "runs/logs/$PREFIX-s$s.out" 2>&1 &
  sleep 20
  .venv/bin/python scripts/spectator.py "$PREFIX-s$s" --every 15 --steps 600 \
      > "runs/logs/$PREFIX-s$s.spectator.out" 2>&1 &
done
wait
echo "$(date -Is) all done"
