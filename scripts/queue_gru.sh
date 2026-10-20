#!/usr/bin/env bash
# GRU experiment queue (Linux RTX 5090 machine).
#  phase 1: AD + rarity β=2 + GRU memory (configs/ppo_ad_gru.yaml), seeds 0-2 in parallel
#  phase 2: AD + rarity β=2 without GRU, seeds 1-2, same flags as the home laptop's ad-rarity2-s0
#           (micro-batch 256 / aux 128), so the baseline also has 3 seeds
# Each run has a spectator filming its latest checkpoint every 15 min. Logs: runs/logs/.
set -u
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # less fragmentation with 3 runs per GPU
mkdir -p runs/logs
spect() { .venv/bin/python scripts/spectator.py "$1" --every 15 --steps 600 > "runs/logs/$1.spectator.out" 2>&1 & }

echo "$(date -Is) phase 1: GRU seeds 0-2"
for s in 0 1 2; do
  scripts/queue.sh --config ppo_ad_gru --bonus 2.0 --prefix ad-rarity2-gru --seeds "$s" \
      --vram_frac 0.30 --aux_micro_batch 128 > "runs/logs/ad-rarity2-gru-s$s.out" 2>&1 &
  sleep 20
  spect "ad-rarity2-gru-s$s"
done
wait
echo "$(date -Is) phase 2: baseline seeds 1-2"
for s in 1 2; do
  scripts/queue.sh --config ppo_ad --bonus 2.0 --prefix ad-rarity2 --seeds "$s" \
      --micro_batch 256 --aux_micro_batch 128 --vram_frac 0.30 > "runs/logs/ad-rarity2-s$s.out" 2>&1 &
  sleep 20
  spect "ad-rarity2-s$s"
done
wait
echo "$(date -Is) all done"
