#!/usr/bin/env bash
# Survival-signal experiments: AD + rarity β=2 + VitalsShaping, seeds 0-2 in parallel, same flags as
# the 3-seed AD + rarity β=2 baseline (micro-batch 256 / aux 128).
#   scripts/queue_vitals.sh                                    -> ad-rarity2-vit1  (γΦ' − Φ, α = 1)
#   scripts/queue_vitals.sh ad-rarity2-vit1u --vitals_gamma 1  -> no per-step cost (Φ' − Φ)
set -u
PREFIX=${1:-ad-rarity2-vit1}
shift || true
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p runs/logs
echo "$(date -Is) start: $PREFIX seeds 0-2 (extra: $*)"
for s in 0 1 2; do
  scripts/queue.sh --config ppo_ad --bonus 2.0 --prefix "$PREFIX" --seeds "$s" --vitals 1.0 "$@" \
      --micro_batch 256 --aux_micro_batch 128 --vram_frac 0.30 > "runs/logs/$PREFIX-s$s.out" 2>&1 &
  sleep 20
  .venv/bin/python scripts/spectator.py "$PREFIX-s$s" --every 15 --steps 600 \
      > "runs/logs/$PREFIX-s$s.spectator.out" 2>&1 &
done
wait
echo "$(date -Is) all done"
