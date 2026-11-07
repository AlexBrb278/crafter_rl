#!/usr/bin/env bash
# Starts the map-memory experiment (queue_map.sh) once the world-model Stage 1e jobs have freed the GPU.
set -u
cd "$(dirname "$0")/.."
until grep -q "stage 1e semantic done" runs/wm/logs/semantic.log 2>/dev/null; do sleep 60; done
echo "$(date -Is) GPU free, starting the map experiment"
scripts/queue_map.sh
