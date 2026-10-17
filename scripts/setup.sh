#!/usr/bin/env bash
# Re-create the game/ folder and the Python env from scratch.
# game/ is gitignored, so run this after a fresh clone.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

CRAFTER_COMMIT=e04542a   # danijar/crafter, 2023-12-13 (v1.8.3)
CRAFTAX_COMMIT=c3c2e0d   # MichaelTMatthews/Craftax, 2026-06-20 (v1.6.1)

mkdir -p game
[ -d game/crafter ] || git clone https://github.com/danijar/crafter.git game/crafter
[ -d game/craftax ] || git clone https://github.com/MichaelTMatthews/Craftax.git game/craftax
git -C game/crafter checkout -q "$CRAFTER_COMMIT"
git -C game/craftax checkout -q "$CRAFTAX_COMMIT"

[ -d .venv ] || python3 -m venv .venv
.venv/bin/pip install -q --upgrade pip setuptools wheel
.venv/bin/pip install -q "jax[cuda12]"   # RTX 5090 (sm_120) needs CUDA 12.8+ wheels
.venv/bin/pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu128
.venv/bin/pip install -q -e game/crafter -e game/craftax
.venv/bin/pip install -q -r requirements.txt

.venv/bin/python scripts/smoke_test.py
