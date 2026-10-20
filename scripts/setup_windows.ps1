# Windows counterpart of setup.sh: re-create the gitignored game/, .venv/ and method
# upstream clones, install everything, and run the smoke test.
#   powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1
# Differences from setup.sh:
#   * JAX has no CUDA wheels for native Windows, so jax is CPU-only here (Craftax works, slowly).
#     PyTorch uses the GPU (cu128 wheels; tested on an RTX 3050 Laptop, 4 GB).
#   * numba is installed so opensimplex (Crafter world generation) is JIT-compiled:
#     env.reset() drops from ~1 s to ~50 ms. Maps are bit-identical.
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$CrafterCommit = "e04542a"   # danijar/crafter, 2023-12-13 (v1.8.3)
$CraftaxCommit = "c3c2e0d"   # MichaelTMatthews/Craftax, 2026-06-20 (v1.6.1)
$ADCommit      = "9abbce6"   # snu-mllab/Achievement-Distillation, 2025-02-25

New-Item -ItemType Directory -Force game, runs, logs, checkpoints, wandb | Out-Null
if (-not (Test-Path game\crafter)) { git clone https://github.com/danijar/crafter.git game\crafter }
if (-not (Test-Path game\craftax)) { git clone https://github.com/MichaelTMatthews/Craftax.git game\craftax }
$ad = "methods\05_achievement_distillation\upstream"
if (-not (Test-Path $ad)) { git clone https://github.com/snu-mllab/Achievement-Distillation.git $ad }
git -C game\crafter checkout -q $CrafterCommit
git -C game\craftax checkout -q $CraftaxCommit
git -C $ad checkout -q $ADCommit

# Python 3.13 (3.14 also works for torch, but 3.13 has the widest wheel coverage).
if (-not (Test-Path .venv)) { py -3.13 -m venv .venv }
$py = ".venv\Scripts\python.exe"
& $py -m pip install -q --upgrade pip setuptools wheel
& $py -m pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu128
& $py -m pip install -q -e game\crafter -e game\craftax
& $py -m pip install -q -r requirements.txt
& $py scripts\smoke_test.py --craftax_envs 64 --craftax_steps 200
