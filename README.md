# crafter_rl

Reinforcement-learning experiments on **Crafter** (Hafner, ICLR 2022) and its JAX version **Craftax / Craftax-Classic** (Matthews et al., ICML 2024).
The goal is to reproduce the strongest published methods, then try our own ideas, and write it up.

## Layout
```
game/        Crafter + Craftax sources (gitignored; recreate with scripts/setup.sh)
methods/     one folder per RL method (01_ppo_rnn … 07_world_model_itc, _template)
common/      shared code: Crafter score, wrappers, logging
scripts/     setup.sh, smoke_test.py
results/     small committed summaries (CSV/plots)
runs/        per-run logs, final weights (agent-e250.pt) and final episode, plus report/;
             resume state, earlier checkpoints and training clips stay local (see .gitignore)
paper/       research_survey.md (literature), work_log.md (what we tried, append-only)
```

## Setup
```bash
./scripts/setup.sh            # clones the games at pinned commits, builds .venv, runs the smoke test
source .venv/bin/activate
```
On Windows (no WSL): `powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1`. PyTorch uses the GPU; JAX/Craftax runs on CPU only.
Model weights and videos (`*.pt`, `*.mp4`) are stored with Git LFS: run `git lfs install` before cloning, or `git lfs pull` afterwards.

## Tools
- `scripts/dashboard.py`: live dashboard on http://localhost:8765 (curves, achievement bars, agent clips).
- `scripts/spectator.py <run>`: films the latest checkpoint of a running job every ~15 min.
- `scripts/queue.ps1` (Windows) / `scripts/queue.sh` (Linux): runs one configuration for several seeds back to back. Finished runs are skipped; interrupted runs resume.
- `scripts/plot_results.py`, `scripts/aggregate_seeds.py`: PNG plots and CSV summaries into `results/`.
- `scripts/report_data.py` + `scripts/make_report.py`: the self-contained HTML report (`runs/report/`).

## Status (2026-10-20)
`methods/05_achievement_distillation` runs on an RTX 3050 Laptop (4 GB). Our addition, a *rarity bonus* for rarely unlocked achievements, lifts PPO from 15.75 ± 2.99 to 20.99 ± 0.87 (3 seeds, β = 2) and Achievement Distillation from 17.71 to 23.08 (seed 0).
See `paper/work_log.md` and `results/`.

## Protocol
- **Crafter / Craftax-Classic:** 1M env steps. Report the Crafter score (geometric mean of per-achievement success rates over *training* episodes; `common/crafter_score.py`), reward, and the 22 success rates.
- **Craftax:** reward as % of 226, on the 1M and 1B tracks, symbolic obs.
- Use at least 3 seeds per configuration. Pin every upstream commit in the method's README.
