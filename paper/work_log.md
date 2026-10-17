# Work log

Append-only, newest entry at the bottom. One entry per working session: what we tried, the exact config/commit, the result, and what's next.
Record negative results too. They go in the paper.

---

## 2026-10-17 — Project setup and literature survey

**Done**
- Created the repo layout: `game/` (gitignored), `methods/` (7 shortlisted methods + `_template`), `common/`, `scripts/`, `results/`, `paper/`.
- Cloned the games into `game/`:
  - Crafter @ `e04542a` (v1.8.3)
  - Craftax @ `c3c2e0d` (v1.6.1, includes Craftax-Classic)
- Created a venv at `.venv/` (Python 3.12.3) with jax/jaxlib 0.11.2 (CUDA 12), flax 0.12.10, torch 2.11.0+cu128, numpy 2.5.4.
  - GPU: RTX 5090 Laptop (24 GB, sm_120) is visible to both JAX and PyTorch.
- `scripts/smoke_test.py` results:
  - Crafter: 64×64×3 obs, 17 actions, 22 achievements, **334 SPS** (1 env, CPU).
  - Craftax-Classic-Symbolic: **~0.2M SPS** (random actions, 1024 and 8192 envs).
  - Craftax-Symbolic: **~0.1M SPS** at 1024 envs, and **<50k SPS** at 8192 envs (shown as "0.0M" by the script).
- Added `common/crafter_score.py` (the official geometric-mean score). Achievement names were checked against `crafter.constants`.
- Literature survey: `paper/research_survey.md`.

**Observations / open issues**
- Craftax-Classic runs at ~0.2M SPS, about half the paper's ~406k (4096 envs, desktop GPU). Full Craftax gets slower at 8192 envs than at 1024. Possible causes: laptop power limits, JAX 0.11 vs the paper's version, or random-action rollouts not being representative. Profile this before the 1B runs.
- Legacy `gym` is not installed, so `crafter.Env().action_space` is a bare namedtuple with no `.sample()`. Sample with numpy, or use a gymnasium wrapper.

**Next**
- Review the survey with the supervisor and pick which methods to run first.
