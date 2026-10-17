# crafter_rl

Reinforcement-learning experiments on **Crafter** (Hafner, ICLR 2022) and its JAX version **Craftax / Craftax-Classic** (Matthews et al., ICML 2024).
The goal is to reproduce the strongest published methods, then try our own ideas, and write it up.

## Layout
```
game/        Crafter + Craftax sources (gitignored; recreate with scripts/setup.sh)
methods/     one folder per RL method (01_ppo_rnn … 07_world_model_itc, _template)
common/      shared code: Crafter score, wrappers, logging
scripts/     setup.sh, smoke_test.py
results/     small committed summaries (CSV/plots); raw runs go to runs/ (gitignored)
paper/       research_survey.md (literature), work_log.md (what we tried, append-only)
```

## Setup
```bash
./scripts/setup.sh            # clones the games at pinned commits, builds .venv, runs the smoke test
source .venv/bin/activate
```

## Protocol
- **Crafter / Craftax-Classic:** 1M env steps. Report the Crafter score (geometric mean of per-achievement success rates over *training* episodes; `common/crafter_score.py`), reward, and the 22 success rates.
- **Craftax:** reward as % of 226, on the 1M and 1B tracks, symbolic obs.
- Use at least 3 seeds per configuration. Pin every upstream commit in the method's README.
