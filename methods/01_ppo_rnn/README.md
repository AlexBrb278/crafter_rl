# 01 — PPO-RNN (GRU) on Craftax

**Status:** done (baseline reproduced, 2026-10-18)
**Family:** model-free · **Framework:** JAX (Craftax_Baselines)
**Code:** https://github.com/MichaelTMatthews/Craftax_Baselines · **Paper:** https://arxiv.org/abs/2402.16801

**Targets to reproduce**
- Craftax-1B (symbolic): 15.3% of max reward
- Craftax-Classic-1B (symbolic): ~90% of max reward
- Craftax-1M: 2.3%

**Why first:** minutes to ~2 h per run. It validates the JAX/RTX 5090 stack, logging, and scoring before anything expensive.
Also covers the cheap baselines in the same repo: PPO (MLP), PPO-RND, ICM, E3B.

## Setup
- Upstream: `upstream/` = Craftax_Baselines @ `7ce36fa` (2026-06-16), unmodified (gitignored).
- `train.py` runs upstream `ppo_rnn.py` as-is. It sets wandb to offline mode and appends every update to `<run_dir>/metrics.jsonl`.
- `summarize.py` gives final reward, per-achievement success rates and score, and writes a plot and CSV to `results/ppo_rnn/`.
- Queue: `scripts/overnight_ppo_rnn.sh`.

## Hyperparameters (upstream defaults, unchanged)
| param | value | param | value |
|---|---|---|---|
| num_envs | 1024 | lr | 2e-4, linearly annealed |
| num_steps (rollout) | 64 | update_epochs | 4 |
| num_minibatches | 8 | gamma | 0.99 |
| gae_lambda | 0.8 | clip_eps | 0.2 |
| ent_coef | 0.01 | vf_coef | 0.5 |
| max_grad_norm | 1.0 | network | 512-wide MLP + GRU, tanh |
| optimistic resets | on, ratio 16 | obs | symbolic |

## Throughput on our laptop (RTX 5090 Laptop, "balanced" profile, GPU capped at 95 W)
- Craftax-Classic-Symbolic: ~92k SPS, so 1B steps take ~3.0 h.
- Craftax-Symbolic: ~55k SPS, so 1B steps take ~5.1 h.

## Runs
| date | env | steps | seed | reward % max | score | wall-clock | notes |
|---|---|---|---|---|---|---|---|
| 2026-10-17 | Classic-Symbolic | 20M | 0 | 66.9 | 37.3 | 4 min | smoke test |
| 2026-10-17 | Craftax-Symbolic | 10M | 0 | 4.4 | – | 4 min | smoke test |
| 2026-10-17 | Classic-Symbolic | 1B | 0 | 88.3 | 87.5 | 2 h 55 min | 95k SPS; diamond 30.5%, eat_plant 20.1% |
| 2026-10-17 | Craftax-Symbolic | 1B | 0 | **16.1** | – | 5 h 04 min | 55k SPS; paper reports 15.3%. Reaches floor 1 (dungeon), never floor 2+ |
| 2026-10-18 | Classic-Symbolic | 1B | 1 | 87.7 | 85.1 | 2 h 56 min | diamond 20.2%, eat_plant 16.0% |

**Conclusion:** we reproduce the paper. Classic reaches ~88% (paper ~90%) and Craftax-1B 16.1% (paper 15.3%, one seed).
The final policies are saved under `runs/ppo_rnn/<run>/wandb/offline-run-*/files/policies/`. The plot is `results/ppo_rnn/learning_curves.png`.
