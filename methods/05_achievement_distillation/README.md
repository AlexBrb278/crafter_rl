# 05 — Achievement Distillation (PPO + contrastive achievement prediction)

**Status:** done (2026-10-20). 15 runs; best AD + rarity bonus β=2 = 23.08 (seed 0); PPO + β=2 = 20.99 ± 0.87 (3 seeds)
**Family:** representation learning · **Framework:** PyTorch, original Crafter
**Code:** https://github.com/snu-mllab/Achievement-Distillation @ `9abbce6` · **Paper:** Moon et al., *Discovering Hierarchical Achievements in Reinforcement Learning via Contrastive Learning*, NeurIPS 2023, https://arxiv.org/abs/2307.03486

**Target:** Crafter 1M, score 21.79 ± 1.37, reward 12.60 ± 0.31 (paper, 10 seeds). PPO baseline from the same paper: 15.60 ± 1.66.

## Why this method (for an RTX 3050 Laptop, 4 GB, Windows)
- It is the best published **model-free** result on the canonical Crafter protocol (see `paper/research_survey.md`).
  World models (DreamerV3, Δ-IRIS, TWM/ITC) score higher but need 1–4 days per seed on far bigger GPUs.
- It is cheap: 8 environments, a 9.2M-parameter IMPALA-ResNet policy, PyTorch. That fits in 4 GB with micro-batching.
- The Craftax/JAX methods (01–04) cannot use the GPU on native Windows (JAX has no Windows CUDA wheels), so the numpy Crafter + PyTorch stack is the practical choice here.

## Method in one paragraph
PPO with a ResNet encoder (γ = 0.95, λ = 0.65, "modern PPO" from the same paper). Every 8 PPO iterations, an **auxiliary phase** (6 epochs over the last 8 rollouts) trains the encoder with two contrastive losses:
- **Intra-trajectory achievement prediction:** from (state, action), predict the latent of the *next achievement* the agent will unlock.
- **Cross-trajectory matching:** achievements matched across episodes by entropic partial optimal transport are pulled together.

KL and value-distillation terms keep the policy and value heads from drifting during the aux phase (as in PPG). The policy also receives a "memory": the normalised latent difference of the last unlocked achievement.

## Hyperparameters (paper = `configs/ppo_ad.yaml`, unchanged)
| | |
|---|---|
| envs × steps per rollout | 8 × 512 (4096 steps), 250 rollouts = 1.024M steps |
| γ / GAE λ | 0.95 / 0.65 |
| PPO | 3 epochs, 8 minibatches (512), clip 0.2, vf coef 0.5, entropy 0.01, Adam 3e-4, grad-norm 0.5 |
| model | IMPALA CNN [64, 128, 128] × 2 blocks, 256 → 1024 hidden, LayerNorm (9.23M params) |
| aux phase | every 8 PPO iterations, 6 epochs, temperature 0.1, π-distill 1.0, V-distill 1.0, OT reg 0.05, 16 target trajectories |

## Local changes (none change the algorithm)
| change | why |
|---|---|
| `common/crafter_vec_env.py` replaces SB3 `SubprocVecEnv` + `VecMonitor` | No `gym`/SB3 on Python 3.13; Windows `spawn`-safe. Same auto-reset / episode-stat semantics. |
| `gym` → `gymnasium` (module alias), stubbed `achievement_distillation.wrapper` | Upstream only uses `gym.spaces` and imports the SB3 wrapper as a type hint. |
| **micro-batching** (`algos.py`) | A 512 minibatch needs 4.3 GB in fp32. It is split into 2 × 256 (PPO) / 4 × 128 (aux) with gradient accumulation, which gives the same gradient. The value normaliser's EWMA β is adjusted to β^(1/k) so its time constant per optimizer step is unchanged. |
| **`fast_aux.py`** | The aux phase spent most of its time recomputing things that cannot change: the frozen old policy on every sample in every epoch, the same anchor goal embeddings for every step of a trajectory, and 16 separate tiny GPU calls per source trajectory. The same math is computed once and reused. Verified against upstream on real data: losses agree to 1e-6 and gradients to TF32 precision (`--check_fast_aux`). |
| numba for opensimplex | Crafter `reset()` 1 s → 50 ms; maps bit-identical (verified). Without it the 8 lock-stepped envs ran at 180 SPS instead of 1600. |
| `--vram_frac 0.85` + `empty_cache()` before the aux phase | Windows WDDM pages VRAM to host RAM instead of raising OOM, which silently slows training ~10×. |
| TensorBoard / JSONL / `status.json` instead of W&B; random MP4 clips; resumable checkpoints | Local monitoring (`scripts/dashboard.py`). |

## Commands
```bash
# from the repo root
.venv/Scripts/python methods/05_achievement_distillation/train.py --config ppo_ad --seed 0 --micro_batch 256 --aux_micro_batch 128
.venv/Scripts/python methods/05_achievement_distillation/train.py --config ppo    --seed 0 --micro_batch 256   # PPO baseline
.venv/Scripts/python scripts/dashboard.py        # live dashboard on http://localhost:8765
.venv/Scripts/python scripts/plot_results.py ad-ppo_ad-s0 ad-ppo-s0 --labels "PPO + AD" "PPO"
```
Add `--resume` to continue from `runs/<run>/checkpoints/latest.pt` (saved every 10 rollouts; it includes the rarity bonus window). `latest.pt` is not committed, so to resume on another machine copy the whole `runs/<run>/` folder over.
A run folder that already has logs is only reused by `--resume` with a `latest.pt`; otherwise `train.py` refuses to start, so a reused run name cannot mix two runs. For a new experiment pick a new `--run_name`.
Outputs go to `runs/<run>/`: `metrics.jsonl` (per rollout), `episodes.jsonl` (per episode, upstream format + step), `tb/`, `videos/` (random clips + `final_episode.mp4`), `checkpoints/`, `final.json`. The logs, `agent-e250.pt` and `final_episode.mp4` are committed (weights and video via Git LFS); the rest stays local (see `.gitignore`).

## Runs
| date | env | steps | seeds | score | reward | wall-clock | notes |
|---|---|---|---|---|---|---|---|
| 2026-10-18 | Crafter 1.8.3 | 1.024M | 0 | **17.71** | 8.72 all / 11.76 last 100 eps | 6.9 h (RTX 3050 Laptop) | paper config; last-100-episode score 25.1. Stuck at stone pickaxe (1.6%) → iron 0.2%, iron tools / diamond 0%. `runs/ad-ppo_ad-s0` |
| 2026-10-19 | Crafter 1.8.3 | 1.024M | 0 | **22.80** | 9.24 all / 13.01 last 100 eps | 6.9 h | **+ rarity bonus β=1** (ours, `--rarity_bonus 1.0`): stone pickaxe 19.1%, stone sword 29%, iron 3.3%, skeleton 21%; wake_up drops 88 → 69%. Last-100 score 33. `runs/ad-rarity1-s0` |
| 2026-10-19 | Crafter 1.8.3 | 1.024M | 0 | 12.40 | 7.46 / 9.01 | 1.2 h | PPO baseline (`--config ppo`); paper PPO 15.60. `runs/ppo-s0` |
| 2026-10-19 | Crafter 1.8.3 | 1.024M | 0 | 20.30 | 8.43 / 11.37 | 1.2 h | PPO + rarity bonus: stone pickaxe 19.9%, iron 3.3%, first iron tools. `runs/ppo-rarity1-s0` |
| 2026-10-19 | Crafter 1.8.3 | 1.024M | 0,1,2 | 15.75 ± 2.99 | | 1.2–1.4 h | PPO, seeds 12.40 / 16.67 / 18.17 |
| 2026-10-19 | Crafter 1.8.3 | 1.024M | 0,1,2 | 18.58 ± 1.62 | | 1.2–1.4 h | PPO + rarity β=1: 20.30 / 18.35 / 17.08 |
| 2026-10-19 | Crafter 1.8.3 | 1.024M | 0,1,2 | **20.99 ± 0.87** | | 1.3 h | PPO + rarity β=2: 20.85 / 20.20 / 21.92 |
| 2026-10-19 | Crafter 1.8.3 | 1.024M | 0,1,2 | 20.14 ± 1.43 | | 1.3 h | PPO + rarity β=4: 21.22 / 18.53 / 20.67 |
| 2026-10-20 | Crafter 1.8.3 | 1.024M | 0 | **23.08** | 9.18 / 13.07 | 6.9 h | AD + rarity β=2: stone pickaxe 25.2%, iron 5.6%, skeleton 22.7%. `runs/ad-rarity2-s0` |

## Notes
- Throughput on the RTX 3050 Laptop: rollout ≈ 6 s and PPO update ≈ 11 s per 4096 steps. Upstream's aux phase took 934 s; see the work log for the fast path.
- Crafter is not bit-reproducible across processes: creatures live in Python `set`s ordered by memory address, and `_balance_object` picks which one to despawn from that order.
