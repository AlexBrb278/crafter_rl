# RL on Crafter / Craftax: literature survey

_Compiled 2026-10-17. Numbers were checked against the primary PDFs where marked ✔. ~ means self-reported, third-party or converted. ✗ means unverified._
_"Reward %" means episode reward / 22 (Crafter, Craftax-Classic) or / 226 (full Craftax)._

## 1. The benchmarks

### Crafter
Hafner, ICLR 2022 — https://arxiv.org/abs/2109.06780 · https://github.com/danijar/crafter

| | |
|---|---|
| Observation | 64×64×3 RGB only: a 9×7 local top-down view plus an inventory/vitals strip. Health, food and tools are read from pixels. |
| World | Procedural 64×64 map. Privileged state (semantic map, inventory) exists but must not be observed. |
| Actions | 17 discrete: noop, 4 moves, do, sleep, 4 place, 6 make |
| Episode | Ends at health 0 or 10,000 steps |
| Achievements | 22. **Collect:** coal, diamond, drink, iron, sapling, stone, wood. **Defeat:** skeleton, zombie. **Eat:** cow, plant. **Make:** wood/stone/iron pickaxe and sword. **Place:** furnace, plant, stone, table. **Wake up.** |
| Reward | +1 the first time each achievement unlocks per episode; −0.1 / +0.1 per health point lost / regained. Max ≈ 22. |
| Protocol | **1M env steps.** s_i = % of *all training episodes* that unlocked achievement i. |
| Score | **S = exp(mean ln(1+s_i)) − 1**. A geometric mean, so hard achievements dominate. Report score and success rates. Implemented in `common/crafter_score.py`. |
| Humans | Score 50.5 ± 6.8%, reward 14.3 ± 2.3 |
| Speed | Pure Python, ≈1.6k SPS best case. We measured 334 SPS for 1 env on our machine. |

### Craftax-Classic and Craftax
Matthews et al., ICML 2024 — https://arxiv.org/abs/2402.16801 · https://github.com/MichaelTMatthews/Craftax · baselines https://github.com/MichaelTMatthews/Craftax_Baselines

- **Craftax-Classic** is a JAX near-replica of Crafter: the same 22 achievements and 17 actions.
  - Up to 250× faster (≈406k SPS with 4096 envs).
  - Observations: 63×63×3 pixels, or a 1345-dim symbolic vector.
  - Dynamics differ slightly from Crafter, so its numbers are "near-replica", not identical.
  - PPO at 1B steps reaches about 90% of max reward, so Classic is considered solved at 1B except for diamond and eat-plant.
- **Craftax (full)** is a NetHack-like extension.
  - 9 floors, 65 achievements weighted 1/3/5/8, max reward 226, 43 actions.
  - Observations: 110×130×3 pixels, or 8268-dim symbolic.
  - Two official challenges on symbolic obs: **Craftax-1B** (≈1 h for PPO-RNN on a 4090) and **Craftax-1M** (sample efficiency).
  - UED (DR, PLR, ACCEL) made no material progress.
  - Bug fixes landed in v1.5.0 and v1.6.0, so pin the version you report. We use **v1.6.1 @ c3c2e0d**.

## 2. Crafter leaderboard (1M steps)

| Method (family) | Score % | Reward | Notes | ✔ |
|---|---|---|---|---|
| Human experts | 50.5 ± 6.8 | 14.3 ± 2.3 | Hafner 2022 | ✔ |
| **ITC** (TWM + optimal-transport decoding, 2026) | **35.6 ± 0.9** | 72.5% (~15.9) | Craftax-Classic pixels. https://arxiv.org/abs/2605.16457 | ✔ |
| Dedieu et al. TWM "slow" (ICML 2025) | 31.77 ± 1.43 | 69.66% | Craftax-Classic pixels; first agent above human reward. https://arxiv.org/abs/2502.01591 | ✔ |
| Dedieu et al. TWM "fast" | 27.91 ± 0.63 | 67.42% | same | ✔ |
| SPRING (GPT-4 zero-shot) | 27.3 ± 1.2 | 12.3 ± 0.7 | No RL; different protocol. https://arxiv.org/abs/2305.15486 | ✔ |
| AdaRefiner (GPT-4 + Llama2-7B) | 15.8 (1M) / 28.2 (5M) | 12.3 / 12.9 | https://arxiv.org/abs/2309.17176 | ✔ |
| **Achievement Distillation** (NeurIPS 2023) | **21.79 ± 1.37** | 12.60 ± 0.31 | Best model-free on true Crafter. https://arxiv.org/abs/2307.03486 | ✔ |
| Curious Replay (DreamerV3 + prioritised replay) | 19.4 ± 1.6 | – | https://arxiv.org/abs/2306.15934 | ✔ |
| Dedieu MFRL (tuned PPO + GRU) | 16.77 ± 1.11 | 55.49% | Craftax-Classic pixels; 15 min on an A100 | ✔ |
| PPO (ResNet, "modern PPO") | 15.60 ± 1.66 | 10.32 | Moon et al. 2023 | ✔ |
| DreamerV3 | 14.5 ± 1.6 | 11.7 ± 1.9 | https://arxiv.org/abs/2301.04104 (v1) | ✔ |
| LSTM-SPCNN | 12.1 ± 0.8 | – | https://arxiv.org/abs/2208.03374 | ✔ |
| DART | – | 12.2 | https://arxiv.org/abs/2406.01361 | ✔ |
| EDE | 11.7 ± 1.0 | – | https://arxiv.org/abs/2306.05483 | ✔ |
| OC-SA | 11.1 ± 0.7 | – | Stanić et al. | ✔ |
| DreamerV2 | 10.0 ± 1.2 | 9.0 | Hafner | ✔ |
| Δ-IRIS | 9.30 | 35.0% | Eval-episode score. https://arxiv.org/abs/2406.19320 | ✔ |
| IRIS | 6.66 | 25.0% | https://arxiv.org/abs/2209.00588 | ✔ |
| PPO (2021 SB3) | 4.6 | 4.2 | Hafner | ✔ |
| MuZero + SPR | 4.4 (16.4 pre-trained) | 8.5 / 12.7 | https://arxiv.org/abs/2302.04009 | ✔ |
| Rainbow | 4.3 | 6.0 | Hafner | ✔ |
| Plan2Explore / RND / Random | 2.1 / 2.0 / 1.6 | – | Hafner | ✔ |

**At 10M steps:** EMERALD scores **58.1%** (first above human score; https://arxiv.org/abs/2507.04075 ✔). Δ-IRIS scores 42.5, DreamerV3-XL 39.6.
STORM, DIAMOND and TWISTER report no Crafter results. ELLM and OMNI use modified Crafter variants, so they are not comparable.

## 3. Craftax leaderboards

**Craftax-1B (symbolic, % of 226)**

| Method | % max | ✔ |
|---|---|---|
| AHC (rule-based curriculum on GTrXL; preprint Oct 2026, https://arxiv.org/abs/2610.15328) | ~17.1 at 1B, ~20.7 at 2B | ~ |
| DiCode (LLM-written curricula, ICML 2026, https://arxiv.org/abs/2602.08194) | ~20–21 | ✗ |
| PPO-GTrXL | 18.3 (20.6 at 4B) | ~ |
| PQN-RNN | 16.0 | ✔ |
| PPO-RNN | 15.3 | ✔ |
| PPO-RND / PPO / ICM / E3B | 12.0 / 11.9 / 11.9 / 11.0 | ✔ |

Full Craftax is unsolved: the best result is about 18–21%, and the deepest floors are never reached.

**Craftax-1M (% of 226):**

| Method | % max | ✔ |
|---|---|---|
| Dedieu TWM | 7.20 | ✔ |
| ITC | 7.09 | ✔ |
| Simulus (https://github.com/leor-c/Simulus) | 6.6 | ✔ |
| Dedieu MFRL | 4.63 | ✔ |
| PPO-RNN | 2.3 | ✔ |

## 4. Method families

| Family | Best result | Compute (reported) | Code | Feasibility on our laptop |
|---|---|---|---|---|
| **Model-free (PPO, PQN, + memory)** | AD 21.8 (Crafter); GTrXL 18.3% (Craftax-1B) | Minutes to hours | Craftax_Baselines, purejaxql, transformerXL_PPO_JAX | Easy; ~1–3 h per Craftax-1B run (estimate) |
| **World models** (DreamerV3, IRIS family, Dedieu TWM, ITC, EMERALD) | ITC 35.6 (Classic 1M) | 14–94 h per 1M on an A100 / 3090 / 4090 | dreamerv3, delta-iris, Simulus, ITC, EMERALD | Feasible but slow: 1–4 days per seed |
| **Intrinsic motivation** (RND, ICM, E3B, EDE) | Small or no gains | ≈ PPO | Craftax_Baselines, ede | Easy |
| **Representation / aux losses** (Achievement Distillation, OC-SA) | AD 21.8 | Hours per seed | Achievement-Distillation | Easy to moderate |
| **LLM-guided** (SPRING, ELLM, AdaRefiner, SGRL, DiCode) | SPRING 27.3 zero-shot (privileged knowledge) | API-bound | SPRING, ellm, AdaRefiner, omni | Possible with a local 7–14B LLM; won't match GPT-4 numbers |

## 5. Shortlist (easiest → most ambitious)

These map to `methods/01…07`.

1. **PPO-RNN** (Craftax_Baselines): validates the stack against official numbers.
2. **PQN-RNN**: the strongest cheap value-based agent.
3. **PPO-GTrXL**: the strongest cheap model-free agent, and the backbone for curriculum ideas.
4. **Dedieu tuned PPO+GRU, Classic pixels 1M**: the best accuracy per GPU-hour; good for many seeds and ablations.
5. **Achievement Distillation**: the canonical-Crafter model-free SOTA, with official code.
6. **DreamerV3 (+ Curious Replay)**: the standard world-model reference.
7. **ITC / Dedieu TWM**: current 1M SOTA; the base for any new-SOTA claim.

**Key takeaways**
- On true Crafter at 1M: Achievement Distillation scores 21.8 (model-free) and Curious Replay 19.4 (model-based).
- On Craftax-Classic at 1M: transformer world models pass human *reward* but stay well below human *score* (35.6 vs 50.5).
- Full Craftax is wide open.
