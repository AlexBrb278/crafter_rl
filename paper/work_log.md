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

---

## 2026-10-18 — Windows laptop setup, Achievement Distillation (AD) on Crafter

**Machine:** i5-12500H (12 cores / 16 threads), 16 GB RAM, **RTX 3050 Laptop (4 GB, sm_86)**, Windows 11, no WSL.

**Setup**
- Re-created the gitignored folders: `game/crafter` @ `e04542a`, `game/craftax` @ `c3c2e0d`, `methods/05_achievement_distillation/upstream` @ `9abbce6`, `.venv/` (Python 3.13.4), `runs/`, `logs/`, `checkpoints/`, `wandb/`.
- Installed: torch 2.11.0+cu128 (GPU OK), jax/jaxlib 0.11.2 (**CPU only**: no CUDA wheels for native Windows), flax 0.12.10, numpy 2.5.2, numba 0.68.0, POT 0.9.7.
- Added `scripts/setup_windows.ps1` (Windows counterpart of `setup.sh`).
- Added `--craftax_envs/--craftax_steps` to `scripts/smoke_test.py`, so the CPU-only check finishes. The default 1024 × 1000 run had not finished after 15 min on CPU.
- Smoke test: Crafter OK. Craftax-Classic-Symbolic 0.003M SPS and Craftax-Symbolic 0.001M SPS (64 envs, CPU).
  → **Craftax/JAX methods (01–04) are not practical on this machine.** Method 05 (PyTorch, original Crafter) is.

**Findings**
1. **Crafter `reset()` takes ~1 s** with a pure-Python `opensimplex` (0.4.5): world generation calls `noise3` 82k times.
   - With 8 lock-stepped envs and early episodes of ~170 steps, resets dominate: 180 SPS total.
   - `opensimplex` uses numba when it is installed: reset → 50 ms, 8 envs → **1,620 SPS** (9×).
   - Noise values and generated maps are bit-identical with and without numba (checked on 3 seeds + md5 of the material map).
2. **Crafter is not bit-reproducible across processes.**
   - `World._chunks` holds creatures in Python `set`s, ordered by memory address.
   - `Env._balance_object` picks which creature to despawn from that order.
   - Trajectories for a fixed seed are stable run to run in the same environment, but change when the memory layout changes (e.g. importing numba). The game rules are unchanged.
3. **A 512 minibatch of the AD network needs 4.3 GB in fp32.** On Windows (WDDM), going past dedicated VRAM pages to host RAM instead of raising OOM: 68 samples/s vs ~1,050.
   - Fix: gradient micro-batching (2 × 256 PPO, 4 × 128 aux), which gives an identical gradient.
   - The value normaliser's EWMA β is set to β^(1/k), so its per-step time constant is unchanged.
   - Also a `set_per_process_memory_fraction(0.85)` cap and `empty_cache()` before the aux phase. Without them the aux phase ran ~2× slower: the PPO phase's cached blocks plus the aux phase's varied shapes over-subscribed VRAM.
4. **AD aux phase cost.** Upstream recomputes the frozen old policy on every sample in each of the 6 aux epochs, recomputes identical anchor-goal embeddings per step, and makes 16 tiny GPU calls per source trajectory.
   - `fast_aux.py` does the same math once and reuses it.
   - Equivalence on real buffers (GPU, TF32 convs): losses agree to ~1e-5; relative gradient difference ~1e-6 (prediction) and 7e-4–3e-2 (matching).
     The matching anchors are *normalised differences of two encodings of consecutive frames*, which amplifies TF32 rounding.
     In exact fp32 on CPU the differences are 7.7e-7 (prediction) and 2.7e-5 (matching), so the fast path is exact up to float rounding.
   - Gotcha (upstream): `Buffer.insert` stores `storage[key].cpu()`, which is a *copy* on GPU but the *same tensor* on CPU.
     On CPU the buffer aliases the rollout storage and the goal assertion in `get_goals` fails. Irrelevant for GPU training.
   - One aux phase (8 rollouts, ~190 trajectories): **934 s upstream → 544 s**. The prediction step now runs at the card's fp32 limit (~1,000 samples/s, two gradient passes per sample).
   - channels_last and cudnn.benchmark did not help (1,005 / 1,122 vs 1,125 samples/s).
5. Per-rollout cost: 6 s env + 11 s PPO update (4096 steps). Estimated wall-clock for 1M steps: ~6 h.

**Run started:** `ad-ppo_ad-s0`, paper config `configs/ppo_ad.yaml`, seed 0, `--micro_batch 256 --aux_micro_batch 128` (fast aux on).
Live view: `scripts/dashboard.py` (http://localhost:8765) and TensorBoard in `runs/ad-ppo_ad-s0/tb`.

**Mid-run observation (AD s0, ~740k steps, last 100 episodes).**
- The tech tree is blocked at `make_stone_pickaxe` (1%), even though `make_stone_sword` (11%) needs the same inputs (1 wood + 1 stone, next to a table) and `place_furnace` reached 69%.
- Without a stone pickaxe there is no iron, so iron tools and diamond stay at 0%. `eat_plant` is also 0% (saplings are planted 81% of the time, but the agent never returns to eat the ripe plant).
- Episodes last ~236 steps on average.

**Next experiment (stated before running it): AD + rarity bonus.**
- When achievement *i* is first unlocked in an episode, add β·(1 − p_i) to the reward, where p_i is the fraction of the last 200 episodes that unlocked *i* (`train.py --rarity_bonus 1.0`, class `RarityBonus`).
- Hypothesis: the score is a geometric mean, so it is dominated by the ~0% achievements. Doubling the reward for rare unlocks should raise stone pickaxe / iron / eat_plant above zero and improve the score, at little cost to common achievements.
- Logged rewards and the score use the unshaped environment reward.
- Same seed, same config otherwise.

**Result: `ad-ppo_ad-s0` (AD, paper config, seed 0, 1.024M steps, 6.9 h)**
- Crafter score **17.71**; paper: 21.79 ± 1.37 over 10 seeds.
- Mean episode reward 8.72 over all episodes, 11.76 over the last 100 (paper: 12.60).
- Score over the last 100 episodes rose from 20 to 25 over the final 300k steps, so the agent was still improving when the budget ran out.
- Success over all episodes: wood 92, table 87, wood pickaxe 76, stone 56, coal 37, furnace 36, skeleton 15, stone sword 5.6, **stone pickaxe 1.6, iron 0.2, eat plant 0.1, diamond / iron tools 0**.
- Aux phase 531 → ~690 s over training (more goals per episode → more matched pairs). Cost per 8 rollouts ≈ 8 × 17 s + ~650 s.
- Possible reasons for the gap to the paper, none confirmed:
  - single-seed variance;
  - micro-batching changes the order of the value-normaliser EWMA updates;
  - TF32 numerics.
  Crafter 1.8.1 → 1.8.3 changes no game logic (checked the git history).

---

## 2026-10-19 — Rarity bonus result

**`ad-rarity1-s0`** (AD + rarity bonus β = 1, window 200; otherwise identical to `ad-ppo_ad-s0`; 6.9 h)
- **Crafter score 22.80, vs 17.71 without the bonus (+5.1, +29%)**. That is slightly above the paper's 10-seed AD mean (21.79). Single seed.
- Reward (unshaped environment reward): 9.24 over all episodes, 13.01 over the last 100 (AD: 8.72 / 11.76; paper: 12.60).
- Score over the last 100 episodes at the end: 33 (AD: 25).
- Prediction check:
  - Stone pickaxe 1.6 → **19.1**, stone sword 5.6 → 29.0, iron 0.2 → **3.3**, skeleton 14.8 → 21.1, coal 36.7 → 41.3, furnace 35.8 → 41.6.
  - Eat plant stayed at 0 (prediction wrong for this one). Iron tools / diamond stayed at 0.
  - Cost: wake_up 87.9 → 68.6. The agent spends less time sleeping and more time on the tech tree.
- Timeline: identical to AD until ~100k steps; the lead grows from +0.5 (300k) to +2.3 (650k, when the stone pickaxe takes off) to +5.1 at the end.
  The bonus mostly speeds up *discovery* of the stone tier (furnace 29% vs 13% at 356k). The stone pickaxe then went from 1% to 30% between 600k and 650k.
- Caveats: one seed per arm. The bonus is non-stationary (it shrinks as an achievement becomes common), so late training is close to the plain reward for common achievements.

**Next:** PPO baseline and PPO + rarity bonus (`scripts/queue_ppo.ps1`, ~1.3 h each), to see whether the bonus also helps without AD.

**PPO baseline and PPO + rarity bonus** (`ppo-s0`, `ppo-rarity1-s0`; paper PPO config `configs/ppo.yaml`, seed 0, 1.2 h each)
- PPO: **12.40** (paper 15.60 ± 1.66). PPO + rarity bonus: **20.30** (+7.9, +64%). That beats plain AD (17.71) at one sixth of the wall-clock.
- Stone pickaxe 0.9 → 19.9, furnace 2.6 → 29.2, coal 14.1 → 30.5, skeleton 2.9 → 17.9, iron 0 → 3.3, iron pickaxe 0 → 2 episodes, iron sword 0 → 1 episode. wake_up 85.7 → 72.6.
- The two effects add up, roughly: AD alone +5.3 over PPO, the bonus alone +7.9, both +10.4 (AD + bonus 22.80). Iron tools appeared only in bonus runs (3 PPO + bonus episodes, 1 AD + bonus episode, none without).

**Seed-0 summary (1.024M steps)**

| run | score | reward (all / last 100) | stone pickaxe | iron | wall-clock |
|---|---|---|---|---|---|
| PPO | 12.40 | 7.46 / 9.01 | 0.9 | 0.0 | 1.2 h |
| PPO + rarity bonus | 20.30 | 8.43 / 11.37 | 19.9 | 3.3 | 1.2 h |
| AD | 17.71 | 8.72 / 11.76 | 1.6 | 0.2 | 6.9 h |
| AD + rarity bonus | **22.80** | 9.24 / 13.01 | 19.1 | 3.3 | 6.9 h |

**Running:** seeds 1 and 2 of PPO and PPO + rarity bonus (`scripts/queue_ppo.ps1 1 2`).

**Seeds 1 and 2 of PPO and PPO + rarity bonus** (1.2–1.4 h each; the CPU ran at ~70–77% clock late in the session, slowing rollouts)

| config | seed 0 | seed 1 | seed 2 | mean ± sd |
|---|---|---|---|---|
| PPO | 12.40 | 16.67 | 18.17 | **15.75 ± 2.99** (paper 15.60 ± 1.66) |
| PPO + rarity bonus (β = 1) | 20.30 | 18.35 | 17.08 | **18.58 ± 1.62** |

- The 3-seed PPO mean matches the paper, so the Windows / 4 GB / micro-batching setup reproduces the published baseline.
- Paired gain from the bonus: +7.90, +1.68, −1.09 (mean +2.83, paired t ≈ 1.07, **not significant with n = 3**).
- The gain is large when the baseline never finds the stone pickaxe (seeds 0 and 1: pickaxe 0.9% / ~2%). It is gone when the baseline finds it unaided (seed 2: 26% recent by the end).
- The bonus halves the across-seed spread, and its worst seed (17.08) beats the baseline mean: it raises the floor more than the ceiling.
- Seed 0 is an unlucky seed for the baselines, so the seed-0 AD comparison (17.71 → 22.80) likely overstates the typical AD gain.

**Running:** PPO + rarity bonus β = 2, seeds 0–2 (`scripts/queue.ps1 -Config ppo -Bonus 2.0 -Prefix ppo-rarity2 -Seeds 0,1,2`).
Mistake on the way: `[int[]]` parsed `-Seeds 0,1,2` (one string under `powershell -File`) as 12 via thousands separators. The run was killed after 16 updates and deleted; seeds are now a comma-split string.

**Stronger bonus: PPO + rarity bonus β = 2** (`ppo-rarity2-s{0,1,2}`)

| config | seed 0 | seed 1 | seed 2 | mean ± sd |
|---|---|---|---|---|
| PPO | 12.40 | 16.67 | 18.17 | 15.75 ± 2.99 |
| + rarity β = 1 | 20.30 | 18.35 | 17.08 | 18.58 ± 1.62 |
| + rarity β = 2 | 20.85 | 20.20 | 21.92 | **20.99 ± 0.87** |

- Paired gain of β = 2 over PPO: +8.44, +3.53, +3.74 (mean +5.24, sd 2.78, paired t = 3.26, df = 2; one-sided p ≈ 0.04). Consistent on every seed, unlike β = 1.
  It also wins on seed 2, where the baseline found the stone pickaxe unaided (21.92 vs 18.17).
- Every β = 2 seed beats every PPO seed, and the across-seed spread keeps shrinking (2.99 → 1.62 → 0.87).
- PPO + β = 2 (1.3 h per run) is within 1 point of published AD (21.79, ~7 h per run here).
- Dose-response β = 0 / 1 / 2 → 15.75 / 18.58 / 20.99. **Running:** β = 4, seeds 0–2, to find where it saturates. Next: AD + best β.

**β = 4** (`ppo-rarity4-s{0,1,2}`): 21.22, 18.53, 20.67 → **20.14 ± 1.43**.
- Dose-response over 3 PPO seeds: β = 0 / 1 / 2 / 4 → 15.75 / 18.58 / **20.99** / 20.14. The effect saturates around β = 2.
- β = 4 converts less reliably to stone tools: on seeds 1–2 it lagged β = 2 on stone pickaxe throughout. All bonus settings are well above the baseline.
- **Running:** AD + rarity bonus β = 2, seed 0 (`ad-rarity2-s0`, ~7 h). Compare with AD 17.71 and AD + β = 1 22.80 on the same seed.

---

## 2026-10-20 — AD + rarity bonus β = 2, wrap-up

**`ad-rarity2-s0`** (AD, β = 2, seed 0, 6.9 h): **23.08**, the best score in this project (AD 17.71, AD + β = 1 22.80, paper AD 21.79).
- Final success over all episodes: stone pickaxe 25.2 (AD 1.6), iron 5.6 (0.2), skeleton 22.7 (14.8), stone sword 22.3 (5.6). wake_up 70.1 (87.9). eat_plant, iron tools and diamond are all 0.
- β = 2 with AD trailed β = 1 until ~600k steps (e.g. 10.22 vs 11.76 at 410k, below plain AD too), then overtook once the stone pickaxe took off (63% vs 51% of recent episodes at 780k).

**Summary of all 15 runs (1.024M steps each, 36.0 h of training)**

| config | seeds | score |
|---|---|---|
| PPO | 0, 1, 2 | 15.75 ± 2.99 |
| PPO + rarity β = 1 | 0, 1, 2 | 18.58 ± 1.62 |
| PPO + rarity β = 2 | 0, 1, 2 | **20.99 ± 0.87** |
| PPO + rarity β = 4 | 0, 1, 2 | 20.14 ± 1.43 |
| AD | 0 | 17.71 |
| AD + rarity β = 1 | 0 | 22.80 |
| AD + rarity β = 2 | 0 | **23.08** |

**Takeaways**
1. The Windows / 4 GB port reproduces the PPO baseline (3-seed mean 15.75 vs paper 15.60). Seed 0 is a weak seed for both baselines.
2. A rarity bonus β·(1 − p_i) for first unlocks is a cheap, effective fix for Crafter's stone-pickaxe bottleneck.
   It peaks around β = 2: +5.2 on PPO (3/3 seeds improved), and it brings PPO (1.3 h per run) to within 1 point of published AD (~7 h per run here).
3. It does not solve eat_plant, iron tools or diamond.

**Open / next**
- More AD seeds per arm (≈ 28 GPU-hours on this laptop for 2 more seeds × 2 arms).
- Delayed-reward achievements (eat_plant) need memory or longer-horizon credit assignment, not a bonus.
- The `RarityBonus` is method-agnostic, so it could be tried on the Craftax-Classic methods (01–04) on a Linux GPU box.
