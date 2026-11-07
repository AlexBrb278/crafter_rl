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

**Repo and resume fixes (same day)**
- Pushed to GitHub: code, results, and for each run its logs, final weights (`agent-e250.pt`) and `final_episode.mp4` (weights and videos via Git LFS), plus the HTML report. `latest.pt`, earlier checkpoints and training clips stay local.
- Bug: `latest.pt` did not store the rarity bonus window. A resumed bonus run started with an empty window, so every first unlock (even collect_wood) paid the full β until 200 episodes had finished.
  Fixed: the checkpoint now stores `rarity_hist` and `rarity_total`.
  Tested by resuming a 2-epoch PPO + β = 2 run for a third epoch: the 45 saved episodes were restored and the window grew to 68, and the bonus total went 101.07 → 154.12 instead of restarting.
  None of the 15 runs was ever resumed (no "resumed from" in any `train.log`), so their results are unaffected.
- `scripts/setup.sh` (Linux) did not clone the AD upstream that `train.py` imports. It now clones it @ `9abbce6`, as `setup_windows.ps1` already did.
- Guard: `train.py` appends to `metrics.jsonl` / `episodes.jsonl`, so a fresh run under a used name (e.g. the committed `ad-rarity2-s0`) would have mixed into the old logs and averaged its reward over both runs.
  It now refuses to start if the folder has logs, unless `--resume` finds `checkpoints/latest.pt`.
- `scripts/queue.sh`: Linux version of `queue.ps1` (same skip / resume behaviour). It does not force micro-batching, which the 24 GB RTX 5090 does not need.
- Tested end to end from Git Bash: a 2-seed queue, a rerun that skips both finished seeds, the three guard cases, and a refusal on the committed `ppo-s0`, which was left untouched.

---

_Entries below were written on the Linux RTX 5090 machine in parallel with the Windows entries above._

## 2026-10-17 (evening) — PPO-RNN baseline, overnight 1B runs

**Why this first:** it's the official Craftax baseline with verified numbers (Craftax-1B: 15.3% of max; Classic-1B: ~90%). Matching them validates our stack (JAX on the RTX 5090, logging, scoring) before we try anything new.

**Setup**
- Cloned Craftax_Baselines @ `7ce36fa` into `methods/01_ppo_rnn/upstream/` and left it unmodified.
- Added deps: distrax 0.1.9, optax 0.2.8, gymnax 0.0.9, chex 0.1.92, orbax-checkpoint 0.12.7. Upstream pins orbax 0.5.0, but the newer version works; it only emits a deprecation warning when saving.
- Wrote `methods/01_ppo_rnn/train.py`. Upstream only logs to wandb, so we run wandb offline and patch `Run.log` to also write `metrics.jsonl`.
  - First attempt patched `wandb.log`, which logged nothing: `wandb.init()` rebinds the module-level `wandb.log`.
- Hyperparameters: all upstream defaults (table in `methods/01_ppo_rnn/README.md`).

**Smoke tests** (seed 0)
- Classic-Symbolic, 20M steps (4 min):
  - reward 14.7/22 (66.9%), score 37.3 (averaged over updates, not the official protocol).
  - 18 of 22 achievements unlocked. Never unlocked: diamond, eat_plant, iron pickaxe, iron sword.
- Craftax-Symbolic, 10M steps (4 min): 4.4% of max reward.
- Checkpoint saving (`--save_policy`) works with orbax 0.12.

**Throughput finding**
- PPO-RNN trains at 92k SPS on Classic and 55k on full Craftax, about 3× slower than the ~280k the paper implies for an RTX 4090.
- The GPU is 98% utilised but capped at 95 W (max 175 W). The power profile is "balanced", on AC.
- Try the "performance" profile and re-measure. The user has to change it.

**Overnight queue** (`scripts/overnight_ppo_rnn.sh`, started 20:06, run under `systemd-inhibit` against sleep and lid close)
1. `classic_1b_s0`: Craftax-Classic-Symbolic, 1B steps, seed 0 (~3 h)
2. `craftax_1b_s0`: Craftax-Symbolic, 1B steps, seed 0 (~5 h)
3. `classic_1b_s1`: Craftax-Classic-Symbolic, 1B steps, seed 1 (~3 h)

Outputs go to `runs/ppo_rnn/<name>/`. Summarise with `python methods/01_ppo_rnn/summarize.py`.

**Next (morning)**
- Compare against the paper and note the per-achievement gaps (diamond, eat_plant, deeper Craftax floors).
- Then decide between more seeds of PPO-RNN or moving on to PQN / GTrXL.

---

## 2026-10-19 — Overnight PPO-RNN results: baseline reproduced

All 3 runs finished with rc=0 (queue ran 2026-10-17 20:06 → 2026-10-18 07:01).

| run | steps | reward | % of max | score* | paper | wall-clock | SPS |
|---|---|---|---|---|---|---|---|
| classic_1b_s0 | 1B | 19.43 / 22 | 88.3 | 87.5 | ~90% | 2 h 55 min | 95k |
| classic_1b_s1 | 1B | 19.28 / 22 | 87.7 | 85.1 | ~90% | 2 h 56 min | 95k |
| craftax_1b_s0 | 1B | 36.45 / 226 | **16.1** | – | 15.3% | 5 h 04 min | 55k |

\* Rates averaged over the last 1% of updates (≈10M steps), not the official all-training-episodes protocol.

**Observations**
- **Classic:** 20 of 22 achievements are at ≥94%. The only bottlenecks are **collect_diamond** (30.5 / 20.2%) and **eat_plant** (20.1 / 16.0%), exactly as the Craftax paper reports. The two seeds agree closely (±0.3% reward).
  - Most of the learning happens in the first ~50M steps (≈80% of max). The remaining 950M add only ~8 points.
- **Full Craftax:** the agent masters the overworld and floor 1 (enter_dungeon 89%, bow/arrows ~87%, orc kills 55–70%).
  - It **never enters floor 2** (gnomish mines 0%) and never learns magic (fireball/iceball 0%) or enchanting.
  - Diamond 14.8%, ruby/sapphire ~12.5%, iron pickaxe only 18%.
  - Reward is still slowly rising at 1B, so it has not plateaued.
- All runs saved final checkpoints.

**Implications for what to try next**
- Our stack is validated, so new methods can be compared against these numbers.
- Full Craftax is where RL headroom is: the bottleneck is deep exploration and long-horizon credit assignment (floor 2+, magic). On Classic, the remaining problem is just two rare achievements.

---

## 2026-10-20 — Death diagnosis: why no agent gets near a diamond

**Question.** Diamond is 0.00% in all 15 runs. The training logs show that every episode ends in death (none reaches the 10,000-step limit), with a median length of 220–236 steps. The first night is darkest at step 210. So we need to know what kills the agent.

**Method** (`scripts/diagnose_deaths.py`, `scripts/analyze_deaths.py`; Linux RTX 5090 machine, from the committed `agent-e250.pt` weights)
- Hooked Crafter's three damage paths so every lost health point is logged with its source:
  - `Zombie.update`: 7 damage if the player is asleep, 2 if awake.
  - `Arrow.update`: skeletons, 2 damage.
  - `Player.update`: lava (health → 0), or hunger/thirst/fatigue (health −1 every ~15 steps while food, drink or energy is 0).
- Cause of death = the source of the hit that took health to 0.
- 200 fresh episodes per run, all 15 runs (3,000 episodes), using the stochastic policy as in training. Each run takes ~50 s on the GPU with 32 envs in lockstep.
- Night = daylight < 0.3, which is ~30% of each 300-step cycle (Crafter: `daylight = 1 − |cos(π·((t/300) mod 1 + 0.3))|³`).
- Also needed here: numba, imageio-ffmpeg, POT, and git-lfs (taken from the Ubuntu .deb into ~/.local/bin, no sudo) to get the real `.pt`/`.mp4` files.

**Results** (`results/deaths/summary.csv`, `causes.png`, `timing.png`)

| config | n | median steps | alive at 300 | zombie awake | zombie asleep | arrow | thirst | hunger/fatigue | lava | water 0 at death |
|---|---|---|---|---|---|---|---|---|---|---|
| PPO (3 seeds) | 600 | 234 | 22.0% | 46% | 2% | 23% | 17% | 8% | 4% | 38.5% |
| PPO + rarity β=2 (3 seeds) | 600 | 230 | 17.0% | 52% | 4% | 18% | 12% | 6% | 8% | 39.8% |
| AD (1 seed) | 200 | 258 | 22.0% | 52% | 0% | 15% | 22% | 5% | 5% | 58.0% |
| AD + rarity β=2 (1 seed) | 200 | 240 | 17.5% | 49% | 0% | 18% | 18% | 6% | 8% | 50.0% |

Over all 3,000 episodes:
- 100% die.
- Zombies cause 52% of deaths; 69% of those happen at night.
- In 54% of deaths, food, water or energy is at 0 (water alone: 42%).
- The agent spends only **4% of night-time asleep**, and only 5% of deaths happen in sleep.
- Not one diamond or iron pickaxe.

**Interpretation**
- The expected failure, "sleeps in the open and gets eaten", is rare. The agent mostly **stays awake at night** on open ground, where zombies spawn more in the dark, and is killed awake.
- Neglecting vitals is the second big factor. At 0 food, water or energy, health stops regenerating, so any zombie or arrow becomes lethal.
- The rarity bonus reduces sleep further for AD: episodes with any sleep drop from 84.5% to 49.5%, which matches wake_up falling from 87.9% to 70.1% in training.
- Skeleton arrows (15–23%) and lava (4–8%) hit the agent in caves, which is exactly where iron and diamonds are.
- Likely root causes:
  - γ = 0.95: effective horizon ~20 steps, but a night lasts ~90.
  - A memoryless CNN policy that cannot remember where water was.

**Report:** added a "Why the agent dies" section to `runs/report/crafter_report.html`: tiles, a cause-of-death chart, a death-timing chart with night shading, a per-config table and 6 death videos. The videos are re-encoded copies in `runs/report/media/<run>/videos/death_*.mp4`; the originals are in `runs/<run>/videos/death_*.mp4`, with an index in `deaths.jsonl`.

**Next:** γ = 0.99 (horizon), then memory (GRU), then a tech-tree-aware bonus. Use PPO + rarity β=2 as the base, 3 seeds each. Track survival and cause of death alongside the score.

---

## 2026-10-20 (afternoon) — Longer horizon γ = 0.99: negative result

**Hypothesis** (from the death diagnosis): γ = 0.95 gives an effective horizon of ~20 steps, but a night lasts ~90. So the agent cannot value surviving the night, and γ = 0.99 (~100 steps) should make it survive longer.

**Setup** (Linux RTX 5090 machine, real Crafter 1.8.3, home code `methods/05_achievement_distillation/train.py`)
- Added `--gamma` / `--gae_lambda` overrides to train.py. The value is recorded in `config.json`.
- PPO + rarity β = 2, the same as the home `ppo-rarity2` runs. Only γ changes: 0.95 → 0.99. λ stays 0.65.
- Runs: `ppo-rarity2-g99-s0/1/2`, plus the control `ppo-rarity2-5090-s0` (γ = 0.95 rerun on this machine).
- No micro-batching here. Without it each run reserves 6–7 GB of GPU memory, so the 4th run hit OOM when started alongside the other three. Seed 0 was then started after the control finished.
- ~1 h per run with 3 in parallel (0.43 h alone). Spectator checkpoint videos every 10 min. Death diagnosis on all 4 (200 episodes each).

**Results** (`results/gamma_summary.csv`, `runs/report/gamma.html`)

| config | seeds | score | per seed | eval median steps | alive at 300 | zombie deaths | stone pickaxe | iron |
|---|---|---|---|---|---|---|---|---|
| PPO + rarity β=2, γ 0.95 (home) | 3 | **20.99 ± 0.87** | 20.85 / 20.20 / 21.92 | 230 | 17.0% | 56% | 17.4% | 2.5% |
| same, rerun here | 1 | 20.90 | 20.90 | 232 | 13.5% | 50% | 19.4% | 4.2% |
| PPO + rarity β=2, **γ 0.99** | 3 | 16.94 ± 2.55 | 19.79 / 14.90 / 16.12 | 228 | 18.5% | 55% | 9.1% | 1.2% |

- **The machine check passes.** The control scores 20.90 here against 20.85 at home, so results from both laptops can be compared directly.
- **γ = 0.99 is worse on every seed** (−4.0 on the mean) and **does not change survival**: same median life, same causes of death, still only ~5% of night-time asleep.
- Seeds 1 and 2 fall back behind the stone-pickaxe wall (4.8% / 1.6%) that the rarity bonus had broken. One hypothesis: with γ = 0.99 the return sums many more future rewards, so the one-off bonus is a smaller share of the value, and the policy gradient is noisier.
- Caveat: λ = 0.65 keeps the advantage short-sighted whatever γ is. A complete horizon test would be γ = 0.99 with λ = 0.95. Low priority given this result.

**Observation from the videos** (`runs/report` section "Longer horizon")
- Agents walk right past diamonds. In the control's 901k-step checkpoint episode, it ends 4 tiles from a diamond and next to 2 iron ores, with only a wood pickaxe: it made a stone sword but no stone pickaxe.
- In a γ = 0.99 death video, a zombie kills it at night between two furnaces it built, with a diamond one tile away.
- This supports the memory / "remember where the diamond, iron and water are" idea, and the duplicate-furnace and sword-before-pickaxe complaints.

**Conclusion:** the night problem is not a horizon problem by itself. Next candidates:
1. Memory (GRU) in the PPO + rarity agent.
2. Measure how often a diamond/iron/water tile is in view and then lost (needs the semantic view logged in the diagnosis).
3. A tech-tree-aware bonus (pickaxe > sword).

---

## 2026-10-20 (afternoon) — AD + rarity β=2 + GRU memory: implementation, runs started

**Motivation:**
- Agents walk past diamonds and die of thirst near water they saw earlier.
- The AD network only sees the current frame plus the embedding of its last achievement. It has no memory of the world.
- The user's idea: "attention inside one run", i.e. remember where the diamond / water / iron is.

**Implementation** (`methods/05_achievement_distillation/recurrent.py`, `configs/ppo_ad_gru.yaml`; γ unchanged at 0.95)
- `PPOADGRUModel`:
  - `h_t = GRUCell(1024→512)(encode(obs_t), h_{t-1}·[same episode])`. The heads see [latents, AD state, h_t]. 11.61M parameters (AD: 9.23M).
  - AD's goal embeddings (`get_states`) and next-goal predictor stay encoder-only, so its contrastive losses are unchanged.
- `RecurrentRolloutStorage`:
  - Stores each step's input state (zeroed at episode starts).
  - The PPO loader yields whole 64-step chunks (64 chunks per rollout → 8 minibatches × 8 chunks = 512 samples, as in the paper) with the stored chunk-start state, for BPTT within the chunk.
- `RecurrentBuffer`: the aux buffer also keeps the per-step stored state (`old_hiddens`). `fast_aux.py` passes it to `forward()`, so pi/V distillation uses it as a fixed input.
- `test_recurrent.py` (real Crafter rollouts, 8×512 steps). All passed:
  - Chunk replay reproduces the rollout log-probs and values to <1e-5 (PPO ratio = 1).
  - The per-step aux inputs reproduce them too (12 complete trajectories).
  - The state is zero exactly at episode starts.
  - Zeroing the memory changes the policy.
- Memory: 3 parallel runs at a 30% VRAM cap went OOM in the aux phase (6.3–6.6 GB + fragmentation). The fix: `--aux_micro_batch 128` (gradient accumulation) plus `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. Peak is now 4.9 GB per run.
  - Aux micro-batching would update the value normaliser's EWMA 4× per step, so for the GRU runs β is set to β^(1/4) during the aux phase (`aux_norm_k`).
  - The cost: an aux phase takes ~390 s with 3 runs in parallel, instead of ~170 s alone.
- `--ckpt` option for `diagnose_deaths.py`. The spectator and the diagnosis carry the GRU state.

**Runs** (`scripts/queue_gru.sh`, started 13:48, sleep-inhibited, spectator videos every 15 min)
1. `ad-rarity2-gru-s0/1/2`: AD + rarity β=2 + GRU, in parallel. ~5 h estimated.
2. Then `ad-rarity2-s1/2`: the AD + rarity β=2 baseline without GRU, home flags (micro 256 / aux 128). The baseline currently has only seed 0 (23.08), so this gives a 3 vs 3 comparison.

**Results** (2026-10-20 19:10; `results/gru_summary.csv`, report section "Memory")

| config | seeds | score | per seed | eval median life | thirst deaths | water 0 at death | wake up (train) | iron pickaxe (train) |
|---|---|---|---|---|---|---|---|---|
| AD + rarity β=2 (home) | 1 | 23.08 | 23.08 | 240 | 18% | 50% | 70.1% | 0 episodes |
| AD + rarity β=2 + GRU | 3 | 20.28 ± 1.97 | 20.91 / 21.86 / 18.07 | 234 | 21% | 48% | 46.9 / 42.4 / 70.1% | ~4 / ~2 / 0 episodes |

- **Memory does not raise the score at 1M steps.**
  - The GRU runs learn more slowly and trail through most of training.
  - Final last-100 score: 30.8 / 32.4 / 29.9, against 34.3 for the baseline.
- **Survival is unchanged.** Same median life, same causes of death. Water is still empty in about half of all deaths, so the memory has not taught it to return to water within 1M steps.
- **Two of three seeds sleep much less.**
- **The best GRU seeds reach iron tools slightly more often.**
  - Iron pickaxes in training: GRU s0 ~4, s1 ~2. That is the most of any run, but iron tools stay at a handful of episodes everywhere (PPO runs had 1–2).
  - Highlight search (`diagnose_deaths.py --keep_if`, 800 fresh episodes, final GRU weights s0/s1): 3 iron pickaxes, 1 also with an iron sword, 0 diamonds.
  - In the best episode the agent makes both iron tools at step 403, with a diamond 4 tiles away and 1 health, water 0, energy 0. It dies of thirst and fatigue on the next step. The agents that get furthest die of neglected vitals, not missing skills.
- **Videos** (report): two iron-pickaxe highlights, the GRU s0 final episode, the s1 901k checkpoint, and an s2 "stone sword but no stone pickaxe" checkpoint. Also visible: duplicate furnaces (3) and 6 stone swords in one inventory.
- **Preliminary:** the baseline has 1 seed. `ad-rarity2-s1/s2` (home flags) started at 19:09 and should finish ~midnight; then update the table and text.
- **Next idea suggested by the videos:** a survival / vitals signal (keep water, food and energy up). The memory and skills alone are not the bottleneck for the best agents.

**Final (2026-10-20 22:45)**: the baseline seeds 1–2 finished, so this is now 3 vs 3.

| config | seeds | score | per seed | last-100 score at end | eval median life | thirst / water 0 | wake up | iron pickaxes train / 800 eval | iron swords train / 800 eval | collected iron in 800 eval |
|---|---|---|---|---|---|---|---|---|---|---|
| AD + rarity β=2 | 3 | **23.65 ± 1.05** | 23.08 / 24.87 / 23.01 | 34.3 / 32.9 / 32.5 | 242 | 18% / 45% | 70–74% | ~1 / 1 | ~3 / 1 | 204 |
| + GRU memory | 3 | 20.28 ± 1.97 | 20.91 / 21.86 / 18.07 | 30.8 / 32.4 / 29.9 | 234 | 21% / 48% | 47 / 42 / 70% | ~6 / 3 | ~1 / 1 | 206 |

- **The GRU is worse on every seed** (−2.2 / −3.0 / −4.9). It learns more slowly and does not change survival or how the agent dies.
- **Iron tools are equally rare.** The earlier "slightly more iron tools" impression came from comparing against one baseline seed; with 3 seeds and the same highlight search on the baseline, the counts are tiny and mixed.
- **Conclusion:** at 1M steps memory is not the bottleneck. The best agents die of neglected vitals.
- **Side result: the rarity bonus on AD holds over 3 seeds.** 23.65 ± 1.05 vs the paper's AD 21.79 ± 1.37 (10 seeds). Our own plain AD has only 1 seed (17.71).
- **Report updated:**
  - Lede and seeds table: AD + rarity now 3 seeds, plus γ = 0.99 and GRU rows.
  - GRU section: final text and table, plus the baseline's iron-pickaxe highlight for comparison.
  - Death section: now 24 runs, 4,800 episodes.

**Next candidate:** a survival / vitals signal. Keep water, food and energy up; that is what kills the agents that reach iron.

---

## 2026-10-20 (night) — Report: pooled tech-tree entries, zoomable learning curves

- **Tech tree:** two new entries pooled over 3 seeds each, "AD + rarity β=2 (3 seeds)" and "AD + rarity β=2 + GRU (3 seeds)" (`tree_extra` in narrative.json, `pooled_rates()` in make_report.py). Boxes below 1% also show the episode count.
  - Iron pickaxe: GRU 6 eps vs 1 without memory.
  - Iron sword: 1 vs 3.
  - Diamond: 0 in both.
- **Learning curves:** window buttons (all / 0–300k / 300k–700k / 700k–1M / last 120k). The y-axis re-fits to the lines in view; legend clicks toggle runs.

---

## 2026-10-21 (night) — Survival signal: AD + rarity β=2 + potential-based vitals shaping

**Motivation:**
- 54% of all 4,800 diagnosed deaths happen with food, water or energy at 0 (water alone ~45%).
- The furthest-reaching agents (iron pickaxe and iron sword, diamond 4 tiles away) die of thirst and fatigue.
- Agents sleep through only ~4% of the night.

**Method** (`VitalsShaping` in `methods/05_achievement_distillation/train.py`, flag `--vitals α`)
- Potential Φ(s) = α Σ_{food, drink, energy} (1 − ((9 − v)/9)²). Shaping F = γΦ(s') − Φ(s), with Φ = 0 after death (timeouts are not deaths).
- This is potential-based shaping (Ng, Harada & Russell 1999), so the optimal policy is unchanged. It pays most for topping up a nearly empty vital, costs a little as vitals run down, and costs Φ(s) on death. Logged rewards and the score stay the game's.
- Unit test passes: the discounted sum of F over an episode = −Φ(s₀) (−3.0 for α = 1), whatever the agent does.
- **Bug found by the smoke test:** upstream AD finds goal steps from rewards > 0.1 and asserts that this agrees with the success flags. The shaping breaks that, since drinking when thirsty gives +0.1 with no unlock.
  - Fix: with `--vitals`, `Buffer.get_goals` uses the success flags only (`algos.goals_from_successes`).
  - Verified identical to upstream on unshaped reward. Runs without `--vitals` keep the upstream code path.
- **Config:** α = 1, γ 0.95, rarity β = 2, AD paper config, micro 256 / aux 128 (same as the `ad-rarity2-s*` baseline, 23.65 ± 1.05).
- **Runs:** `ad-rarity2-vit1-s0/1/2` via `scripts/queue_vitals.sh`, started 00:51, ~4.5 h estimated, spectators every 15 min.

**Results** (2026-10-21 06:02; `results/vitals_summary.csv`, report section "Survival signal")

| config (3 seeds each, same settings) | score | per seed | eval median life | water 0 at death | thirst deaths | zombie deaths | drink (train) | stone pickaxe (train) | iron tools in 800 eval eps | collected iron in 800 eval eps |
|---|---|---|---|---|---|---|---|---|---|---|
| AD + rarity β=2 | **23.65 ± 1.05** | 23.08 / 24.87 / 23.01 | 242 | 45% | 18% | 46% | 54–57% | 25–31% | 2 | 204 |
| + GRU memory | 20.28 ± 1.97 | 20.91 / 21.86 / 18.07 | 234 | 48% | 21% | 48% | – | 10–27% | 4 | 206 |
| + survival signal α=1 | 17.80 ± 4.11 | 22.26 / 16.98 / 14.17 | 232 | **39%** | **15%** | 52% | **54–67%** | 5–21% | 0 | 130 |

- **The signal hits its own target.** The agent drinks more; there are fewer deaths with water at 0 and fewer thirst deaths.
- **But it doesn't live longer.** Zombie deaths rise, the tech tree is delayed, the score drops, and the seed spread grows.
- **Likely mechanism:** with γ = 0.95, γΦ' − Φ includes a −(1−γ)Φ ≈ −0.15 cost per step alive, while death costs ≈ −3. With GAE λ = 0.65 the advantage is short-sighted, so staying alive looks barely better than dying. Seeds 1–2 indeed had more, shorter episodes (5,013 / 5,214 vs ~4,600). The optimal policy is unchanged only in theory.
- **Next variant if we continue this line:** undiscounted Φ(s') − Φ(s) (no per-step drift), or a refill-only bonus. Possibly smaller α.
- **Videos:**
  - The seed-0 final agent stands next to a diamond at step 157 with 3 stone pickaxes and 3 wood swords but no iron pickaxe.
  - A 456-step, 18-achievement life ends in hunger and fatigue beside 2 furnaces and a lake.
  - Asleep in the open, killed by a zombie.
- **Report:** new section, table, 4 videos; tech tree gets a "survival (3 seeds)" entry; seeds table has a new row.

**Overall status of the improvement experiments** (AD + rarity β=2 = 23.65 ± 1.05 remains our best, above the paper's AD at 21.79):
- γ = 0.99: worse.
- GRU memory: worse.
- Survival shaping α=1: worse.
- **Common thread:** anything that adds learning signal or capacity without directly helping the tech tree slows progress within 1M steps.

---

## 2026-10-21 (morning) — Survival signal without the per-step cost

**Change:** `--vitals_gamma 1.0` uses F = Φ(s') − Φ(s) instead of γΦ(s') − Φ(s).
- No reward while vitals are unchanged (previously −(1−γ)Φ ≈ −0.15 per step alive).
- Drinking from empty: +0.21. Death: −Φ(s).
- The formal policy-invariance guarantee is lost under γ = 0.95. Relative to the invariant form, it adds (1−γ)Φ(s') per step, a small "comfortable and alive" bonus.
- Unit test: 0 when nothing changes; the episode sum is still −Φ(s₀).

**Runs:** `ad-rarity2-vit1u-s0/1/2`. Otherwise identical to `ad-rarity2-vit1` and the `ad-rarity2` baseline (α = 1, rarity β = 2, γ 0.95, micro 256 / aux 128). Started 09:09 via `scripts/queue_vitals.sh ad-rarity2-vit1u --vitals_gamma 1.0`, now generalised to take a prefix and extra flags.

**Report restructure (2026-10-21, requested by the user)**
- "Our change: reward rare achievements" moved to right after the header. It explains the idea every later section uses.
- Its outdated "next step: two more AD seeds" line was replaced with the 3-seed result.
- Two claims corrected:
  - Iron tools also appeared once in plain PPO seed 2, so not "only in bonus runs".
  - Eat plant is near zero, not zero.
- "What it took to run this on a 4 GB laptop" was renamed "Engineering notes: the first runs on a 4 GB laptop" and moved to the end. Its intro now says which 15 runs used the 4 GB laptop, and that later runs used the RTX 5090.
- Title "Crafter on 4 GB" → "Crafter RL report"; h1 → "Chasing the diamond in Crafter".
- The lede now mentions both machines and the three follow-ups. The date runs to 10-21.

**Results** (2026-10-21 14:22; `results/vitals_summary.csv`, report section "Survival signal without the per-step cost")

| config (3 seeds, same settings) | score | per seed | eval median life | alive at 300 | water 0 at death | thirst | lava | arrows | hunger/fatigue | zombies | stone pickaxe (train) | iron collected / iron tools in 800 eval eps |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| AD + rarity β=2 | **23.65 ± 1.05** | 23.08 / 24.87 / 23.01 | 242 | 21.5% | 45% | 18% | 10% | 20% | 7% | 46% | 25–31% | 204 / 2 |
| + survival, γΦ'−Φ | 17.80 ± 4.11 | 22.26 / 16.98 / 14.17 | 232 | 19.8% | 39% | 15% | 8% | 20% | 6% | 52% | 5–21% | 130 / 0 |
| **+ survival, Φ'−Φ (no drift)** | 22.07 ± 1.35 | 20.64 / 23.32 / 22.25 | **266** | **36.0%** | **30%** | 14% | **4%** | **13%** | 20% | 49% | 17–29% | **264** / 0 |

- **Removing the per-step cost fixes the first version.** +4.3 points, and the seeds agree again.
- **It is the first intervention that measurably lengthens life:**
  - median +10%;
  - +67% of agents alive at step 300;
  - training episodes 284 vs 252 steps.
- **Deaths shift from thirst, lava and arrows to hunger and fatigue,** which is consistent with living longer.
- **It collects iron most often of any configuration but crafts no iron tools.** An iron pickaxe needs a table + furnace nearby + wood + coal.
- **The longest evaluated life of any run:** 515 steps into the second night (s1, `death_hunger_1.mp4`).
- **Score is still 1.6 below the baseline,** because the tech tree is slightly slower.
- **Candidate building block:** survival without drift + something that turns extra time into iron tools (e.g. Plan C, a tech-tree subgoal planner).
- **Report:** new section, table, 4 videos; tech-tree entry "+ survival, no drift (3 seeds)"; seeds table row; lede updated.

---

## 2026-10-22 — Dashboard: runs grouped by configuration, in creation order

- `scripts/dashboard.py` `/api/runs` now returns each run's configuration (prefix of `<config>-s<seed>`), a description (`CONFIGS`), the seed, and its start time (first TensorBoard event file, else config.json mtime). Runs are sorted by configuration (first start) and then by seed.
- `scripts/dashboard.html`:
  - The run dropdown uses numbered `<optgroup>`s (1. AD paper config … 12. survival no-drift). Each option shows seed · final score · start date.
  - The score / recent-score charts show the selected configuration's seeds, with a "Compare: all runs" switch. Previously they drew all 30 runs.
- **Status of the improvement search:** AD + rarity β=2 (23.65 ± 1.05) is still the best. γ 0.99, GRU, and both survival variants are below it. Only the no-drift survival signal improves survival.
- **Proposed next direction:** our own world-model agent with rarity-guided imagination. Stage 1 is a "dream vs reality" world model without RL. A design note comes first, for the user to challenge.
- Design note for the user to review: `paper/designs/world_model_stage1.md` (Stage 1 "dream vs reality": VQ tokenizer 8×8, Transformer dynamics with a separate inventory head, rarity-weighted training as the testable twist, probes and pre-registered predictions P1–P5, bar for Stage 2). No code yet.

---

## 2026-10-22 — World model Stage 1: build and launch

- **Decisions:**
  - 8×8 tokens and 64-step context (user).
  - Rare = unlocked in <30% of episodes, weight 5×. The 10% first pick only selected events with 1–2 examples.
  - The user decides go/no-go for Stage 2 after the probes and videos.
- **Code:** `methods/08_world_model/` (record, tokenizer, cache_tokens, dynamics, evaluate, run_stage1.sh, README).
- **Recording:** ~3,940 episodes, mostly final-weight agents.
  - In the training set, stone pickaxe appears in 74.5% of episodes and iron in 27.3%; iron pickaxe and diamond in 0.1% each, so there is at least one diamond in the data.
- **Tokenizer smoke test** (1k steps, MSE loss): all 512 codes used, test PSNR 25.5. Terrain is fine, but small sprites (cows, coal, furnaces, digits) blurred away, so the loss was switched to L1 for the real run (20k steps, ~85 min at ~1,000 frames/s).
- **Dynamics plumbing test** (provisional tokenizer, 300 steps): all losses fall on held-out data. 25.6M params, ~230 steps/min.
- **Running:** `run_stage1.sh` (sleep-inhibited): recording → tokenizer → token cache → uniform + rarity dynamics in parallel (30k steps each). ~6 h in total.
- **Delay (my mistake):** `run_stage1.sh` waited with `pgrep -f "record.py --envs"`, which also matched the shell that had created the script, so it never started the tokenizer. Recording itself finished fine: 3,940 episodes, 880,879 train / 48,970 test frames. The wait now matches only the interpreter (`^.venv/bin/python [^ ]*record\.py`). Relaunched 00:26, ~1.5 h lost.
- **Tokenizer** (20k steps, L1, 76 min): test PSNR 28.7, 511/512 codes used, perplexity ~300 (`results/world_model/tokenizer_reconstructions.png`).
  - Kept: terrain, trees, tables, furnaces, coal, water, sand.
  - Still lost: cows, some zombies, saplings.
  - Some inventory digits are wrong (health 2 → 7), which supports predicting inventory as a separate output.
- **Dataset fact:** exactly **one diamond in 3,940 episodes** (`ad-rarity2-s0_agent-e250/ep_00304`). The best agent crafts an iron pickaxe at step 202 and mines a diamond at step 207, at night, with 18 achievements, then dies at step 404. Video: `results/world_model/diamond_episode.mp4`.
- **Dynamics started 01:47.** Train 3,548 / test 392 episodes. Rare set (<30%): collect_iron, iron pickaxe, iron sword, diamond, eat_plant. Rarity run: 21% of sampled windows contain a rare unlock.
- **Evaluation dry run** on the uniform model at 10k/30k steps (20 dreams; not results, a bug check). It found an unfair baseline:
  - Pixel PSNR was compared against copying the *real* last frame (31.3 dB), but no dream can beat the tokenizer ceiling (~28.7 dB).
  - Now reported: `psnr_copy` = last frame through the tokenizer, `psnr_copy_raw`, and `psnr_ceiling` = the real future frame through the tokenizer.
- **Early signals:**
  - Crafting unlocks are predicted next-step with 95–100% recall.
  - Creature events are weak (zombie 46%, cow 58%), matching what the tokenizer drops.
  - Water memory 5% (the detector agrees with the game's labels 95% of the time). Night correlation 0.40.
  - Dreams follow reality for ~5–15 steps, then drift.
- **Videos:** labelled REALITY / DREAM / DIFFERENCE, with steps ahead and real vs model inventory (orange when wrong).
- **Dynamics training done** (05:55, 247 min each, run in parallel).
  - Final test token CE: 0.876 (uniform) and 0.905 (rarity).
  - **Overfitting in the small heads:** test inventory loss is best at 7–9k steps (0.045–0.046) and ends at 0.054. Test unlock loss goes 0.012 → 0.018.
  - The frame predictor's test loss was still improving at 30k.
  - Only final weights were saved. Next time: early stopping / separate regularisation for the heads.
- **Stage 1 results** (06:00–06:30): see the Results section of `paper/designs/world_model_stage1.md` and the report section.
  - The model beats "nothing changes" at every horizon. It is good for ~5 steps (items exact 91% vs 54%), drifts by 15 (56%), and loses the scene by 50.
  - Crafting is near perfect one step ahead. Creatures are weak, off-screen memory fails (15%), and night tracking is weak (0.42).
  - Rarity weighting does not help recall.
  - Bar not met.
  - Added the `items_exact` metric (12 material/tool slots), because vitals change on hidden timers. The report is 13.3 MB, near the 16 MB limit.
- **Stage 1b plan** for the user to challenge: `paper/designs/world_model_stage1b.md`.
  - Five fixes: hard-pixel tokenizer loss (semantic-weighted only as a diagnostic), head re-fit with early stopping, skip-ahead memory objective, 8-step unrolled training, episode-clock input.
  - Predictions P6–P10, the same evaluation, and a no-memory ablation.
  - Open questions Q5–Q7. No code yet.

---

## 2026-10-25 — World model Stage 1b: implementation and launch (autonomous; the user left Q5–Q7 open)

- **Decisions:** see the Decisions section of `paper/designs/world_model_stage1b.md`. Semantic map for diagnostics only; clock yes; no extra ablation.
- **Recording plan B** (`record.py --plan b`, 3,990 episodes, new seeds, agents not used before; folders `<source>_b`).
- **Code:**
  - `data.py`: source filter (`stage1` / `b`), plus `fit` / `val` splits. The Stage 1 test set is still 392 episodes.
  - `tokenizer.py`: `--loss hard` (top-10% pixels ×5) or `sem` (creature tiles ×10, diagnostic). Creature- and resource-tile PSNR. `tile_mask` (7×7 px per tile, world view in rows 0–48), checked visually on cows.
  - `dynamics.py`: `--clock` (episode step mod 300), `--skip` (grid k ∈ {4,8,16,32} ahead from h_t + mean action embedding), `--unroll 0.25` (last 8 inputs from own S=1 predictions, no grad), `--refit` (re-init heads on the frozen model, early stopping on val, patience 1000). Stage 1 models still load.
    - Smoke test: all losses fall; 26.4M params; ~135 steps/min (+70% cost).
  - `evaluate.py`: loads each model with its own tokenizer and flags; fixed Stage 1 test set and starts; creature-tile PSNR inside dreams (Stage 1 uniform: ~14.7 dB at 1 step, copy 14.2); skip-ahead water probe.
- **Running:** `run_stage1b.sh` (sleep-inhibited). It waits for plan B, then:
  1. tokenizer_v2 (25k) + tokenizer_sem (12k) in parallel;
  2. tokens_v2;
  3. v2 + v2_nomem dynamics (30k each, in parallel);
  4. head re-fit;
  5. evaluation (stage1b_eval.json, videos).
  - ~10 h in total.
- **Tokenizer comparison on the same 4,096 test frames** (Stage 1 tokenizer measured directly; v2 and sem at partial training):
  - Stage 1 tokenizer: overall 28.94, creature tiles 15.67, resource tiles 23.03.
  - v2 (hard-pixel) at 5k/25k: 30.8 / 18.1 / 22.9.
  - sem (diagnostic) at 4.5k/12k: 25.3 / 22.7 / 18.0. Strong creature weighting trades off everything else.
- **Tokenizers final** (same 4,096 test frames):
  - Stage 1: 28.94 overall / 15.67 creature tiles / 23.03 resource tiles.
  - **v2 (hard-pixel, 25k): 34.84 / 19.50 / 27.17.**
  - sem (diagnostic, 12k): 28.17 / 24.27 / 22.40.
  - v2 visibly restores cows, iron ore and night zombies (`results/world_model/tokenizers_compared.png`). sem keeps creatures but loses iron ore.
  - P6, tokenizer part (≥3 dB on creatures): met (+3.8 dB).
- **tokens_v2 cached** 14:04. v2 and v2_nomem dynamics started.
- **Early check, v2 at 10k/30k** (60 dreams, no head re-fit):
  - One-step unlock recall: zombie 64% (Stage 1: 35%), cow 76% (40%), iron 86% (39%), all 92% (81%).
  - Water memory via the skip-ahead output: 34% (dream 13%, Stage 1 7% on the same cases).
  - Items exact at 15 steps: 53%, vs 67% for Stage 1 on this sample. Watching.
- **Metric fix:** `evaluate.psnr` averaged per-frame PSNR, so perfectly reconstructed frames (~138 dB) inflated the mean. The v2 "ceiling" read 44 dB, impossible given its 34.8 dB test score. It now uses the PSNR of the mean squared error. The final evaluation re-scores Stage 1 the same way.
- **Stage 1b results** (20:29): see the Results section of `paper/designs/world_model_stage1b.md` and the report section "World model Stage 1b".
  - Fixes 1, 2 and 4 helped:
    - creatures 15.7 → 19.5 dB;
    - iron recall 39 → 92%, cow 40 → 74%, zombie 35 → 54%;
    - items @5 93%, @15 62%;
    - inventory loss 0.054 → 0.028;
    - night 0.42 → 0.66 with the clock.
  - The memory signal had no effect on dreams (17% with and without); the skip-ahead output alone gets 34%.
  - Bar not met (memory). Report 13.8 MB.
- **Anchored dreaming** (evaluation only; `evaluate.py --anchor 8`: every 8th dreamed frame up to 32 steps is replaced by the skip-ahead prediction from the last real state). **Negative result:**
  - water memory 17% → 21%;
  - items exact @15: 62% → 51%, because the inserted frame disagrees with the running inventory;
  - tokens @15: 42% → 39%.
  - Videos in `runs/wm/eval/v2_model_refit_anchor8/`.
- **Longer training launched** (`run_stage1b_long.sh`, 21:05): v2_long continues v2 from 30k for another 30k steps, lr 1e-4, skip weight 1.0. Then re-fit and evaluation.

---

## 2026-10-25 (evening) — World model Stage 1c (diagnostic): does knowing its position give the model memory?

- **Hypothesis:** off-screen memory fails because the model must infer from pixels how far the agent walked.
- **Positions without re-recording** (`derive_positions.py`): a move action either moves one tile or is blocked. Comparing consecutive 9×7 tile maps (materials only) under both hypotheses resolves 98.7% of moves; the 1.3% ambiguous ones count as moved.
  - Consistency check: when a world tile is seen again, its material matches in 99.9% of 553k revisits.
  - Positions for all 7,930 episodes in `runs/wm/pos/`.
- **`dynamics.py --pos`:**
  - Position relative to the window start (±32 tiles) as an input.
  - Odometry output (stay / ±x / ±y) trained on the derived moves.
  - Dreams and unrolled training update the position from the model's own odometry; the real context uses the derived path.
  - `--init` now loads with strict=False, so the new parts start fresh. Diagnostic only: the positions come from the game's tile maps.
- **Runs (in parallel):**
  - `v2_long`: v2 + 30k steps, lr 1e-4, skip weight 1.0.
  - `v2_pos`: init v2, `--pos`, 20k steps, lr 2e-4, skip weight 1.0.
  - Then head re-fit and a joint evaluation (`stage1c_eval.json`).
- Note: DST ended today, so log timestamps switch from +03 to +02.
- **Early Stage 1c check** (v2_pos at 10k/20k, 80 dreams, no re-fit):
  - Skip-ahead water recall 43% (v2: 34%).
  - Dream water 15% (unchanged).
  - Items @5/15: 90% / 60%. Night 0.53. Unlock recall 91%.
  - **Reading:** position helps the direct memory output, but the dream itself still doesn't redraw remembered terrain.
  - **Likely architectural reasons:**
    1. The temporal model only sees one compressed vector per past frame.
    2. 8 px tokens don't align with Crafter's 7 px tiles, so a one-tile move isn't a whole-token shift and remembered patches can't be copied.
  - **Stage 1d idea:** tile-aligned tokens (one per 7×7 tile, 9×9 per frame) plus an explicit top-down map memory written at world positions and read back when predicting.
- Design note for review: `paper/designs/world_model_stage1d.md` (tile-aligned tokens + top-down map memory, predictions P11–P14, open questions Q8–Q10). No code yet.
- **Stage 1c + long-training results** (02:51; `results/world_model/stage1c_eval.json`, report section "World model Stage 1c"):
  - Dream water: v2 17% · v2_long 23% · v2_pos 21%.
  - Skip-ahead water: 34% · 39% · 38%.
  - Items exact @5/15/50: v2 93/62/15.5 · long 93.5/57/15 · pos 94.5/62/19.5.
  - Unlock precision: 73 · 73 · 56%. The v2_pos re-fit landed worse.
  - **Conclusion:** position is not the main bottleneck (≈ the gain from 2× training). This supports the compressed-frame / tile-misalignment explanation, i.e. Stage 1d.

---

## 2026-10-26 (night) — World model Stage 1d: tile tokens and a map memory

- **Tile tokenizer** (`tile_tokenizer.py`): 63×63 crop cut into 9×9 tiles of 7×7 px (rows 0–6 world, 7–8 inventory). Tiling round-trips exactly. Per-tile MLP, EMA VQ, hard-pixel L1. 40k steps take **4 minutes**.
  - Same 4,096 test frames:
    - 1,024 codes: **36.24 dB overall, 31.79 creature tiles, 34.58 resource tiles**.
    - 2,048 codes: 38.06 / 33.68 / 34.87.
    - v2 (8×8): 34.84 / 19.50 / 27.17.
  - Chosen: 1,024 codes (`runs/wm/tokenizer_tile`). Creature tiles +12 dB over v2. **P13 met.** Reconstructions are visually near-perfect except night speckle.
- **Tile tokens cached** (`tokens_tile`, 9×9 per frame).
- **Map cache** (`build_maps.py`): for every step, the 7×9 map window at the next position, built from the whole episode so far: last code seen there and age, or unseen. Uses the Stage 1c paths.
  - The map already knows 97.9% of the next frame's world tiles.
  - **Alignment check** (daytime): after a move, map code = real next code in 66.8% of tiles, vs 41.2% if deliberately shifted the wrong way. Alignment is correct.
  - **Code jitter:** in daytime with no move, consecutive frames keep the same code for only 74.7% of tiles (27.1% at night). Crafter changes lighting slightly every step and adds noise at night, and 1,024 codes include near-duplicates. So the map is fed to the model as **codebook vectors** (lighting-robust) plus an age embedding, not as code IDs.
- **Code:**
  - `dynamics.py`: any grid/codebook (tile 9×9 / 1,024 codes, or 8×8 / 512). `--map` adds the map window: codebook vectors through a linear layer, an "unseen" vector, and 11 age buckets, added to the decoder's first 63 (world) positions.
  - `build()` reads grid/codebook from the tokenizer. Old models load unchanged.
  - `evaluate.py`: the dream builds the map from the whole real history, then each step moves by its own odometry, reads the map there, predicts, and writes the predicted frame back.
  - Smoke test: v3 test token loss 2.1 after 100 steps (v2: ~4.2), ~200 steps/min.
- **Launched** `run_stage1d.sh` (sleep-inhibited): v3 (tile + pos + clock + map) and v3_nomap (same without the map), 30k each in parallel. Then re-fit, then a joint evaluation with v2_long (`stage1d_eval.json`).
- Stage 1d at 5k steps (04:10): test token CE v3 (map) 0.454 vs v3_nomap 0.706 (same tile tokens); odometry test CE 0.0003 (moves are exact one-tile shifts with tile tokens). ~94 steps/min each in parallel -> done ~09:20.
- **Early check at 10k steps** (`results/world_model/early_v3_10k.json`, 80 dreams, same starts):

  | | map (v3) | no map (v3_nomap) | best before (v2_long, 30k) |
  |---|---|---|---|
  | dream water memory | **56%** | 10% | 23% |
  | items exact @5/15/50 | 96/**79**/28% | 89/63/14% | 93.5/57/15% |
  | PSNR @5/15/50 | 24.0/19.6/15.9 | 20.1/16.7/14.3 | |
  | night brightness corr. | 0.65 | 0.39 | |

  - The video (`runs/wm/eval/v3_model_10k/dream_00.mp4`) shows the agent walking back to a lake and table at step 50: the map model redraws both, while the no-map model invents open grass. At step 100 (night, near stone) the map model keeps the dark scene; the no-map model turns it to daylight grass.
  - P11 (≥60%) is nearly met at a third of training. P12 (≥70% @15) is already met.
  - A ghost cow appears at step 1 (P14 risk), to be checked on the final model.
  - **Train and test loss are close** (one-batch train estimates 0.36–0.43 vs test 0.43 at 12–13k), so the model is not clearly data-limited yet. Recording more episodes is lower priority than longer training or more capacity. To revisit when the curve flattens.

### Positions from pixels only (pixel_odometry.py)
- **Why:** the map needs the agent's position. Until now it came from the recorded semantic tile maps (game internals). An agent playing from pixels would not have that.
- **Method:**
  - For a move action, compare frame t+1 with frame t shifted one tile (moved) vs unshifted (blocked), using a trimmed mean of per-tile pixel error.
  - Ties (uniform grass, forest, water) go to a **self-learned walkability table**: for each tile code, how often a clear move onto it succeeded, learned from the data itself with no labels.
  - **Sleep:** Crafter renders sleep as a grey view and ignores actions until energy is full. A grey frame therefore means no move, except on the wake-up step, which is compared in luminance after undoing the sleep tint.
- **Checked against the game's true player position** (80 new random-policy episodes, ~14k steps, 19% at night, 13% asleep):

  | | step correct | night | position still exact |
  |---|---|---|---|
  | semantic tile maps (old) | 99.71% | 99.6% | 94.1% |
  | pixels, first version | 94.7% | | 49% |
  | + walkability table | 99.6% | | 76% |
  | + sleep handling (final) | **99.77%** | 98.9% | **99.2%** |

  - Dead ends: weighting tiles towards the noise-free centre of the night vignette (worse); a noise-relative tie threshold (worse at night).
  - The old semantic path has the same sleep blind spot.
- **On the recorded agent data:** 99.85% of steps agree with the old path. Map material alignment (semantic labels, used only as a check): **99.75%** of seen tiles correct (old 99.76%). For tiles last seen 20+ steps ago: 99.52% vs 99.80%.
- So the map memory can run on pixels alone at essentially no cost. Written to `runs/wm/pos_pix`, maps to `runs/wm/maps_tile_pix` (`build_maps.py --pos pos_pix --out maps_tile_pix`).
- **Side finding:** tile *codes* of a tile seen 20+ steps ago match the code shown now only 8% of the time in daytime, even though the material matches 99.8%. Slow lighting changes re-pick near-duplicate codes. The model has to translate old codes to the current lighting. A lighting-invariant map is a candidate for later.

### Stage 1e queued (`run_stage1e.sh`, starts when 1d is evaluated)
- v4 = map + pixel positions + unrolled training (25% of windows end with 8 self-fed steps; `self_feed` now reads the map window).
- v4_nou = the same without unrolling.
- Also v3 re-evaluated with pixel positions instead of semantic ones (no retraining).
- Output: `stage1e_eval.json`, `stage1e_v3pix.json`.

### Semantic grader for dreams (`semantic_eval.py`), for P14 (ghost creatures)
- Every dreamed frame is decoded, re-encoded with the tile tokenizer, and each world tile classified through a table code → most common semantic class, built from real training frames. Labels are used only for grading.
- **Grader on real frames:** terrain 99.5%, creature recall 89–97%, creature precision 100%. It is trustworthy.
- **At 10k steps** (40 dreams; terrain accuracy @1/5/15/50; ghost creatures per frame):
  - map: 96/94/79/57%, 0.25/0.38/0.40/0.23 ghosts.
  - no map: 94/88/69/38%, 0.38/2.4/3.9/6.4 ghosts.
  - "Nothing changes" copy: 88/73/57/38%.
- **P14 runs the other way:** the map does not create ghost creatures; without a map the model hallucinates more and more of them.
- Creature *positions* are predicted poorly beyond one step by both (recall 47% @1, ~5% @5). Cows walk randomly, so this is largely unpredictable.
- Queued for the final models (`run_semantic.sh` → `stage1d_semantic.json`, `stage1e_semantic.json`).

### Stage 1d final results (08:52)
- Details in `paper/designs/world_model_stage1d.md` (Results). Short version (map vs no map vs v2_long):
  - Dream water memory **60%** vs 22.5% vs 23%: P11 met.
  - Items @50: 31% vs 18% vs 15%. Terrain @50: 65% vs 58% vs 54%.
  - Ghosts @50: 0.26 vs 0.65 per frame: P14 refuted.
  - Unlock precision drops to 59%.
- Report: new section "World model Stage 1d: a map the model can read". It has a table and a stacked comparison video (`media/wm/videos/dream1d_03_compare.mp4`, 0.35 MB). Report now 14.46 MB.
- Videos sent: v3/v3_nomap dream_03 and v3 dream_00.
- Stage 1e started at 08:52 (v4, v4_nou on pixel positions).

### EMERALD code review (repo pulled to ~/Desktop/EMERALD, burchim/EMERALD @665948d)
- **License:** CC BY-NC-SA 4.0. Copying code would force the same license and attribution on our repo. Plan: re-implement ideas (most are DreamerV3's) in our own code and cite.
- **Budget:** the repo's defaults run 50 epochs × 12,500 training steps, each followed by one step of 16 envs, so about 10M env steps. `media/figure.png` confirms the x-axis goes to 10M.
  - Their reported per-achievement rates are at 10M: iron pickaxe 32%, diamond 0.9%, score ≈ 55%.
  - At 1M (first point of their curves) the score is ≈ 3% and iron is ≈ 0.
  - So EMERALD's strength is at 10M, not the 1M Crafter benchmark we target.
- **World model:**
  - CNN encoder to a 4×4 grid of 32×32 categorical latents (Dreamer-style KL losses, free nats 1.0).
  - Causal transformer over 64 steps with a KV cache.
  - Spatial MaskGIT (3 decoding steps) for the next latent.
  - Reward and continue heads.
  - No map or long-term memory. Context is 64 steps, the same limit Stage 1–1c had.
- **Agent:** actor-critic purely in imagination, as in DreamerV3:
  - Horizon 15 from every 4th real state; λ-returns (λ 0.95, γ 0.997).
  - Return normalisation by an EMA of the 5–95% percentiles; critic regularised towards a slow EMA copy; entropy 3e-4.
  - Losses weighted by the predicted continue probability.
  - World model and agent are trained online, interleaved with play.
- **Takeaways:** our world model is already more specialised (tile tokens + map memory). EMERALD's useful part for us is the learning-in-imagination machinery for Stage 2, plus the KV cache for fast dreaming.

### Stage 1e at 10k steps (11:14; `early_v4_10k.json`, same 80 dreams as `early_v3_10k.json`)

| | water | items @5/15/50 | PSNR @15 | night |
|---|---|---|---|---|
| v3 (map, semantic positions) | 56% | 96/79/28% | 19.6 | 0.65 |
| v4_nou (map, pixel positions) | 57.5% | 99/78/31% | 19.3 | 0.63 |
| v4 (map, pixel positions + unroll) | 55% | 97.5/76/30% | 20.3 | 0.65 |

- **Pixel-only positions cost nothing**: v4_nou ≈ v3 on every measure. The map model no longer needs game internals.
- **Unrolled training:** no clear effect yet (PSNR @15 +1 dB, the rest within noise of 80 dreams). The final 200-dream evaluation decides.

---

## 2026-10-26 — The agent gets its own map (AD + rarity β=2 + pixel map)

**Idea:** the user's in-run memory idea ("remember where the water or the diamond was"), done the honest way. The world-model work showed that a top-down map built from pixels alone is accurate. Here the RL agent itself gets that map as a second input image.

**Map** (`common/pixel_map.py`, no game internals; runs inside each env worker):
- **Odometry** as in `pixel_odometry.py`: tile-shift comparison, grey-screen sleep handling, wake-up step in luminance.
- The **walkability table is learned online** from the env's own clear moves. Key: the mean colour of the tile stepped onto, quantised to 16 levels; the world model's version used tokenizer codes, which an online agent does not have.
- **Memory:** every world tile in view is stored at its world position as a 2×2-px thumbnail (not while asleep).
- **Input:** an egocentric 32×32-tile window (half the world), 64×64×3, never-seen tiles black, the agent at the centre.
- **Check:** at 2×2 px per tile, material colours stay apart; the closest pair is stone/iron, mean difference 11 levels; coal/iron 27.
- **Against the true position** (120 random-policy episodes, one map kept across episodes as in training): 99.0–99.4% of steps right; exact position kept 93–94% of the time after a 40-episode warm-up.
  - Slightly below the offline tokenizer-keyed version (99.2%). Alternative keys (2×2 thumbnail, 4×4 sample) were no better.
  - After an odometry slip the map stays locally consistent, because new frames overwrite the area around the agent.
- Cost: ~120 µs per step.

**Model** (`methods/05_achievement_distillation/map_model.py`, `PPOADMapModel`):
- Observation = 6 channels (frame, map).
- The frame goes through the original IMPALA encoder; the map through a smaller one (channels 32/64/64, also 256 out).
- Both feature vectors are concatenated before the latent layer, so policy, value and AD's goal states see both.
- 10.94M parameters (AD: ~9.5M).

**Plumbing:**
- `CrafterVecEnv(pixel_map=True)`.
- `train.py --pixel_map 1`.
- `record_episode` and `spectator.py` support the map model.
- Training clips and evaluation videos show the map to the right of the game, with the agent's tile outlined. The outline is for viewers only, not the agent's input.
- CPU smoke test passed (1 epoch, clips and final video OK).

**Queued:** `scripts/run_map_after_wm.sh` → `scripts/queue_map.sh`: `ad-rarity2-map-s{0,1,2}`, 1M steps, same flags as the AD + rarity β=2 baseline, plus spectators. Starts when Stage 1e's semantic evaluation has finished.

### Stage 1e final results (15:24; `stage1e_eval.json`, `stage1e_v3pix.json`, `stage1e_semantic.json`)
- **v3 with pixel positions at evaluation only:** identical to v3 (water 60%, items 97/72.5/30.5%).
- **v4_nou (trained on pixel positions):**
  - water 58.5%, items @5/15/50 97/75.5/32%;
  - terrain @50 65.5%, ghosts 0.27;
  - unlock P/R 59/92%.
- **v4 (+ unrolled training):**
  - water 56%, items 97/76/33.5%;
  - terrain @50 65.6%, ghosts 0.26;
  - unlock P/R **70**/89%.
- **Conclusion:** the map memory runs on pixels alone at no cost. Unrolling gives no real gain in dreams; the unlock precision difference likely comes from the re-fit.
- Report: section "World model Stage 1e: the map from pixels alone" (table only, no video).
- 17:31: map runs at ~300k steps. Score 7.58 / 7.63 / 9.99 vs the AD + rarity β=2 baseline 8.25 / 10.26 / 8.81 at the same step: on par. Episode length 218/196/229 vs 183/214/201. The spectator video at 287k shows the map filling in correctly (lake, stone with lava, night regions); sent to the user.

### Map agent results (2026-10-26, 23:16; `results/map_summary.csv`, report section "The agent gets its own map")
- **Score:** 22.31 ± 1.00 (21.46, 22.06, 23.42) vs AD + rarity β=2 23.65 ± 1.05. About 7.8 h per run.
- **Training success rates (mean of 3 seeds) vs baseline:**
  - drink 62.0 vs 55.7%; iron 3.7 vs 7.4%;
  - stone pickaxe 21.3 vs 28.0%; furnace 38.9 vs 37.1%.
  - Iron pickaxe 0.01% (one episode); iron sword 0.03%; diamond 0.
- **Death diagnosis** (600 fresh episodes, final weights) vs baseline:
  - median life 248 vs 242; alive at 300: 27.5 vs 21.5%;
  - zombie deaths while awake 39 vs 44%; **thirst deaths 24 vs 18%**; drink at 0 at death 43.8 vs 44.8%.
- **Blank-map ablation** (same 600 seeds, map input zeroed):
  - achievements per episode 12.73 vs 14.00; stone pickaxe 67 vs 72%;
  - median life 237 vs 246.
  - So the agent does use the map. Causes of death were "unknown" in this run (no damage hooks), so only lengths and achievements count.
- **Water memory check** (600 fresh episodes, own loop; water = blue thumbnails in the map): 282 died with drink 0.
  - 17 had never seen water; 104 had water in the map but never drank; 161 had drunk earlier and did not return.
  - Water was seen in 95% of all episodes (median first sighting at step 25).
  - **So memory is not the bottleneck: the agent has the information but no incentive to act on it in time** (γ = 0.95, delayed thirst damage).
- **Correction:** from two thirst-death videos I first concluded the agent "never saw water". The 600-episode check shows that is the rare case (17/282).
- In `death_thirst_1.mp4` (seed 0) the agent stands next to a diamond at step 179 with only stone tools.
- **Next candidate:** combine the map with the no-drift survival signal, the only change that cut thirst deaths (18 → 14%).
- `diagnose_deaths.py` and `analyze_deaths.py` support the map model. Report now 15.50 MB, close to the 16 MB limit.

### Map + no-drift survival signal (started 2026-10-27)
- **Why:** the map agent has seen water in 95% of its games but still dies of thirst (24% of deaths): it knows where water is but has no reason to go back in time. The no-drift survival signal (Φ'−Φ on the vitals, α = 1, γ_vitals = 1) is the only change so far that cut thirst deaths (18 → 14%). Together, the map supplies *where* and the signal supplies *why now*.
- **Prediction (to check):** thirst deaths below both parents (< 14%), more lives past step 300 than either; score ≥ the map agent's 22.31. Refuted if thirst deaths are not lower than with the survival signal alone.
- **Runs:** `ad-rarity2-map-vit1u-s{0,1,2}`, 1M steps, via `scripts/queue_vitals.sh ad-rarity2-map-vit1u --vitals_gamma 1.0 --pixel_map 1` (same flags as both parents: rarity β = 2, γ 0.95, micro 256 / aux 128), with spectators. Labels added to `compare_configs.py`, `analyze_deaths.py`, `dashboard.py`.
- 12:49, 0.25M steps (cumulative score, mean of 3 seeds): map + survival 6.16 (5.96 / 6.76 / 5.75) vs baseline 7.99, map 7.25, survival no-drift 6.44. Episode length 199/201/186 vs baseline 227/173/187. A little behind but within early-run noise (the survival-only runs were also behind here and ended at 22.07). Trainer ETA ~5 h.
- 14:50, 0.5M steps (cumulative score, mean of 3 seeds): map + survival 11.40 (10.51 / 11.70 / 12.00) vs baseline 13.95, map 13.16, survival no-drift 12.55. But the **longest lives of all four**: mean episode length 273/259/243 vs baseline 230/234/208 and map 233/236/222. Recent-100 score 17.8 / 19.8 / 24.0, so the cumulative score is still catching up. ETA ~3.6 h.

### Map + survival results (2026-10-27, 18:54; `results/map_vit1u_summary.csv`, report section "Map + survival signal together")
- **Score:** 20.53 ± 0.46 (20.00, 20.78, 20.81), the lowest of the four: baseline 23.65, map 22.31, survival no-drift 22.07. ~8 h per run.
- **Training success rates (mean of 3 seeds)** vs baseline / map / survival:
  - drink 67.9 vs 55.7 / 62.0 / 65.3%; wake up 75.9 vs 71.7%;
  - stone pickaxe **15.0** vs 28.0 / 21.3 / 22.7%; iron **1.9** vs 7.4 / 3.7 / 5.2%; eat cow 50.8 vs 58.4%; defeat zombie 40.9 vs 59.7%.
  - Iron pickaxe 0.02%; **one diamond** (seed 0, training episode 4353, 252 steps, 18 achievements incl. iron pickaxe; ended at env step 1,001,136, i.e. in the last epoch, which like every run overshoots 1M by up to 24k steps). Not filmed.
- **Death diagnosis** (600 fresh episodes, final weights) vs map / survival no-drift:
  - median life 256 vs 248 / 266; alive at 300: 35.2 vs 27.5 / 36.0%;
  - thirst 17 vs 24 / 14%; hunger/fatigue 12 vs 6 / 20%; zombie awake 42 vs 39 / 49%; **zombie while asleep 6%** (highest of any config); drink 0 at death 31.7 vs 43.8 / 29.5%.
- **Both predictions refuted:** thirst deaths are not below the survival signal alone (17 vs 14%), and the score is below the map agent's.
- **Reading:** each addition trades tech-tree progress for safety, and the trades add up. Longer lives buy fewer tools, not more. Neither memory nor a survival incentive is what blocks the iron → diamond chain; the agent reaches iron less often the safer it plays.
- Report: section with table and 2 videos (`death_zombie_asleep_1.mp4`: iron at step 205, killed asleep at 210; `death_thirst_1.mp4`: water on the map from the start, dies of thirst at 261), re-encoded at CRF 37. Tech-tree entry "+ own map + survival, no drift (3 seeds)". **Report 15.95 MB: at the 16 MB limit**; the next section needs space freed (e.g. re-encode older videos).

## 2026-10-28 — Is the dream an easier world? Zombies and damage in dreams vs reality

- **Why:** the user asked whether the world model "just reduces zombies and makes it easier". A Stage 2 agent trained in a safer dream would learn to ignore the night.
- **Code:**
  - `dynamics.py predict_next(..., temperature=0.0)`: 0 = greedy (unchanged default), > 0 = sample tokens and inventory from the softmax.
  - `evaluate.SAMPLE_T` passes it to `dream()`.
  - `creature_eval.py` (new): 600 dreams × 50 steps from held-out moments, starts stratified half day / half night (real daylight < 0.3), the same for every model and decoding. Per dream step: zombies / cows / skeletons in view (semantic_eval grader on dreamed frames; reality counted both from true labels and through the same grader), zombies within reach (Manhattan distance 1 = where Crafter zombies hit), walk-ins (no zombie at the start, one appears later), health lost (start health minus the inventory output), terrain accuracy.
  - `creature_report.py` (new): report table + inline SVG. 16 min on the RTX 5090 for 3 models × 3 temperatures.
- **Results** (`results/world_model/creature_eval.json`; map model v4; reality = real frames through the grader):

  | | walk-ins | zombies/frame night @15 / @50 | within reach night @50 | health lost @50 day / night | terrain @15 / @50 |
  |---|---|---|---|---|---|
  | reality | 44% | 0.26 / 0.51 | 0.15 | 1.30 / 2.09 | – |
  | greedy | **5%** | 0.35 / 0.61 | 0.13 | **0.39 / 0.45** | 84 / 63% |
  | sampled T = 0.7 | 14% | 0.32 / 1.05 | 0.22 | 0.59 / 1.06 | 82 / 60% |
  | sampled T = 1 | 51% | 0.49 / 1.07 | 0.37 | 1.46 / 2.37 | 78.5 / 54% |
  | no map, greedy | 4% | 1.09 / 1.11 | 0.11 | 0.22 / 0.36 | 77 / 54% |

- **Correction:** I had told the user (from Stage 1e creature recall / precision) that creatures fade to 25–51% of reality in dreams. True for cows by day (greedy 0.03 vs 0.20 per frame @50), **not for zombies at night**: greedy map dreams show about the real number, also within reach.
- **The dream is easier through damage and arrivals, not zombie count:** greedy decoding never picks the minority outcome, so health almost never drops (about a quarter of real damage at night) and new zombies rarely walk in. Even the no-map model, with 4× too many zombies, loses almost no health, so the dreamed health counter is not tied to the zombies drawn.
- **Sampling at T = 1** matches reality for ~5 steps and gives realistic damage and walk-ins, but overpopulates later (2× zombies, 6× cows by day @50) and costs terrain (63 → 54% @50). T = 0.7 is not a compromise.
- **For Stage 2:** sample (T = 1) and keep imagination short (≤ 15 steps): danger is then about right or higher than reality, never lower. Greedy dreams must not be used to train an agent. Possible later fix for long dreams: calibrate creature spawning (e.g. sample only where the model is uncertain, or a separate spawn head).
- Report: section "Is the dream an easier world?" with chart and table. Report 15,972,391 bytes (limit 16,000,000).
- **Videos** (`creature_video.py v4:model_refit.pt --videos 3` → `runs/wm/eval/creatures/night_0{0,1,2}.mp4`): night starts where the real player loses ≥ 3 health within 30 steps; REALITY | GREEDY | SAMPLED T = 1 with each one's health.
  - night_00: zombie next to the player; real health 8 → 4, greedy 9, sampled 2.
  - night_01: real 9 → 4, greedy 9, sampled 7.
  - night_02 is the counter-case: greedy predicts death (0) and sampled stays at 9. Single dreams vary; the 600-dream averages are the result.
  - Not embedded in the report (28 KB left under the 16 MB limit); sent to the user.

## 2026-10-29 — Stage 2 design note (for review, no code)
- `paper/designs/world_model_stage2.md`: Dyna-style PPO. The AD + rarity β=2 agent keeps its real 1M steps unchanged and adds short (H = 15) sampled (T = 1) dreamed rollouts from real states, half from "frontier" states just before the next rare step (stone pickaxe without iron, …).
  - Dream reward: the unlock head gated by the inventory output.
  - AD's auxiliary phase uses real episodes only.
  - Two stages: 2a with the frozen v4 world model (diagnostic, not 1M-comparable), 2b with a world model learned from the agent's own steps (honest).
  - Predictions P15–P18 (P15: iron ≥ 15%, iron pickaxe ≥ 0.5% in 2a); open questions Q11–Q15; first step is a batched-dream speed benchmark.
- The two "stopped" background tasks reported at session start were the expired 30-min watchers of the map + survival runs, which had finished on 27 Oct (18:52). Nothing to resume.

## 2026-10-29 — Stage 2a: implementation and launch (decisions on Q11–Q15 in the design note)
- **Code:**
  - `methods/08_world_model/dream_env.py` (new):
    - `RealTracker` follows the 8 real envs: tile tokens, inventory, clock, pixel-odometry position per frame.
    - `Dreamer` runs D dreams in parallel with a frozen world model: cached frame embeddings (the cached history equals `WorldModel.history`, max diff 0.0000) and one `TileMap` per dream.
    - `DreamRollout` fills a `RolloutStorage` with the policy acting in the dreams.
    - `MixedStorage` gives PPO minibatches that are half real, half dreamed; AD's buffer and auxiliary phase see real data only.
  - `train.py --dream v4:model_refit.pt` (+ `--dream_envs 256 --dream_steps 16 --dream_temp 1 --dream_frontier 0.5 --dream_gate 2 --dream_video_every 25`). Per-epoch dream vs real unlock / death rates go to `metrics.jsonl`; dream videos `videos/dream_XXX.mp4`.
  - `dynamics.predict_next(..., inv_sampled=)`. `scripts/queue_dream.sh`. Labels in compare/analyze/dashboard.
- **Speed:** 256 dreams × 16 steps = 4,096 dreamed steps in ~1–2 s alone (4,600 steps/s), 2.5–8 s with 3 runs sharing the GPU. No KV cache needed.
- **Found and fixed in smoke tests** (6-epoch runs, dream vs real rates per 1,000 steps, real counted from episode step 63 on because dreams need a 64-step context):
  1. **Death and unlock probabilities were 10× too likely when sampled.** The heads were trained with BCE `pos_weight=10`, which multiplies the odds; harmless at a 0.5 threshold, wrong for sampling. Logits are now corrected by −log 10. Dreamed deaths went from ~14 to ~10 per 1k (real ~9).
  2. **The unlock head never pays easy first-time unlocks late in an episode** (collect wood 0.04 vs real 2.7 per 1k): it learned from competent agents, who take them in their first steps, and it does not know the episode's unlocked set. **Change to Q12 (gate mode 2):** collect/make unlocks (12 achievements) are now read off the dreamed inventory (count rises, not yet unlocked this episode). The head, checked against the inventory, stays for eat/place/drink; defeat and wake-up use the head alone.
  3. **Sampling item counts at T = 1 produced phantom diamonds** (a near-random agent "mined" 1.4 diamonds per 1k dreamed steps). Now only the vitals and saplings (a 10% drop from grass) are sampled; other items take the most likely count.
  - After the fixes: collect wood 2.2 vs 2.7, sapling 2.7 vs 2.1 per 1k; no phantom iron or diamonds. Head-based unlocks (wake up, place plant, drink, table) are still under-paid 2–10×. The dream pays at most what reality pays, which leaves nothing for the policy to exploit.
- **First launch crashed** (out of GPU memory): embedding 256 × 64 context frames at once needs 2.5 GB, over the 7 GB per-run cap (this is a 24 GB laptop RTX 5090). The embedding is now chunked. The failed run folders were deleted and the runs relaunched at 14:08.

### 2026-10-29 15:30 — Stage 2a checkpoint at 0.25M steps
- **Score (cumulative, 3 seeds):** dream 7.84 / 9.04 / 10.27 (mean 9.05) vs baseline ad-rarity2 7.15 / 9.06 / 8.04 (mean 7.99) at the same step. Recent-episode score 11.1 / 16.0 / 17.6 vs 10.9 / 14.3 / 11.2.
- **Chain so far (success % over all training episodes):**
  - Place furnace 2.5 / 2.8 / 3.2 vs 0.2 / 1.3 / 0.6. This is the clearest early gain.
  - Collect coal 4.3 / 4.6 / 7.8 vs 1.6 / 6.1 / 4.6.
  - Stone pickaxe 0 / 0.8 / 3.2 vs 0.15 / 1.0 / 1.4.
  - Iron 0 / 0.07 / 0 in both. Iron pickaxe and diamond are 0 in both, as expected this early.
- **Survival-type achievements:** wake_up is lower (69–71% vs 81–86%).
- **Dream health (P17):** the dream pays 11–16 unlocks per 1k steps vs 14–23 in real play (from step 63 on), with no achievement over-paid. Gate rejections are 0.7–1.2 per 1k.
- **Dream danger:** dream deaths per 1k are 7–11 vs real 7–8. They spike to 25–65 per 1k in some epochs with frontier starts, which are often deep in caves near lava.
- **Frontier starts:** they only exist once episodes reach the stone pickaxe: seed 0 none yet, seed 1 in 11 of 63 epochs, seed 2 in 24 of 62.
- **Speed:** ~30 s per epoch; the trainer's ETA is ~4 h more.
- **Videos:** `dream_050.mp4` and spectator clips in each run's `videos/` folder.

### 2026-10-29 17:10 — Stage 2a checkpoint at 0.5M steps
- **Score (cumulative):** dream 15.56 / 16.98 / 19.35 (mean 17.30) vs baseline 12.77 / 16.23 / 13.19 (mean 14.06) at the same step. Recent-episode score 28.1 / 28.1 / 29.8 vs 20.4 / 26.8 / 21.3.
- **The iron chain moves (success %, all episodes; recent 100 in brackets):**
  - Place furnace 22.7 / 21.2 / 26.9 vs 7.7 / 15.3 / 5.5.
  - Stone pickaxe 4.2 / 9.1 / 17.1 (43 / 44 / 58) vs 2.5 / 9.7 / 6.7 (15 / 33 / 28).
  - **Collect iron 0.66 / 1.29 / 3.41 (recent 6 / 10 / 18) vs 0.04 / 1.19 / 0.31 (recent 0 / 7 / 1).**
  - First iron pickaxe: seed 2, 0.04%. Diamond still 0.
- **Cost: wake_up falls,** 52–56% (recent 24–43) vs 79–83%. Likely cause: the dream under-pays wake_up (head-only, about 0.1 vs 0.08 per 1k now, but under-paid 2–10× in the smoke tests) while dreamed sleep still takes zombie damage. The value of sleeping is learned lower. Episodes are longer (242–308 vs 223–229 steps).
- **Dream health (P17):** over 0.4–0.5M the dream pays 15–17 unlocks per 1k vs 24–28 real. Only coal, sapling and iron are paid above the real rate. Iron is 0.49–0.89 vs 0.21–0.76 per 1k (1.2–2.4×). This comparison is biased: 53–71% of dream starts are frontier states, which is where iron is, so a higher dreamed iron rate is expected. Gate rejections are 0.3–0.4 per 1k.
- **Dream deaths:** 7–8 per 1k vs real 5.6–5.8.
- **P15 needs iron ≥ 15% over all of training,** so it is not reachable for seeds 0–1 at their current pace. Seed 2's recent 18% shows the rate is there late. The cumulative number at 1M is what counts.

### 2026-10-29 18:45 — Stage 2a checkpoint at 0.75M steps
- **Score (cumulative):** dream 21.76 / 22.10 / 24.12 (mean 22.66) vs baseline 18.91 / 21.59 / 19.01 (mean 19.84) at the same step. Recent-episode score 34.1 / 31.2 / 34.1 vs 31.2 / 31.2 / 31.7.
- **Iron chain (all episodes %; recent 100 in brackets):**
  - **Collect iron 7.2 / 5.1 / 10.7 (46 / 20 / 45) vs 1.9 / 5.3 / 2.0 (13 / 20 / 13).**
  - Stone pickaxe 20.8 / 23.6 / 32.9 (81 / 69 / 81) vs 12.8 / 20.3 / 16.2 (50 / 44 / 64).
  - Iron sword 0.03 / 0.12 / 0.15 and iron pickaxe 0 / 0.03 / 0.03 vs 0 in the baseline. Diamond 0.
- **Wake up still the cost:** 43 / 53 / 46% (recent 18 / 62 / 21) vs 75–77%.
- **Dream health (0.65–0.75M):** 18 dream unlocks per 1k vs 28–29 real. Iron is 1.15–2.08 vs 1.10–1.87 per 1k (1.0–1.3×), with 80–84% frontier starts. Rejections ~0. Dream deaths 6.3–6.5 vs real 5.0–5.4.
- **P15 outlook:** iron ≥ 15% over all of training at 1M looks reachable for seed 2 only if the recent ~45% holds, likely ~12–20% cumulative. Iron pickaxe is nowhere near 0.5%.

### 2026-10-29 — Report videos re-encoded to free space
- Every embedded report video was re-encoded from its original (`runs/<run>/videos/`) with the same CRF 37 but `-preset veryslow -movflags +faststart` (imageio-ffmpeg's ffmpeg 7.0.2). Raw video bytes went from 11.3 MB to 4.4 MB, and the report from 15.97 MB to **6.74 MB**. Frames were checked side by side, including the small HUD text in the world-model videos; there is no visible difference. A copy is kept only when it is smaller than the old one.
- 3 old copies have no original left (`ad-rarity1-s0` spectator_015, `ad-ppo_ad-s0` spectator_000, `ad-rarity2-s0` spectator_018) and were left as they were.

### 2026-10-29 21:00 — Stage 2a final results (1M steps, 3 seeds)
**Not comparable to the benchmark:** the frozen v4 world model was trained on 6,753 episodes from other agents (see design note, "The 1M rule"). This is the diagnostic stage.
- **Score 26.10 ± 0.75** (25.62, 25.73, 26.96) vs AD + rarity β=2 23.65 ± 1.05. Every dream seed is above the best baseline seed. Wall clock 6.56 h per seed (3 in parallel).
- **Iron chain, all training episodes (dream vs baseline):**
  - Place furnace 47.1 vs 37.1%, stone pickaxe 38.5 vs 28.0%.
  - **Collect iron 15.4 vs 7.4%** (15.5 / 12.8 / 18.0).
  - Iron sword 0.18 vs 0.02%, iron pickaxe 0.09 vs 0.01%. Diamond 0 in both.
- **Lost:** wake_up 44.2 vs 71.7%. Eat cow, defeat zombie, drink and place plant are within ±3 points.
- **Predictions:**
  - **P15: iron part holds** (15.4% mean, seed 1 12.8%). **Iron pickaxe part fails** (0.09% vs ≥ 0.5%).
  - **P16 holds** (26.10 ≥ 22.6).
  - **P17 holds** (the dream pays at most 1.3× real, for iron, with 85% frontier starts).
  - **P18 mixed** (see the death diagnosis).
- **The dream never paid an iron sword, iron pickaxe or diamond** in ~3M dreamed steps per seed (0 in all 750 seed-epochs). Real iron-tool episodes rose anyway (in 3–10 epochs per seed). The dream teaches the way to iron; the last crafts come from real play only. Open question: can v4 dream that craft at all from a real moment just before one?
- **Dream health over the whole run:**
  - 15 dream vs 24–25 real unlocks per 1k steps.
  - Dream/real ratio per achievement: coal 1.0, iron 1.2–1.3, sapling 1.3–1.5. Everything else is under-paid: place_plant 0.04–0.10, place_table 0.15–0.18, wake_up 0.31–0.37, eat_cow 0.44–0.48, defeat_zombie 0.45–0.48, stone pickaxe 0.45–0.51.
  - Gate rejections 0.4–0.5 per 1k. Dream deaths 7.8–8.9 vs real 5.9–6.2 per 1k.
  - Frontier starts 50–63% overall, 84–88% after 0.75M.
  - Dreaming takes ~6 s of a ~30 s epoch.
- **Death diagnosis** (final weights, 200 fresh episodes per seed):
  - Median length 251 (baseline 242), alive at step 300 28.7% (21.5%).
  - Zombie awake 36% (44%), thirst 25% (18%), **zombie asleep 8% (2%)**, lava 10% (10%).
  - 14.2% of deaths are asleep and 52.5% have drink 0 at death (44.8%).
  - Video `death_zombie_asleep_2`: iron at step 93, then at step 262 it sleeps in a cave **by day** with drink 0 and health 4, and is killed asleep at 278. The agent sleeps more often in bad places (cave, low health, no water) and wakes up less.
  - Plausible cause: the dream under-pays wake_up (0.35×) and pays nothing for drinking, and a 16-step dream cannot see the cost of sleeping in a bad spot.
- **Files:** table `runs/report/dream.html`, `results/dream_summary.csv`; death rows in `results/deaths/ad-rarity2-dream-s*.jsonl`; death videos in `runs/ad-rarity2-dream-s0/videos/death_*.mp4`.
- **Highlight search** (`diagnose_deaths.py --keep_if make_iron_pickaxe collect_diamond --out_dir results/highlights_dream`, final weights, 800 fresh episodes per seed):
  - Collect iron in 383 / 349 / 381 of 800 episodes (46% overall).
  - Iron pickaxe 5 / 3 / 0, so 8 in 2,400 (GRU agent: 3 in 1,600). One of them also made the iron sword. **0 diamonds.**
  - 7 of the 8 iron-pickaxe episodes end in an empty-vital death (thirst / hunger / fatigue), 1 by arrow.
  - Report video: `ad-rarity2-dream-s1/highlight_make_iron_pickaxe_1.mp4`. Both iron tools at about step 240 with drink 0 and energy 2; dies at step 338 of hunger, thirst and fatigue.

### 2026-10-30 — Can the world model dream an iron pickaxe? (check before Stage 2b)
**Question:** in Stage 2a the dream never paid an iron tool or a diamond. Is that the policy not trying in the dream, or the world model not knowing the craft?
**Method** (`methods/08_world_model/craft_check.py`):
- Take the policy out: start dreams from real moments k = 0 / 3 / 7 / 15 steps before a real item rise, i.e. 1 / 4 / 8 / 16 dreamed steps. Replay the real actions and count a hit when the dreamed inventory shows the rise by the real step + 2.
- Uses Stage 2a's `Dreamer`: 64-step context, map, T = 1 (vitals and saplings sampled, other counts most likely), 32 dreams per moment. Also records the inventory head's probability of the rise.
- **Held-out moments:** `record_probe.py` (new) played 3,000 fresh episodes per Stage 2a seed (final weights) and kept only those with an iron tool: 44 + 12 + 22 = 78 episodes in `runs/wm/probe/`, outside `runs/wm/data`, so no world model trains on them. They are tokenized and positioned online with the pixel odometry, as in training.
- `record.py` gained `out_root` and `keep_if`.

**v4's own data** (7,134 episodes, 1.76M steps): iron 2,440, stone pickaxe 4,726, **iron pickaxe 4, iron sword 0, diamond 1**.

**Result, held-out episodes (T = 1; T = 0 is the same):**

| item | cases | dream shows it, 1 / 4 / 8 / 16 steps | head p(rise) |
|---|---|---|---|
| coal (control) | 100 | 99 / 96 / 78 / 56% | 0.96 → 0.55 |
| iron (control) | 100 | 81 / 81 / 75 / 56% | 0.76 → 0.55 |
| stone pickaxe (control) | 74 | 81 / 85 / 78 / 55% | 0.80 → 0.54 |
| **iron pickaxe** | **41** | **0 / 0 / 0 / 0%** | ≤ 0.004 |
| **iron sword** | **47** | **0 / 0 / 0 / 0%** | ≤ 0.004 |
| diamond | 1 | 0% | 0.000 |

- **v4's own episodes:**
  - Its 4 training iron-pickaxe crafts are recalled 50% one step ahead, falling to 7% at 16 steps. That is memorised.
  - Its 1 test craft is never recalled (p 0.001). Its 1 diamond is never recalled.
  - Test controls: coal 75–97%, iron 58–82%, stone pickaxe 47–88%.
  - File: `results/world_model/craft_check_traintest.json`.
- **Video** `runs/wm/eval/crafts/iron_pickaxe_0{0,1}.mp4` (`craft_video.py`): the dream keeps terrain, table and furnace right. At the craft step, reality goes wood 1 → 0, coal 2 → 1, iron 1 → 0, iron pickaxe 0 → 1; the dream inventory does not change.
- **The Stage 2a agents' fresh episodes contain one diamond** (1 in 9,000).
- **Real-play crafts per 1M steps** (`episodes.jsonl`), i.e. what a Stage 2b world model would learn from:
  - AD + rarity: iron pickaxe 0 / 1 / 0, iron sword 1 / 1 / 1.
  - Stage 2a: iron pickaxe 5 / 5 / 3, iron sword 6 / 10 / 8.

**Conclusion:**
- The world model, not the dream setup or the policy, is the ceiling for the last crafts. It learns crafting per item from that item's own examples and has 4 for iron pickaxes.
- Stage 2b as designed (a world model learned from the agent's own steps) would have 0–5 examples and the same ceiling.
- Options that could lift it:
  1. An inventory output that shares the craft rule across tools: action → per-item delta, conditioned on what is in reach. Wood and stone tools then teach the iron ones.
  2. Heavy oversampling of the rare crafts when training the world model.
  3. Hand-coding Crafter's crafting rules in the dream. This is the strongest, but it is no longer a learned model and changes what the method claims.
- Report: section "Can the world model dream an iron pickaxe? No: it has seen four" (table `runs/report/craft_check.html`, 1 video). Report 7.45 MB.

### 2026-10-30 — Design note: a shared craft rule (`paper/designs/world_model_craft_rule.md`)
- **Data behind it** (v4 training data, per craft action: pressed / requirements met by Crafter's rule / tool count rose):
  - wood pickaxe 27,514 / 7,633 / 7,433; stone pickaxe 38,999 / 5,468 / 4,726.
  - **iron pickaxe 17,471 / 4 / 4**.
  - wood sword 30,689 / 7,770 / 7,541; stone sword 45,869 / 5,123 / 4,567.
  - **iron sword 15,594 / 0 / 0**.
  - "Requirements met" means items in the inventory plus table / furnace in the 3×3 of `sem` around the player.
  - No craft ever worked without its requirements. When they are met, 86–97% work; the rest are most likely presses while asleep (Crafter replaces the action with sleep), not verified.
  - Missing on iron-pickaxe presses: furnace 90%, table 84%, iron 81%, wood 52%, coal 38%.
- **Proposal:** a craft layer for the 6 make actions.
  - A learned AND over item thresholds and stations in reach, plus a learned fixed effect.
  - Station tiles are identified from the agent's own placing events.
  - HUD rows are redrawn from the dreamed inventory.
  - Trained on the frozen v4 trunk.
  - Q16 (pick: tie pickaxe and sword per tier, `make_X` gives `X`) decides whether iron sword can be learned at all.
- **Predictions:** P19 (recipes match `data.yaml`), P20 (craft_check iron tools ≥ 75%, false crafts ≤ 1 in 10k), P21 (Stage 2a rerun: iron pickaxe and iron sword ≥ 0.5% of training episodes, score ≥ 25.35), P22 (diamonds still not, risk).
- Waiting for the user's review before code.

### 2026-10-30 — The shared craft rule: built and tested (`methods/08_world_model/craft_layer.py`)
Decisions on Q16–Q20: tied tiers, no collect rule, a HUD lookup, a v4 rerun first, placing events allowed. Full detail is in `paper/designs/world_model_craft_rule.md`, sections "Changes while building" and "Tests".

**Data** (`craft_layer.py extract`, `runs/wm/craft/events_{train,test}.npz`): every craft press with tokens and inventory; placing steps; HUD codes per (item, count). `sem` and `daylight` are kept only for reporting.

**What had to change from the note, and why:**
1. **Station codes from placing alone were wrong (35% pure).** At night the noise changes all four neighbours on every frame.
   - Fix: keep placing steps with one changed neighbour (4,652 tables, 3,871 furnaces) and track each station's world position with the pixel odometry. Count its code in every later frame that shows it (94–95% of tracked tiles really are the station).
   - Station codes are those with a relative share above 0.2. Codes at stations score 0.38–1; everything else ≤ 0.05.
   - Result: 14 table and 11 furnace codes, 99.7% and 98.8% of real stations in reach found, **0 false detections** on the test split.
2. **Coal is not identifiable for the iron tools.** No iron press failed with only coal missing (counts of presses failing with exactly one requirement missing: wood 38, iron 35, table 365, furnace 167, coal 0).
   - Fix: requirement = max(amount used up, learned extra with L1). The 4 iron crafts show coal used up.
3. **1,727 presses met everything and failed.** All had energy < 9, i.e. the player was asleep.
   - Fix: an "awake" term, a linear read-out of v4's frame embedding (which does not see the action).
4. **Tools change only on their own make action** (24,271 of 24,271 changes in the data). In the dream, tool counts are now carried over on non-make steps.
5. **v4's inventory input rows for pairs it never saw** (1 iron sword, 2+ iron pickaxes) are untrained random vectors. With the layer, the world model reads them as the nearest seen count below. The dream keeps the true count.

**Fit:** 3,000 full-batch Adam steps on the 176,136 train presses with the frozen v4 frame embeddings, about 1 minute. BCE 1e-5.

**Tests** (`craft_layer.py check` → `results/world_model/craft_layer.json`; `craft_check.py --craft` → `results/world_model/craft_check_layer.json`):

| test | result |
|---|---|
| 1. recipes (tied) | all 6 match `data.yaml` (items, amounts, table / furnace, effects). **P19 holds** |
| 1. recipes (untied) | 5 of 6; iron sword learns nothing, as predicted |
| 2. held-out iron pickaxe / sword, 1 step ahead | **97% / 99%** (was 0 / 0); 16 steps ahead 63 / 69%. **P20 holds** |
| 2. controls | stone pickaxe 99% (was 81%); coal, iron unchanged |
| 3. false crafts | 0.012 per 10,000 presses missing a requirement |
| recall | p(works) 0.999 on real crafts, 0.045 on asleep presses |
| 4. HUD lookup | 100% of cells right where the pair is known |

- **Code changes:**
  - `dream_env.Dreamer(..., craft=)`.
  - `train.py --dream_craft tied|untied`.
  - `craft_check.py --craft [--untied]`.
  - `craft_video.py --craft --item`.
  - Labels added in dashboard, compare and analyze.
- **Smoke test:** 6 epochs ran fine. Dreamed wood crafts are paid at or below the real rate.
- **Launched 16:35:** `scripts/queue_dream.sh ad-rarity2-dream-craft --dream_craft tied` (3 seeds, 1M steps, about 7 h).

### 2026-11-01 — Craft-layer rerun finished; iron-in-hand starts launched
- **Craft-layer rerun (`ad-rarity2-dream-craft`, finished 2026-10-30 23:15):** score 26.98 ± 0.76 (Stage 2a 26.10 ± 0.75); collect iron 17.6%, make iron pickaxe 0.11%, iron sword 0.18%, one diamond (seed 2, step 335k, after an iron pickaxe). P21: score part holds, iron-tool part (≥ 0.5%) missed. P22 (no diamonds) broken. Full write-up, deaths and videos not done yet (the user asked to hold them).
- **Diagnosis:** the dream paid about 22 iron pickaxes in total over the 3 seeds; per 1,000 steps the dreamed rate (0.002–0.012) equals the real one (0.004–0.008). The layer fixed what the model can dream; the dream starts are now the limit.
- **Untied layer, probe split (T = 1):** iron pickaxe 93/86/75/62% at k = 0/3/7/15, iron sword 0%, as predicted.
- **Launched 23:22:** iron-in-hand starts, see `paper/designs/world_model_stage2.md` "Iron-in-hand dream starts" (P23). Smoke test 3 epochs OK; picker unit-tested on synthetic candidates.

### 2026-11-02 05:52 — Iron-in-hand dream starts: result (`ad-rarity2-dream-iron`)

| | craft-layer rerun | iron-in-hand starts |
|---|---|---|
| score | 26.98 ± 0.76 | 26.11 ± 0.93 |
| collect iron | 17.6% | 14.8% (seed 0 only 9.9%) |
| make iron pickaxe | 0.11% (0.12 / 0.15 / 0.07) | **0.23%** (0.02 / 0.24 / 0.41) |
| make iron sword | 0.18% | **0.37%** (0.05 / 0.16 / 0.92) |
| collect diamond | 1 episode (seed 2) | 1 episode (seed 1) |
| wake_up | 43.0% | 33.2% |
| dreamed iron pickaxes, total over 3 seeds | 22 | 52 |
| dream starts with iron in hand | – | 24–29% (real moments: 8–14%) |

- **P23:** score ≥ 25.35 holds; dreamed iron pickaxes ×2.4 (target ×5) and real iron pickaxe 0.23% (target 0.3%) missed. Seed 2 alone reached 0.41% / 0.92%.
- **Reading:** iron tools doubled, but with 3 seeds and seed 0 far behind (it found iron late, so it had few iron starts) the gain is not clear-cut. About 0.2% of iron-in-hand dreams end in an iron pickaxe: starting near the craft is not enough, the policy rarely puts wood, coal, a table and a furnace together within 16 steps.
- wake_up fell 10 points; dreamed deaths 7.8–8.7 per 1,000 steps vs 7.2–8.3 in the rerun. Not explained yet.
- Not done yet: deaths on 600 episodes, videos, report section.

### 2026-11-02 — Why iron-in-hand dreams rarely end in an iron tool (`methods/08_world_model/iron_diag.py` → `results/world_model/iron_diag.json`)
Final weights of both dream runs (6 policies), 300 real episodes each: every step with iron in hand and an iron tool missing (122,249 steps). From 60 sampled moments per policy: 8 real futures (copies of the game) and 8 dreams (v4 + craft layer), 16 steps, same policy.

**What those moments have (all 6 policies):**

| | share of iron-in-hand steps |
|---|---|
| wood and coal in hand | 46% |
| enough wood for a new table / stone for a new furnace | 30% / 33% |
| table in reach (3 × 3) / furnace in reach | 18% / 14% |
| **table and furnace both in reach** | **2.0%** |
| both within 4 tiles | 17% |
| everything for the craft | **0.14%** |
| nearest table / furnace (median distance) | 7 / 5 tiles; 14% never placed a furnace |

- **The bottleneck is putting the stations together.** The iron craft needs a table and a furnace both next to the player. The agents place furnaces often (a furnace needs no table nearby in Crafter), but rarely next to a table, and hold iron mostly elsewhere.
- **Policy:** when everything is in place it presses make iron 22% of steps (iron-start seeds 1–2: 35–40%; craft-run seeds: 12–14%). It also presses make iron when nothing is in place: 6.4% of all actions, 36% of 16-step futures. It has learned the button, not the condition.
- **Dream vs real from the same moments:** actions, tables / furnaces placed and deaths match closely. Iron tools made: real 0.49% (14 / 2,880), dream 0.14% (4 / 2,880); "everything in place at some step": real 1.3%, dream 0.9%. The dream under-pays a little, but it is not the main limit.
- **What the iron starts changed:** pressing when craftable (35–40% vs 12–14% in two seeds), not how often the stations are together (1–3% in both runs).

### 2026-11-02 — Ready-to-craft starts, checked before training (`iron_diag.py --near 4 --p_start 0.3 --H 32` → `results/world_model/iron_diag_ready.json`)
Same 6 policies and v4 + craft layer as above, but the starts are only "ready-to-craft" moments: iron, wood and coal in hand, an iron tool missing, a table and a furnace both within 4 tiles (349 starts; 8% of them already have both in reach). 8 real futures and 8 dreams of 32 steps each.

| policy | real: iron tool in 16 / 32 steps | dream: 16 / 32 | pressed make-iron when craftable (real) |
|---|---|---|---|
| iron-start s0 | 1.0% / 1.5% | 1.0% / 1.3% | 8% |
| iron-start s1 | 11.9% / 12.7% | 11.0% / 11.7% | 39% |
| iron-start s2 | 11.5% / 14.0% | 12.3% / 14.8% | 31% |
| craft-run s0 | 4.0% / 4.0% | 5.0% / 5.0% | 15% |
| craft-run s1 | 6.3% / 7.3% | 5.4% / 6.0% | 15% |
| craft-run s2 | 4.4% / 4.8% | 2.7% / 3.5% | 16% |
| **all** | **6.7% / 7.6%** | **6.4% / 7.2%** | 23% |

- **Ready starts pay about 45× more often in the dream than any iron-in-hand start** (6.4% vs 0.14% at 16 steps), and the dream now matches the real game closely (6.4% vs 6.7%).
- **Longer dreams barely help:** 16 → 32 steps adds about 1 point. Most crafts happen within 4 steps (4.5%), so they come mostly from starts already next to both stations. Within 32 steps, only 14% of futures ever have everything in place.
- **The payoff follows the policy's pressing rate** (1–14% across policies). Ready starts would mainly train *pressing make-iron at the right moment*. They don't teach *bringing the table and furnace together*; that needs its own fix.

### 2026-11-02 15:34 — Ready-to-craft dream starts launched (`ad-rarity2-dream-ready`, seeds 0–2)
- New option `--dream_ready` in `methods/08_world_model/dream_env.py` and `train.py` (`ready_to_craft`). Design and reasons: `paper/designs/world_model_stage2.md`, "Ready-to-craft dream starts".
- Rule check on held-out probe episodes: true on 8.3% of iron-in-hand frames, and at 88 of 89 real iron crafts. The start picker was tested on synthetic candidates, including the fallback when no ready moments exist; a 2-epoch smoke run passed (and was deleted).
- Run: `scripts/queue_dream.sh ad-rarity2-dream-ready --dream_craft tied --dream_iron 0.25 --dream_ready 0.15`. Same setup as `ad-rarity2-dream-iron` otherwise. Log: `runs/logs/queue_dream_ready.out`.
- Labels added in `scripts/dashboard.py`, `compare_configs.py`, `analyze_deaths.py`.

### 2026-11-02 22:08 — Ready-to-craft dream starts: result (`ad-rarity2-dream-ready`)
Success rates over all training episodes (Crafter protocol), 3 seeds each.

| | craft-layer rerun | iron-in-hand starts | ready-to-craft starts |
|---|---|---|---|
| score | 26.98 ± 0.62 | 26.11 ± 0.76 | 26.25 ± 2.23 (27.92 / 23.10 / 27.74) |
| collect iron | 17.6% | 14.8% | 13.7% |
| make stone pickaxe | 41.8% | 40.9% | 34.7% |
| make iron pickaxe | 0.11% | 0.23% | **0.66%** (0.66 / 0.00 / 1.33) |
| make iron sword | 0.18% | 0.37% | **0.93%** (2.06 / 0.02 / 0.72) |
| episodes with an iron pickaxe (seeds 0 / 1 / 2) | – | – | 30 / 0 / 61 |
| collect diamond | 1 episode | 1 episode | 2 episodes (seed 2) |
| wake_up | 43.0% | 33.2% | 42.1% |
| dreamed iron pickaxes, total over 3 seeds | 22 | 52 | **752** (257 / 4 / 491) |
| dream starts ready to craft (last 100 epochs) | – | – | 10–15% (real moments: 0.5–1.3%) |

- **P24 holds on the mean:** iron pickaxe ×2.9 and iron sword ×2.5 over the iron-start run, ×6 and ×5 over the craft-layer rerun. Seed 2 makes an iron pickaxe in 1.33% of its episodes.
- **The dream rewards feed back:** in seeds 0 and 2, dreamed iron pickaxes grow every 50 epochs (seed 2: 0, 0, 53, 161, 277), so the real game produces more ready moments and the dreams pay more.
- **Seed 1 never takes off.** It found iron late (epoch 92, against 59 and 63) and had its first ready start at epoch 122. It dreamed only 4 iron pickaxes and made none in the real game. Its score (23.10) and stone pickaxe rate (26%) are low too. This is the same pattern as seed 0 of the iron-start run: a late iron finder gets too few iron starts.
- **The score did not move** (26.25 vs 26.98): the iron gains are a few tenths of a percent, and the stone pickaxe rate fell 7 points. Still diagnostic, not 1M-comparable (v4 was trained on other agents' data).
- Not done yet (on hold until the user asks): deaths on 600 episodes, videos, report section.

### 2026-11-03 — Why seed 1 of the ready-start run never took off; design proposed
- Seed 1 had about 3,100 ready starts (seeds 0 and 2: about 4,500–4,900), but they gave 0.7 dreamed iron tools per 100, against 16–24.
- `iron_diag.py --near 4 --H 16` on the run's final weights (`results/world_model/iron_diag_ready_run.json`): from ready moments, an iron tool in 39% / 4% / 38% of real futures; make-iron pressed when craftable 57% / 8% / 69%; dream ≈ real in every seed (37% / 4% / 34%).
- Reading: seeds 0 and 2 entered a self-reinforcing loop; seed 1 found iron late and never got enough rewarded presses to start it.
- Proposed fix (not built, for review): craftable starts plus a start pool across epochs. See `paper/designs/world_model_stage2.md`, "Proposed: craftable starts plus a start pool across epochs".

### 2026-11-04 — Craftable starts validated (`iron_diag.py --reach --p_start 1 --per_ep 5 --episodes 600 --H 16` → `results/world_model/iron_diag_reach.json`)
Final weights of `ad-rarity2-dream-ready`. Starts only at craftable moments: iron, wood and coal in hand, table and furnace both in the 3 × 3.

| seed | starts | iron tool made, real (≤ 4 / ≤ 16 steps) | dream (≤ 4 / ≤ 16) | died in 16 steps (real) |
|---|---|---|---|---|
| 0 | 60 | 89% / 97.5% | 90% / 97.3% | 9% |
| 1 | 42 | 19% / **28.3%** | 21% / 27.7% | 25% |
| 2 | 60 | 84% / 95.0% | 83% / 92.7% | 8% |

- **Go rule met:** seed 1 reaches 28% (rule: at least 20%), against 4% from ready starts and 0.7 per 100 ready starts during training. Its pressing rate when craftable is unchanged (7%), but a craftable start gives it many chances before the moment passes.
- The dream matches the real game in every seed, within 2.3 points.
- **Caveat:** seed 1's 42 starts come from few episodes (up to 5 consecutive craftable steps per episode), so they are correlated. Its 25% death rate in those futures is higher than the other seeds' and is not explained.
- Next: build Parts A and B (design: `paper/designs/world_model_stage2.md`).

### 2026-11-04 — Craftable starts and start pool built (Parts A and B)
- Code: `methods/08_world_model/dream_env.py` (`craftable_now`, `Track.unlocks`, the `DreamRollout` pool) and `train.py` (`--dream_reach`, `--dream_pool`, `--dream_pool_age`; metrics `dream_reach_share`, `dream_pool_share`, `dream_pool_size`, `dream_pool_reach`, `dream_pool_state_cos`, `real_reach_share`).
- Smoke test: resumed the final weights of `ad-rarity2-dream-ready-s0` for 8 epochs (pool 64, age 3), then deleted it. Pool starts 5–14%, craftable starts 5–7%, recomputed AD state vs storage cosine 0.993–1.000.
- Not launched yet. Proposed run and details: `paper/designs/world_model_stage2.md`.

### 2026-11-04 08:52 — Start-pool run launched (`ad-rarity2-dream-pool`, seeds 0–2)
- `scripts/queue_dream.sh ad-rarity2-dream-pool --dream_craft tied --dream_iron 0.25 --dream_ready 0.15 --dream_reach 0.05 --dream_pool 512 --dream_pool_age 25`. Log: `runs/logs/queue_dream_pool.out`.
- Prediction: P25 in `paper/designs/world_model_stage2.md`. Compare against `ad-rarity2-dream-ready`; seed 1 is the case to watch.
- Labels added in `scripts/dashboard.py`, `compare_configs.py`, `analyze_deaths.py`.
