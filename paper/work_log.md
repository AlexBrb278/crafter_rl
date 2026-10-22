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
- **Git:** the 4 commits pushed from the Windows laptop carry `Co-Authored-By: Claude Opus 5.5` trailers, and GitHub credits that address to Claude. The user rewrites the messages and force-pushes. Also needed: the author email on their GitHub account, and `"attribution": {"commit": "", "pr": ""}` in `~/.claude/settings.json` on both machines.

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
