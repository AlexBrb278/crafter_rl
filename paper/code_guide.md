# Code guide: how this Crafter agent works, file by file

This guide explains the whole project so you can present it and answer questions about it.
Part 1 is the code you hand in (`submission/`), read line by line. Part 2 is the larger research
repository around it. Part 3 lists questions you are likely to be asked, with answers.

---

## 0. The big picture in one page

**Goal.** Train an agent that plays Crafter from pixels for 1M environment steps and gets the highest
*Crafter score*.

**Final method.** PPO (an actor-critic policy-gradient algorithm) + Achievement Distillation (AD, an
auxiliary representation-learning phase from Moon et al., NeurIPS 2023) + our **rarity bonus** (extra
reward for rare achievements).

**Result.** 23.65 ± 1.05 over 3 seeds, against 15.75 ± 2.99 for PPO, 1.55 for a random agent and
21.79 for the published AD.

**Two codebases in this repo.**

| | where | what it is |
|---|---|---|
| research code | `methods/05_achievement_distillation/` (+ `upstream/`, `common/`, `scripts/`) | What produced all the results in the deck. It imports the official AD code (cloned into `upstream/`, git-ignored) and adds our changes on top. |
| submission code | `submission/` | A compact, self-contained rewrite of the same algorithm (~1,000 lines with comments). This goes in the zip. It was checked against the official code: identical network outputs on a trained checkpoint, same losses and gradients. |

**One training update, in words** (`submission/train.py`):
1. The current policy plays 8 Crafter games for 512 steps each (4,096 steps). Every step is stored.
2. Our rarity bonus is added to the reward of steps where an achievement is unlocked for the first time.
3. GAE turns rewards and value predictions into advantages and value targets.
4. PPO runs 3 passes over the 4,096 steps in minibatches of 512 (24 gradient steps).
5. Every 8th update, the AD auxiliary phase runs: 6 passes over the last 8 rollouts, cut into whole
   episodes, training the encoder with two contrastive losses.
6. Logging: score, reward, losses. After 245 updates (≈1M steps): save, then evaluate on fresh worlds.

---

## 1. Crafter: what the code receives from the game

```python
import crafter
env = crafter.Env(seed=123)
obs = env.reset()                         # uint8 array (64, 64, 3): the agent's view + inventory bar
obs, reward, done, info = env.step(a)     # a in 0..16
```

- **Observation:** a 64×64 RGB picture. The top 7/9 is a 9×7-tile view of the world around the player, the
  bottom shows the inventory (health, food, drink, energy, materials, tools).
- **17 actions:** `noop, move_left, move_right, move_up, move_down, do` (interact: collect, attack,
  drink, eat), `sleep, place_stone, place_table, place_furnace, place_plant,
  make_wood_pickaxe, make_stone_pickaxe, make_iron_pickaxe, make_wood_sword, make_stone_sword, make_iron_sword`.
- **`info`** is a dict. We use three keys:
  - `info["achievements"]`: `{name: count}` for the 22 achievements;
  - `info["inventory"]`: `{"health": 9, "food": 9, "drink": 9, "energy": 9, "wood": 0, ...}`;
  - `info["discount"]`: 0 if the player died this step, else 1.
- **Reward** (`crafter/env.py`): `(health_now − health_before) / 10`, plus **+1** on a step where at least one
  achievement is unlocked for the first time in this episode. So the episode reward is roughly
  "number of different achievements unlocked", minus 0.9 when the agent dies (health 9 → 0).
- **Episode end:** death (health ≤ 0) or 10,000 steps. Our agents always die first, around 250 steps.
- **Score** (Hafner 2022): for each achievement, its success rate `s_i` = % of *all training episodes*
  in which it was unlocked at least once. Then `S = exp(mean_i ln(1 + s_i)) − 1`. Because of the
  logarithm, going from 0% to 1% on one achievement adds as much as going from 10% to 21% on another.
  The rare achievements dominate the score.

---

## Part 1 — the submission code

Dependency order: `envs.py` and `model.py` are independent; `algo.py` uses `model.py`; `train.py`
uses all three; `evaluate.py` uses `envs.py` and `model.py`.

### 1.1 `envs.py`: environments, score and reward changes

**`ACHIEVEMENTS`**: the 22 names in alphabetical order. Every 22-long vector in the code uses this order.

**`crafter_score(rates)`**: the formula above. `np.log1p(s)` is `ln(1 + s)`. Input is in percent.

**`_worker(remote, seed)`**: the function that runs inside each child process.
- It creates one `crafter.Env`, then waits for commands on a `multiprocessing.Pipe`.
- `"reset"` sends back the first observation.
- `"step"` plays one action, keeps the episode's running return and length, and copies the three
  `info` keys we need (sending the full `info` through the pipe would be slower).
- **Auto-reset:** when `done`, it adds `info["episode"] = {"r": return, "l": length}`, resets the game and
  sends the *first frame of the next episode* as `obs`, while `info` still describes the step that ended.
  This is the standard "vector env" convention (same as stable-baselines' `SubprocVecEnv` + `VecMonitor`).

**`VecCrafter`**: starts `num_envs` worker processes.
- `"spawn"` start method: safe on Windows, and the parent never imports Crafter.
- Per-env seeds are drawn from `RandomState(seed)`, so `--seed` fixes the worlds.
- `step(actions)` sends one action to each worker and then collects all results, so the 8 games run in
  parallel. It returns `obs (8, 64, 64, 3) uint8`, `rews (8,)`, `dones (8,)`, `infos` (list of 8 dicts).

**`RarityBonus`**: our contribution.
- `hist`: a `deque(maxlen=200)`; each finished episode appends one row of 22 booleans (unlocked or not).
- `rates()` = column means, i.e. `p_i` = share of the last 200 episodes that unlocked achievement *i*.
- `__call__(rews, dones, infos)`, called once per environment step for all 8 envs:
  - `unlocked` = achievements with count > 0 so far in this episode; `self.prev[i]` = the same set one step
    earlier. Their difference is what was unlocked *on this step*.
  - For each new achievement *a*: `rews[i] += beta * (1 - p[a])`.
  - On `done`, it appends the episode's row to `hist` and clears `prev`.
- **Worked example** (β = 2): `collect_wood` is unlocked in 92% of recent episodes, so its first
  unlock pays 1 (game) + 2 × 0.08 = 1.16. `make_stone_pickaxe` at 2% pays 1 + 2 × 0.98 = 2.96.
  Never-seen `collect_diamond` would pay 3.
- It only changes the reward the **learner** sees. `train.py` logs the game's reward, and the score
  uses the achievement flags, so results are comparable with other methods.
- `self.total` is just a running sum, logged as `rarity_bonus_total`.

**`VitalsBonus`** (ablation, off by default): potential-based reward shaping.
- Potential `Φ(s) = α Σ_{food, drink, energy} (1 − ((9 − v)/9)²)`: 3α when all vitals are full, lower as
  they run down (quadratically, so refilling an almost-empty vital pays most). Φ = 0 after death.
- Shaping `F = γ_v Φ(s') − Φ(s)` is added to the reward.
- With `γ_v` = the agent's γ, this is the classic policy-invariant shaping (Ng, Harada & Russell 1999).
  In practice it hides a cost of −(1−γ)Φ ≈ −0.15 for every step alive, which made agents worse
  (17.8). With `γ_v = 1` ("no drift", the default of `--vitals_gamma`) it scored 22.1 and lengthened lives.

### 1.2 `model.py`: the network

Shapes for one observation (batch dimension omitted):

```
obs 3×64×64 (floats in [0,1])
 └ ImpalaCNN
    stack 0: conv 3→64 (3×3) → maxpool 3×3/2 → GroupNorm → 2 residual blocks   → 64×32×32
    stack 1: GN → conv 64→128  → maxpool → GN → 2 blocks                         → 128×16×16
    stack 2: GN → conv 128→128 → maxpool → GN → 2 blocks                         → 128×8×8
    flatten 8192 → LayerNorm → Linear → ReLU                                      → 256
 └ linear: LayerNorm → Linear 256→1024 → ReLU                                     → latent z (1024)
 heads see x = [z, g] (2048), where g is the 1024-d achievement state (AD)
    pi_head: Linear 2048→17 → log_softmax                                         → log π(a|s)
    vf_head: Linear 2048→1                                                        → normalised V(s)
```

**`FanInLayer`**: the building block: optional normalisation (`"group"` = GroupNorm with 1 group,
`"layer"` = LayerNorm), then conv or linear, then optional ReLU. The weights of each output unit are
rescaled to have L2 norm `scale` ("fan-in" initialisation from the IMPALA / VPT code). When there is a
norm layer, the conv/linear has no bias (the norm already has one).

**`ResBlock`**: `x + conv1(conv0(x))`, each conv preceded by GroupNorm and followed by ReLU. Scales are
chosen so the residual sum keeps its size at initialisation.

**`DownStack`** and **`ImpalaCNN`**: the encoder of IMPALA (Espeholt et al. 2018) with normalisation.
Each stack halves the resolution. 9.23M parameters in total, most of them in the 8192 → 256 dense layer
(2.1M) and the AD MLPs.

**`MLP`**: two layers (LayerNorm + Linear + ReLU, then Linear), used by AD.

**`PolicyHead`**: a linear layer with tiny orthogonal initialisation (gain 0.01), so the initial policy
is almost uniform; returns **log-probabilities** (`log_softmax`). Everything downstream works in log space.

**`Normalizer`**: running mean and variance of the value targets (returns), as an exponential moving
average with β = 0.99, debiased like Adam (`debiasing_term` starts at 0 and grows to 1).
- In training mode, every call updates the statistics with the batch it sees and returns `(x − mean)/std`.
- The critic is trained to predict *normalised* returns; `denormalize` converts its output back when acting.
- Why: returns grow from ~1 to ~10+ during training; normalised targets keep the value loss at a stable
  scale, which matters because the value loss is added to the policy loss.

**`ValueHead`**: linear layer (gain 0.1) + `Normalizer`. `loss(pred, target)` = squared error between
the prediction and the *normalised* target (per sample).

**`Agent`**: the whole model.
- `encode(obs)` → z. `forward(obs, states)` → dict with `latents`, `pi_logits`, `vpreds` (normalised).
  With `ad=False` the heads see z only.
- `act(obs, states)` (no gradients): samples an action from the categorical distribution, returns its
  log-probability and the de-normalised value. The policy is stochastic during training *and* evaluation.
- **`get_states(goal_obs, goal_next_obs)`**: AD's *achievement embedding*:
  `normalise(enc(o_after) − enc(o_before))`, where `o_before`/`o_after` are the frames just before and
  just after an unlock. The difference of two encodings describes *what changed* (e.g. a pickaxe
  appeared in the inventory). An all-zero frame means "no frame" and is encoded as a zero vector (used for
  the "no more achievements in this episode" anchor).
- **`next_goal_pred(latents, actions, states)`**: predicts the embedding of the *next* achievement.
  The one-hot action goes through `action_mlp`, which outputs (γ, β) for **FiLM** modulation:
  `(1 + γ) ⊙ z + β`. The result, concatenated with the current achievement state, goes through
  `next_goal_pred_mlp` and is normalised to unit length.

**`kl(new, old)`**: `Σ_a π_new(a) (log π_new(a) − log π_old(a))` = KL(π_new ‖ π_old), from log-probabilities.

**The achievement state g (AD's "memory").** The policy sees one extra 1024-d vector: the embedding of the
*last* achievement unlocked in the current episode (zero at the start). It tells the policy "where in the
tech tree am I" without a recurrent network. It is computed during the rollout (`Rollout.insert`) and in
evaluation, never learned directly.

### 1.3 `algo.py`: rollouts, GAE, PPO and the AD phase

#### `Rollout`: what is stored
All tensors live on the GPU, with time first, then env: `obs (513, 8, 3, 64, 64)`, `actions (512, 8, 1)`, etc.
Index convention: **position t holds the observation o_t and everything the agent did at o_t**
(action, its log-probability, value, the reward it received). Arrays with 513 entries hold the state
after the last step too.
- `masks[t] = 0` means o_t is the first frame of a new episode (the previous step ended an episode).
  `masks[t+1] = 1 − done_t`.
- `states[t]`: the achievement state g fed to the heads at step t.
- `successes[t]`: the 22 unlocked flags at o_t (reset to 0 at episode starts).

**`insert(t, model, out, obs, rewards, masks, successes)`**, called after every env step:
1. If the unlocked flags changed between o_t and o_{t+1} (a new achievement), the new state is
   `normalise(enc(o_{t+1}) − z_t)`; otherwise the state is carried over.
2. If the episode ended, state and flags are zeroed.
3. Everything is written at t (action, reward, log-prob, value) or t+1 (next obs, mask, state, flags).

**`compute_returns(gamma, lam)`**: Generalised Advantage Estimation, backwards in time:
```
δ_t = r_t + γ V(s_{t+1}) m_{t+1} − V(s_t)          (TD error; m = 0 cuts across episode ends)
A_t = δ_t + γ λ m_{t+1} A_{t+1}                      (exponentially weighted sum of TD errors)
R_t = A_t + V(s_t)                                   (value target)
```
`vpreds[512]` (the value of the state after the last step) is filled in by `train.py` before this call,
so the last steps bootstrap from it. Advantages are then normalised over the whole rollout
(mean 0, std 1). γ = 0.95 and λ = 0.65 are short horizons (≈20 and ≈3 steps); they come from the AD paper.

**`minibatches(nbatch)`**: shuffles the 4,096 steps and yields 8 minibatches of 512 (flattening time and env).

**`segment()`**: what the AD buffer keeps of a rollout, on the CPU. Frames are stored as uint8
(4× less memory, and exact: they were uint8/255 to begin with).

**`reset()`**: the last stored step becomes step 0 of the next rollout, so episodes continue across rollouts.

#### `PPO.update(roll)`
1. If AD is on, store `roll.segment()` in `self.buffer` (a deque that keeps the last 8 rollouts).
2. `ppo_loss(mb)` for one minibatch:
   ```python
   logp  = log π_θ(a|s)                       # new policy, from the stored action
   ratio = exp(logp − log π_old(a|s))         # π_θ / π_old; log π_old was stored during the rollout
   pi_loss = −mean(min(ratio·A, clip(ratio, 0.8, 1.2)·A))
   entropy = −Σ π log π, averaged
   vf_loss = mean((V_θ(s) − normalise(R))²)
   loss = pi_loss + 0.5·vf_loss − 0.01·entropy
   ```
   The clipping removes the incentive to move the probability ratio outside [0.8, 1.2]: this is what
   keeps PPO's updates small and lets it reuse each rollout for 3 epochs.
3. 3 epochs × 8 minibatches = 24 optimiser steps per update (Adam, lr 3e-4, gradient norm clipped to 0.5).
4. Every 8th update (`self.updates % 8 == 0`), `aux_phase()`.

**`_step(opt, batch, loss_fn)`**: one optimiser step. With `--micro_batch k`, the batch is cut into chunks;
each chunk's loss is multiplied by (chunk size / batch size) and backpropagated, so the summed gradient
equals the full-batch gradient (all losses are means). This is gradient accumulation; it was needed on a
4 GB GPU. The value normaliser would otherwise update k times per step, so its β is set to β^(1/k) during
the step, which keeps its time constant.

#### The AD auxiliary phase (`aux_phase`)
1. **`_episodes()`**: concatenates the last 8 rollouts in time (8 × 512 = 4,096 steps per env), finds
   the episode starts (`masks == 0`) in each env's stream and cuts out every *complete* episode (the
   partial ones at the edges are dropped). For each episode it finds the **goal steps**: the indices g
   where the unlocked flags change, i.e. `o_g` is the first frame after an unlock. It stores
   `goal_obs = o_{g−1}` and `goal_next = o_g` (before / after frames of every achievement).
   Typically 120–190 episodes; only episodes with at least one achievement are used.
2. **Old policy.** The model, *before* any auxiliary update, computes `old_logits` on every stored step,
   once. These are the targets of the KL term. (The reference code re-ran a frozen copy of the model on
   every sample in every epoch; computing it once gives the same numbers and is much faster.)
3. **6 epochs**, each: all matching batches (with `match_opt`), then all prediction batches (with `pred_opt`).
   Each loss has its own Adam optimiser, separate from the PPO one, as in the paper.

**`_contrast(anchor, pos, neg)`**: the contrastive loss used by both: logits
`[⟨anchor, pos⟩, ⟨anchor, neg⟩] / τ` with τ = 0.1, cross-entropy with target 0 ("the positive is the right
one"). All vectors are unit length, so the dot product is the cosine similarity.

**`_distill(out, old_logits, returns)`**: `KL(π_θ ‖ π_old)` and `(V_θ − normalise(R))²`. Both auxiliary
losses add these with weight 1, so that reshaping the encoder does not change what the policy does or
breaks the value function (the idea comes from Phasic Policy Gradient).

**Prediction loss (`_pred_batches`, `_pred_loss`), within one episode.**
- For each step t, the **anchor** is the embedding of the *next* achievement unlocked after t.
  `seg = searchsorted(goal_steps, t, right=True)` gives the index of that next goal; steps after the last
  achievement get a special anchor: the final frame against an empty frame.
  Example: goal steps [40, 95] → steps 0–39 use goal 0, steps 40–94 use goal 1, steps 95+ use the end anchor.
- **Positive:** the prediction from the true (o_t, a_t, g_t). **Negative:** the prediction from a random
  other step of the same episode.
- The loss teaches the encoder (and the predictor) features that tell *which achievement comes next* from
  the current frame and action: a notion of progress through the tech tree.
- Anchors are computed without gradient (they are targets); each distinct goal is encoded once per batch.

**Matching loss (`_match_batches`), across episodes.**
- For a source episode, pick up to 16 other episodes. Embed all their achievements (no gradient).
- For each target episode, solve an **entropic partial optimal transport** problem between the source's
  achievements and the target's (POT library; cost = 1 − cosine similarity, regularisation 0.05, each
  achievement has mass 1, total transported mass = the smaller count). The transport plan pairs up
  achievements that look alike; entries above 0.5 are taken as matches. Partial transport lets
  achievements that exist in only one of the two episodes stay unmatched.
- **Anchor:** a source achievement (with gradient). **Positive:** its match in the target.
  **Negative:** a random achievement of the same target.
- This makes "the same achievement" have the same embedding in different worlds, even though the frames
  around it look different. That consistency is what makes the achievement state g useful to the policy.
- The KL / value terms are evaluated on random steps of the source episode.

### 1.4 `train.py`: the main loop

- `parse_args()`: every hyperparameter is a flag; **the defaults are the best configuration**
  (AD on, `--rarity_bonus 2`, paper hyperparameters). `python train.py` trains it, as the assignment requires.
- Run folder `runs/<method>-s<seed>`; refuses to overwrite a folder with logs unless `--resume`.
- Seeds for Python, NumPy and PyTorch. (Crafter itself is not bit-reproducible across processes: creatures
  are stored in Python sets ordered by memory address.)
- 245 updates × 4,096 = 1,003,520 steps (the smallest whole number of updates ≥ 1M).
- **Rollout loop** (512 steps): `model.act` → `venv.step` → record finished episodes in
  `episodes.jsonl` with the *game* reward → apply the reward shapers (rarity, vitals) → `roll.insert`.
  After the loop, the value of the last observation is stored for bootstrapping.
- **Learn:** `compute_returns`, `algo.update`, `roll.reset`.
- **Log** one line per update to `metrics.jsonl`: cumulative score over all training episodes so far (the
  Crafter protocol), the score over the last 100 episodes, mean reward and length of the episodes that
  ended in this rollout, losses, and all 22 success rates.
- **Checkpoint** every 10 updates (`checkpoint.pt`: model, optimisers, counters, rarity window), so a
  crashed run can `--resume` with the same bonus statistics.
- **End:** `agent.pt`, then `evaluate()` on 100 fresh episodes, all written to `final.json`.

### 1.5 `evaluate.py`

- `evaluate(model, episodes, ...)`: plays fresh worlds (different seeds from training) with the
  stochastic policy and the same achievement-state update as training. Each env plays the same number of
  episodes, because "stop after the first N episodes" would favour short (bad) episodes.
  `model=None` plays uniformly random actions: that is how the random baseline (1.55 score, 1.41 reward,
  1,008 episodes) was measured.
- `record_video`: one episode in-process at 384×384, saved as MP4.

### 1.6 `plot.py`

Reads `metrics.jsonl` / `episodes.jsonl` of several runs, groups seeds by label, and draws the score and
reward curves (mean ± sd) and the per-achievement success rates (log scale).

---

## Part 2 — the research repository

### 2.1 Layout

```
common/        crafter_score.py, crafter_vec_env.py (vector env), video.py (clips, HUD), pixel_map.py (agent's own map)
methods/
  01_ppo_rnn/                 JAX PPO-RNN on Craftax (the official baseline, used to validate the GPU stack)
  02–04, 06, 07               READMEs only: methods surveyed but not run
  05_achievement_distillation/  the main line (all deck results)
     train.py                 runner: rarity bonus, vitals shaping, GRU / map / dream options, logging
     algos.py                 micro-batched PPO / AD updates (subclasses of upstream's)
     fast_aux.py              faster, identical AD auxiliary phase
     recurrent.py             GRU-memory variant (+ test_recurrent.py)
     map_model.py             variant that also sees its own top-down map
     configs/*.yaml           the paper's hyperparameters (ppo, ppo_ad, ppo_ad_gru)
     upstream/                official AD code @ 9abbce6 (git-ignored; cloned by scripts/setup.sh)
  08_world_model/             our world model and "dreaming" (Stage 1 → 2)
paper/         research_survey.md, work_log.md (every experiment, dated), designs/ (world-model design notes),
               deck/ (slide sources), code_guide.md (this file)
results/       small summaries: CSVs, death analyses, plots
runs/          one folder per training run: logs, final weights (Git LFS), videos; runs/report/ = HTML report
scripts/       dashboard, spectator, death diagnosis, plots, report builder, queue scripts, setup
submission/    the compact code to hand in
```

### 2.2 `methods/05_achievement_distillation/` (the code behind the results)

- **`train.py`**: same loop as the submission, but it builds the official AD objects
  (`PPOADModel`, `RolloutStorage`, `sample_rollouts`) from `upstream/` and wraps our env in
  `TorchVecEnv` (adapter to upstream's interface). It holds `RarityBonus` and `VitalsShaping` (same as in
  `submission/envs.py`), video recording, `--resume`, and the options for every experiment:
  `--gamma`, `--vitals`, `--vitals_gamma`, `--pixel_map`, `--config ppo_ad_gru`, `--dream*`.
  It ran 250 updates (1.024M steps).
- **`algos.py`**: `MBPPOAlgorithm` / `MBPPOADAlgorithm` subclass upstream's algorithms and add
  gradient micro-batching (`_step`) for the 4 GB GPU. `goals_from_successes` replaces upstream's goal
  detection when vitals shaping is on (upstream also detected goals from `reward > 0.1`, which shaping breaks).
- **`fast_aux.py`**: the same AD losses with less recomputation: old-policy logits once per phase, each
  distinct goal encoded once, the 16 target episodes encoded in one batch. One aux phase: 934 s → 544 s on
  the 4 GB laptop. `check_equivalence()` compares losses and gradients with upstream.
- **`recurrent.py`**: `PPOADGRUModel` (a GRUCell 1024→512 between encoder and heads), a rollout storage
  that keeps the hidden state per step, and PPO over 64-step chunks with back-propagation through time.
  `test_recurrent.py` checks that replaying chunks reproduces the rollout's log-probabilities.
- **`map_model.py`**: `PPOADMapModel`, input = frame + the agent's own map image (6 channels), the map
  through a second, smaller IMPALA encoder.
- **`upstream/achievement_distillation/`** (official code, read-only): `model/ppo_ad.py` (network and AD
  losses), `algorithm/ppo_ad.py` (`Buffer` that cuts episodes, OT matching, aux loop), `storage.py`
  (rollout storage with GAE and the achievement-state update), `sample.py` (rollout loop),
  `impala_cnn.py`, `torch_util.py` (`FanInInitReLULayer`), `normalize.py`, `action_head.py`, `mse_head.py`.
  The submission's `model.py` and `algo.py` are compact equivalents of these files (same parameter names,
  so the official checkpoints load into `submission/model.py`).

### 2.3 `common/`

- `crafter_vec_env.py`: the research version of `VecCrafter` (several envs per process, an optional
  high-resolution frame of one env for videos, optional pixel map).
- `video.py`: `HUD` (text overlay with step, vitals, achievements), `write_mp4`, `ClipRecorder` (random
  400-step clips during training).
- `pixel_map.py`: the agent's own map. Odometry from pixels: after a move action, it compares the next
  frame with the current frame shifted by one tile ("moved") and unshifted ("blocked"); ties go to a
  walkability table learned online; sleeping (grey screen) means no move. Every seen tile is stored as a
  2×2-pixel thumbnail at its world position; `render()` gives a 64×64 egocentric map. 99% of steps correct.

### 2.4 `scripts/`

- `dashboard.py` + `dashboard.html`: live web dashboard (http://localhost:8765) with curves, success
  rates and clips of running jobs. `spectator.py` films the latest checkpoint of a running job.
- `diagnose_deaths.py`: monkey-patches Crafter's zombie, arrow and player-update code to log the cause of
  every lost health point, plays fresh episodes with final weights, writes `results/deaths/<run>.jsonl`
  (also the evaluation data in the deck) and death videos. `analyze_deaths.py` aggregates them
  (`results/deaths/summary.csv`, plots).
- `compare_configs.py`, `aggregate_seeds.py`, `plot_results.py`: tables and plots across seeds.
- `report_data.py` + `make_report.py` + `report_template.html`: the self-contained HTML report in `runs/report/`.
- `setup.sh` / `setup_windows.ps1`: clone Crafter, Craftax and the AD code at pinned commits, create the venv.
- `queue*.sh` / `queue*.ps1`: run one configuration for seeds 0–2 (in parallel on the big GPU), skip finished
  runs, resume interrupted ones.

### 2.5 `methods/08_world_model/`: the dreaming line (beyond the 1M budget)

A world model is a network that predicts the next frame (and inventory, unlocks, death) from the past
frames and the action. With one, an agent can practise "in its head". Pipeline:
1. `record.py`: our trained agents play ~7,000 episodes; frames, actions, inventories are saved.
2. `tile_tokenizer.py`: each 7×7-pixel game tile → one of 1,024 codes (a 9×9 grid per frame;
   earlier: `tokenizer.py`, a conv VQ-VAE with an 8×8 grid). `cache_tokens.py` stores the codes.
3. `pixel_odometry.py` / `build_maps.py`: positions from pixels and a top-down map memory per step.
4. `dynamics.py`: a causal Transformer over 64 past frames (each frame summarised by a small spatial
   Transformer), with heads for the next frame's tokens (MaskGIT-style decoder), inventory, unlocks and
   death; the map memory is added to the decoder input. 25.6M parameters.
5. `evaluate.py`, `semantic_eval.py`, `creature_eval.py`: how long dreams stay faithful, whether they
   remember terrain, whether they are "easier" than reality (greedy dreams are: too little damage).
6. `craft_layer.py`: a learned crafting rule shared across tool tiers, so the model can dream iron tools
   after seeing only 4 real examples. `craft_check.py` / `iron_diag.py`: diagnostics.
7. `dream_env.py`: used by `train.py --dream`: each update, 256 dreams × 16 steps start from real
   moments (biased to the iron chain) and the policy acts in them; PPO trains on real + dreamed steps.
   Scores 26–28, but the world model was trained on 1.76M steps of other agents' play, so this is not a
   valid 1M result and the deck reports it as an exploration.

The design notes in `paper/designs/` explain each stage's intent and pre-registered predictions, and
`paper/work_log.md` has the dated results.

---

## Part 3 — likely questions, with answers

**Why PPO and not DQN?** PPO is on-policy and stable with large CNNs and parallel environments, and it is
the base of the strongest model-free Crafter results (AD). Crafter's reward is sparse and the
interesting behaviour is long chains of actions; a stochastic policy explores them naturally, and PPO
needs no replay buffer of images.

**What does the clipping in PPO do?** The surrogate `ratio · A` would push the probability of good actions
up without limit. Clipping the ratio at 1 ± 0.2 makes the objective flat once the new policy has moved
20% away from the data-collecting policy for that sample, so several epochs on the same data stay safe.

**What is GAE and why λ = 0.65?** It mixes n-step returns of all lengths with weights λ^n. λ trades
bias (small λ: relies on the value function) for variance (large λ: relies on actual rewards). 0.65 and
γ = 0.95 are AD's tuned values. We tried γ = 0.99: worse on every seed.

**Why normalise value targets?** Returns change scale a lot during training; the critic learns
normalised targets, so the value loss (added to the policy loss) keeps a stable size.

**What is "Achievement Distillation" in one sentence?** An auxiliary phase that trains the encoder so
that each achievement has a consistent embedding (matching across episodes) and so that the current
frame and action predict the next achievement (prediction within an episode); the policy also receives the
embedding of the last achievement as input.

**Why optimal transport for matching?** Two episodes unlock achievements in different orders and different
numbers. OT finds the best one-to-one pairing between the two sets by similarity; the *partial* version
lets achievements without a counterpart stay unmatched. Then matched pairs are pulled together.

**Why the KL and value terms in the aux phase?** The aux losses change the shared encoder. Without a
constraint, that would also change the policy and value outputs, undoing PPO's work. The KL keeps π close
to what it was before the phase, the value term keeps V fitting the returns.

**Is the rarity bonus "reward hacking" or cheating?** No: it only changes the reward used for learning,
from information the agent itself observes (its own achievement statistics). It does not touch the
environment, and all reported rewards and scores are the game's. Unlike potential-based shaping, it *does*
change the optimal policy on purpose: it values rare achievements more, which is what the score measures.

**Why does the bonus fade?** `p_i` is measured on the agent's recent episodes. Once an achievement is
common, `1 − p_i` is small and it is worth ~+1 again. So it acts as an automatic curriculum.

**Why β = 2?** We measured β = 0, 1, 2, 4 on PPO with 3 seeds each: 15.75, 18.58, 20.99, 20.14. β = 2 is
the best and the most consistent across seeds.

**Is the improvement significant?** On PPO, β = 2 beats the baseline on all 3 seeds (paired differences
+8.4, +3.5, +3.7; t = 3.26, one-sided p ≈ 0.04). On AD, 3 seeds of ours vs 1 seed of plain AD, so the
cleanest evidence is the PPO comparison; ours is also above the published AD (21.79 ± 1.37 over 10 seeds).

**Why do agents never get diamonds?** A diamond needs an iron pickaxe, which needs a table *and* a furnace
next to the player plus wood, coal and iron. Agents rarely bring the two stations together (2% of the
moments when they hold iron) and they die young: all evaluation episodes end in death, mostly from zombies
at night and empty vitals.

**What did not work and why?** γ = 0.99 (the bonus becomes a smaller part of a longer return; more noise),
a GRU memory (more parameters to learn in 1M steps; agents already see water but don't go back), survival
shaping (the γΦ' − Φ form adds a per-step cost; the no-drift form helps survival but slows the tech tree),
the agent's own map (the information is there, the incentive to use it in time is not).

**What is the difference between training and evaluation score?** Training score (Crafter's protocol)
averages over *all* episodes during learning, including the early bad ones. Evaluation uses only the final
weights on fresh worlds, so it is higher (33.4 for ours).

**How do you know the compact code is the same algorithm?** We loaded a trained checkpoint into both the
official AD model and our `model.py`: the outputs are identical (difference 0.0). PPO and prediction
losses and gradients agree to 1e-7, the matching loss to 6e-5 (floating-point rounding in a different
batching).

---

## Glossary

- **Rollout:** a block of experience collected with a fixed policy (here 8 × 512 steps).
- **Advantage A:** how much better an action was than the value function expected.
- **Entropy bonus:** rewards a spread-out policy, to keep exploring.
- **Contrastive loss:** a classification loss that tells a positive pair from negatives; here with one negative.
- **FiLM:** feature-wise linear modulation; the action scales and shifts the latent features.
- **GroupNorm / LayerNorm:** normalisation layers that work per sample (no batch statistics).
- **Potential-based shaping:** adding γΦ(s') − Φ(s) to the reward; it does not change the optimal policy.
- **Seed:** the random seed of a run; different seeds give different worlds and initial weights.
