# Design note: our own world model, Stage 1 ("dream vs reality")

_Draft 2026-10-22, for review before any code. Open questions are marked **Q**._

## Goal
Learn a model of Crafter that, given what the agent has seen and the actions it takes, predicts what it will see next. Stage 1 trains **no agent**. We only find out how well we can make Crafter "dream", and where the dream breaks. Stage 2 (practise inside the dream) is worth building only if Stage 1 passes the bar at the end of this note.

## Why
- Our best model-free agent (AD + rarity β=2, 23.65) makes an iron pickaxe in ~0.01% of episodes.
- EMERALD, a world-model agent, does it in 41.5%, at 10× our step budget.
- A world model lets the agent rehearse the long chain (stone pickaxe → iron + coal + furnace → iron pickaxe → diamond) without having to stumble on it for real.

## What we build (our design, not a copy)
1. **Data.** About 2,000 episodes (~500k frames) recorded from agents we already have: the final weights of AD + rarity β=2 and of the no-drift survival agents, plus earlier checkpoints for weaker, more varied play. Each step stores the 64×64 frame, the action, the reward, the inventory (16 counts), the achievements and death.
2. **Tokenizer.** A small VQ autoencoder turns each frame into an **8×8 grid of discrete codes** (codebook of 512), about 5M parameters. **Q1:** 8×8 (finer, close to Crafter's tile grid) or 4×4 (EMERALD's choice, cheaper)?
3. **Dynamics model.** A Transformer reads the last **32 steps** (token grids plus actions) and predicts:
   - the next 8×8 grid, all tokens at once, with optional refinement passes;
   - **the next inventory as its own output** (16 counts);
   - reward and death.

   About 20M parameters. **Q2:** a 32-step memory, or 64 (EMERALD's) at twice the cost?
4. **Our twist, testable already in Stage 1: rarity-weighted training.** Sequences that contain rare events (stone pickaxe, iron, furnace, iron tools) are sampled 4–8× more often during training. We compare against uniform sampling. If rare transitions are predicted much better, that's the first evidence for "rarity-guided dreaming" in Stage 2. **Q3:** what weight, and which events count as rare?

The separate inventory output is a deliberate difference from EMERALD. Inventory *is* the crafting state, and Stage 2 needs it to compute our rarity bonus and survival signal inside the dream.

## How we test it
- **Dream-vs-reality videos.** Start from a real moment, feed the real actions, and play reality | dream | difference side by side for 100 steps.
- **Accuracy against horizon** (1, 5, 15, 50 and 100 steps): token accuracy, pixel error, exact inventory accuracy, and achievement/death prediction (precision and recall).
- **Four targeted probes:**
  1. **Night:** does the dream darken on the right step?
  2. **Off-screen memory:** walk away from water and come back; is the water still there?
  3. **Crafting:** after a table and a pickaxe, does the inventory update correctly?
  4. **Rare chain:** given the real actions that made an iron pickaxe, does the dream make one too?

## What we expect (written down before running)
- **P1 (short dreams):** at 1 step, token accuracy above 95% and inventory above 98%; by 50 steps the frame blurs or drifts.
- **P2 (inventory):** inventory stays right longer than the picture does, because it changes rarely and only on known actions.
- **P3 (memory):** off-screen memory works within the 32-step context and fails beyond it.
- **P4 (night):** night timing is learned, since it is a regular 300-step cycle.
- **P5 (rarity weighting):** with uniform sampling the iron-pickaxe probe fails (too few examples); rarity weighting fixes it without hurting common transitions by more than a few points.

## Bar for going to Stage 2
- At 15 steps, the imagination horizon we would use: inventory accuracy ≥ 90%, and dreams that still look plausible in the videos.
- Achievement recall ≥ 70% at 1 step.
- If we miss the bar, we fix the model first, or stop and say why.

## Not in Stage 1
Training an agent in the dream (Stage 2), planning, online data collection, any score claims.

## Cost
About one day of code. GPU time: ~1 h recording data, ~1 h tokenizer, ~3–4 h dynamics (each sampling variant), ~1 h evaluation and videos on the RTX 5090. All results, videos and the decisions on Q1–Q3 go into the work log and a new report section.

**Q4 (yours):** what would convince *you* that the dream is good enough? Name one probe or video you would want to see before Stage 2.

---

## Decisions (2026-10-22)
- **Q1:** 8×8 token grid (user).
- **Q2:** 64-step context (user). It covers the length of a typical nightfall-to-dawn stretch (~90 steps) only partly, so P3 now predicts off-screen memory up to 64 steps.
- **Q3** (provisional, open to change): "rare" is decided from the recorded data. Any achievement unlocked in fewer than 10% of episodes counts as rare. Training sequences that contain such an unlock are sampled 5× more often. Both a uniform and a rarity-weighted model are trained, since the comparison is the experiment.
- **Q4**: the user decides go/no-go for Stage 2 after seeing the four probes and the dream-vs-reality videos.
- **Data honesty:** Stage 1 data come from agents that each trained for 1M steps, which is fine for studying the model. A Stage 2 agent must learn from its own 1M environment steps only, or its score is not comparable to the 1M protocol.
- **Data mix:** about 4,000 episodes (~1M frames) from these checkpoints:
  - final weights of AD + rarity β=2 (s0–s2) and no-drift survival (s0–s2);
  - epochs 50/100/150 of the 5090 runs, for weaker and more varied play;
  - final weights of PPO + rarity β=2 and plain AD.
  - Stochastic policy throughout. 10% of episodes held out for testing.
- **Q3 revised after the data came in:** with a 10% threshold, the rare set was only diamond, eat plant and the iron tools. Each is unlocked in ≤0.1% of the recorded episodes (1–2 examples), and those windows were 0.1% of all training windows, so 5× weighting would change almost nothing. The threshold is now **30%**, which adds **collect iron** (27% of episodes), the gateway to iron tools and diamond. It is still decided from the data.

---

## Results (2026-10-23): does not yet meet the Stage 2 bar
200 held-out dreams per model, driven by real actions (`results/world_model/stage1_eval.json`, report section "Our own world model, Stage 1").

| steps ahead | tokens right (copy) | items exact (copy) | PSNR (copy / ceiling) |
|---|---|---|---|
| 1 | 74% (62%) | 96% (81%) | 28.5 dB (24.5 / 31.2) |
| 5 | 51% (30%) | 91% (54%) | 23.5 (18.5 / 31.2) |
| 15 | 30% (15%) | 56% (32%) | 18.7 (15.7 / 30.9) |
| 50 | 12% (8%) | 14% (14%) | 15.5 (13.8 / 29.9) |

- **P1 fails:** 74% of tokens right at 1 step, not ≥95%. Inventory, all 16 slots, is exact in 90% of cases.
- **P2 holds only for materials and tools.** Vitals change on hidden timers.
- **P3 fails:** off-screen water comes back in only 15% of dreams (median gap 14 steps; detector agrees with labels in 97%).
- **P4 is weak:** night brightness correlation 0.42 (rarity model 0.47).
- **P5 is not supported:**
  - Collect-iron recall: uniform 39%, rarity 36%; precision 49% → 66%.
  - Only one iron-pickaxe event in the test set.
- **Bar:** items exact at 15 steps is 56% (needed ≥90%). One-step unlock recall is 81% (needed ≥70%) ✓, with crafting at 95–100% and creatures at 21–44%.
- **Also found:** the inventory/unlock heads overfit after ~8k of 30k steps.
- **Options:**
  - **(a) Stage 1b**, fixing the weaknesses: creature-aware tokenizer, early stopping for the heads, a memory training signal.
  - **(b) Stage 2 with 5-step imagination,** where the model is already good.
  - The user decides (Q4).
