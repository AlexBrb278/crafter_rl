# Design note: world model Stage 1b, fixing what Stage 1 got wrong

_Draft 2026-10-23, for review before any code. Builds on `world_model_stage1.md`. Open questions are marked **Q**._

## Where Stage 1 left us
- The dream is good for ~5 steps (materials and tools exact in 91% of dreams), drifts by 15 (56%), and is lost by 50.
- Crafting is predicted almost perfectly.
- Three specific failures, plus drift:
  1. **Creatures disappear.** The tokenizer drops cows, zombies and saplings, so the dynamics model cannot predict zombie kills (35% recall) or eating cows (40%).
  2. **The small outputs overfit.** Inventory and unlock losses on test data are best at ~8k of 30k steps, then get worse.
  3. **No off-screen memory.** Water comes back in only 15% of dreams after a median of 14 steps away, although 14 steps is well inside the 64-step context. So capacity is not the problem; nothing in training *forces* the model to use the past, because the next frame is almost entirely predictable from the current one.
  4. **Drift.** The model is trained only on real inputs (teacher forcing), but dreams feed it its own predictions. Small errors pile up, and it has never practised recovering from them.

## The fixes (one per failure)
**Fix 1: creature-aware tokenizer.** Reconstruction errors on creatures are few pixels but matter most.
- **Main version:** weight the pixel loss by its own error ("hard-pixel" L1: the 10% worst pixels of each frame count 5×). It needs no extra information.
- **Diagnostic version:** weight the pixels of creature and sapling tiles 10×, using the game's semantic map. It shows the best a loss change can do.
- **Q5:** the semantic map is information the agent never sees. Using it to train the *tokenizer* is fine for Stage 1, but a Stage 2 agent learning from its own 1M steps could not use it. Keep it as a diagnostic only (my pick), or allow it?

**Fix 2: stop the small outputs at their best.** Train the whole model as before. Then freeze it and re-fit only the inventory, unlock and death outputs on its features, with early stopping on held-out data. The frame predictor keeps its 30k steps of training; the small outputs keep their best point.

**Fix 3: a training signal that needs memory.** Add a *skip-ahead* output. From the state at step t, plus the actions the agent takes next, predict the token grid **k steps later**, with k drawn from {4, 8, 16, 32}. To predict that a lake reappears 16 steps from now, the model must remember the lake. The next-frame objective never asks for this.

**Fix 4: practise on its own dreams.** For a quarter of the training windows, dream the last 8 steps from the model's own predictions instead of the real tokens, and train on the result ("unrolled" training). This directly targets drift at 5–15 steps.

**Fix 5: tell it the time.** Crafter's night follows a fixed 300-step clock from the episode start. Stage 1 never told the model how far into the episode a window starts, so it could only guess the time from brightness. Add the episode step (as its phase in the 300-step cycle) to every input. An agent can count its own steps, so this is not privileged information.
- **Q6:** allow this input?

**Dropped:** rarity weighting. It did not help in Stage 1 and adds a variable. It can come back in Stage 2, where it can steer *where the agent dreams*.

## How we test it
Exactly the Stage 1 evaluation: same 392 held-out episodes, same 200 dream starts, same probes and baselines. That makes every number directly comparable. Two new measurements:
- **Creature-tile PSNR:** reconstruction quality on tiles that hold a creature, from the semantic map; used for measuring only.
- **Skip-ahead accuracy at k = 16:** the memory probe, scored directly.

**Ablation:** one model with all fixes, plus one without Fix 3, so we can tell whether the memory signal does anything on its own. **Q7:** also an ablation without Fix 4 (+4 h)?

## What we expect (written down before running)
- **P6 (creatures):** creature-tile PSNR improves by ≥3 dB, and zombie-kill recall rises from 35% to ≥60%.
- **P7 (small outputs):** test inventory loss ≤0.045, and items exact at 5 steps ≥93% (from 91%).
- **P8 (memory):** water comes back in ≥50% of dreams (from 15%) with Fix 3, and stays below 25% in the ablation without it.
- **P9 (drift):** items exact at 15 steps ≥70% (from 56%).
- **P10 (night):** night brightness correlation ≥0.8 (from 0.42).

## Bar
- **Stage 2 with 15-step dreams:** unchanged bar, items exact ≥90% at 15 steps.
- **Lower bar, Stage 2 with 5-step dreams:** items exact ≥90% at 5 steps (Stage 1 is already at 91%) *and* P8 holds. A short dream is only useful if the model also remembers what it saw.

## Cost
- About 1 day of code.
- GPU: tokenizer v2 ~1.5 h, plus the diagnostic tokenizer in parallel. Dynamics v2 ~5 h for the full model and the no-memory ablation in parallel (the extra losses cost ~20%). Head re-fit and evaluation ~1 h.
- About 8 h of GPU in total, run unattended. Same 3,940-episode dataset; no new recording.

---

## Decisions (2026-10-25)
- **Q5:** the semantic creature map is used only in the diagnostic tokenizer (`tokenizer_sem`), never in the main pipeline.
- **Q6:** episode clock input, yes.
- **Q7:** no "without unrolled training" ablation for now (GPU time). The ablation is still "without the memory signal" (`v2_nomem`).
- **More data:** recording plan B adds 3,990 episodes with new seeds and agents not used before:
  - GRU (final and epoch 150);
  - survival v1 (final and epoch 100);
  - AD + rarity β=2 and no-drift survival (new seeds);
  - PPO γ 0.99, the PPO rerun, plain PPO, PPO β=1/4, AD β=1;
  - two early checkpoints.
- **Dataset:** ~7,930 episodes. The Stage 1 test set (392 episodes) stays fixed; new held-out episodes form a second test set. 5% of the training episodes are held out as "val" for the head re-fit.
- **Pipeline:** `methods/08_world_model/run_stage1b.sh`.

---

## Results (2026-10-25): creatures and short dreams fixed, memory not yet
Same 392 test episodes and 200 dream starts as Stage 1 (`results/world_model/stage1b_eval.json`; PSNR from mean MSE, Stage 1 re-scored).

| measure | Stage 1 | **1b (v2, re-fit)** | 1b no memory | target |
|---|---|---|---|---|
| tokenizer creature tiles | 15.7 dB | **19.5 dB** | same | +3 dB ✅ |
| zombie kill recall | 35% | 54% | 57% | ≥60% ❌ |
| cow / iron recall | 40% / 39% | 74% / **92%** | 71% / 83% | – |
| test inventory loss | 0.054 | **0.028** | 0.029 | ≤0.045 ✅ |
| items exact @5 | 90.5% | **93.0%** | 95.5% | ≥93% ✅ |
| items exact @15 | 55.5% | 62.0% | 60.0% | ≥70% ❌ |
| night correlation | 0.42 | 0.66 | 0.57 | ≥0.80 ❌ |
| water back in dream | 14.5% | 17.0% | 17.0% | ≥50% ❌ (ablation: no effect) |
| water via skip-ahead output | – | 34% | – | ≥50% ❌ |

- The re-fit chose step 3,200 for both models. It mainly raised unlock precision (60% → 73%) at a small cost in recall (89% → 85%).
- **Bar:** 15-step dreams are not ready. The 5-step bar is met on accuracy (93%), but not on memory.
- **Memory is now the main gap.** The skip-ahead signal did not transfer into the dreams.
- **Hypothesis for why memory fails:** the model is never given the agent's position, so it must infer how far it walked from pixels.
- **Next:** a Stage 1c with an explicit top-down map memory positioned by the agent's moves, or Stage 2 with 5-step dreams.
- **Correction:** a thumbnail first suggested Stage 1b kept the scene on one start where Stage 1 drifted. A larger frame showed both lose the lake within ~15 steps. The report caption says so.
