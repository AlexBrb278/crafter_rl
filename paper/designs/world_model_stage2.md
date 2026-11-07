# Design note: world model Stage 2, practising in the dream

_Draft 2026-10-29, for review before any code. Open questions are marked **Q**._

## Why
- **The wall is the iron chain, not survival or memory.** Every change on top of AD + rarity β=2 (23.65) so far has made the agent safer or better informed but no better at crafting:
  - GRU memory, survival signals, own map, map + survival: all score lower (22.3 → 20.5).
  - All of them reach iron less often (7.4% → 1.9–5.2%).
  - Iron pickaxe stays at ~0.01%, diamond at ~0.
- **The chain is long and rare in real play.** Stone pickaxe → find iron + coal → table + furnace nearby → iron pickaxe → find a diamond. A real episode reaches the iron step in about 7% of games. The agent gets too few real examples to learn the later steps within 1M steps.
- **The world model is good at exactly this part.**
  - Crafting and inventory are predicted almost exactly: items exact 97% at 5 steps; unlock recall ~90%, precision 70% (v4).
  - Terrain is remembered: map memory, water 60%.
- **The world model is bad at long dreams**: terrain 63% right at step 50, and creatures drift. Those are fine if dreams are short.
- **Since 28 Oct we know how to keep the dream honest about danger:** sample (T = 1) instead of greedy, and stay at ≤ 15 steps. Greedy dreams are the easier world (a quarter of the real damage at night) and must not be used for training.

## Idea: Dyna-style PPO, dreams as extra practice from real moments
Keep our best agent and its real play. Add short dreamed rollouts that start from real states, mainly states close to the next rare step of the chain.

1. **Real play stays exactly as now.** 8 envs × 512 steps per epoch, AD + rarity β=2, pixels in. The 1M real steps and the score are measured exactly as before.
2. **Dream envs.** A `DreamVecEnv` with the same interface as `CrafterVecEnv`.
   - Each dream env is reset to a real state from the last epochs' rollouts: that frame, plus its 64-step context, inventory and map.
   - It then steps the world model with the **current policy's** actions, sampled at T = 1.
   - Each step returns the decoded 64×64 frame, a reward and a done flag.
   - After H = 15 steps the dream is cut off (a *truncation*, bootstrapped with the value head, not a terminal) and the env resets to a new real state.
   - Because the policy acts in the dream, the rollouts are on-policy and PPO's update needs no change.
3. **Reward in the dream**, the same reward as in real play: +1 for a first unlock this episode, 0.1 × change in health, plus the rarity bonus.
   - Unlocks come from the unlock head (sampled Bernoulli). An unlock only counts if the inventory output agrees where the achievement has an inventory trace: a collect or make achievement needs that item's count to rise. This guards against the 30% false unlocks (**Q12**).
   - The episode's already-unlocked set carries over from the real start state, so the dream cannot re-earn old achievements.
4. **Rarity-guided starts, our twist.** Half of the dream starts are sampled uniformly from the real buffer. The other half come from **frontier states**: the agent has the item that unlocks the next rare step but not the step itself. Examples: a stone pickaxe but no iron; iron and coal but no iron pickaxe; an iron pickaxe but no diamond. Most practice then goes where real experience is thinnest (**Q13**).
5. **What AD's auxiliary phase sees.** AD's goal matching and its distillation use **real episodes only**. Dreamed transitions enter only the PPO policy and value loss, so a false dreamed unlock can never become an AD goal state.

The agent does not need its own map for this: it is plain AD + rarity β=2 (the best agent), and the world model keeps its own map internally (**Q14**).

## Why not EMERALD's / DreamerV3's agent
- An actor-critic on world-model latents, trained only in imagination, is the established recipe. We take its proven settings: 15-step imagination, sampled latents, value bootstrap at the cut, continue-weighting through the death head.
- **We don't switch agents:**
  - At 1M steps EMERALD scores ≈ 3%. Its strength shows at 10M.
  - Replacing the agent *and* adding imagination changes two things at once.
  - Dyna-style keeps the 23.65 agent as the floor: real play is untouched.
- **Fallback:** if Dyna-style dreaming helps but the decoded-pixel gap hurts, a latent actor-critic is the next step.

## The 1M rule: two stages
- **The world model is trained on someone else's experience.** Our world model (v4) was trained on 6,753 recorded episodes from agents that each trained for 1M steps. An agent using it has indirectly seen far more than 1M steps, so its score is **not comparable** to the benchmark.
- **Stage 2a, diagnostic (not comparable):** the frozen v4 world model and the agent's own 1M real steps. It answers one question cheaply: *can practice in our dreams teach the iron chain at all?* If it can't, even with a world model trained on 1.7M frames of good play, the honest version won't either, and we stop here.
- **Stage 2b, honest:** the world model is learned from the agent's own steps.
  - The first ~200k steps are real-only and fill a replay buffer. Then the tile tokenizer (4 min) and dynamics are trained on it, and dreaming starts.
  - The world model keeps training on the growing buffer alongside the agent, as EMERALD does.
  - Its score is comparable to every other number in the report.

## How we test it
- **Same protocol as every other config:** 3 seeds, 1M real steps, the same flags as AD + rarity β=2. Death diagnosis on 600 fresh episodes, videos, a report section.
- **Dream health checks during training,** logged every epoch:
  - **dreamed unlock rate per step vs the real unlock rate** at the same kind of state. A dream that pays out much more often than reality is being exploited.
  - Share of dream rewards rejected by the inventory check.
  - Dreamed health lost per step vs real.
- **Videos:** real clips as now, plus a few dream clips (decoded frames, with the dream's reward shown) at checkpoints.

## What we expect (written down before running)
- **P15 (the reason to build it):** in Stage 2a, collect iron in training episodes rises from 7.4% to **≥ 15%**, and make iron pickaxe from 0.01% to **≥ 0.5%**.
- **P16:** the score does not fall below 23.65 − 1 sd (22.6), because real play is unchanged and dreams add practice only.
- **P17 (risk, model exploitation):** without the inventory check the agent learns actions that make the *dream* pay (fake unlocks), visible as a dreamed unlock rate ≫ real. With the check this rate stays within 2× of real.
- **P18:** deaths do not rise, because sampled dreams are as dangerous as reality up to 15 steps.
- **Bar for Stage 2b:** P15 holds in 2a. If not, we report a negative result and stop.

## Open questions
- **Q11:** 2a first (diagnostic, fast, not comparable), or go straight to the honest 2b? My pick: 2a first. It costs one 3-seed run and can stop us early.
- **Q12:** reward from the unlock head gated by the inventory output (my pick), or from inventory changes alone? The second can't express `defeat_zombie` or `wake_up`.
- **Q13:** how many dreamed steps per real step: 1:1 (my pick, 4,096 dreamed steps per epoch) or more? And 50% frontier starts, or more?
- **Q14:** plain AD + rarity β=2 (my pick), or the map agent? The map agent would need its map rebuilt from decoded dream frames.
- **Q15:** should the survival signal (no drift) also be paid in dreams? My pick: no, to keep one change at a time.

## Cost
- **Code:** about 2 days.
  - Batched dreaming: many dreams in parallel, each with its own map. Today's dreams run one at a time at ~280 steps/s.
  - `DreamVecEnv`, truncation handling in the rollout, the reward gate, frontier sampling, the dream health logs and dream videos.
- **First measurement before any training:** batched dream speed. A 1:1 ratio needs ~4k dreamed steps per epoch. If that costs more than the ~20 s of a real epoch (11 s rollout + 8 s update in today's logs), a cache of past keys and values (EMERALD's KV cache) comes first.
- **GPU:** Stage 2a is three seeds in parallel, likely 10–14 h instead of 8, since the world model shares the GPU. Stage 2b adds online world-model training, so about one more day.

---

## Decisions (2026-10-29)
- **Q11:** Stage 2a first, with the frozen v4 world model (diagnostic, not 1M-comparable).
- **Q12:** dream reward from the sampled unlock head, gated by the inventory output.
- **Q13:** 1:1, 4,096 dreamed steps per epoch (256 dreams × 16 steps), 50% frontier starts. Frontier = the episode has unlocked make_stone_pickaxe but not collect_diamond.
- **Q14:** plain AD + rarity β=2.
- **Q15:** no survival signal in dreams.
- **Implementation choices:**
  - Real and dreamed samples share every PPO minibatch (half each). AD's buffer and auxiliary phase get real data only.
  - A dreamed death resets that dream slot to a new real start; the death cuts the return (mask 0). The end of the 16 steps is bootstrapped with the value head, as at the end of every real rollout.
  - Dream starts need a full 64-step real context, so they come from real states at episode step ≥ 63.
  - Positions for the world model's map come from the online pixel odometry (`common/pixel_map.py`), not the offline one it was trained with (99.0–99.4% vs 99.77% of steps right).

## Changes after the smoke tests (2026-10-29; details in the work log)
- **Q12, changed:** the plain gate ("head, paid only if the inventory agrees") almost never paid easy unlocks late in an episode. The unlock head learned from competent agents, who take them in their first steps, and it cannot see the episode's unlocked set. Collect and make unlocks are now read off the dreamed inventory instead (count rises, not yet unlocked). Eat, place and drink use the head checked against the inventory; defeat and wake-up use the head alone.
- **Sampling:** tokens, vitals and saplings are sampled at T = 1; other item counts take their most likely value. Sampling them produced phantom diamonds from noise.
- **Calibration:** the unlock and death heads' odds are divided by 10 (their training `pos_weight`) before sampling.
- **Result of the fixes:** the dream now pays at most the real rate for every achievement (P17 check). The head-based ones are still under-paid 2–10×.

## Stage 2a result (2026-10-29, details in the work log)
- **Score 26.10 ± 0.75 vs 23.65 ± 1.05.** Collect iron 15.4% vs 7.4%, iron pickaxe 0.09% vs 0.01%. Not 1M-comparable.
- **P15:** iron part met, iron pickaxe part missed. **P16, P17:** hold. **P18:** mixed. More are alive at step 300, fewer awake-zombie deaths, but more thirst and asleep deaths, and wake_up drops to 44%.
- **The dream never paid an iron tool or a diamond.** It teaches the way to iron, not the last crafts.
- **Before Stage 2b, check:**
  1. Can v4 dream the iron-pickaxe craft from a real moment just before one? If not, Stage 2b inherits the same ceiling.
  2. Pay wake_up and drink correctly in dreams (calibrated against real rates), to win back the sleep loss.

## Check before Stage 2b: can v4 dream the iron crafts? (2026-10-30)
- **No.** On 41 held-out iron-pickaxe and 47 iron-sword crafts, with real actions, 0 dreams show the tool (controls: coal, iron and stone pickaxe 55–99%). v4 saw 4 iron pickaxes and no iron sword in 1.76M steps.
- **Consequence for 2b:** the agent's own 1M steps hold 0–5 iron pickaxes, so a world model learned from them has the same ceiling.
- **Next design question (Q16):**
  - (a) A shared craft rule in the inventory output (action → item deltas), so common tools teach rare ones.
  - (b) Oversample rare crafts.
  - (c) Hand-code the crafting rules in the dream (not a learned model; changes the claim).

## Iron-in-hand dream starts (launched 2026-11-01 23:22)
- **Why:** with the craft layer the dream *can* make iron tools, but in the craft-layer rerun it paid them at the real rate (0.002–0.012 vs 0.004–0.008 per 1,000 steps). Frontier starts ("stone pickaxe made") are too far from the craft: iron, coal, wood, a table and a furnace must all come together within 16 dreamed steps.
- **Change (one):** a quarter of all dream starts (half of the frontier share) come from real moments with iron in hand and an iron tool missing (`dream_env.iron_in_hand`, `train.py --dream_iron 0.25`). Uniform starts stay at half. Everything else as in the craft-layer rerun.
- **P23:** dreamed iron-pickaxe payouts rise at least 5× over the craft-layer rerun; make iron pickaxe in training episodes ≥ 0.3% (was 0.11%); score ≥ 25.35.
- Run: `scripts/queue_dream.sh ad-rarity2-dream-iron --dream_craft tied --dream_iron 0.25`, 3 seeds, about 7 h.
- **Result (finished 2026-11-02 05:52):** score 26.11 ± 0.93 (craft-layer rerun 26.98 ± 0.76). Make iron pickaxe 0.23% (seeds 0.02 / 0.24 / 0.41; was 0.11%), iron sword 0.37% (was 0.18%), one diamond (seed 1). Dreamed iron pickaxes 52 in total (was 22), 2.4×. Iron-in-hand starts 24–29% of all starts. **P23:** score part holds; the 5× dream part and the 0.3% real part are missed. Cost: wake_up 33% (was 43%).

## Ready-to-craft dream starts (launched 2026-11-02 15:34)
- **Why:** iron-in-hand starts rarely pay, because the table and furnace are rarely both next to the player (`iron_diag.py`; work log 2026-11-02). Checked before training: starts with wood and coal in hand and both stations within 4 tiles end in an iron tool in 6.4% of 16-step dreams (real game 6.7%), against 0.14% for iron-in-hand starts in general. 32-step dreams add only about 1 point, so dreams stay at 16 steps.
- **Change:** `dream_env.ready_to_craft`: iron in hand, wood ≥ 1, coal ≥ 1, and a table code and a furnace code (the craft layer's `station` codes) both in the 9 × 7 map rows of the start frame's tokens. `--dream_ready` is the part of `--dream_iron` started there; it falls back to iron-in-hand, then frontier, then uniform starts when there are none. On the held-out probe episodes, the rule is true on 8.3% of iron-in-hand frames and at 88 of 89 real iron crafts.
- **Expected effect:** more pressing of make-iron at the right moment (P24: iron pickaxe and sword rates above the iron-start run, `ad-rarity2-dream-iron`). It does not teach bringing the table and furnace together, which stays the main real-game bottleneck (2% of iron-in-hand steps).
- **Run:** `scripts/queue_dream.sh ad-rarity2-dream-ready --dream_craft tied --dream_iron 0.25 --dream_ready 0.15`, 3 seeds. Diagnostic, not 1M-comparable (v4 was trained on other agents' data).
- **Result (2026-11-02 22:08):** P24 holds on the mean: iron pickaxe 0.66% (0.23% with iron-in-hand starts), iron sword 0.93% (0.37%), dreamed iron pickaxes 752 (52). Seed 1 found iron late and never took off (0.00%). The score did not change (26.25 ± 2.23). Details in the work log.

## Proposed: craftable starts plus a start pool across epochs (for review, not built)
**The problem is not what it looked like.** Seed 1 of `ad-rarity2-dream-ready` did not mainly lack ready starts. It got about 3,100 of them, against 4,500–4,900 in seeds 0 and 2. But it turned only 0.7 per 100 into a dreamed iron tool, against 16–24. With its final weights (`iron_diag.py --near 4`, `results/world_model/iron_diag_ready_run.json`):

| seed | iron tool made, 16 real steps from a ready moment | presses make-iron when the craft is possible | reaches a craftable step |
|---|---|---|---|
| 0 | 39% | 57% | 42% |
| 1 | 4% | 8% | 14% |
| 2 | 38% | 69% | 40% |

Seeds 0 and 2 caught fire: the first dreamed crafts raised p(make iron) at the right moment, which brought more crafts. Seed 1 found iron late (epoch 92, against 59 and 63), and its few rewarded presses were not enough to start that loop before training ended. A late finder also gets fewer ready moments per epoch: 33 of seed 1's last 100 epochs had none.

**Part A — craftable starts (`--dream_reach`, a part of `--dream_ready`).** Start some dreams where the craft is possible right now: iron, wood and coal in hand, with a table code and a furnace code both in the 3 × 3 around the player. This is the same window the craft rule uses (`tokens[2:5, 3:6]`). Pressing make-iron then pays within a step or two, so even a policy at 8% per step gets rewarded. From ready starts, the policy has to walk to the stations first, then press, inside 16 steps. Proposed shares: frontier 0.5 ⊃ iron 0.25 ⊃ ready 0.15 ⊃ reach 0.05 (13 dreams per epoch).

**Part B — a start pool across epochs, for the rare tiers (ready and reach).**
- **Why:** craftable moments are about 10× rarer than ready ones. Roughly 1–3 per epoch for a good seed and about 1 for seed 1 (4,096 real steps × 20% iron-in-hand × 0.14–0.45% craftable). Without a pool the reach tier would be empty in most epochs.
- **What it keeps:** the most recent 1,024 ready or reach moments from up to the last 25 epochs. Each entry holds:
  - the `(track, t)` reference (the Track stays alive; the world model needs its whole map, and episodes are a few hundred KB);
  - the storage row: obs as uint8, successes, timesteps;
  - the two frames behind the AD state.
- **AD state:** recomputed with the current encoder when a dream starts, so a 25-epoch-old state is not stale.
- **How starts are drawn:** the rare tiers sample from this epoch's candidates plus the pool. The other tiers stay as now.
- **Memory:** about 30 MB of storage rows, plus the Tracks the pool keeps alive (about 100–300 MB of CPU RAM).

**Validation before any training (cheap, ~5 min):** add `--reach` to `iron_diag.py` (starts with both stations in the 3 × 3) and run it on the final weights of all three seeds. Go if seed 1 makes an iron tool from at least 20% of craftable starts, real and dream alike, against 4% from ready starts. If seed 1 doesn't press there either, Part A won't work, and the issue is exploration of the make action itself. That case needs a different design, for example a small entropy floor on make actions in dreams.

**Risks:**
- **Few distinct moments, dreamed many times:** about 50 craftable moments over 25 epochs, each started repeatedly. Sampling at T = 1 makes the dreams differ, and the reach share stays small (5%).
- **Over-pressing:** the policy may press make-iron too eagerly in the real game (6.4% of all actions already). Watch `make_iron_pressed` and the death rate.
- **Seed noise:** with 3 seeds, one late finder can hide or fake an effect. Rerunning seeds 0–2 keeps the comparison paired; 5 seeds would be safer, at about 11 h.

**What it does not fix:** bringing the table and furnace together in the real game when no ready moment exists (2% of iron-in-hand steps). The ready starts did help a little here: craftable within 16 steps rose to 40% in seeds 0 and 2, from 14% before.

**P25 (if built):**
1. Every seed reaches at least 5 dreamed iron tools per 100 ready starts within 50 epochs of its first ready start.
2. Every seed makes an iron pickaxe in at least 0.3% of its episodes, with the mean at or above 0.66%.
3. The score is at least 25.35.
- **Validated and built (2026-11-04).** `iron_diag.py --reach`: seed 1 makes an iron tool from 28% of craftable starts (go rule: 20%).
  - **Code:** `dream_env.craftable_now`, `Track.unlocks` (u, frame u − 1, frame u), `DreamRollout` pool (`_pick`, `_place`, `_fresh_states`, `_remember`). Options: `--dream_reach`, `--dream_pool`, `--dream_pool_age`.
  - **Deviations from the note:** the pool cap applies per kind, so `--dream_pool N` keeps up to N ready moments and N craftable ones; craftable moments are not evicted by the far more common ready ones. The AD state's two frames are real frames, stored in the Track at each unlock.
  - **Check metric:** `dream_pool_state_cos`, the mean cosine between each new pool entry's recomputed state and the one in storage.
  - **Smoke test:** resumed the final weights of `ad-rarity2-dream-ready-s0` for 8 epochs with pool 64, age 3. The pool filled and evicted as expected; pool starts were 5–14% of dreams, craftable 5–7%, and the state cosine 0.993–1.000.
  - **Proposed run:** `scripts/queue_dream.sh ad-rarity2-dream-pool --dream_craft tied --dream_iron 0.25 --dream_ready 0.15 --dream_reach 0.05 --dream_pool 512 --dream_pool_age 25`, seeds 0–2, about 7 h.
