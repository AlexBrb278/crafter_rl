# Design note: a shared craft rule in the world model

_Draft 2026-10-30, for review before any code. Open questions are marked **Q**._

## Why
- **Stage 2a lifted iron, not the iron tools.** Score 26.10 vs 23.65 and collect iron 15.4% vs 7.4%, but the dream never paid an iron pickaxe, an iron sword or a diamond.
- **The check of 30 Oct shows the world model is the ceiling.** On 41 held-out iron-pickaxe and 47 iron-sword crafts, replayed with the real actions, 0 dreams show the tool (controls: coal, iron, stone pickaxe 55–99%). In the dream the agent stands at a table and furnace with wood, coal and iron, presses the craft action, and nothing happens.
- **Stage 2b as designed would hit the same ceiling.** The agent's own 1M steps hold 0–5 iron pickaxes.

## What the data says (v4's training data: 7,134 episodes, 1.76M steps)
For each craft action: how often it was pressed, how often Crafter's requirements were met (items in the inventory, table and furnace in the 3×3 around the player), and how often the tool count rose.

| action | pressed | requirements met | tool count rose |
|---|---|---|---|
| make wood pickaxe | 27,514 | 7,633 | 7,433 |
| make stone pickaxe | 38,999 | 5,468 | 4,726 |
| **make iron pickaxe** | **17,471** | **4** | **4** |
| make wood sword | 30,689 | 7,770 | 7,541 |
| make stone sword | 45,869 | 5,123 | 4,567 |
| **make iron sword** | **15,594** | **0** | **0** |

- **The agents press the iron crafts all the time;** they almost never have everything. Missing on iron-pickaxe presses: furnace in 90%, table 84%, iron 81%, wood 52%, coal 38%.
- **The world model is not wrong about its data.** "Make iron pickaxe" really did nothing in 17,467 of 17,471 presses. It has no reason to learn that it *would* work if all five requirements held. Its inventory output predicts each item's count with its own weights, so 12,000 working wood and stone crafts say nothing about iron.
- **The rule is the same for every tool,** and it can be read off the data, not only off Crafter's source:
  - A craft action works if certain items are at hand and a station is within reach.
  - The items are used up and the tool count rises by 1.
  - Wood and stone tools (25,000 working crafts) show this cleanly. No craft ever worked without its requirements met.
  - When the requirements are met, the count rose in 86–97% of presses. The rest were most likely presses while asleep, which the game replaces with sleep; this is not yet verified.
  - Pickaxe and sword of the same tier have identical requirements.

## Idea: a craft layer with the rule's shape, learned from data
Add a small module next to the inventory output. It handles the 6 craft actions only; everything else (vitals, collecting, placing) stays as now.

1. **When does a craft work?** A learned *conjunction*, one per action `a`:
   - `p(works) = σ(β · min(m_a,1 … m_a,16, s_a,table, s_a,furnace))`.
   - Item margins: `m_a,k = inv_k − r_a,k + ½`, with a learned requirement `r_a,k ≥ 0` (most go to 0, meaning not needed).
   - Station terms: `s_a,j` is positive when station `j` is within reach, or when the action doesn't need it (learned).
   - This has ~20 numbers per action instead of a free function of a 512-d history. Under this form, 4 working iron-pickaxe crafts plus 17,467 failures identify the recipe. Failures with everything but one item present tell which items are needed.
2. **What does it do?** A learned effect per action: the mean inventory change over the presses that worked, rounded to whole items. Iron pickaxe, from its 4 crafts: wood −1, coal −1, iron −1, iron pickaxe +1.
3. **What is in reach?** Read from the dreamed tile codes in the 3×3 around the player, which sits at a fixed place in the view.
   - Which codes mean "table" or "furnace" is learned from the agent's own placing events. When place_table succeeds (wood count falls), the neighbouring tile whose code changed is a table. The same works for furnaces (stone count falls).
   - That gives thousands of labelled station tiles with no simulator labels. The furnace no longer has to be learned from 4 crafts.
4. **Sharing across tools (Q16).** Tie the pickaxe and sword of each tier to one recipe, and state that the product of `make_X` is `X`.
   - Without this, iron sword has 0 working examples and stays impossible.
   - With it, iron sword is learned from iron pickaxe, the way the data shows for wood and stone.
5. **Keep the HUD consistent (Q18).** The agent sees pixels, and its inventory is drawn in the bottom two rows of the frame.
   - Today those rows come from the token decoder, so a craft the layer allows would not show in the frame. The craft video already shows the dream's HUD without the pickaxe.
   - The two HUD rows of a dreamed frame are replaced by tokens looked up from the dreamed inventory: cell, item and count → the most common code, learned from real frames.
6. **Where it plugs in.**
   - Trained on the frozen v4 trunk, like the existing refit: the inputs are the inventory, action and tile codes at step t, the target is the count change. That takes minutes.
   - In `Dreamer.step`, when the action is a craft, the layer's sampled outcome overrides the item counts. The vitals still come from the inventory output.
   - Reward (gate mode 2) is unchanged: a rising tool count is the unlock.

## How much is learned, and how much we put in
This is the question to settle before building. From least to most prior knowledge:

| level | what we put in | iron pickaxe | iron sword | still "a learned model"? |
|---|---|---|---|---|
| now | nothing (free output per item) | 0% | 0% | yes |
| oversample rare crafts | nothing | memorises the 4 | impossible (0 examples) | yes |
| **this note, untied** | the rule's *shape*: an AND over items and stations, a fixed effect | learned from 4 | impossible | yes, with a strong inductive bias |
| **this note, tied (Q16)** | + pickaxe and sword of a tier share a recipe; `make_X` gives `X` | learned from 4 | learned from the pickaxe | mostly; the tie and the name come from us |
| hand-coded rules | the recipes themselves (data.yaml) | exact | exact | no |

My pick is the tied version, and the report says plainly what was put in.

The tie is the kind of prior a person brings to the game ("a sword is made like the pickaxe of the same material"). The data supports it at the two tiers where it can be checked. The recipes themselves (which items, which stations) are still learned. A test checks that they come out as in Crafter's source.

## Tests (before any agent training)
1. **Recipes:** the learned requirements and effects for all 6 actions match Crafter's `data.yaml` (items, amounts, stations).
2. **The same craft_check** on the held-out probe episodes (41 iron pickaxes, 47 iron swords), with the controls as before.
3. **No false crafts:** among presses whose requirements are *not* met (from real episodes), the layer allows at most 1 in 10,000.
4. **HUD:** on real frames, the looked-up HUD rows match the real HUD tokens in at least 95% of cells.

## What we expect (written down before building)
- **P19:** test 1 holds for all 6 actions in the tied version. In the untied version it holds for all but iron sword.
- **P20:** in craft_check, iron pickaxe and iron sword are dreamed in ≥ 75% of cases 1 step ahead (stone pickaxe today: 81%), and false crafts are ≤ 1 in 10,000.
- **P21 (Stage 2a rerun with the layer, 3 seeds, 1M steps, still not 1M-comparable):**
  - Iron pickaxe in ≥ 0.5% of training episodes (the half of P15 that failed), and iron sword in ≥ 0.5%.
  - Score not below 26.10 − 0.75.
- **P22 (risk):** the agent learns to craft iron tools in dreams but still not diamonds. Finding and mining a diamond is a separate rule with 1 example in the data (see Q17).

## Open questions
- **Q16:** tie the pickaxe and sword of a tier, and name the product, or leave them untied? My pick: tied, stated in the report. Untied, iron sword stays impossible.
- **Q17:** also give "collect" (do + material in front + required pickaxe) the same rule shape, for diamonds?
  - My pick: not yet. The data has 1 diamond.
  - Even with the shape, which pickaxe a diamond needs would come from one example. Tying it across tiers (stone → wood pickaxe, iron → stone pickaxe, diamond → iron pickaxe) is much more prior knowledge than Q16.
  - We look at it after P21.
- **Q18:** HUD rows from a learned lookup (my pick), or drawn with Crafter's own renderer? The renderer is exact but is hand-coded display.
- **Q19:** first test on the frozen v4 (Stage 2a rerun, fast, not comparable), then Stage 2b? My pick: yes. If P20 fails on v4, the honest version has no chance either.
- **Q20:** may placing events (wood or stone count falls, a neighbouring tile changes) be used to find station tiles? My pick: yes. They come from the agent's own pixels and inventory, with no simulator labels.

## Cost
- **Code:** about 1 day.
  - The craft layer and its training on the frozen trunk.
  - Station codes from placing events and the HUD lookup.
  - Plugging into `Dreamer.step`, and craft_check / craft_video for the layer.
- **Tests 1–4:** under an hour of compute.
- **Stage 2a rerun:** 3 seeds in parallel, about 7 h, as last time.

---

## Decisions (2026-10-30)
- **Q16:** tied. Pickaxe and sword of a tier share one recipe, and `make_X` gives `X`.
- **Q17:** no collect rule yet.
- **Q18:** HUD rows from a learned lookup.
- **Q19:** first a Stage 2a rerun on the frozen v4.
- **Q20:** placing events may be used to find station tiles.

## Changes while building (2026-10-30, details in the work log)
- **Stations: tracked by position, not read off the placing step alone.**
  - At night the noise changes every tile from one frame to the next, so a placing step shows four "changed" neighbours.
  - Only placing steps with exactly one changed neighbour are used: 4,652 tables and 3,871 furnaces. Each station's world position comes from the pixel odometry, and its code is counted in every later frame that shows that position, by night too.
  - A code counts as a station if its share of tiles at tracked station positions is above 0.2 of the best code's. Codes at stations score 0.38–1, all other codes ≤ 0.05.
- **Requirements: what a craft uses up must be at hand.** Requirement = max(amount used up, a learned extra with an L1 cost).
  - The data cannot show coal is needed for the iron tools. No iron press failed with only coal missing, and the 4 successes all had coal.
  - The 4 successes do show coal being used up.
- **Asleep: an "awake" term.** It is a linear read-out of v4's own embedding of the current frame.
  - 1,727 presses met every requirement and still failed. All had energy below 9, which fits Crafter replacing presses with sleep while asleep.
  - The frame embedding does not see the action, so the term cannot learn "iron crafts fail".
- **Tools change only through the layer.**
  - In v4's data, all 24,271 tool-count changes happened on the tool's own make action.
  - In the dream, tool counts are carried over on every other step.
- **Counts the world model never saw.**
  - Its inventory input has a learned row per (item, count). Rows it never trained on (1 iron sword, 2+ iron pickaxes, …) are still at their random start.
  - With the layer on, the world model reads such a pair as the nearest count below that it did see ("seen" = at least 20 training frames, as for the HUD lookup).
  - The dream's own inventory, the reward and the HUD keep the true count.
- **HUD:** redrawn on every dream step, not only crafts, so the frame always matches the dreamed inventory. A pair seen in fewer than 20 training frames keeps the decoder's cell. This affects the iron sword (none in v4's data).

## Tests 1, 3, 4 (held-out test split: 796 episodes, 20,476 presses)
| test | result | bar |
|---|---|---|
| 1. recipes, tied | **all 6 match `data.yaml`**: items, amounts, stations, effects | all 6 |
| 1. recipes, untied | 5 of 6; iron sword learns nothing (no example) | P19: all but iron sword |
| 3. false crafts, tied | **0.012 per 10,000** presses that miss a requirement (17,632); highest p 2·10⁻⁶ | ≤ 1 |
| 3. false crafts, untied | 3.5 per 10,000, all on iron sword (it "works" with an empty effect) | |
| recall | p(works) 0.999 on the 2,703 crafts that worked; 0.045 on the 141 that met everything but failed (asleep) | |
| 4. HUD lookup | **100%** of cells right where the pair is known; unknown pairs are 0.004% of test cells | ≥ 95% |
| stations | table 99.7% found, furnace 98.8%, **0 false detections** (semantic labels used only here) | |

## Test 2: craft_check with the layer (held-out probe episodes, T = 1, 32 dreams per moment)
| item | cases | dream shows it 1 / 4 / 8 / 16 steps ahead, with the layer | before (v4 alone) |
|---|---|---|---|
| **iron pickaxe** | 41 | **97 / 89 / 77 / 63%** | 0 / 0 / 0 / 0% |
| **iron sword** | 47 | **99 / 92 / 81 / 69%** | 0 / 0 / 0 / 0% |
| stone pickaxe (control) | 74 | 99 / 95 / 91 / 68% | 81 / 85 / 78 / 55% |
| iron (control, not a craft) | 100 | 82 / 82 / 76 / 55% | 81 / 81 / 75 / 56% |
| coal (control, not a craft) | 100 | 96 / 95 / 86 / 69% | 99 / 96 / 78 / 56% |
| diamond | 1 | 0% | 0% |

- **P20 holds:** iron tools ≥ 75% one step ahead (97% and 99%), and false crafts ≤ 1 in 10,000 (0.012).
- **The drop with distance is the dream's, not the rule's.** Over 16 dreamed steps the dream loses track of the table, the furnace or the items about as often as it does for the controls.
- **Collecting is unchanged,** as intended (Q17).
- Videos: `runs/wm/eval/crafts/iron_{pickaxe,sword}_layer_0{0,1}.mp4`. The dream's inventory and HUD show the tool on the real craft step. The iron sword's HUD cell is the decoder's, since there is no example of it.

## Stage 2a rerun with the layer (launched 2026-10-30 16:35)
`scripts/queue_dream.sh ad-rarity2-dream-craft --dream_craft tied`: 3 seeds, 1M steps, otherwise exactly as Stage 2a. P21 is checked against it.
