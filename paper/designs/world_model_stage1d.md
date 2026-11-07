# Design note: world model Stage 1d, a map memory

_Draft 2026-10-25, written while Stage 1b-long and 1c train. For review before any code. Open questions are marked **Q**._

## Why
- Every stage so far fails at the same thing: **remembering what left the screen.**
- Water that the agent walked away from comes back in only 15–17% of dreams (Stage 1, 1b, 1c).
- Giving the model the agent's position (Stage 1c) helps a direct memory query (34% → 43%) but not the dream.
- Two architectural reasons:
  1. **The temporal model sees each past frame as one compressed vector.** To redraw a lake from 14 steps ago at its new screen position, it must rebuild spatial detail from that summary.
  2. **Our 8×8-pixel tokens don't line up with Crafter's 7×7-pixel tiles.** When the agent walks one tile, the picture shifts by 7 pixels, which is not a whole number of tokens, so a remembered patch can never simply be copied to its new place.

## Idea
Do what a human player does: keep a **map**, not 64 screenshots.
1. **Tile tokens.** Crop the 64×64 frame to 63×63 and cut it into 9×9 tiles of 7×7 pixels: 7 rows of world plus 2 rows of inventory, one token per tile. A one-tile move is now an exact one-token shift.
   - **Q8:** encode each tile on its own (simple, exact alignment), or with a little context from its neighbours (helps at night, when Crafter adds noise)?
2. **A top-down map memory.** As the model looks at real frames, and later at its own dreamed frames, it writes every world tile token into a 2D map at that tile's world position (agent position + offset in view), with the step it was seen.
3. **Reading the map when predicting.** For the next frame, the model reads the 9×7 map window around the agent's next position. Each tile gets "the last token seen here, how long ago", or "never seen". The frame decoder receives this as an extra input next to the usual MASK tokens. When the agent walks back to a lake, the water tiles are *in the map*, and the decoder only has to copy them.
4. **Position.** Stage 1c's odometry output already predicts moves almost perfectly (test loss 0.02). It updates the position inside dreams.
   - **Q9:** start with the position from the tile maps (diagnostic, as in Stage 1c), or go straight to positions the model estimates itself from how the picture shifts between frames? With tile tokens, a shift is exact, so self-estimated odometry should be easy.

## What stays the same
The data (7,930 episodes), the test set and dream starts, the clock, unrolled training, the head re-fit, and the evaluation. Results will be directly comparable to Stage 1, 1b and 1c.

## What we expect (written down before running)
- **P11:** water comes back in **≥60%** of dreams (from 15–17%). This is the reason to build it.
- **P12:** items exact at 15 steps ≥70% (1b: 62%). The map also stops terrain from drifting.
- **P13:** the tile tokenizer reconstructs at least as well as tokenizer v2 (34.8 dB), because tiles are Crafter's own unit.
- **P14 (risk):** creatures move, so the map holds stale copies of them. If dreams show "ghost" cows where a cow used to be, creature tiles must not be written into the map. Q10: drop creatures from the map from the start?

## Cost
- Code: about 1–1.5 days (tile tokenizer, map write/read, decoder input, evaluation hooks).
- GPU: tile tokenizer ~1.5 h, token cache ~15 min, dynamics ~5 h, re-fit and evaluation ~1 h.
- No new recording needed.

---

## Evidence and decisions (2026-10-26, 03:00)
- **Stage 1c result:** with the agent's position, dream water memory is 21% and skip-ahead 38%. Training 1b twice as long gives 23% / 39%. So position alone is not the bottleneck. This supports the compressed-frame / tile-misalignment explanation above.
- **Q8:** each tile is encoded on its own. Simplest, with exact alignment.
- **Q9:** positions from the tile maps first (diagnostic, as in 1c), to isolate whether the map works. Self-estimated odometry comes later.
- **Q10:** all tiles go into the map, including creatures, together with *how long ago* each was seen, so the model can learn to distrust old creature sightings.
- **Change from the draft:** the map is built from the agent's **whole episode history**, not just the 64-step context. Training uses a precomputed per-step map window; dreams keep writing their own frames into it.
- **Simplifications for the first version:** no skip-ahead output (the map is the memory now) and no unrolled training. Clock, position and odometry stay.

---

## Results (2026-10-26, 08:52; `results/world_model/stage1d_eval.json`, `stage1d_semantic.json`)
200 dreams on the fixed Stage 1 test starts, re-fitted outputs. v3 = tile tokens + position + clock + map; v3_nomap = identical without the map.

| measure | v2_long (8×8) | v3_nomap | **v3 (map)** |
|---|---|---|---|
| dream water memory | 23% | 22.5% | **60%** |
| items exact @5/15/50 | 93.5/57/15% | 99/72/18% | 96.5/72/**31%** |
| terrain tiles right @1/5/15/50 (copy 89/74/62/49) | 91/84/69/54% | 98/91/79/58% | **98.5/95/85/65%** |
| ghost creatures per frame @5/15/50 | 0.10/0.04/0.01* | 0.33/0.56/0.65 | 0.21/0.25/0.26 |
| creature recall @1 | 14% | 49% | 58% |
| PSNR @1/5/15/50 | 27.9/21.2/17.6/15.1 | 29.4/21.9/18.6/15.4 | **31.6/24.0/19.5/16.2** |
| unlock recall / precision | 85/73% | 89/70% | 92/**59%** |
| night correlation | 0.66 | 0.66 | **0.73** |

\*v2 hardly draws creatures at all.

- **P11 met** (60% ≥ 60%). The map, not the tile tokens, is what does it: v3_nomap stays at 22.5%.
- **P12 met** (72% @15). But v3_nomap reaches it too, so it comes from the tile tokens. The map's gain shows at 50 steps (31% vs 18%).
- **P13 met** earlier (tokenizer 36.2 dB).
- **P14 refuted:** the map does not leave ghosts behind; the model without a map draws more non-existent creatures (0.65 vs 0.26 per frame @50). Video `runs/wm/videos/dream1d_03_compare.mp4`.
- **Weakness:** unlock precision 59% (re-fit lands at a lower threshold; recall 92%). To check whether it is the re-fit or the representation.
- Single-frame test loss barely separates the two models (0.421 vs 0.445). The memory effect only shows in multi-step dreams.
