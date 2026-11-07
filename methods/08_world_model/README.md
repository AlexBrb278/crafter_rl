# 08 — Our own world model (Stage 1: dream vs reality)

**Status:** Stage 1 running (2026-10-22). Design and decisions: `paper/designs/world_model_stage1.md`.

**Idea:** learn a simulator of Crafter that imagines the next frames from what the agent saw and does. Stage 1 trains no agent; it measures how well the model dreams. Stage 2 (an agent that practises in the dream) only if Stage 1 passes the bar in the design note.

## Pipeline (`run_stage1.sh`)
| step | file | what |
|---|---|---|
| 1 | `record.py` | ~3,940 episodes (~1M frames) from our saved agents: final, mid-training and early checkpoints. Per step: frame, action, reward, inventory (16), unlocked achievements (22), daylight, 9×7 semantic view, death. Output in `runs/wm/data/` (gitignored). |
| 2 | `tokenizer.py` | VQ autoencoder, 64×64 frame → **8×8 grid of codes** (512-code book, EMA, dead-code restarts), L1 pixel loss, 11.4M params. |
| 3 | `cache_tokens.py` | Encodes every frame once into `runs/wm/tokens/`. |
| 4 | `dynamics.py` | World model, 25.6M params. Per step: frame tokens → 2 spatial Transformer layers → one vector, plus inventory and action embeddings. A causal Transformer over **64 steps** then predicts the next grid (MaskGIT-style, 3 passes), the **next inventory as its own output**, unlocks and death. Trained twice: **uniform** windows, and **rarity-weighted** (windows with an unlock of an achievement in <30% of episodes, i.e. iron and beyond, sampled 5×). |
| 5 | `evaluate.py` | Held-out dreams with real actions: accuracy against horizon vs a "nothing changes" baseline; unlock/death recall; probes (night, off-screen water, crafting, rare chain); reality / dream / difference videos. |

`data.py`: episode loading; train/test split by path hash (10% test).
