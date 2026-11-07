# Crafter agent: PPO + Achievement Distillation + rarity bonus

Our agent for Crafter (Hafner, 2022), 1M environment steps. PPO with Achievement Distillation
(AD, Moon et al., NeurIPS 2023), plus our change: a **rarity bonus** that pays extra for unlocking the
achievements the agent rarely gets.

| configuration (1M steps) | seeds | Crafter score | episode reward (last 100) |
|---|---|---|---|
| random agent | – | 1.55 | 1.41 |
| PPO | 3 | 15.75 ± 2.99 | 10.3 |
| PPO + rarity bonus β=2 | 3 | 20.99 ± 0.87 | 11.6 |
| AD (paper configuration) | 1 | 17.71 | 11.8 |
| **AD + rarity bonus β=2 (default of `train.py`)** | 3 | **23.65 ± 1.05** | 12.8 |

Published references at 1M steps: PPO 15.60 ± 1.66, AD 21.79 ± 1.37.

## Run

```bash
pip install -r requirements.txt
python train.py                  # best configuration, seed 0 (~4 h on one recent GPU)
python train.py --seed 1         # more seeds
python evaluate.py runs/ad-rarity2-s0/agent.pt --episodes 200 --video episode.mp4
python evaluate.py --random --episodes 1000
python plot.py "AD + rarity=runs/ad-rarity2-s0,runs/ad-rarity2-s1" --out plots
```

Ablations: `--no_ad` (plain PPO network), `--rarity_bonus 0` (game reward), `--gamma 0.99`,
`--vitals 1` (survival shaping). On a GPU with less than ~8 GB, add `--micro_batch 256`
(gradient accumulation; the gradient is unchanged). Install `numba`: Crafter's world generation is
then ~20x faster (1 s → 50 ms per reset) with identical worlds.

## Files

| file | contents |
|---|---|
| `train.py` | command line (defaults = best hyperparameters), rollout loop, logging, checkpoints, final evaluation |
| `algo.py` | rollout storage + GAE, PPO update, AD auxiliary phase (prediction + optimal-transport matching) |
| `model.py` | IMPALA ResNet encoder, policy / value heads, value normaliser, AD's achievement state and predictor |
| `envs.py` | multiprocess Crafter vector env, Crafter score, `RarityBonus` (ours), `VitalsBonus` (ablation) |
| `evaluate.py` | final policy (or random agent) on fresh worlds; optional MP4 of one episode |
| `plot.py` | score / reward curves and per-achievement success rates from the logs |

Outputs go to `runs/<name>/`: `metrics.jsonl` (per update: score, losses, success rates),
`episodes.jsonl` (per training episode), `final.json` (score over all training episodes and the
evaluation of the final policy), `agent.pt`, `checkpoint.pt` (for `--resume`).

## Method

- **PPO:** 8 envs × 512 steps per update, GAE (γ 0.95, λ 0.65), 3 epochs × 8 minibatches, clip 0.2,
  value coefficient 0.5, entropy 0.01, Adam 3e-4, gradient-norm 0.5, normalised value targets.
- **Network:** IMPALA ResNet (64/128/128 channels, 2 residual blocks each, GroupNorm), 256 → 1024
  latent. The heads also see AD's *achievement state*: the normalised latent difference across the last
  unlock in the episode.
- **AD auxiliary phase** (every 8 updates, 6 epochs on the last 8 rollouts split into episodes):
  a contrastive loss that predicts the next achievement's embedding from (state, action), and a
  contrastive loss that matches achievements across episodes by entropic partial optimal transport;
  both with a KL term to the pre-phase policy and value regression.
- **Rarity bonus (ours):** on the first unlock of achievement *i* in an episode the learner gets
  `r + β (1 − p_i)`, with `p_i` the share of the last 200 episodes that unlocked *i* (β = 2). The Crafter
  score is a geometric mean over achievements, so it is dominated by the rare ones; the bonus targets
  exactly those and fades as they become common. Logged rewards and the score use the game reward.

The network, PPO and AD losses were checked against the reference AD implementation
(snu-mllab/Achievement-Distillation) on a trained checkpoint: identical forward pass, and losses and
gradients equal up to float rounding.
