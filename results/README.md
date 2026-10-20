# results/

Small, committed summaries only: CSV/JSON of final scores, per-achievement success rates, and plots used in the paper.
Raw logs, final weights and final-episode videos are in `../runs/<run>/` (weights and videos via Git LFS).

## Contents (2026-10-20)
- `summary.csv`: the five seed-0 headline runs (PPO, PPO + rarity β=2, AD, AD + rarity β=1/2): score, reward, 22 success rates, wall-clock.
- `all_runs_summary.csv`: the same columns for all 15 runs.
- `seeds_summary.csv`: mean ± sd per configuration across seeds.
- `plots/score.png`, `plots/reward_length.png`, `plots/achievements.png`, `plots/achievement_curves.png`: seed-0 headline runs.
- `plots/rarity_beta.png`: PPO score vs rarity-bonus strength β (3 seeds each).
Regenerate with the commands in `methods/05_achievement_distillation/README.md`. The full HTML report is `runs/report/crafter_report.html` (self-contained: download it and open it in a browser).
