# 04 — Tuned PPO + GRU on Craftax-Classic pixels (Dedieu et al. "MFRL")

**Status:** not started
**Family:** model-free · **Paper:** https://arxiv.org/abs/2502.01591 (ICML 2025)
**Code:** no official release. The backbone is reproduced inside the ITC repo (see 07); otherwise reimplement from the paper's Algorithm 2.

**Target:** Craftax-Classic pixels, 1M steps: 55.5% reward, score 16.8. ~15 min per run on an A100.
Best accuracy per GPU-hour, so it suits many seeds and ablations.
