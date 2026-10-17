# 01 — PPO-RNN (GRU) on Craftax

**Status:** not started
**Family:** model-free · **Framework:** JAX (Craftax_Baselines)
**Code:** https://github.com/MichaelTMatthews/Craftax_Baselines · **Paper:** https://arxiv.org/abs/2402.16801

**Targets to reproduce**
- Craftax-1B (symbolic): 15.3% of max reward
- Craftax-Classic-1B (symbolic): ~90% of max reward
- Craftax-1M: 2.3%

**Why first:** minutes to ~2 h per run. It validates the JAX/RTX 5090 stack, logging, and scoring before anything expensive.
Also covers the cheap baselines in the same repo: PPO (MLP), PPO-RND, ICM, E3B.
