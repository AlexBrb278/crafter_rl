"""Consistency checks for recurrent.py on real Crafter rollouts (run from the repo root):

  .venv/bin/python methods/05_achievement_distillation/test_recurrent.py

1. PPO chunks: replaying each 64-step chunk from its stored start state reproduces the
   rollout's log-probs and values (ratio = 1 before any update).
2. Aux phase: the per-step stored input state (RecurrentBuffer old_hiddens) reproduces them too.
3. The GRU state is zero exactly at episode starts, and memory matters (a zeroed state changes
   the policy), so the test would catch a model that ignores h.
"""
import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path[:0] = [HERE, ROOT, os.path.join(HERE, "upstream")]

import gymnasium  # noqa: E402
sys.modules.setdefault("gym", gymnasium)
import numpy as np  # noqa: E402
import torch as th  # noqa: E402
import yaml  # noqa: E402

import train as T  # noqa: E402  (TorchVecEnv, stub for upstream's SB3 wrapper)
stub = types.ModuleType("achievement_distillation.wrapper")
stub.VecPyTorch = T.TorchVecEnv
sys.modules["achievement_distillation.wrapper"] = stub
from achievement_distillation.sample import sample_rollouts  # noqa: E402
from common.crafter_vec_env import CrafterVecEnv  # noqa: E402
from recurrent import PPOADGRUModel, RecurrentBuffer, RecurrentRolloutStorage  # noqa: E402


def main():
    th.manual_seed(0)
    np.random.seed(0)
    dev = th.device("cuda")
    cfg = yaml.safe_load(open(os.path.join(HERE, "configs", "ppo_ad_gru.yaml")))
    nstep, nproc, L = 512, 8, cfg["seq_len"]
    obs_space = gymnasium.spaces.Box(0, 1, (3, 64, 64))
    act_space = gymnasium.spaces.Discrete(17)
    model = PPOADGRUModel(observation_space=obs_space, action_space=act_space, **cfg["model_kwargs"]).to(dev)
    venv = CrafterVecEnv(np.random.randint(0, 2**31 - 1, size=nproc))
    env = T.TorchVecEnv(venv, dev)
    st = RecurrentRolloutStorage(nstep=nstep, nproc=nproc, observation_space=obs_space, action_space=act_space,
                                 hidsize=cfg["model_kwargs"]["hidsize"], device=dev,
                                 gru_size=cfg["model_kwargs"]["gru_size"], seq_len=L)
    st.obs[0].copy_(env.reset())
    stats = sample_rollouts(env, model, st)
    print(f"rollout: {len(stats['episode_lengths'])} episodes ended in {nstep * nproc} steps")
    model.eval()  # same mode as the rollout (no dropout/BN anyway)

    # 3a. hidden is zero exactly where an episode starts
    starts = (st.masks[1:nstep + 1] == 0).squeeze(-1)
    hz = st.hiddens[1:nstep + 1].abs().sum(-1) == 0
    assert th.equal(starts, hz), "GRU state is not reset exactly at episode starts"
    assert starts.any(), "no episode ended; rerun with a longer rollout"

    # 1. chunk replay == rollout
    with th.no_grad():
        worst_lp = worst_v = 0.0
        for b in st.get_data_loader(nbatch=2):
            out = model.forward_seq(b["obs"], b["states"], b["masks"], b["hiddens0"])
            lp = model.pi_head.log_prob(out["pi_logits"], b["actions"].reshape(-1, 1))
            worst_lp = max(worst_lp, (lp - b["log_probs"].reshape(-1, 1)).abs().max().item())
            losses = model.compute_losses(**b)
        # vpreds: compare with storage (denormalised there)
        full = st.get_data_loader(nbatch=1)
        b = next(full)
        out = model.forward_seq(b["obs"], b["states"], b["masks"], b["hiddens0"])
        v = model.vf_head.denormalize(out["vpreds"])
        L_, T_, P_ = L, nstep, nproc
        vst = st.vpreds[:T_].reshape(T_ // L_, L_, P_, 1).transpose(1, 2).reshape(-1, L_, 1)
        # loader permutes chunks; recover the order through hiddens0 is awkward -> compare sorted sets
        worst_v = (v.reshape(-1).sort().values - vst.reshape(-1).sort().values).abs().max().item()
    print(f"1. chunk replay vs rollout: max |dlogp| = {worst_lp:.2e}, max |dV| (sorted) = {worst_v:.2e}, "
          f"loss at ratio 1: pi {losses['pi_loss'].item():.4f}")
    assert worst_lp < 1e-3 and worst_v < 1e-3

    # 2. aux phase: per-step stored states
    buf = RecurrentBuffer(maxlen=1)
    buf.insert({k: st[k].cpu() for k in ["obs", "actions", "states", "returns", "masks", "rewards",
                                          "successes", "hiddens", "log_probs"]})
    buf.parse_segs()
    worst = 0.0
    # reconstruct per-trajectory log-probs from the storage in the same order as parse_segs
    lps = st.log_probs[:nstep].cpu()
    masks = st.masks[:nstep].cpu()
    k = 0
    with th.no_grad():
        for p in range(nproc):
            done = sorted((masks[:, p] == 0).squeeze(-1).nonzero().squeeze(-1).tolist())
            for s, e in zip(done[:-1], done[1:]):
                tr = buf.trajs[k]
                out = model.forward(tr["obs"].to(dev), states=tr["old_states"].to(dev),
                                    hiddens=tr["old_hiddens"].to(dev))
                lp = model.pi_head.log_prob(out["pi_logits"], tr["actions"].to(dev)).cpu()
                worst = max(worst, (lp - lps[s:e, p]).abs().max().item())
                k += 1
    print(f"2. aux per-step states: {k} complete trajectories, max |dlogp| = {worst:.2e}")
    assert k > 0, "no complete trajectory to check"
    assert worst < 1e-3

    # 3b. memory matters
    with th.no_grad():
        b = next(st.get_data_loader(nbatch=1))
        o1 = model.forward_seq(b["obs"], b["states"], b["masks"], b["hiddens0"])["pi_logits"]
        o2 = model.forward_seq(b["obs"], b["states"], th.zeros_like(b["masks"]), th.zeros_like(b["hiddens0"]))["pi_logits"]
    print(f"3. zeroing the memory changes logits by up to {(o1 - o2).abs().max().item():.3f} (must be > 0)")
    assert (o1 - o2).abs().max().item() > 1e-4
    venv.close()
    print("all checks passed")


if __name__ == "__main__":
    main()
