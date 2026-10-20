"""Improvement experiment (not in the paper): Achievement Distillation with a GRU memory.

The upstream PPOADModel only sees the current frame plus one "memory" vector: the
embedding of the last achievement it unlocked. It cannot remember the world (where the
water, the iron or a diamond was once they leave the 9x7 view). Here a GRU sits between
the encoder and the heads:

    latents = encode(obs)                        (unchanged, also used for AD's goal states)
    h_t     = GRU(latents, h_{t-1} * not_new_episode)
    heads   <- [latents, AD state, h_t]

Rollouts carry h step by step and store each step's (already reset) input state. The PPO
update replays 64-step chunks from their stored start state with BPTT, keeping the paper's
minibatch size (8 minibatches of 512 samples). The AD auxiliary phase only needs the policy
and value outputs to stay close to the old model, so it uses the stored per-step input
state as a fixed input (no BPTT there).
"""
from typing import Dict, Iterator

import torch as th
import torch.nn as nn
import torch.nn.functional as F

from achievement_distillation.action_head import CategoricalActionHead
from achievement_distillation.algorithm.ppo_ad import Buffer
from achievement_distillation.model.ppo_ad import PPOADModel
from achievement_distillation.mse_head import ScaledMSEHead
from achievement_distillation.storage import RolloutStorage


class PPOADGRUModel(PPOADModel):
    def __init__(self, observation_space, action_space, hidsize, gru_size=512,
                 action_head_kwargs: Dict = {}, mse_head_kwargs: Dict = {}, **kwargs):
        super().__init__(observation_space, action_space, hidsize,
                         action_head_kwargs=action_head_kwargs, mse_head_kwargs=mse_head_kwargs, **kwargs)
        assert self.use_memory, "the GRU model keeps AD's achievement state as an input"
        self.gru_size = gru_size
        self.gru = nn.GRUCell(hidsize, gru_size)
        insize = 2 * hidsize + gru_size
        num_actions = getattr(self.action_space, "n")
        self.pi_head = CategoricalActionHead(insize=insize, num_actions=num_actions, **action_head_kwargs)
        self.vf_head = ScaledMSEHead(insize=insize, outsize=1, **mse_head_kwargs)

    def initial_hiddens(self, n, device):
        return th.zeros(n, self.gru_size, device=device)

    def _heads(self, latents, states, h):
        core = th.cat([latents, states, h], dim=-1)
        return {"latents": latents, "pi_latents": core, "vf_latents": core,
                "pi_logits": self.pi_head(core), "vpreds": self.vf_head(core), "hiddens": h}

    def forward(self, obs, states, hiddens=None):
        """One step. `hiddens` is the input state (already zeroed at episode starts)."""
        latents = self.encode(obs)
        if hiddens is None:
            hiddens = self.initial_hiddens(len(obs), obs.device)
        return self._heads(latents, states, self.gru(latents, hiddens))

    def forward_seq(self, obs, states, masks, hiddens0):
        """Chunks: obs (B, L, C, H, W), states (B, L, D), masks (B, L, 1) where 0 marks the
        first step of a new episode, hiddens0 (B, G) = stored input state of step 0.
        Returns outputs flattened to (B*L, ...) in (b, t) order."""
        B, L = obs.shape[:2]
        latents = self.encode(obs.reshape(B * L, *obs.shape[2:])).reshape(B, L, -1)
        h, hs = hiddens0, []
        for t in range(L):
            if t > 0:
                h = h * masks[:, t]
            h = self.gru(latents[:, t], h)
            hs.append(h)
        h = th.stack(hs, dim=1)
        return self._heads(latents.reshape(B * L, -1), states.reshape(B * L, -1), h.reshape(B * L, -1))

    def compute_losses(self, obs, states, actions, log_probs, vtargs, advs, masks, hiddens0,
                       clip_param=0.2, **kwargs):
        out = self.forward_seq(obs, states, masks, hiddens0)
        flat = lambda x: x.reshape(-1, x.shape[-1])
        actions, log_probs, vtargs, advs = map(flat, (actions, log_probs, vtargs, advs))
        new_log_probs = self.pi_head.log_prob(out["pi_logits"], actions)
        ratio = th.exp(new_log_probs - log_probs)
        ratio_clipped = th.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param)
        pi_loss = -th.min(advs * ratio, advs * ratio_clipped).mean()
        entropy = self.pi_head.entropy(out["pi_logits"]).mean()
        vf_loss = self.vf_head.mse_loss(out["vpreds"], vtargs).mean()
        return {"pi_loss": pi_loss, "vf_loss": vf_loss, "entropy": entropy}

    # AD auxiliary losses: same as upstream, with the stored per-step GRU state as input.
    def compute_pred_losses(self, *a, **k):
        raise NotImplementedError("use fast_aux (default); it passes the stored GRU states")

    def compute_match_losses(self, *a, **k):
        raise NotImplementedError("use fast_aux (default); it passes the stored GRU states")


class RecurrentRolloutStorage(RolloutStorage):
    def __init__(self, *args, gru_size=512, seq_len=64, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.nstep % seq_len == 0, "nstep must be a multiple of seq_len"
        self.seq_len = seq_len
        self.hiddens = th.zeros(self.nstep + 1, self.nproc, gru_size, device=self.device)

    def get_inputs(self, step):
        inputs = super().get_inputs(step)
        inputs["hiddens"] = self.hiddens[step]
        return inputs

    def insert(self, hiddens, masks, **kwargs):
        step = self.step
        super().insert(masks=masks, **kwargs)
        # input state for the next step: zero where a new episode starts
        self.hiddens[step + 1].copy_(hiddens * masks)

    def reset(self):
        super().reset()
        self.hiddens[0].copy_(self.hiddens[-1])

    def get_data_loader(self, nbatch) -> Iterator[Dict[str, th.Tensor]]:
        """Minibatches of whole chunks: nstep*nproc/seq_len chunks, split into nbatch groups."""
        L, T, P = self.seq_len, self.nstep, self.nproc
        nchunk = (T // L) * P
        assert nchunk % nbatch == 0, f"{nchunk} chunks do not split into {nbatch} minibatches"

        def chunks(x):  # (T(+1), P, ...) -> (nchunk, L, ...), chunk c = (time block c // P, env c % P)
            x = x[:T]
            x = x.reshape(T // L, L, P, *x.shape[2:]).transpose(1, 2)
            return x.reshape(nchunk, L, *x.shape[3:])

        data = {
            "obs": chunks(self.obs), "states": chunks(self.states), "masks": chunks(self.masks),
            "actions": chunks(self.actions), "vtargs": chunks(self.returns),
            "log_probs": chunks(self.log_probs), "advs": chunks(self.advs),
        }
        h0 = self.hiddens[:T:L]  # (T // L, P, G): stored input state at the start of each chunk
        data["hiddens0"] = h0.reshape(nchunk, -1)
        perm = th.randperm(nchunk, device=self.device)
        size = nchunk // nbatch
        for b in range(nbatch):
            idx = perm[b * size:(b + 1) * size]
            yield {k: v[idx] for k, v in data.items()}


class RecurrentBuffer(Buffer):
    """Upstream Buffer, plus each trajectory's stored per-step GRU input state (old_hiddens)."""

    def parse_segs(self):
        super().parse_segs()
        hiddens = th.cat([seg["hiddens"][:-1] for seg in self.segs], dim=0)
        masks = th.cat([seg["masks"][:-1] for seg in self.segs], dim=0)
        k = 0
        for p in range(hiddens.shape[1]):  # same order as Buffer.parse_segs
            done = sorted((masks[:, p] == 0).squeeze(-1).nonzero().squeeze(-1).tolist())
            for start, end in zip(done[:-1], done[1:]):
                traj = self.trajs[k]
                assert len(traj["obs"]) == end - start
                traj["old_hiddens"] = hiddens[start:end, p]
                k += 1
        assert k == len(self.trajs)
