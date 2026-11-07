"""Rollout storage with GAE, the PPO update, and Achievement Distillation's auxiliary phase.

PPO (Schulman et al., 2017), per minibatch:
    L = -E[min(rho * A, clip(rho, 1 - eps, 1 + eps) * A)] + c_v * E[(V(s) - R)^2] - c_e * E[H(pi(.|s))]
    rho = pi(a|s) / pi_old(a|s), A = GAE(gamma, lambda) normalised, R = A + V_old (value targets).

Achievement Distillation (Moon et al., 2023), every `aux_every` PPO updates, on the last
`aux_every` rollouts split into complete episodes. Two contrastive losses shape the encoder:
  * prediction (within an episode): from (s_t, a_t) predict the embedding of the NEXT achievement
    the agent unlocks; a random other step of the same episode is the negative;
  * matching (across episodes): achievements of two episodes are paired by entropic partial
    optimal transport on their embeddings; a matched pair is pulled together against a random one.
Both are cross-entropies over [positive, negative] similarities / temperature, plus
    + KL(pi || pi_before_aux) + (V(s) - R)^2     (keep the policy and value from drifting, as in PPG).
"""
import time
from collections import deque

import numpy as np
import torch as th
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

from model import kl


def as_float(obs_u8, device):
    return obs_u8.to(device).float().div_(255.0)


class Rollout:
    """`nstep` steps of `nproc` envs. Index t holds o_t and what the agent did at o_t."""

    def __init__(self, nstep, nproc, hidsize, device):
        z = lambda *shape, dtype=th.float32: th.zeros(*shape, dtype=dtype, device=device)
        self.nstep = nstep
        self.obs = z(nstep + 1, nproc, 3, 64, 64)
        self.states = z(nstep + 1, nproc, hidsize)          # AD achievement state, input to the heads
        self.successes = z(nstep + 1, nproc, 22, dtype=th.long)
        self.masks = th.ones(nstep + 1, nproc, 1, device=device)  # 0: o_t is the first frame of an episode
        self.actions = z(nstep, nproc, 1, dtype=th.long)
        self.rewards, self.log_probs = z(nstep, nproc, 1), z(nstep, nproc, 1)
        self.vpreds = z(nstep + 1, nproc, 1)
        self.returns, self.advs = z(nstep, nproc, 1), z(nstep, nproc, 1)

    def insert(self, t, model, out, obs, rewards, masks, successes):
        states = self.states[t]
        new = (successes != self.successes[t]).any(-1, keepdim=True)
        if model.ad and new.any():  # new unlock: state = normalise(enc(o_{t+1}) - enc(o_t))
            with th.no_grad():
                diff = F.normalize(model.encode(obs) - out["latents"], dim=-1)
            states = th.where(new, diff, states)
        done = masks == 0
        self.obs[t + 1] = obs
        self.states[t + 1] = th.where(done, 0, states)
        self.successes[t + 1] = th.where(done, 0, successes)
        self.masks[t + 1] = masks
        self.actions[t], self.rewards[t] = out["actions"], rewards
        self.log_probs[t], self.vpreds[t] = out["log_probs"], out["vpreds"]

    def compute_returns(self, gamma, lam):
        gae = 0
        for t in reversed(range(self.nstep)):
            delta = self.rewards[t] + gamma * self.vpreds[t + 1] * self.masks[t + 1] - self.vpreds[t]
            gae = delta + gamma * lam * self.masks[t + 1] * gae
            self.returns[t] = gae + self.vpreds[t]
            self.advs[t] = gae
        self.advs = (self.advs - self.advs.mean()) / (self.advs.std() + 1e-8)

    def minibatches(self, nbatch):
        n = self.nstep * self.obs.shape[1]
        flat = lambda x: x[:self.nstep].reshape(n, *x.shape[2:])
        data = {"obs": flat(self.obs), "states": flat(self.states), "actions": flat(self.actions),
                "returns": flat(self.returns), "log_probs": flat(self.log_probs), "advs": flat(self.advs)}
        for idx in th.randperm(n, device=self.obs.device).chunk(nbatch):
            yield {k: v[idx] for k, v in data.items()}

    def segment(self):
        """What the AD buffer keeps of this rollout (on the CPU, frames as uint8)."""
        return {"obs": (self.obs[:-1] * 255).round().byte().cpu(), "states": self.states[:-1].cpu(),
                "actions": self.actions.cpu(), "returns": self.returns.cpu(),
                "masks": self.masks[:-1].cpu(), "successes": self.successes[:-1].cpu()}

    def reset(self):  # the last step becomes the first step of the next rollout
        for x in (self.obs, self.states, self.successes, self.masks):
            x[0] = x[-1]


class PPO:
    def __init__(self, model, lr=3e-4, epochs=3, nbatch=8, clip=0.2, vf_coef=0.5, ent_coef=0.01,
                 max_grad_norm=0.5, aux_every=8, aux_epochs=6, pi_dist_coef=1.0, vf_dist_coef=1.0,
                 micro_batch=0):
        self.model = model
        self.epochs, self.nbatch, self.clip = epochs, nbatch, clip
        self.vf_coef, self.ent_coef, self.max_grad_norm = vf_coef, ent_coef, max_grad_norm
        self.aux_every, self.aux_epochs = aux_every, aux_epochs
        self.pi_dist_coef, self.vf_dist_coef = pi_dist_coef, vf_dist_coef
        self.micro_batch = micro_batch
        self.opt = th.optim.Adam(model.parameters(), lr=lr)
        if model.ad:  # AD gives each auxiliary loss its own Adam
            self.match_opt = th.optim.Adam(model.parameters(), lr=lr)
            self.pred_opt = th.optim.Adam(model.parameters(), lr=lr)
        self.buffer = deque(maxlen=aux_every)
        self.updates = 0

    def optimizers(self):
        return {k: getattr(self, k) for k in ("opt", "match_opt", "pred_opt") if hasattr(self, k)}

    def _step(self, opt, batch, loss_fn):
        """One optimizer step on `batch`. With `micro_batch`, gradients are accumulated over chunks
        (same gradient, less GPU memory); the value normaliser's EWMA is then updated k times with
        beta^(1/k), which keeps its time constant per optimizer step."""
        n = len(next(iter(batch.values())))
        mb = self.micro_batch or n
        norm = self.model.vf_head.normalizer
        beta0, norm.beta = norm.beta, norm.beta ** (1.0 / -(-n // mb))
        opt.zero_grad()
        stats = {}
        for s in range(0, n, mb):
            w = min(mb, n - s) / n
            loss, parts = loss_fn({k: v[s:s + mb] for k, v in batch.items()})
            (loss * w).backward()
            for k, v in parts.items():
                stats[k] = stats.get(k, 0.0) + w * v.item()
        norm.beta = beta0
        clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        opt.step()
        return stats

    def update(self, roll):
        model = self.model
        model.train()
        if model.ad:
            self.buffer.append(roll.segment())

        def ppo_loss(mb):
            out = model(mb["obs"], mb["states"])
            logp = out["pi_logits"].gather(-1, mb["actions"])
            ratio = (logp - mb["log_probs"]).exp()
            adv = mb["advs"]
            pi_loss = -th.min(ratio * adv, ratio.clamp(1 - self.clip, 1 + self.clip) * adv).mean()
            entropy = -(out["pi_logits"].exp() * out["pi_logits"]).sum(-1).mean()
            vf_loss = model.vf_head.loss(out["vpreds"], mb["returns"]).mean()
            loss = pi_loss + self.vf_coef * vf_loss - self.ent_coef * entropy
            return loss, {"pi_loss": pi_loss, "vf_loss": vf_loss, "entropy": entropy}

        stats, n = {}, 0
        for _ in range(self.epochs):
            for mb in roll.minibatches(self.nbatch):
                for k, v in self._step(self.opt, mb, ppo_loss).items():
                    stats[k] = stats.get(k, 0.0) + v
                n += 1
        stats = {k: v / n for k, v in stats.items()}
        self.updates += 1
        if model.ad and self.updates % self.aux_every == 0:
            stats.update(self.aux_phase())
        return stats

    # ------------------------------------------------------------------ AD auxiliary phase
    def _episodes(self):
        """Split the buffered rollouts into complete episodes and find their achievement steps."""
        cat = {k: th.cat([s[k] for s in self.buffer]) for k in self.buffer[0]}
        trajs = []
        for p in range(cat["obs"].shape[1]):
            starts = (cat["masks"][:, p, 0] == 0).nonzero().squeeze(-1).tolist()
            for s, e in zip(starts[:-1], starts[1:]):  # drop the partial first / last episode
                tr = {k: v[s:e, p] for k, v in cat.items() if k != "masks"}
                succ = tr["successes"]
                g = (succ[1:] != succ[:-1]).any(-1).nonzero().squeeze(-1) + 1  # o_g = first frame after an unlock
                tr["goal_steps"], tr["goal_obs"], tr["goal_next"] = g, tr["obs"][g - 1], tr["obs"][g]
                trajs.append(tr)
        return trajs

    def aux_phase(self):
        t0, model = time.time(), self.model
        dev = next(model.parameters()).device
        trajs = [tr for tr in self._episodes() if len(tr["goal_steps"]) > 0]
        model.eval()  # the policy before the aux phase, for the KL term (no grad, computed once)
        with th.no_grad():
            for tr in trajs:
                tr["old_logits"] = th.cat([model(as_float(tr["obs"][i:i + 1024], dev), tr["states"][i:i + 1024].to(dev))
                                           ["pi_logits"].cpu() for i in range(0, len(tr["obs"]), 1024)])
        model.train()
        totals, counts = {}, {"match": 0, "pred": 0}
        for _ in range(self.aux_epochs):
            for kind, opt, batches in (("match", self.match_opt, self._match_batches(trajs, dev)),
                                       ("pred", self.pred_opt, self._pred_batches(trajs, dev))):
                for batch, loss_fn in batches:
                    for k, v in self._step(opt, batch, loss_fn).items():
                        totals[k] = totals.get(k, 0.0) + v
                    counts[kind] += 1
        n = max(counts["match"] + counts["pred"], 1)
        return {"match_loss": totals.get("match_loss", 0) / max(counts["match"], 1),
                "pred_loss": totals.get("pred_loss", 0) / max(counts["pred"], 1),
                "pi_dist": totals.get("pi_dist", 0) / n, "vf_dist": totals.get("vf_dist", 0) / n,
                "aux_episodes": len(trajs), "aux_s": time.time() - t0}

    def _distill(self, out, old_logits, returns):
        """KL to the pre-aux policy and value regression to the rollout's value targets."""
        return (kl(out["pi_logits"], old_logits).mean(),
                self.model.vf_head.loss(out["vpreds"], returns).mean())

    def _contrast(self, anchor, pos, neg):
        logits = th.stack([(anchor * pos).sum(-1), (anchor * neg).sum(-1)], -1) / self.model.temperature
        return F.cross_entropy(logits, th.zeros(len(logits), dtype=th.long, device=logits.device))

    def _pred_batches(self, trajs, dev, bs=512):
        """Within-episode prediction: anchor = embedding of the next achievement after step t
        (after the last one: the final frame against an empty frame)."""
        model = self.model
        for i in th.randperm(len(trajs)).tolist():
            tr = trajs[i]
            T = len(tr["obs"])
            seg = th.searchsorted(tr["goal_steps"], th.arange(T), right=True)  # index of the next goal
            anc_obs = th.cat([tr["goal_obs"], tr["obs"][-1:]])
            anc_next = th.cat([tr["goal_next"], th.zeros_like(tr["obs"][-1:])])
            neg_all = th.randperm(T)
            for idx in th.randperm(T).split(bs):
                neg = neg_all[idx]
                u, inv = seg[idx].unique(return_inverse=True)
                with th.no_grad():
                    anchor = model.get_states(as_float(anc_obs[u], dev), as_float(anc_next[u], dev))[inv]
                pick = lambda j: {"obs": as_float(tr["obs"][j], dev), "states": tr["states"][j].to(dev),
                                  "actions": tr["actions"][j].to(dev), "returns": tr["returns"][j].to(dev),
                                  "old_logits": tr["old_logits"][j].to(dev)}
                batch = {"anchor": anchor, **pick(idx), **{"neg_" + k: v for k, v in pick(neg).items()}}
                yield batch, self._pred_loss

    def _pred_loss(self, mb):
        model = self.model
        pos = model(mb["obs"], mb["states"])
        neg = model(mb["neg_obs"], mb["neg_states"])
        p = model.next_goal_pred(pos["latents"], mb["actions"], mb["states"])
        q = model.next_goal_pred(neg["latents"], mb["neg_actions"], mb["neg_states"])
        pred_loss = self._contrast(mb["anchor"], p, q)
        both = {k: th.cat([pos[k], neg[k]]) for k in ("pi_logits", "vpreds")}
        pi_dist, vf_dist = self._distill(both, th.cat([mb["old_logits"], mb["neg_old_logits"]]),
                                         th.cat([mb["returns"], mb["neg_returns"]]))
        loss = pred_loss + self.pi_dist_coef * pi_dist + self.vf_dist_coef * vf_dist
        return loss, {"pred_loss": pred_loss, "pi_dist": pi_dist, "vf_dist": vf_dist}

    def _match_batches(self, trajs, dev, bs=512, ntarget=16, reg=0.05):
        """Cross-episode matching: each source episode's achievements are paired with those of up to
        16 other episodes by entropic partial optimal transport (POT); pairs with plan weight > 0.5 count."""
        from ot.partial import entropic_partial_wasserstein
        model = self.model
        for i in th.randperm(len(trajs)).tolist():
            src = trajs[i]
            others = [trajs[j + (j >= i)] for j in th.randperm(len(trajs) - 1)[:ntarget].tolist()]
            if not others:
                continue
            g_obs, g_next = as_float(src["goal_obs"], dev), as_float(src["goal_next"], dev)
            b_obs = as_float(th.cat([t["goal_obs"] for t in others]), dev)
            b_next = as_float(th.cat([t["goal_next"] for t in others]), dev)
            with th.no_grad():
                s_src, bank = model.get_states(g_obs, g_next), model.get_states(b_obs, b_next)
            anc, pos, neg, off = [], [], [], 0
            for t in others:
                n_t = len(t["goal_steps"])
                cost = (1 - s_src @ bank[off:off + n_t].T).cpu().numpy()
                plan = entropic_partial_wasserstein(np.ones(len(s_src)), np.ones(n_t), cost, reg=reg, numItermax=100)
                rows, cols = th.where(th.from_numpy(plan) > 0.5)
                anc.append(rows)
                pos.append(cols + off)
                neg.append(th.randint(n_t, (len(cols),)) + off)
                off += n_t
            anc, pos, neg = th.cat(anc), th.cat(pos), th.cat(neg)
            if len(anc) == 0:
                continue
            steps = th.randint(len(src["obs"]), (len(anc),))  # random steps of the source for distillation

            def loss_fn(mb, g_obs=g_obs, g_next=g_next):
                u, inv = mb["anc"].unique(return_inverse=True)  # each distinct source goal encoded once
                anchor = model.get_states(g_obs[u], g_next[u])[inv]
                match_loss = self._contrast(anchor, mb["pos"], mb["neg"])
                pi_dist, vf_dist = self._distill(model(mb["obs"], mb["states"]), mb["old_logits"], mb["returns"])
                loss = match_loss + self.pi_dist_coef * pi_dist + self.vf_dist_coef * vf_dist
                return loss, {"match_loss": match_loss, "pi_dist": pi_dist, "vf_dist": vf_dist}

            for k, idx in enumerate(th.randperm(len(anc)).split(bs)):
                p_i, n_i = pos[idx].to(dev), neg[idx].to(dev)
                if k == 0:  # the bank was encoded with the current parameters
                    p_s, n_s = bank[p_i], bank[n_i]
                else:       # parameters changed: re-encode the goals this batch uses
                    used, inv = th.cat([p_i, n_i]).unique(return_inverse=True)
                    with th.no_grad():
                        st = model.get_states(b_obs[used], b_next[used])[inv]
                    p_s, n_s = st[:len(idx)], st[len(idx):]
                j = steps[idx]
                yield {"anc": anc[idx].to(dev), "pos": p_s, "neg": n_s, "obs": as_float(src["obs"][j], dev),
                       "states": src["states"][j].to(dev), "old_logits": src["old_logits"][j].to(dev),
                       "returns": src["returns"][j].to(dev)}, loss_fn
