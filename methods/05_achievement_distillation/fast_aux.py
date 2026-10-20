"""Faster, mathematically identical auxiliary phase for Achievement Distillation.

Upstream (algorithm/ppo_ad.py, model/ppo_ad.py) re-runs work whose result cannot change:
  * old_model (frozen for the whole aux phase) is re-evaluated on every sample in every
    one of the 6 aux epochs -> here its policy logits are computed once per aux phase;
  * anchor goal embeddings are computed once per *sample*, although a trajectory only
    has (#goals + 1) distinct anchors -> here each distinct goal is encoded once per
    batch and gathered (with gradients flowing through the gather in the matching loss);
  * positive / negative goal embeddings in the matching loss: likewise, unique goals only.
The encoder has no dropout / batch-norm (GroupNorm is per-sample), so encoding a goal
once and indexing gives the same values and gradients as encoding every copy.
`check_equivalence()` verifies losses and gradients against upstream on real data.
"""
import torch as th
import torch.nn.functional as F
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

from ot.partial import entropic_partial_wasserstein
import numpy as np


@th.no_grad()
def precompute_old_logits(buffer, old_model, device, bs=1024):
    for traj in buffer.trajs:
        out = []
        for s in range(0, len(traj["obs"]), bs):
            o = traj["obs"][s:s + bs].to(device)
            st = traj["old_states"][s:s + bs].to(device)
            out.append(old_model.forward(o, states=st)["pi_logits"].cpu())
        traj["old_pi_logits"] = th.cat(out)


def _segment_anchors(traj):
    """Per-step index of the 'next goal' anchor, and the (G+1) distinct anchor pairs."""
    obs, goal_steps = traj["obs"], traj["goal_steps"]
    T = len(obs)
    seg = th.searchsorted(goal_steps.sort().values, th.arange(T), right=True)
    anc_obs = th.cat([traj["goal_obs"], obs[-1:]], 0)
    anc_next = th.cat([traj["goal_next_obs"], th.zeros_like(obs[-1:])], 0)
    return seg, anc_obs, anc_next


def pred_batches(buffer, model, device, max_batch_size=512):
    """Same sampling as Buffer.get_pred_data_loader, anchors encoded once per distinct goal."""
    ntraj = len(buffer.trajs)
    for i in th.randperm(ntraj):
        traj = buffer.trajs[i]
        if len(traj["goal_steps"]) == 0:
            continue
        obs, actions = traj["obs"], traj["actions"]
        old_states, old_vtargs, old_logits = traj["old_states"], traj["old_vtargs"], traj["old_pi_logits"]
        seg, anc_obs, anc_next = _segment_anchors(traj)
        ndata = len(obs)
        rand_steps = th.randperm(ndata)
        sampler = BatchSampler(SubsetRandomSampler(range(ndata)), batch_size=max_batch_size, drop_last=False)
        for inds in sampler:
            inds = th.as_tensor(inds)
            neg = rand_steps[inds]
            u, inv = th.unique(seg[inds], return_inverse=True)
            with th.no_grad():  # current params, as upstream (computed right before the step)
                anc_states = model.get_states(anc_obs[u].to(device), anc_next[u].to(device))[inv]
            yield {
                "anc_states": anc_states,
                "pos_obs": obs[inds].to(device), "pos_actions": actions[inds].to(device),
                "pos_old_states": old_states[inds].to(device), "pos_old_vtargs": old_vtargs[inds].to(device),
                "pos_old_logits": old_logits[inds].to(device),
                "neg_obs": obs[neg].to(device), "neg_actions": actions[neg].to(device),
                "neg_old_states": old_states[neg].to(device), "neg_old_vtargs": old_vtargs[neg].to(device),
                "neg_old_logits": old_logits[neg].to(device),
            }


def pred_losses(model, anc_states, pos_obs, pos_actions, pos_old_states, pos_old_vtargs, pos_old_logits,
                neg_obs, neg_actions, neg_old_states, neg_old_vtargs, neg_old_logits):
    """PPOADModel.compute_pred_losses with precomputed anchors / old logits."""
    pos_out = model.forward(pos_obs, states=pos_old_states)
    pos_preds = model.get_next_goal_preds(pos_out["latents"], pos_actions, pos_old_states)
    neg_out = model.forward(neg_obs, states=neg_old_states)
    neg_preds = model.get_next_goal_preds(neg_out["latents"], neg_actions, neg_old_states)
    logits = th.stack([th.einsum("bk,bk->b", anc_states, pos_preds),
                       th.einsum("bk,bk->b", anc_states, neg_preds)], dim=-1) / model.temperature
    pred_loss = F.cross_entropy(logits, th.zeros(len(logits), dtype=th.long, device=logits.device))
    pi = th.cat([pos_out["pi_logits"], neg_out["pi_logits"]], 0)
    old_pi = th.cat([pos_old_logits, neg_old_logits], 0)
    pi_dist = model.pi_head.kl_divergence(pi, old_pi).mean()
    vf_dist = model.vf_head.mse_loss(th.cat([pos_out["vpreds"], neg_out["vpreds"]], 0),
                                     th.cat([pos_old_vtargs, neg_old_vtargs], 0)).mean()
    return {"pred_loss": pred_loss, "pi_dist": pi_dist, "vf_dist": vf_dist}


def match_batches(buffer, model, device, max_batch_size=512, ntarget=16):
    """Same matching / sampling as Buffer.get_match_data_loader.

    The goals of the 16 target trajectories are encoded in ONE batched call (upstream: one
    call per target; same parameters, so same values) and reused as positive / negative
    states for the first batch of a source (upstream recomputes them inside the loss with
    the same, not-yet-updated parameters). Later batches of that source re-encode them.
    Yields (batch, src_goals): batch rows index the source goals (anc_idx) and the target
    goal bank (pos_idx / neg_idx); src_goals = (goal_obs, goal_next_obs) of the source.
    """
    trajs = [t for t in buffer.trajs if len(t["goal_steps"]) > 0]
    ntraj = len(trajs)
    for i in th.randperm(ntraj):
        traj_s = trajs[i]
        gobs_s = traj_s["goal_obs"].to(device)
        gnext_s = traj_s["goal_next_obs"].to(device)
        targets = []
        for j in th.randperm(ntraj - 1)[:ntarget]:
            if j >= i:
                j += 1
            targets.append(trajs[j])
        sizes = [len(t["goal_obs"]) for t in targets]
        bank_obs = th.cat([t["goal_obs"] for t in targets]).to(device)
        bank_next = th.cat([t["goal_next_obs"] for t in targets]).to(device)
        with th.no_grad():
            states_s = model.get_states(gobs_s, gnext_s)
            bank_states = model.get_states(bank_obs, bank_next)
        anc_idx, pos_idx, neg_idx = [], [], []
        offset = 0
        for n_t in sizes:
            states_t = bank_states[offset:offset + n_t]
            a = np.ones(len(states_s))
            b = np.ones(n_t)
            M = 1 - th.einsum("ik,jk->ij", states_s, states_t).cpu().numpy()
            T = entropic_partial_wasserstein(a, b, M, reg=0.05, numItermax=100)
            T = th.from_numpy(T).float()
            row_inds, col_inds = th.where(T > 0.5)
            if len(row_inds) > 0:
                anc_idx.append(row_inds)
                pos_idx.append(col_inds + offset)
                neg_idx.append(th.randint(n_t, (len(col_inds),)) + offset)
            offset += n_t
        if len(anc_idx) == 0:
            continue
        anc_idx = th.cat(anc_idx)
        pos_idx = th.cat(pos_idx).to(device)
        neg_idx = th.cat(neg_idx).to(device)
        ndata = len(anc_idx)
        rand_inds = th.randint(len(traj_s["obs"]), (ndata,))
        obs = traj_s["obs"][rand_inds]
        old_states = traj_s["old_states"][rand_inds]
        old_vtargs = traj_s["old_vtargs"][rand_inds]
        old_logits = traj_s["old_pi_logits"][rand_inds]
        sampler = BatchSampler(SubsetRandomSampler(range(ndata)), batch_size=max_batch_size, drop_last=False)
        for k, inds in enumerate(sampler):
            inds = th.as_tensor(inds)
            p_i, n_i = pos_idx[inds.to(device)], neg_idx[inds.to(device)]
            if k == 0:
                pos_states, neg_states = bank_states[p_i], bank_states[n_i]
            else:  # parameters changed since the bank was encoded
                used, inv = th.unique(th.cat([p_i, n_i]), return_inverse=True)
                with th.no_grad():
                    st = model.get_states(bank_obs[used], bank_next[used])[inv]
                pos_states, neg_states = st[:len(inds)], st[len(inds):]
            batch = {
                "anc_idx": anc_idx[inds].to(device),
                "pos_states": pos_states, "neg_states": neg_states,
                "obs": obs[inds].to(device), "old_states": old_states[inds].to(device),
                "old_vtargs": old_vtargs[inds].to(device), "old_logits": old_logits[inds].to(device),
            }
            yield batch, (gobs_s, gnext_s)


def match_losses(model, src_goals, anc_idx, pos_states, neg_states, obs, old_states, old_vtargs, old_logits):
    """PPOADModel.compute_match_losses; anchors = distinct source goals encoded once (with grad)."""
    u, inv = th.unique(anc_idx, return_inverse=True)
    anc_states = model.get_states(src_goals[0][u], src_goals[1][u])[inv]
    logits = th.stack([th.einsum("bk,bk->b", anc_states, pos_states),
                       th.einsum("bk,bk->b", anc_states, neg_states)], dim=-1) / model.temperature
    match_loss = F.cross_entropy(logits, th.zeros(len(logits), dtype=th.long, device=logits.device))
    out = model.forward(obs, states=old_states)
    pi_dist = model.pi_head.kl_divergence(out["pi_logits"], old_logits).mean()
    vf_dist = model.vf_head.mse_loss(out["vpreds"], old_vtargs).mean()
    return {"match_loss": match_loss, "pi_dist": pi_dist, "vf_dist": vf_dist}


def _rel_grad_diff(m1, m2):
    """max |g1 - g2| / max |g1| over all parameter tensors."""
    num = max((p1.grad - p2.grad).abs().max().item() for p1, p2 in zip(m1.parameters(), m2.parameters())
              if p1.grad is not None)
    den = max(p1.grad.abs().max().item() for p1 in m1.parameters() if p1.grad is not None)
    return num / den


def check_equivalence(model, old_model, buffer, device):
    """Compare losses + gradients of the fast path against upstream on one real batch each."""
    import copy
    report = {}
    traj = next(t for t in buffer.trajs if len(t["goal_steps"]) > 1)
    n = min(len(traj["obs"]), 64)
    inds = th.arange(n)
    neg = th.flip(inds, [0])
    nxt_obs, nxt_next = buffer.get_next_goals(traj["goal_steps"], traj["goal_obs"], traj["goal_next_obs"], traj["obs"])
    up = {k: v.to(device) for k, v in {
        "anc_goal_obs": nxt_obs[inds], "anc_goal_next_obs": nxt_next[inds],
        "pos_obs": traj["obs"][inds], "pos_actions": traj["actions"][inds],
        "pos_old_states": traj["old_states"][inds], "pos_old_vtargs": traj["old_vtargs"][inds],
        "neg_obs": traj["obs"][neg], "neg_actions": traj["actions"][neg],
        "neg_old_states": traj["old_states"][neg], "neg_old_vtargs": traj["old_vtargs"][neg]}.items()}
    seg, anc_obs, anc_next = _segment_anchors(traj)
    u, inv = th.unique(seg[inds], return_inverse=True)
    m1, m2 = copy.deepcopy(model), copy.deepcopy(model)
    l1 = m1.compute_pred_losses(**up, old_model=old_model)
    with th.no_grad():
        anc = m2.get_states(anc_obs[u].to(device), anc_next[u].to(device))[inv]
    l2 = pred_losses(m2, anc, up["pos_obs"], up["pos_actions"], up["pos_old_states"], up["pos_old_vtargs"],
                     traj["old_pi_logits"][inds].to(device), up["neg_obs"], up["neg_actions"],
                     up["neg_old_states"], up["neg_old_vtargs"], traj["old_pi_logits"][neg].to(device))
    sum(l1.values()).backward()
    sum(l2.values()).backward()
    g = _rel_grad_diff(m1, m2)
    report["pred"] = {k: (l1[k].item(), l2[k].item()) for k in l1} | {"rel_grad_diff": g}

    # matching loss: anchors are source goals repeated, positives/negatives arbitrary goals
    G = len(traj["goal_obs"])
    anc_idx = th.arange(n) % G
    m1, m2 = copy.deepcopy(model), copy.deepcopy(model)
    gobs, gnext = traj["goal_obs"].to(device), traj["goal_next_obs"].to(device)
    pidx, nidx = (th.arange(n) + 1) % G, (th.arange(n) + 2) % G
    l1 = m1.compute_match_losses(gobs[anc_idx], gnext[anc_idx], gobs[pidx], gnext[pidx], gobs[nidx], gnext[nidx],
                                 up["pos_obs"], up["pos_old_states"], up["pos_old_vtargs"], old_model=old_model)
    with th.no_grad():
        ps = m2.get_states(gobs[pidx], gnext[pidx])
        ns = m2.get_states(gobs[nidx], gnext[nidx])
    l2 = match_losses(m2, (gobs, gnext), anc_idx.to(device), ps, ns, up["pos_obs"], up["pos_old_states"],
                      up["pos_old_vtargs"], traj["old_pi_logits"][inds].to(device))
    sum(l1.values()).backward()
    sum(l2.values()).backward()
    g = _rel_grad_diff(m1, m2)
    report["match"] = {k: (l1[k].item(), l2[k].item()) for k in l1} | {"rel_grad_diff": g}
    return report
