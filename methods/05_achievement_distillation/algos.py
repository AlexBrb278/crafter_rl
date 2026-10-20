"""Micro-batched versions of the upstream PPO / PPO+AD update steps.

The math is the upstream one (achievement_distillation/algorithm/{ppo,ppo_ad}.py @ 9abbce6).
The only change: every minibatch is split into micro-batches of at most `micro_batch`
samples, and their gradients are accumulated before one optimizer step. Each loss is a
mean over the batch, so weighting by (micro size / batch size) gives the same gradient
as the full minibatch while using a fraction of the GPU memory (needed on 4 GB cards).

Optional `amp=True` runs forward passes under fp16 autocast with a GradScaler.
"""
import copy
import time

import torch as th
from torch.nn.utils import clip_grad_norm_

from achievement_distillation.algorithm.ppo import PPOAlgorithm
from achievement_distillation.algorithm.ppo_ad import PPOADAlgorithm


def split(batch, micro_batch):
    n = next(iter(batch.values())).shape[0]
    mb = micro_batch or n
    for s in range(0, n, mb):
        yield {k: v[s:s + mb] for k, v in batch.items()}, min(mb, n - s) / n


class _MicroBatchMixin:
    def _setup_mb(self, micro_batch, amp, aux_micro_batch=None):
        self.micro_batch = micro_batch
        self.aux_micro_batch = aux_micro_batch or micro_batch
        self.amp = amp
        self.scaler = th.amp.GradScaler("cuda", enabled=amp)

    def _step(self, optimizer, batch, loss_fn, micro_batch=None):
        """One optimizer step on `batch`; loss_fn(mb) -> (loss, {stat: tensor})."""
        optimizer.zero_grad()
        stats = {}
        for mb, w in split(batch, micro_batch or self.micro_batch):
            with th.autocast("cuda", dtype=th.float16, enabled=self.amp):
                loss, parts = loss_fn(mb)
            self.scaler.scale(loss * w).backward()
            for k, v in parts.items():
                stats[k] = stats.get(k, 0.0) + w * v.item()
        self.scaler.unscale_(optimizer)
        clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.scaler.step(optimizer)
        self.scaler.update()
        return stats

    def _ppo_phase(self, storage):
        totals, n = {}, 0

        def loss_fn(mb):
            losses = self.model.compute_losses(**mb, clip_param=self.clip_param)
            loss = (losses["pi_loss"] + self.vf_loss_coef * losses["vf_loss"]
                    - self.ent_coef * losses["entropy"])
            return loss, losses

        for _ in range(self.ppo_nepoch):
            for batch in storage.get_data_loader(self.ppo_nbatch):
                for k, v in self._step(self.optimizer, batch, loss_fn).items():
                    totals[k] = totals.get(k, 0.0) + v
                n += 1
        return {k: v / n for k, v in totals.items()}


class MBPPOAlgorithm(_MicroBatchMixin, PPOAlgorithm):
    def __init__(self, model, micro_batch=None, amp=False, **kwargs):
        super().__init__(model, **kwargs)
        self._setup_mb(micro_batch, amp)

    def update(self, storage):
        self.model.train()
        return self._ppo_phase(storage)


class MBPPOADAlgorithm(_MicroBatchMixin, PPOADAlgorithm):
    """aux_micro_batch: micro-batch for the aux phase, where each sample costs several
    encoder passes (pred: positive + negative + anchors; match: anchor/pos/neg goals)."""

    def __init__(self, model, micro_batch=None, amp=False, aux_micro_batch=None,
                 fast_aux=True, check_fast_aux=False, **kwargs):
        super().__init__(model, **kwargs)
        self._setup_mb(micro_batch, amp, aux_micro_batch)
        self.fast_aux = fast_aux              # fast_aux.py: same math, less recomputation
        self.check_fast_aux = check_fast_aux  # print an equivalence report on the first aux phase

    def update(self, storage):
        self.model.train()

        keys = ["obs", "actions", "states", "returns", "masks", "rewards", "successes"]
        self.buffer.insert({key: storage[key].cpu() for key in keys})

        train_stats = self._ppo_phase(storage)
        self.ppo_count += 1
        if self.ppo_count % self.aux_freq != 0:
            return train_stats

        # ---- auxiliary phase (achievement distillation) ----
        self.buffer.parse_segs()
        self.buffer.preprocess_trajs()
        old_model = copy.deepcopy(self.model)
        old_model.eval()

        totals = {"match_loss": 0.0, "pred_loss": 0.0, "pi_dist": 0.0, "vf_dist": 0.0}
        n_match = n_pred = 0

        t_aux = time.time()
        aux_mb = self.aux_micro_batch
        if self.fast_aux:
            return self._fast_aux_phase(train_stats, old_model, t_aux)

        def match_fn(mb):
            out = self.model.compute_match_losses(**mb, old_model=old_model)
            loss = (out["match_loss"] + self.pi_dist_coef * out["pi_dist"]
                    + self.vf_dist_coef * out["vf_dist"])
            return loss, out

        def pred_fn(mb):
            out = self.model.compute_pred_losses(**mb, old_model=old_model)
            loss = (out["pred_loss"] + self.pi_dist_coef * out["pi_dist"]
                    + self.vf_dist_coef * out["vf_dist"])
            return loss, out

        for _ in range(self.aux_nepoch):
            for batch in self.buffer.get_match_data_loader(self.model):
                s = self._step(self.match_optimizer, batch, match_fn, aux_mb)
                totals["match_loss"] += s["match_loss"]
                totals["pi_dist"] += s["pi_dist"]
                totals["vf_dist"] += s["vf_dist"]
                n_match += 1
            for batch in self.buffer.get_pred_data_loader():
                s = self._step(self.pred_optimizer, batch, pred_fn, aux_mb)
                totals["pred_loss"] += s["pred_loss"]
                totals["pi_dist"] += s["pi_dist"]
                totals["vf_dist"] += s["vf_dist"]
                n_pred += 1

        # Upstream divides by these counts unguarded; an empty buffer (no complete
        # episode in the last aux_freq rollouts) would raise ZeroDivisionError.
        train_stats.update({
            "match_loss": totals["match_loss"] / max(n_match, 1),
            "pred_loss": totals["pred_loss"] / max(n_pred, 1),
            "pi_dist": totals["pi_dist"] / max(n_match + n_pred, 1),
            "vf_dist": totals["vf_dist"] / max(n_match + n_pred, 1),
            "aux_trajs": float(len(self.buffer.trajs)),
            "aux_s": time.time() - t_aux,
        })
        del old_model
        return train_stats

    def _fast_aux_phase(self, train_stats, old_model, t_aux):
        import fast_aux as fa
        device = next(self.model.parameters()).device
        # Release blocks cached by the PPO phase: the aux phase uses many different batch
        # shapes, and on Windows (WDDM) over-subscribing VRAM silently pages to host RAM.
        th.cuda.empty_cache()
        fa.precompute_old_logits(self.buffer, old_model, device)
        if self.check_fast_aux:
            self.check_fast_aux = False
            if any(len(t["goal_steps"]) > 1 for t in self.buffer.trajs):
                print("fast_aux equivalence check:", fa.check_equivalence(self.model, old_model, self.buffer, device),
                      flush=True)
        c = (self.pi_dist_coef, self.vf_dist_coef)
        totals = {"match_loss": 0.0, "pred_loss": 0.0, "pi_dist": 0.0, "vf_dist": 0.0}
        n_match = n_pred = 0
        for _ in range(self.aux_nepoch):
            for batch, src in fa.match_batches(self.buffer, self.model, device):
                def match_fn(mb, src=src):
                    out = fa.match_losses(self.model, src, **mb)
                    return out["match_loss"] + c[0] * out["pi_dist"] + c[1] * out["vf_dist"], out
                s = self._step(self.match_optimizer, batch, match_fn, self.aux_micro_batch)
                for k in ("match_loss", "pi_dist", "vf_dist"):
                    totals[k] += s[k]
                n_match += 1
            for batch in fa.pred_batches(self.buffer, self.model, device):
                def pred_fn(mb):
                    out = fa.pred_losses(self.model, **mb)
                    return out["pred_loss"] + c[0] * out["pi_dist"] + c[1] * out["vf_dist"], out
                s = self._step(self.pred_optimizer, batch, pred_fn, self.aux_micro_batch)
                for k in ("pred_loss", "pi_dist", "vf_dist"):
                    totals[k] += s[k]
                n_pred += 1
        train_stats.update({
            "match_loss": totals["match_loss"] / max(n_match, 1),
            "pred_loss": totals["pred_loss"] / max(n_pred, 1),
            "pi_dist": totals["pi_dist"] / max(n_match + n_pred, 1),
            "vf_dist": totals["vf_dist"] / max(n_match + n_pred, 1),
            "aux_trajs": float(len(self.buffer.trajs)),
            "aux_s": time.time() - t_aux,
        })
        del old_model
        return train_stats
