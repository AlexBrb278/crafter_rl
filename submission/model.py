"""Policy / value network: IMPALA ResNet encoder, actor and critic heads, and the extra parts
Achievement Distillation (AD, Moon et al., NeurIPS 2023) needs.

Parameter names follow the AD reference code, so its checkpoints load into this model.
"""
import math

import torch as th
import torch.nn as nn
import torch.nn.functional as F


class FanInLayer(nn.Module):
    """[norm] -> conv or linear -> [ReLU]. Each output unit's weights are rescaled to norm `scale`."""

    def __init__(self, cin, cout, kind="conv", scale=1.0, norm=None, act=True, **kw):
        super().__init__()
        self.norm = {"group": lambda: nn.GroupNorm(1, cin), "layer": lambda: nn.LayerNorm(cin)}[norm]() if norm else None
        self.layer = (nn.Conv2d if kind == "conv" else nn.Linear)(cin, cout, bias=self.norm is None, **kw)
        w = self.layer.weight.data
        w *= scale / w.norm(dim=tuple(range(1, w.ndim)), p=2, keepdim=True)
        if self.layer.bias is not None:
            self.layer.bias.data.zero_()
        self.act = act

    def forward(self, x):
        if self.norm is not None:
            x = self.norm(x)
        x = self.layer(x)
        return F.relu(x, inplace=True) if self.act else x


class ResBlock(nn.Module):
    def __init__(self, ch, scale):
        super().__init__()
        s = math.sqrt(scale)
        self.conv0 = FanInLayer(ch, ch, scale=s, norm="group", kernel_size=3, padding=1)
        self.conv1 = FanInLayer(ch, ch, scale=s, norm="group", kernel_size=3, padding=1)

    def forward(self, x):
        return x + self.conv1(self.conv0(x))


class DownStack(nn.Module):
    """conv -> 3x3 max-pool (stride 2) -> GroupNorm -> `nblock` residual blocks."""

    def __init__(self, cin, cout, nblock, scale, first_norm):
        super().__init__()
        self.firstconv = FanInLayer(cin, cout, norm="group" if first_norm else None, kernel_size=3, padding=1)
        self.n = nn.GroupNorm(1, cout)
        self.blocks = nn.ModuleList([ResBlock(cout, scale / math.sqrt(nblock)) for _ in range(nblock)])

    def forward(self, x):
        x = self.n(F.max_pool2d(self.firstconv(x), kernel_size=3, stride=2, padding=1))
        for b in self.blocks:
            x = b(x)
        return x


class ImpalaCNN(nn.Module):
    """64x64x3 -> [64, 128, 128] channels at 32/16/8 px -> LayerNorm + dense 8192 -> 256."""

    def __init__(self, cin=3, chans=(64, 128, 128), nblock=2, outsize=256):
        super().__init__()
        self.stacks = nn.ModuleList()
        for i, c in enumerate(chans):
            self.stacks.append(DownStack(cin, c, nblock, 1 / math.sqrt(len(chans)), first_norm=i > 0))
            cin = c
        self.dense = FanInLayer(cin * 8 * 8, outsize, kind="linear", scale=1.4, norm="layer")

    def forward(self, x):
        for s in self.stacks:
            x = s(x)
        return self.dense(x.flatten(1))


class MLP(nn.Module):
    """insize -> hidsize (LayerNorm, ReLU) -> outsize (linear)."""

    def __init__(self, insize, hidsize, outsize):
        super().__init__()
        self.layers = nn.ModuleList([FanInLayer(insize, hidsize, "linear", scale=1.4, norm="layer"),
                                     FanInLayer(hidsize, outsize, "linear", act=False)])

    def forward(self, x):
        return self.layers[1](self.layers[0](x))


class PolicyHead(nn.Module):
    def __init__(self, insize, n):
        super().__init__()
        self.linear = nn.Linear(insize, n)
        nn.init.orthogonal_(self.linear.weight, gain=0.01)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        return F.log_softmax(self.linear(x), dim=-1)  # log pi(a|s)


class Normalizer(nn.Module):
    """Running (EWMA, debiased) mean / variance of the value targets. The critic predicts
    normalised returns; the normaliser updates every time it sees targets in training mode."""

    def __init__(self, beta=0.99, eps=1e-2):
        super().__init__()
        self.beta, self.eps = beta, eps
        self.register_buffer("running_mean", th.zeros(1))
        self.register_buffer("running_mean_sq", th.zeros(1))
        self.register_buffer("debiasing_term", th.tensor(0.0))

    def stats(self):
        d = self.debiasing_term.clamp(min=self.eps)
        mean, mean_sq = self.running_mean / d, self.running_mean_sq / d
        return mean, (mean_sq - mean ** 2).clamp(min=1e-2).sqrt()

    def forward(self, x):
        if self.training:
            b = self.beta
            self.running_mean.mul_(b).add_((1 - b) * x.detach().mean(0))
            self.running_mean_sq.mul_(b).add_((1 - b) * (x.detach() ** 2).mean(0))
            self.debiasing_term.mul_(b).add_(1 - b)
        mean, std = self.stats()
        return (x - mean) / std

    def denormalize(self, x):
        mean, std = self.stats()
        return x * std + mean


class ValueHead(nn.Module):
    def __init__(self, insize):
        super().__init__()
        self.linear = nn.Linear(insize, 1)
        nn.init.orthogonal_(self.linear.weight, gain=0.1)
        nn.init.zeros_(self.linear.bias)
        self.normalizer = Normalizer()

    def forward(self, x):
        return self.linear(x)  # normalised value

    def loss(self, pred, target):
        return F.mse_loss(pred, self.normalizer(target), reduction="none")


class Agent(nn.Module):
    """Actor-critic on 64x64 frames.

    With `ad=True` (Achievement Distillation) the heads also see a 1024-d "achievement state":
    the normalised latent difference enc(o_{t+1}) - enc(o_t) at the last achievement unlocked in
    the episode (zero before the first one). Two extra MLPs serve AD's auxiliary losses: a FiLM
    action embedding and a predictor of the next achievement's embedding.
    """

    def __init__(self, hidsize=1024, ad=True, temperature=0.1):
        super().__init__()
        self.enc = ImpalaCNN()
        self.linear = FanInLayer(256, hidsize, "linear", norm="layer")
        self.hidsize, self.ad, self.temperature = hidsize, ad, temperature
        head_in = 2 * hidsize if ad else hidsize
        self.pi_head = PolicyHead(head_in, 17)
        self.vf_head = ValueHead(head_in)
        if ad:
            self.action_mlp = MLP(17, hidsize, 2 * hidsize)
            self.next_goal_pred_mlp = MLP(2 * hidsize, hidsize, hidsize)

    def encode(self, obs):
        return self.linear(self.enc(obs))

    def forward(self, obs, states):
        latents = self.encode(obs)
        x = th.cat([latents, states], -1) if self.ad else latents
        return {"latents": latents, "pi_logits": self.pi_head(x), "vpreds": self.vf_head(x)}

    @th.no_grad()
    def act(self, obs, states):
        out = self.forward(obs, states)
        logits = out["pi_logits"]
        out["actions"] = th.distributions.Categorical(logits=logits).sample().unsqueeze(-1)
        out["log_probs"] = logits.gather(-1, out["actions"])
        out["vpreds"] = self.vf_head.normalizer.denormalize(out["vpreds"])
        return out

    # ---- Achievement Distillation ----
    def get_states(self, goal_obs, goal_next_obs):
        """Achievement embedding: normalise(enc(o after the unlock) - enc(o before)).
        An all-zero frame stands for "no frame" and is encoded as a zero vector."""
        def enc(o):
            z = self.encode(o)
            return th.where((o.flatten(1) == 0).all(-1, keepdim=True), 0, z)
        return F.normalize(enc(goal_next_obs) - enc(goal_obs), dim=-1)

    def next_goal_pred(self, latents, actions, states):
        """Predicted embedding of the next achievement from (latent, action, achievement state)."""
        gamma, beta = self.action_mlp(F.one_hot(actions.squeeze(-1), 17).float()).chunk(2, -1)
        x = th.cat([(1 + gamma) * latents + beta, states], -1)  # FiLM by the action
        return F.normalize(self.next_goal_pred_mlp(x), dim=-1)


def kl(logits_new, logits_old):
    """KL(pi_new || pi_old) from log-probabilities, per sample."""
    return (logits_new.exp() * (logits_new - logits_old)).sum(-1)
