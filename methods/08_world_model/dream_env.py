"""Stage 2 (paper/designs/world_model_stage2.md): many dreams in parallel, as an environment for PPO.

RealTracker   follows the real envs during training: each episode's tile tokens, inventory, actions, clock
              and pixel-odometry position (common/pixel_map.py), so any real moment can start a dream.
Dreamer       steps D dreams at once with a frozen world model (tile tokens + map, Stage 1e v4). Frame
              embeddings of the 64-step context are cached (the spatial part of `history` is per frame),
              each dream keeps its own TileMap; tokens and the vitals are sampled at `temperature`, item
              counts take their most likely value. With `craft` (craft_layer.CraftLayer): make actions follow
              the learned shared craft rule, tool counts change only through it, the HUD rows are redrawn
              from the dreamed inventory, and the world model reads (item, count) pairs it never saw in
              training as the nearest count it did.
DreamRollout  fills a RolloutStorage with the current policy acting in D dreams of `nstep` steps, starting
              from real states of the last epoch (a share of them "frontier" states, see FRONTIER; optionally
              part of that share where iron is in hand, see iron_in_hand, part of that where the craft is
              ready, see ready_to_craft, and part of that where it is possible now, see craftable_now; with
              `pool_size` the last two also draw from a pool of such moments kept across epochs). Reward
              is the game's (first unlocks + 0.1 x health change) plus the rarity bonus; an unlock counts
              only if the inventory output agrees (GATE). A dreamed death resets that slot to a new start.
"""
import math
import os
import sys

import numpy as np
import torch as th
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from build_maps import TileMap  # noqa: E402
from dynamics import DAY, POS_CLIP  # noqa: E402

INVENTORY = ["health", "food", "drink", "energy", "sapling", "wood", "stone", "coal", "iron", "diamond",
             "wood_pickaxe", "stone_pickaxe", "iron_pickaxe", "wood_sword", "stone_sword", "iron_sword"]
INIT_INV = np.array([9, 9, 9, 9] + [0] * 12, np.uint8)
CONTEXT = 64
POS_WEIGHT = 10.0  # dynamics.py trains the unlock and death heads with this pos_weight: their odds are x10
# achievement -> (inventory item, +1 the count must rise / -1 fall); missing = no inventory trace (always counts)
GATE = {"collect_coal": ("coal", 1), "collect_diamond": ("diamond", 1), "collect_drink": ("drink", 1),
        "collect_iron": ("iron", 1), "collect_sapling": ("sapling", 1), "collect_stone": ("stone", 1),
        "collect_wood": ("wood", 1), "eat_cow": ("food", 1), "eat_plant": ("food", 1),
        "make_iron_pickaxe": ("iron_pickaxe", 1), "make_iron_sword": ("iron_sword", 1),
        "make_stone_pickaxe": ("stone_pickaxe", 1), "make_stone_sword": ("stone_sword", 1),
        "make_wood_pickaxe": ("wood_pickaxe", 1), "make_wood_sword": ("wood_sword", 1),
        "place_furnace": ("stone", -1), "place_plant": ("sapling", -1), "place_stone": ("stone", -1),
        "place_table": ("wood", -1)}


# gate mode 2: these unlocks are read off the dreamed inventory (count rises, not yet unlocked this episode).
# The unlock head cannot know the episode's unlocked set, and it learned from competent agents that e.g.
# wood is never a *new* unlock late in an episode; the inventory is unambiguous for these 12 (drink is a
# sampled vital, so its noise would read as drinking: collect_drink stays with the head + check).
INV_DERIVED = ("collect_coal", "collect_diamond", "collect_iron", "collect_sapling", "collect_stone",
               "collect_wood", "make_iron_pickaxe", "make_iron_sword", "make_stone_pickaxe", "make_stone_sword",
               "make_wood_pickaxe", "make_wood_sword")


def frontier(successes, names):
    """(N, 22) unlocked flags -> (N,) bool: on the iron chain (stone pickaxe made, no diamond yet)."""
    return (successes[:, names.index("make_stone_pickaxe")] > 0) & (successes[:, names.index("collect_diamond")] == 0)


def iron_in_hand(inv):
    """(N, 16) inventories -> (N,) bool: iron held and an iron tool still missing (the next step is an iron craft)."""
    i = INVENTORY.index
    return (inv[:, i("iron")] >= 1) & ((inv[:, i("iron_pickaxe")] == 0) | (inv[:, i("iron_sword")] == 0))


def ready_to_craft(inv, tok, station):
    """(N, 16) inventories, (N, 9, 9) tile tokens (tensor), (2, codes) station codes -> (N,) bool array: iron in hand
    (iron_in_hand), wood and coal held, and a table and a furnace both in view (the 9 x 7 map rows: within 4
    tiles sideways, 3 up or down). Measured in iron_diag.py --near 4: such starts end in an iron tool far
    more often than iron-in-hand starts in general."""
    i = INVENTORY.index
    seen = station[:, tok[:, :7].reshape(len(tok), -1)].any(-1).cpu().numpy()  # (2, N): table, furnace in view
    return iron_in_hand(inv) & (inv[:, i("wood")] >= 1) & (inv[:, i("coal")] >= 1) & seen[0] & seen[1]


def craftable_now(inv, tok, station):
    """As ready_to_craft, with the table and the furnace both in the 3 x 3 around the player (tokens[2:5, 3:6],
    the craft rule's window): pressing make-iron works now. Measured in iron_diag.py --reach."""
    near = station[:, tok[:, 2:5, 3:6].reshape(len(tok), -1)].any(-1).cpu().numpy()
    return ready_to_craft(inv, tok, station) & near[0] & near[1]


def calibrate(p):
    """Undo the x POS_WEIGHT odds of a head trained with BCE pos_weight, so sampled events occur at their rate."""
    return th.sigmoid(th.logit(p.clamp(1e-6, 1 - 1e-6)) - math.log(POS_WEIGHT))


class Track:
    """One real episode as the world model sees it: per frame tokens (9, 9), inventory, clock, position;
    act[t] is the action taken at frame t (-1 until known). unlocks: (u, frame u - 1, frame u) at each new
    unlock, the two real frames behind the AD state from frame u on (storage.insert)."""

    def __init__(self):
        self.tok, self.inv, self.time, self.pos, self.act = [], [], [], [], []
        self.unlocks = []

    def add(self, tok, inv, t, pos):
        self.tok.append(tok)
        self.inv.append(inv)
        self.time.append(t)
        self.pos.append(pos)
        self.act.append(-1)


class RealTracker:
    def __init__(self, n_envs, tokenizer, device):
        from common.pixel_map import PixelMap
        self.tok, self.dev = tokenizer, device
        self.maps = [PixelMap() for _ in range(n_envs)]
        self.tracks = [Track() for _ in range(n_envs)]
        self.last = [None] * n_envs  # last real frame and unlocked set per env, for Track.unlocks
        self.had = [set() for _ in range(n_envs)]
        self.refs = []  # per storage step: [(track, t)] for each env, aligned with storage.obs

    def _encode(self, frames):
        with th.no_grad():
            return self.tok.encode(th.from_numpy(frames).to(self.dev)).cpu().numpy().astype(np.int16)

    def current(self):
        return [(tr, len(tr.tok) - 1) for tr in self.tracks]

    def reset(self, frames):  # (N, 64, 64, 3) uint8 first observations
        codes = self._encode(frames)
        for i, f in enumerate(frames):
            self.maps[i].reset_episode()
            self.maps[i].observe(f)
            self.tracks[i] = Track()
            self.tracks[i].add(codes[i], INIT_INV.copy(), 0, self.maps[i].pos.copy())
            self.last[i], self.had[i] = f.copy(), set()
        self.refs = [self.current()]

    def begin_epoch(self):
        self.refs = [self.current()]

    def step(self, frames, actions, dones, infos):
        """frames: observations after the step (the reset frame where done), as the real env returns them."""
        codes = self._encode(frames)
        for i, f in enumerate(frames):
            tr = self.tracks[i]
            tr.act[-1] = int(actions[i])
            if dones[i]:
                self.maps[i].reset_episode()
                self.maps[i].observe(f)
                self.tracks[i] = tr = Track()
                tr.add(codes[i], INIT_INV.copy(), 0, self.maps[i].pos.copy())
                self.had[i] = set()
            else:
                self.maps[i].observe(f, actions[i])
                inv = np.array([infos[i]["inventory"][k] for k in INVENTORY]).clip(0, 255).astype(np.uint8)
                tr.add(codes[i], inv, int(infos[i]["t"]), self.maps[i].pos.copy())
                had = {k for k, v in infos[i]["achievements"].items() if v > 0}
                if had - self.had[i]:
                    tr.unlocks.append((len(tr.tok) - 1, self.last[i], f.copy()))
                self.had[i] = had
            self.last[i] = f.copy()
        self.refs.append(self.current())


class Dreamer:
    def __init__(self, wm, tokenizer, D, device, temperature=1.0, craft=None):
        self.wm, self.tok, self.D, self.dev, self.T = wm, tokenizer, D, device, temperature
        self.craft = craft
        if craft is not None:  # (item, count) -> the nearest count at or below it that the world model was trained on
            seen = (craft.hud >= 0).cpu().numpy()
            seen[:, 0] = True
            remap = np.maximum.accumulate(np.where(seen, np.arange(10)[None], 0), axis=1)
            self.inv_in = th.from_numpy(remap).to(device)
        g = wm.grid
        self.tokens = th.zeros(D, CONTEXT, *g, dtype=th.long, device=device)
        self.emb = None
        self.inv = th.zeros(D, CONTEXT, 16, dtype=th.long, device=device)
        self.act = th.zeros(D, CONTEXT, dtype=th.long, device=device)
        self.time = th.zeros(D, CONTEXT, dtype=th.long, device=device)
        self.pos = th.zeros(D, CONTEXT, 2, dtype=th.long, device=device)
        self.maps = [None] * D

    @th.no_grad()
    def _embed(self, tok, chunk=2048):  # (..., 9, 9) -> (..., d_t), the per-frame part of WorldModel.history
        lead, flat = tok.shape[:-2], tok.reshape(-1, *tok.shape[-2:])
        with th.autocast("cuda", dtype=th.bfloat16):  # in chunks: 256 dreams x 64 frames at once needs ~2.5 GB
            out = th.cat([self.wm.embed_frames(flat[i:i + chunk]).float() for i in range(0, len(flat), chunk)])
        return out.reshape(*lead, -1)

    @th.no_grad()
    def load(self, slots, starts):
        """Fill dream slots from real moments: starts = [(track, t)] with t >= CONTEXT - 1."""
        if self.emb is None:
            self.emb = th.zeros(self.D, CONTEXT, self.wm.temporal.layers[0].linear1.in_features, device=self.dev)
        for d, (tr, t) in zip(slots, starts):
            s = slice(t - CONTEXT + 1, t + 1)
            self.tokens[d] = th.from_numpy(np.stack(tr.tok[s]).astype(np.int64))
            self.inv[d] = th.from_numpy(np.stack(tr.inv[s]).astype(np.int64)).clamp(max=9)
            self.act[d] = th.from_numpy(np.maximum(np.array(tr.act[s]), 0))
            self.time[d] = th.from_numpy(np.array(tr.time[s]))
            self.pos[d] = th.from_numpy(np.stack(tr.pos[s]).astype(np.int64))
            m = TileMap(self.wm.n_codes)
            for k in range(t + 1):
                m.write(tr.tok[k], tr.pos[k], tr.time[k])
            self.maps[d] = m
        idx = th.as_tensor(list(slots), device=self.dev)
        self.emb[idx] = self._embed(self.tokens[idx])

    def _history(self):
        """WorldModel.history with cached frame embeddings -> h (D, d_t) at the last position."""
        wm = self.wm
        inv = self.inv if self.craft is None else self.inv_in[th.arange(16, device=self.dev), self.inv]
        x = self.emb + wm.inv_emb(inv + wm.inv_offset).sum(-2) + wm.act_emb(self.act)
        if wm.use_clock:
            x = x + wm.clock_emb(self.time % DAY)
        if wm.use_pos:
            rel = (self.pos - self.pos[:, :1]).clamp(-POS_CLIP, POS_CLIP) + POS_CLIP
            x = x + wm.pos_x(rel[..., 0]) + wm.pos_y(rel[..., 1])
        x = x + wm.pos_t[:CONTEXT]
        causal = th.triu(th.full((CONTEXT, CONTEXT), float("-inf"), device=x.device), 1)
        return wm.norm_t(wm.temporal(x, mask=causal, is_causal=True))[:, -1].float()

    @th.no_grad()
    def step(self, actions):
        """actions (D,) taken at the current frame -> next frames (D, 64, 64, 3) uint8, inventory (D, 16),
        unlock probabilities (D, 22), death probabilities (D,). Advances every dream by one step."""
        wm = self.wm
        self.act[:, -1] = actions
        with th.autocast("cuda", dtype=th.bfloat16):
            h = self._history()
            move = wm.move_delta[wm.odo_head(h).argmax(-1)]
            nxt = self.pos[:, -1] + move
            now = self.time[:, -1]
            mt, ma = zip(*(m.read(p, t) for m, p, t in zip(self.maps, nxt.cpu().numpy(), now.cpu().numpy())))
            prior = wm.map_prior(th.from_numpy(np.stack(mt).astype(np.int64)).to(self.dev),
                                 th.from_numpy(np.stack(ma).astype(np.int64)).to(self.dev))
            # sampled: vitals (health, food, drink, energy: hidden timers) and saplings (grass drops one 10% of the
            # time); other item counts change only by deliberate actions: most likely count (sampling them makes
            # diamonds appear from noise)
            nt, ni, ach, done = wm.predict_next(h, prior=prior, temperature=self.T, inv_sampled=range(5))
        if self.craft is not None:
            ni = self._craft(ni, actions)
            nt = self.craft.draw_hud(nt, ni)
        nt_np = nt.cpu().numpy()
        for m, g, p, t in zip(self.maps, nt_np, nxt.cpu().numpy(), (now + 1).cpu().numpy()):
            m.write(g, p, t)
        roll = lambda x, new: th.cat([x[:, 1:], new[:, None]], 1)  # noqa: E731
        self.tokens, self.inv = roll(self.tokens, nt), roll(self.inv, ni)
        self.act, self.time, self.pos = roll(self.act, th.zeros_like(actions)), roll(self.time, now + 1), roll(self.pos, nxt)
        self.emb = roll(self.emb, self._embed(nt))
        frames = self.tok.decode(nt)
        return frames, ni, calibrate(ach.float()), calibrate(done.float())

    def _craft(self, ni, actions):
        """Tools keep their count except through a make action (24,271 tool changes in v4's data, all on the
        tool's own make action); on a make action the craft layer decides the non-vital items."""
        ni = ni.clone()
        cur = self.inv[:, -1]
        ni[:, 10:16] = cur[:, 10:16]
        rows = ((actions >= 11) & (actions <= 16)).nonzero()[:, 0]
        if len(rows):
            nxt, p = self.craft.apply(cur[rows], self.tokens[rows, -1], self.emb[rows, -1], actions[rows])
            ni[rows, 4:] = nxt[:, 4:]
            self.last_craft_p = (rows, p)
        return ni


class DreamRollout:
    def __init__(self, wm, tokenizer, D, nstep, device, names, temperature=1.0, frontier_share=0.5,
                 gate=True, seed=0, craft=None, iron_share=0.0, ready_share=0.0, reach_share=0.0, pool_size=0,
                 pool_age=25):
        self.dreamer = Dreamer(wm, tokenizer, D, device, temperature, craft)
        self.D, self.nstep, self.dev, self.names = D, nstep, device, names
        self.iron_share = iron_share  # part of frontier_share started where iron is in hand (iron_in_hand)
        self.ready_share = ready_share  # part of iron_share started where the craft is ready (ready_to_craft)
        self.reach_share = reach_share  # part of ready_share started where the craft works now (craftable_now)
        assert ready_share + reach_share == 0 or craft is not None, "ready / craftable starts need the craft layer's station codes"
        # ready and craftable moments of past epochs (each kind: the last pool_size, at most pool_age epochs old)
        self.pool, self.pool_size, self.pool_age, self.epoch = [], pool_size, pool_age, 0
        self.frontier_share, self.gate = frontier_share, int(gate)  # 0 head only, 1 head + inventory check, 2 see INV_DERIVED
        self.rng = np.random.default_rng(seed)
        self.gate_idx = [(names.index(a), INVENTORY.index(item), sign) for a, (item, sign) in GATE.items()]
        self.derived = {names.index(a) for a in INV_DERIVED}
        self.stats = {}

    def _candidates(self, refs, storage):
        """All (storage step j, env i) whose real moment has a full context; which of them are frontier,
        which have iron in hand, which are ready to craft and which can craft now."""
        js, is_ = [], []
        for j in range(len(refs) - 1):  # the last obs is the next epoch's first; its storage row is reset
            for i, (tr, t) in enumerate(refs[j]):
                if t >= CONTEXT - 1:
                    js.append(j)
                    is_.append(i)
        if not js:
            return None
        js, is_ = np.array(js), np.array(is_)
        succ = storage.successes[js, is_].cpu().numpy()
        inv = np.stack([refs[j][i][0].inv[refs[j][i][1]] for j, i in zip(js, is_)])
        ir, rd, rc = iron_in_hand(inv), np.zeros(len(js), bool), np.zeros(len(js), bool)
        if self.ready_share + self.reach_share > 0 and ir.any():
            tok = th.as_tensor(np.stack([refs[j][i][0].tok[refs[j][i][1]] for j, i in zip(js[ir], is_[ir])]).astype(np.int64)).to(self.dev)
            rd[ir] = ready_to_craft(inv[ir], tok, self.dreamer.craft.station)
            rc[ir] = craftable_now(inv[ir], tok, self.dreamer.craft.station)
        return js, is_, frontier(succ, self.names), ir, rd, rc

    def _pick(self, cand, n):
        """n starts. Tiers nest (craftable now within ready within iron in hand within frontier); the two rare
        tiers also draw from the pool. pick[k] < N: candidate pick[k] of this epoch; else pool entry pick[k] - N."""
        js, is_, fr, ir, rd, rc = cand
        N, P = len(js), self.pool
        rd_all = np.concatenate([np.where(rd)[0], N + np.arange(len(P))]).astype(np.int64)
        rc_all = np.concatenate([np.where(rc)[0], N + np.array([k for k, e in enumerate(P) if e["rc"]], np.int64)]).astype(np.int64)
        k_reach = int(round(n * self.reach_share)) if len(rc_all) else 0
        k_ready = max(int(round(n * self.ready_share)) - k_reach, 0) if len(rd_all) else 0
        k_iron = max(int(round(n * self.iron_share)) - k_reach - k_ready, 0) if ir.any() else 0
        k_front = max(int(round(n * self.frontier_share)) - k_reach - k_ready - k_iron, 0) if fr.any() else 0

        def take(where, k):
            return self.rng.choice(where, k) if k else np.zeros(0, np.int64)
        pick = np.concatenate([take(rc_all, k_reach), take(rd_all, k_ready), take(np.where(ir)[0], k_iron),
                               take(np.where(fr)[0], k_front),
                               self.rng.choice(N, n - k_reach - k_ready - k_iron - k_front)]).astype(np.int64)
        cur, c, q = pick < N, np.minimum(pick, N - 1), np.clip(pick - N, 0, None)
        pf = np.array([e["fr"] for e in P] + [False], bool)  # padded: an empty pool still indexes
        pc = np.array([e["rc"] for e in P] + [False], bool)
        return {"pick": pick, "pool": ~cur, "fr": np.where(cur, fr[c], pf[q]), "ir": np.where(cur, ir[c], True),
                "rd": np.where(cur, rd[c], True), "rc": np.where(cur, rc[c], pc[q])}

    def _fresh_states(self, model, entries, hid):
        """AD state of pool starts with the current encoder: the two frames of the last unlock at or before t
        (zeros if none, as at an episode start)."""
        st = th.zeros(len(entries), hid, device=self.dev)
        last = [(k, next((x for x in reversed(e["track"].unlocks) if x[0] <= e["t"]), None)) for k, e in enumerate(entries)]
        last = [(k, x) for k, x in last if x is not None]
        if last:
            f = np.stack([x[1] for _, x in last] + [x[2] for _, x in last])
            lat = model.encode(th.from_numpy(f).to(self.dev).permute(0, 3, 1, 2).float() / 255.0)
            st[[k for k, _ in last]] = F.normalize(lat[len(last):] - lat[:len(last)], dim=-1)
        return st

    def _place(self, slots, pk, row, cand, dstore, storage, refs, model):
        """Load the picked starts into dream slots and their row `row` of `dstore`."""
        js, is_ = cand[0], cand[1]
        N, pick, P = len(js), pk["pick"], self.pool
        slots = np.asarray(slots)
        self.dreamer.load(slots, [refs[js[p]][is_[p]] if p < N else (P[p - N]["track"], P[p - N]["t"]) for p in pick])
        cur, old = np.where(pick < N)[0], np.where(pick >= N)[0]
        if len(cur):
            s = th.as_tensor(slots[cur], device=self.dev)
            j, i = th.as_tensor(js[pick[cur]], device=self.dev), th.as_tensor(is_[pick[cur]], device=self.dev)
            for name in ("obs", "states", "successes", "timesteps"):
                dstore[name][row, s] = storage[name][j, i]
        if len(old):
            s, es = th.as_tensor(slots[old], device=self.dev), [P[p - N] for p in pick[old]]
            dstore.obs[row, s] = th.stack([e["obs"] for e in es]).to(self.dev).float() / 255.0
            dstore.successes[row, s] = th.stack([e["succ"] for e in es]).to(self.dev)
            dstore.timesteps[row, s] = th.stack([e["ts"] for e in es]).to(self.dev)
            dstore.states[row, s] = self._fresh_states(model, es, dstore.states.shape[-1])

    def _remember(self, cand, refs, storage, model):
        """Add this epoch's ready moments to the pool, then keep per kind (ready / craftable now) the last
        pool_size entries at most pool_age epochs old. Check: the AD state recomputed for the new entries
        matches the one in storage (mean cosine, stats["pool_state_cos"])."""
        js, is_, fr, ir, rd, rc = cand
        new = np.where(rd)[0]
        self.stats["pool_state_cos"] = None
        if len(new):
            j, i = th.as_tensor(js[new], device=self.dev), th.as_tensor(is_[new], device=self.dev)
            obs = storage.obs[j, i].mul(255).round().byte().cpu()
            succ, ts = storage.successes[j, i].cpu(), storage.timesteps[j, i].cpu()
            for k, n in enumerate(new):
                tr, t = refs[js[n]][is_[n]]
                self.pool.append({"track": tr, "t": t, "obs": obs[k], "succ": succ[k], "ts": ts[k], "fr": bool(fr[n]),
                                  "rc": bool(rc[n]), "epoch": self.epoch})
            fresh = self._fresh_states(model, self.pool[-len(new):], storage.states.shape[-1])
            self.stats["pool_state_cos"] = float(F.cosine_similarity(fresh, storage.states[j, i], dim=-1, eps=1e-6).mean())
        keep = [e for e in self.pool if self.epoch - e["epoch"] < self.pool_age]
        self.pool = [e for e in keep if not e["rc"]][-self.pool_size:] + [e for e in keep if e["rc"]][-self.pool_size:]

    @th.no_grad()
    def run(self, model, dstore, storage, refs, rarity_rates=None, beta=0.0):
        """Fill `dstore` (nstep x D) with dreamed experience; returns False if there are no usable starts yet."""
        cand = self._candidates(refs, storage)
        if cand is None:
            return False
        self.epoch += 1
        D, dv = self.D, self.dreamer
        model.eval()
        pk = self._pick(cand, D)
        dstore.step = 0
        self._place(range(D), pk, 0, cand, dstore, storage, refs, model)
        dstore.masks[0].fill_(1.0)
        rate = th.tensor([0.0 if rarity_rates is None else rarity_rates[a] for a in self.names], device=self.dev)
        s = {"steps": 0, "unlocks": 0, "rejected": 0, "deaths": 0, "health_change": 0.0, "reward": 0.0,
             "frontier_starts": int(pk["fr"].sum()), "iron_starts": int(pk["ir"].sum()), "ready_starts": int(pk["rd"].sum()),
             "reach_starts": int(pk["rc"].sum()), "pool_starts": int(pk["pool"].sum()), "starts": D,
             "iron_candidates": int(cand[3].sum()), "ready_candidates": int(cand[4].sum()),
             "reach_candidates": int(cand[5].sum()), "candidates": len(cand[0]), "pool_size": len(self.pool),
             "pool_reach": sum(e["rc"] for e in self.pool)}
        per_ach = th.zeros(len(self.names), device=self.dev)
        rej_ach = th.zeros(len(self.names), device=self.dev)
        for step in range(self.nstep):
            out = model.act(**dstore.get_inputs(step))
            a = out["actions"].squeeze(-1)
            inv_prev = dv.inv[:, -1].clone()
            frames, inv, ach_p, done_p = dv.step(a)
            ev = th.bernoulli(ach_p).bool()
            prev_succ = dstore.successes[step] > 0
            ev &= ~prev_succ
            if self.gate:
                ok = th.ones_like(ev)
                for ai, ii, sign in self.gate_idx:
                    d = inv[:, ii] - inv_prev[:, ii]
                    good = (d > 0) | (inv_prev[:, ii] >= 9) if sign > 0 else (d < 0)
                    ok[:, ai] = good
                    if self.gate == 2 and ai in self.derived:  # the count rose: that is the unlock
                        ev[:, ai] = ((d > 0) | (ev[:, ai] & (inv_prev[:, ii] >= 9))) & ~prev_succ[:, ai]
                s["rejected"] += int((ev & ~ok).sum())
                rej_ach += (ev & ~ok).float().sum(0)
                ev &= ok
            health_d = (inv[:, 0] - inv_prev[:, 0]).float()
            dead = th.bernoulli(done_p).bool() | (inv[:, 0] <= 0)
            r = ev.float().sum(-1) + 0.1 * health_d + beta * (ev.float() * (1 - rate)).sum(-1)
            succ = (prev_succ | ev).long()
            obs = frames.permute(0, 3, 1, 2).float().div_(255.0)
            out["obs"], out["rewards"] = obs, r[:, None]
            out["masks"] = 1.0 - dead.float()[:, None]
            out["successes"] = succ
            dstore.insert(**out, model=model)
            s["steps"] += D
            s["unlocks"] += int(ev.sum())
            s["deaths"] += int(dead.sum())
            s["health_change"] += float(health_d.sum())
            s["reward"] += float(r.sum())
            per_ach += ev.float().sum(0)
            if dead.any() and step < self.nstep - 1:  # new real start in that slot; the mask cuts the return
                slots = dead.nonzero().squeeze(1).cpu().numpy()
                npk = self._pick(cand, len(slots))
                self._place(slots, npk, dstore.step, cand, dstore, storage, refs, model)  # insert advanced step: that row holds the next inputs
                for key, f in (("frontier", "fr"), ("iron", "ir"), ("ready", "rd"), ("reach", "rc"), ("pool", "pool")):
                    s[key + "_starts"] += int(npk[f].sum())
                s["starts"] += len(slots)
        with th.no_grad():  # value of the last dreamed frame, for the bootstrap at the cut
            dstore.vpreds[-1].copy_(model.act(**dstore.get_inputs(self.nstep))["vpreds"])
        s["per_ach"] = {a: float(v) for a, v in zip(self.names, per_ach.cpu().numpy()) if v > 0}
        s["rejected_ach"] = {a: float(v) for a, v in zip(self.names, rej_ach.cpu().numpy()) if v > 0}
        self.stats = s
        if self.pool_size:
            self._remember(cand, refs, storage, model)
        return True


class MixedStorage:
    """Hands PPO minibatches that are half real, half dreamed. Everything else (AD's buffer) sees real data."""

    def __init__(self, real, dream):
        self.real, self.dream = real, dream

    def __getitem__(self, key):
        return self.real[key]

    def __getattr__(self, name):
        return getattr(self.real, name)

    def get_data_loader(self, nbatch):
        for a, b in zip(self.real.get_data_loader(nbatch), self.dream.get_data_loader(nbatch)):
            yield {k: th.cat([a[k], b[k]]) for k in a}
