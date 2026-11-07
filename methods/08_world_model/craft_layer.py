"""A shared craft rule for the world model (paper/designs/world_model_craft_rule.md).

The world model's inventory output predicts each item's count with its own weights, so 4 iron-pickaxe crafts
in 1.76M steps teach it nothing. This layer handles the 6 make actions with the rule's shape, learned from the
same data on the frozen v4 trunk:

  stations   which tile codes are a table / furnace, from the agent's own placing events: place_table that uses
             wood, place_furnace that uses stone; the neighbouring tile whose code changed is the new station.
  awake      which codes of the player's own tile allow acting (a sleeping player's presses become sleep),
             learned from presses that met every other requirement.
  recipe     per tier (wood, stone, iron: pickaxe and sword tied, Q16) a learned requirement r_k >= 0 per item
             and a need weight per station; p(works) = sigmoid(beta * soft-min of the margins).
  effect     per tier the mean item change of the presses that worked, rounded; make_X adds one X (Q16).
  hud        (item, count) -> tile code of its HUD cell, the most common code in real frames (Q18).

Semantic labels (`sem`) are used only to report how well the station codes match the truth, never to fit.

  .venv/bin/python methods/08_world_model/craft_layer.py extract     # events from the train / test split
  .venv/bin/python methods/08_world_model/craft_layer.py fit         # stations, awake, recipes, HUD -> craft_layer.pt
  .venv/bin/python methods/08_world_model/craft_layer.py fit --untied
  .venv/bin/python methods/08_world_model/craft_layer.py check       # tests 1, 3, 4 -> results/world_model/craft_layer.json
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import torch as th
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from data import DATA, ROOT, load_episodes  # noqa: E402

INVENTORY = ["health", "food", "drink", "energy", "sapling", "wood", "stone", "coal", "iron", "diamond",
             "wood_pickaxe", "stone_pickaxe", "iron_pickaxe", "wood_sword", "stone_sword", "iron_sword"]
CRAFTS = {11: "wood_pickaxe", 12: "stone_pickaxe", 13: "iron_pickaxe", 14: "wood_sword", 15: "stone_sword",
          16: "iron_sword"}
TIER = {11: 0, 14: 0, 12: 1, 15: 1, 13: 2, 16: 2}  # Q16: pickaxe and sword of a material share a recipe
TIERS = ("wood", "stone", "iron")
PLACE = {8: ("table", "wood"), 9: ("furnace", "stone")}  # place action -> (station, item it uses)
STATIONS = ("table", "furnace")
SEM_ID = {"table": 11, "furnace": 12}  # only for reporting
WIN = (slice(2, 5), slice(3, 6))  # 3 x 3 tokens around the player (row 3, col 4); tokens are [row = y, col = x]
NEIGH4 = [(1, 0), (1, 2), (0, 1), (2, 1)]  # orthogonal neighbours inside the 3 x 3 window
HUD_CELL = [(7 + k // 9, k % 9) for k in range(16)]  # Crafter's ItemView: item k drawn at (k % 9, k // 9)
OUT = os.path.join(ROOT, "runs", "wm", "craft")
TOKENS = os.path.join(ROOT, "runs", "wm", "tokens_tile")
N_CODES = 1024
POS = os.path.join(ROOT, "runs", "wm", "pos_pix")  # v4's positions: online pixel odometry


def extract(split):
    """Every craft press of a split with what the world model sees (tokens, inventory); the station tiles
    found from placing events; HUD codes per (item, count). `sem` and `daylight` are kept for reporting only."""
    eps = load_episodes(split, keys=("actions", "inventory", "sem", "daylight"), extra_dir=TOKENS)
    C = {k: [] for k in ("inv", "tok", "act", "delta", "sem_table", "sem_furnace", "daylight", "ep", "t")}
    n_all = np.zeros(N_CODES, np.int64)                  # world tiles per code, all frames
    n_station = np.zeros((len(STATIONS), N_CODES), np.int64)  # ... at a tracked station position
    sem_check = np.zeros((len(STATIONS), 2), np.int64)  # tracked tiles: (not, is) that station in sem
    hud = np.zeros((16, 10, N_CODES), np.int64)          # (item, count, code)
    n_place = [0, 0]
    for j, e in enumerate(eps):
        if "tokens" not in e:
            continue
        tok = e["tokens"].astype(np.int64)
        inv, act = e["inventory"].astype(np.int64), e["actions"].astype(np.int64)
        pos = np.load(os.path.join(POS, os.path.relpath(e["path"], DATA)))["pos"].astype(np.int64)
        sem = e["sem"].transpose(0, 2, 1)  # (T+1, 7 y, 9 x)
        np.add.at(n_all, tok[:, :7].ravel(), 1)
        for k, (r, c) in enumerate(HUD_CELL):
            np.add.at(hud[k], (inv[:, k].clip(0, 9), tok[:, r, c]), 1)
        for t in np.where(np.isin(act, list(CRAFTS)))[0]:
            w = sem[t][WIN]
            C["inv"].append(inv[t]); C["tok"].append(tok[t].astype(np.uint16)); C["act"].append(act[t])
            C["delta"].append(inv[t + 1] - inv[t]); C["daylight"].append(e["daylight"][t])
            C["sem_table"].append((w == SEM_ID["table"]).any()); C["sem_furnace"].append((w == SEM_ID["furnace"]).any())
            C["ep"].append(j); C["t"].append(t)
        # a placing step that used up its item, with exactly one neighbour changed (by night the noise changes
        # every tile): that tile is the new station; it stays at that world position for the rest of the episode
        placed = []
        for a, (station, item) in PLACE.items():
            i = INVENTORY.index(item)
            for t in np.where((act == a) & (inv[1:, i] < inv[:-1, i]))[0]:
                w0, w1 = tok[t][WIN], tok[t + 1][WIN]
                changed = [(r, c) for r, c in NEIGH4 if w0[r, c] != w1[r, c]]
                if len(changed) == 1:
                    r, c = changed[0]
                    placed.append((STATIONS.index(station), t + 1, pos[t + 1] + (c - 1, r - 1)))
                    n_place[STATIONS.index(station)] += 1
        for s_, t0, at in placed:  # every later frame that shows that position
            rel = at[None] - pos[t0:]
            row, col = 3 + rel[:, 1], 4 + rel[:, 0]
            ok = (row >= 0) & (row < 7) & (col >= 0) & (col < 9)
            tt = np.arange(t0, len(tok))[ok]
            codes = tok[tt, row[ok], col[ok]]
            np.add.at(n_station[s_], codes, 1)
            truth = sem[tt, row[ok], col[ok]] == SEM_ID[STATIONS[s_]]
            sem_check[s_] += [(~truth).sum(), truth.sum()]
    os.makedirs(OUT, exist_ok=True)
    out = {f"craft_{k}": np.array(v) for k, v in C.items()}
    out.update(n_all=n_all, n_station=n_station, sem_check=sem_check, hud_counts=hud, n_place=np.array(n_place))
    np.savez_compressed(os.path.join(OUT, f"events_{split}.npz"), **out)
    print(split, len(eps), "episodes,", len(C["act"]), "craft presses, placements (table, furnace)", n_place,
          "tracked station tiles right in sem:", (sem_check[:, 1] / sem_check.sum(1)).round(4))


VITALS = 4  # health, food, drink, energy: hidden timers, left to the inventory output
PRODUCT = {a: INVENTORY.index(n) for a, n in CRAFTS.items()}  # Q16: make_X gives X


class CraftLayer(nn.Module):
    """p(a craft works) = sigmoid(soft-min of: item margins gamma * (inv_k - r_k + 1/2), station terms, awake).
    Requirement r_k = max(what the craft uses up, a learned extra >= 0): what a craft consumes must be at hand.
    Station term = max(gamma * (in reach - 1/2), gamma * u): u < -1/2 means the station is needed."""

    def __init__(self, d_emb, tau=0.5):
        super().__init__()
        self.tau = tau
        self.rho = nn.Parameter(th.full((3, 16), -4.0))   # learned extra requirement, softplus -> ~0
        self.u = nn.Parameter(th.zeros(3, len(STATIONS)))
        self.log_gamma = nn.Parameter(th.tensor(math.log(4.0)))
        self.awake = nn.Linear(d_emb, 1)
        nn.init.zeros_(self.awake.weight)
        nn.init.constant_(self.awake.bias, 5.0)
        self.register_buffer("uses", th.zeros(3, 16))        # per tier: items used up (>= 0)
        self.register_buffer("effect", th.zeros(6, 16))      # per craft action 11..16: item change if it works
        self.register_buffer("station", th.zeros(len(STATIONS), N_CODES, dtype=th.bool))
        self.register_buffer("hud", th.full((16, 10), -1, dtype=th.long))
        self.register_buffer("tier", th.tensor([TIER[a] for a in sorted(CRAFTS)]))

    def requirement(self):
        return th.maximum(self.uses, nn.functional.softplus(self.rho))

    def terms(self, inv, tok, emb, act):
        """-> (N, 16 + 2 + 1) terms in logit units, for craft actions act in 11..16."""
        g = self.log_gamma.exp()
        k = self.tier[act - 11]
        items = g * (inv.float() - self.requirement()[k] + 0.5)
        win = tok[:, WIN[0], WIN[1]].reshape(len(tok), -1)
        reach = self.station[:, win].any(-1).T.float()  # (N, 2)
        st = th.maximum(g * (reach - 0.5), g * self.u[k])
        return th.cat([items, st, self.awake(emb.float())], -1)

    def logit(self, inv, tok, emb, act):
        x = self.terms(inv, tok, emb, act)
        return -self.tau * th.logsumexp(-x / self.tau, -1)

    def apply(self, inv, tok, emb, act, sample=True):
        """Dream step for the rows whose action is a craft: (N, 16) next non-vital counts and p(works)."""
        p = self.logit(inv, tok, emb, act).sigmoid()
        works = th.bernoulli(p) if sample else (p > 0.5).float()
        nxt = (inv.float() + works[:, None] * self.effect[act - 11]).clamp(0, 9).long()
        return nxt, p

    def draw_hud(self, tok, inv):
        """Replace each HUD cell whose (item, count) was seen in real frames by its most common code."""
        tok = tok.clone()
        code = self.hud[th.arange(16, device=inv.device)[None], inv.clamp(0, 9)]  # (N, 16)
        for k, (r, c) in enumerate(HUD_CELL):
            tok[:, r, c] = th.where(code[:, k] >= 0, code[:, k], tok[:, r, c])
        return tok


def station_codes(d, min_rel=0.2, min_tiles=20):
    """Codes seen at tracked station positions far more often than elsewhere: share of a code's tiles that are
    at a tracked station, relative to the best code (only a part of all stations is tracked, so the shares are
    small in absolute terms)."""
    out = np.zeros((len(STATIONS), N_CODES), bool)
    for s in range(len(STATIONS)):
        ratio = d["n_station"][s] / np.maximum(d["n_all"], 1)
        rel = ratio / ratio[d["n_station"][s] >= 50].max()
        out[s] = (rel > min_rel) & (d["n_station"][s] >= min_tiles)
    return out


def worked(act, delta):
    return np.array([delta[i, PRODUCT[a]] > 0 for i, a in enumerate(act)])


def effects(act, delta, ok):
    """Per tier: mean change of the non-vital items over the presses that worked, rounded, without the tier's
    own products; then make_X adds one X."""
    tiers = np.array([TIER[a] for a in act])
    uses, eff = np.zeros((3, 16)), np.zeros((6, 16))
    for k in range(3):
        m = (tiers == k) & ok
        mean = delta[m].mean(0) if m.any() else np.zeros(16)
        mean[:VITALS] = 0
        for a in CRAFTS:
            if TIER[a] == k:
                mean[PRODUCT[a]] = 0
        uses[k] = np.maximum(-np.round(mean), 0)
        for a in CRAFTS:
            if TIER[a] == k:
                eff[a - 11] = np.round(mean)
                eff[a - 11, PRODUCT[a]] = 1
    return uses, eff


def hud_lookup(counts, min_frames=20):
    best = counts.argmax(-1)
    best[counts.sum(-1) < min_frames] = -1
    return best, (counts.max(-1) / np.maximum(counts.sum(-1), 1))


@th.no_grad()
def frame_embeddings(wm, tok, dev, chunk=4096):
    out = []
    for i in range(0, len(tok), chunk):
        with th.autocast("cuda", dtype=th.bfloat16):
            out.append(wm.embed_frames(th.from_numpy(tok[i:i + chunk].astype(np.int64)).to(dev)).float())
    return th.cat(out)


def fit(args):
    from evaluate import load
    dev = th.device("cuda")
    th.manual_seed(0)
    wm, _, _ = load(args.run, dev)
    d = dict(np.load(os.path.join(OUT, "events_train.npz")))
    act, inv, delta, tok = d["craft_act"], d["craft_inv"], d["craft_delta"], d["craft_tok"]
    ok = worked(act, delta)
    if args.untied:  # every action its own recipe (Q16 untied): tiers become actions
        global TIER
        TIER = {a: i for i, a in enumerate(sorted(CRAFTS))}
    layer = CraftLayer(wm.temporal.layers[0].linear1.in_features).to(dev)
    if args.untied:
        layer.rho = nn.Parameter(th.full((6, 16), -4.0, device=dev))
        layer.u = nn.Parameter(th.zeros(6, len(STATIONS), device=dev))
        layer.tier = th.arange(6, device=dev)
        layer.uses = th.zeros(6, 16, device=dev)
    uses, eff = effects(act, delta, ok) if not args.untied else effects_untied(act, delta, ok)
    layer.uses.copy_(th.from_numpy(uses).float()); layer.effect.copy_(th.from_numpy(eff).float())
    layer.station.copy_(th.from_numpy(station_codes(d)))
    hud, hud_share = hud_lookup(d["hud_counts"])
    layer.hud.copy_(th.from_numpy(hud))
    emb = frame_embeddings(wm, tok, dev)
    I, T_, A, Y = (th.from_numpy(x).to(dev) for x in (inv, tok.astype(np.int64), act, ok.astype(np.float32)))
    opt = th.optim.Adam([p for p in layer.parameters() if p.requires_grad], lr=args.lr)
    for step in range(args.steps):
        logit = layer.logit(I, T_, emb, A)
        bce = nn.functional.binary_cross_entropy_with_logits(logit, Y)
        l1 = nn.functional.softplus(layer.rho).sum()
        loss = bce + args.l1 * l1
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 500 == 0 or step == args.steps - 1:
            print(f"step {step} bce {bce.item():.5f} extra-req {l1.item():.3f} gamma {layer.log_gamma.exp().item():.2f}")
    name = "craft_layer_untied.pt" if args.untied else "craft_layer.pt"
    th.save({"state": layer.state_dict(), "untied": args.untied, "d_emb": emb.shape[1]},
            os.path.join(ROOT, "runs", "wm", "dynamics", args.run.partition(":")[0], name))
    print("saved", name, "HUD cells known", int((hud >= 0).sum()), "of 160")


def effects_untied(act, delta, ok):
    uses, eff = np.zeros((6, 16)), np.zeros((6, 16))
    for a in CRAFTS:
        m = (act == a) & ok
        mean = delta[m].mean(0) if m.any() else np.zeros(16)
        mean[:VITALS] = 0
        mean[PRODUCT[a]] = 0
        uses[a - 11] = np.maximum(-np.round(mean), 0)
        eff[a - 11] = np.round(mean)
        if m.any():  # untied: the product is only known from the action's own successes
            eff[a - 11, PRODUCT[a]] = 1
    return uses, eff


def load_layer(run, dev, untied=False):
    d = os.path.join(ROOT, "runs", "wm", "dynamics", run.partition(":")[0])
    ck = th.load(os.path.join(d, "craft_layer_untied.pt" if untied else "craft_layer.pt"), map_location=dev)
    layer = CraftLayer(ck["d_emb"]).to(dev)
    if ck["untied"]:
        layer.rho = nn.Parameter(th.zeros(6, 16, device=dev)); layer.u = nn.Parameter(th.zeros(6, 2, device=dev))
        layer.uses = th.zeros(6, 16, device=dev); layer.tier = th.arange(6, device=dev)
    layer.load_state_dict(ck["state"])
    return layer.eval()


RECIPES = {11: ({"wood": 1}, ("table",)), 12: ({"wood": 1, "stone": 1}, ("table",)),
           13: ({"wood": 1, "coal": 1, "iron": 1}, ("table", "furnace")), 14: ({"wood": 1}, ("table",)),
           15: ({"wood": 1, "stone": 1}, ("table",)), 16: ({"wood": 1, "coal": 1, "iron": 1}, ("table", "furnace"))}
# Crafter's data.yaml, used only to check what was learned (test 1) and which presses met it (test 3)


@th.no_grad()
def check(args):
    """Tests 1, 3, 4 of the design note on the test split; writes results/world_model/craft_layer.json."""
    from evaluate import load
    dev = th.device("cuda")
    wm, _, _ = load(args.run, dev)
    R = {}
    for untied in (False, True):
        layer = load_layer(args.run, dev, untied)
        req, g = layer.requirement().cpu().numpy(), layer.log_gamma.exp().item()
        u = layer.u.detach().cpu().numpy()
        rec = {}
        for a in sorted(CRAFTS):
            k = layer.tier[a - 11].item()
            items = {INVENTORY[i]: int(np.ceil(req[k, i] - 1e-3)) for i in range(16) if req[k, i] > 0.5}
            stations = tuple(STATIONS[j] for j in range(len(STATIONS)) if u[k, j] < -0.5)
            eff = {INVENTORY[i]: int(v) for i, v in enumerate(layer.effect[a - 11].cpu().numpy()) if v}
            true_items, true_st = RECIPES[a]
            want_eff = {**{n: -v for n, v in true_items.items()}, CRAFTS[a]: 1}
            rec[CRAFTS[a]] = {"requires": items, "stations": list(stations), "effect": eff,
                              "matches_crafter": items == true_items and stations == true_st and eff == want_eff}
        d = dict(np.load(os.path.join(OUT, "events_test.npz")))
        act, inv, delta, tok = d["craft_act"], d["craft_inv"], d["craft_delta"], d["craft_tok"]
        ok = worked(act, delta)
        met = np.array([all(inv[i, INVENTORY.index(n)] >= v for n, v in RECIPES[a][0].items())
                        and d["craft_sem_table"][i] and ("furnace" not in RECIPES[a][1] or d["craft_sem_furnace"][i])
                        for i, a in enumerate(act)])
        emb = frame_embeddings(wm, tok, dev)
        p = layer.logit(*(th.from_numpy(x).to(dev) for x in (inv, tok.astype(np.int64))), emb,
                        th.from_numpy(act).to(dev)).sigmoid().cpu().numpy()
        per = {}
        for a in sorted(CRAFTS):
            m = act == a
            per[CRAFTS[a]] = {"presses": int(m.sum()), "met": int((m & met).sum()), "worked": int((m & ok).sum()),
                              "p_on_worked": round(float(p[m & ok].mean()), 4) if (m & ok).any() else None,
                              "p_on_met_failed": round(float(p[m & met & ~ok].mean()), 4) if (m & met & ~ok).any() else None,
                              "p_on_not_met": float(p[m & ~met].mean())}
        R["untied" if untied else "tied"] = {
            "gamma": round(g, 2), "recipes": rec, "test_presses": per,
            "false_crafts_per_10k_not_met": round(float(p[~met].mean() * 1e4), 3),
            "not_met_presses": int((~met).sum()),
            "max_p_not_met": float(p[~met].max()),
            "recall_worked": round(float(p[ok].mean()), 4), "worked": int(ok.sum()),
            "p_on_met_failed": round(float(p[met & ~ok].mean()), 4), "met_failed": int((met & ~ok).sum())}
    # test 4: HUD lookup on real test frames
    tr = dict(np.load(os.path.join(OUT, "events_train.npz")))
    hud, _ = hud_lookup(tr["hud_counts"])
    te = np.load(os.path.join(OUT, "events_test.npz"))["hud_counts"]  # (16, 10, codes)
    known = hud >= 0
    right = sum(te[k, n, hud[k, n]] for k in range(16) for n in range(10) if known[k, n])
    total_known = te[known].sum()
    R["hud"] = {"cells_right_where_known": round(float(right / max(total_known, 1)), 4),
                "test_cells_with_unknown_pair": round(float(1 - total_known / te.sum()), 5),
                "unknown_pairs": [(INVENTORY[k], n) for k in range(16) for n in range(10) if not known[k, n] and te[k, n].sum()]}
    st = station_codes(tr)
    R["stations"] = {s: {"codes": int(st[j].sum())} for j, s in enumerate(STATIONS)}
    win = te_tok = np.load(os.path.join(OUT, "events_test.npz"))
    w = win["craft_tok"].astype(int)[:, 2:5, 3:6].reshape(len(win["craft_tok"]), -1)
    for j, s in enumerate(STATIONS):
        det, truth = np.isin(w, np.where(st[j])[0]).any(1), win[f"craft_sem_{s}"]
        R["stations"][s].update(recall=round(float(det[truth].mean()), 4), false_detections=int((det & ~truth).sum()),
                                presses_without=int((~truth).sum()))
    out = os.path.join(ROOT, "results", "world_model", "craft_layer.json")
    json.dump(R, open(out, "w"), indent=1)
    print(json.dumps(R, indent=1))


def main(args):
    if args.cmd == "extract":
        for s in args.splits:
            extract(s)
    elif args.cmd == "fit":
        fit(args)
    elif args.cmd == "check":
        check(args)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["extract", "fit", "check"])
    ap.add_argument("--run", default="v4:model_refit.pt")
    ap.add_argument("--splits", nargs="+", default=["train", "test"])
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--l1", type=float, default=1e-3)
    ap.add_argument("--untied", action="store_true", help="Q16 untied: a recipe per action")
    main(ap.parse_args())
