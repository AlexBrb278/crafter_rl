"""Death diagnosis: replay final checkpoints and record exactly what kills the agent.

Every health loss in Crafter comes from one of four places, and we hook each of them:
  Zombie.update   -> "zombie" (7 damage if the player is asleep, else 2)
  Arrow.update    -> "arrow" (skeletons, 2 damage)
  Player.update   -> "lava" (health set to 0) or "hunger"/"thirst"/"fatigue"
                     (food/drink/energy at 0 drains 1 health every ~15 steps)
The cause of death is the source of the hit that took health to 0.

For each run we play `--episodes` fresh episodes with the final weights (stochastic policy,
as in training) and write one row per episode to results/deaths/<run>.jsonl. For `--video_runs`
we also film the first episodes and keep `--videos_per_cause` MP4s per cause of death in
runs/<run>/videos/death_*.mp4 (index: runs/<run>/videos/deaths.jsonl).

  .venv/bin/python scripts/diagnose_deaths.py ppo-s0 ad-rarity2-s0 --episodes 200 \
      --video_runs ad-rarity2-s0
"""
import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
AD = os.path.join(ROOT, "methods", "05_achievement_distillation")

def _json(o):
    return o.item() if hasattr(o, "item") else str(o)


NIGHT = 0.3  # daylight below this counts as night (~30% of the day/night cycle)


def install_hooks():
    """Patch Crafter so every change to the player's health is logged with its source."""
    from crafter import objects

    def wrap(cls, label_fn):
        orig = cls.update

        def update(self):
            player = self.world._damage_player
            before = player.health
            asleep = player.sleeping
            orig(self)
            if player.health < before:
                player._damage_log.append((label_fn(self, player, asleep), before - player.health, asleep))
        cls.update = update

    def zombie_label(obj, player, asleep):
        return "zombie (asleep)" if asleep else "zombie"

    def arrow_label(obj, player, asleep):
        return "arrow"

    def player_label(obj, player, asleep):
        if obj.world[obj.pos][0] == "lava":
            return "lava"
        inv = obj.inventory
        lack = [n for n, k in (("hunger", "food"), ("thirst", "drink"), ("fatigue", "energy")) if inv[k] == 0]
        return "+".join(lack) if lack else "other"

    wrap(objects.Zombie, zombie_label)
    wrap(objects.Arrow, arrow_label)
    wrap(objects.Player, player_label)


def new_env(seed):
    import crafter
    env = crafter.Env(seed=seed)
    obs = env.reset()
    p = env._player
    env._world._damage_player = p
    p._damage_log = []
    return env, obs


def surroundings(env):
    """How exposed the player is: walkable tiles next to it and zombies within 3 tiles."""
    from crafter import constants, objects
    p, w = env._player, env._world
    x, y = p.pos
    open_n = 0
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        mat, obj = w[(x + dx, y + dy)]
        if mat in constants.walkable and obj is None:
            open_n += 1
    zombies = sum(1 for o in w.objects if isinstance(o, objects.Zombie) and p.distance(o) <= 3)
    return open_n, zombies


def load_model(run, ckpt="agent-e250.pt"):
    import gymnasium
    import torch as th
    from achievement_distillation.model import PPOADModel, PPOModel
    from recurrent import PPOADGRUModel
    cfg = json.load(open(os.path.join(ROOT, "runs", run, "config.json")))["config"]
    cls = {"PPOADModel": PPOADModel, "PPOModel": PPOModel, "PPOADGRUModel": PPOADGRUModel}[cfg["model_cls"]]
    model = cls(observation_space=gymnasium.spaces.Box(0, 1, (3, 64, 64)),
                action_space=gymnasium.spaces.Discrete(17), **cfg["model_kwargs"])
    sd = th.load(os.path.join(ROOT, "runs", run, "checkpoints", ckpt),
                 map_location="cpu", weights_only=False)
    model.load_state_dict(sd.get("model", sd) if isinstance(sd, dict) else sd)
    return model.eval(), cfg


def write_death_video(path, frames, rec, run):
    from common.video import HUD, write_mp4
    hud = HUD(384)
    out, prev = [], set()
    for t, (frame, inv, ach, daylight) in enumerate(frames, 1):
        new = sorted(ach - prev)
        prev = ach
        phase = "NIGHT" if daylight < NIGHT else "day"
        out.append(hud.draw(frame, f"{run}  final weights   t={t}   {phase}",
                            f"health {inv['health']}  food {inv['food']}  drink {inv['drink']}  "
                            f"energy {inv['energy']}   achievements: {len(ach)}/22", new))
    banner = f"DIED: {rec['cause']}" + (" at night" if rec["night"] else " by day")
    last = out[-1]
    for _ in range(30):  # hold the last frame ~2 s with the cause of death
        out.append(hud.draw(last[48:], banner, f"step {rec['length']}, "
                            f"{len(rec['achievements'])} achievements", ()))
    write_mp4(path, out, fps=15)


def diagnose(run, episodes, n_envs, video, videos_per_cause, device, seed0, ckpt="agent-e250.pt",
             keep_if=(), out_dir=None):
    import torch as th
    import torch.nn.functional as F
    model, cfg = load_model(run, ckpt)
    model.to(device)
    is_ad = cfg["model_cls"] in ("PPOADModel", "PPOADGRUModel")
    gru = cfg["model_cls"] == "PPOADGRUModel"  # carry the GRU state per env, zero at episode start
    hid = cfg["model_kwargs"]["hidsize"]
    out_path = os.path.join(out_dir or os.path.join(ROOT, "results", "deaths"), f"{run}.jsonl")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    open(out_path, "w").close()
    vid_dir = os.path.join(ROOT, "runs", run, "videos")
    kept = {}  # cause -> count of videos written
    if video:
        os.makedirs(vid_dir, exist_ok=True)
        if not keep_if:
            open(os.path.join(vid_dir, "deaths.jsonl"), "w").close()

    rng = np.random.default_rng(seed0)
    started = 0
    envs, obs, states, frames, stats, hiddens = [], [], [], [], [], []

    def start(i):
        nonlocal started
        s = int(rng.integers(2**31 - 1))
        env, o = new_env(s)
        rec = {"seed": s, "episode": started, "sleep_steps": 0, "night_steps": 0,
               "night_sleep_steps": 0, "hits_asleep": 0}
        started += 1
        if i < len(envs):
            envs[i], obs[i], frames[i], stats[i] = env, o, [], rec
            states[i].zero_()
            if gru:
                hiddens[i].zero_()
        else:
            envs.append(env); obs.append(o); frames.append([]); stats.append(rec)
            states.append(th.zeros(hid, device=device))
            if gru:
                hiddens.append(model.initial_hiddens(1, device)[0])

    for i in range(min(n_envs, episodes)):
        start(i)
    active = list(range(len(envs)))
    done_n = 0
    t0 = time.time()
    while active:
        x = th.from_numpy(np.stack([obs[i] for i in active])).to(device).permute(0, 3, 1, 2).float() / 255.0
        st = th.stack([states[i] for i in active])
        mem = {"hiddens": th.stack([hiddens[i] for i in active])} if gru else {}
        with th.no_grad():
            out = model.act(x, states=st, **mem)
        acts = out["actions"].squeeze(-1).cpu().numpy()
        next_active = []
        for j, i in enumerate(active):
            env, rec = envs[i], stats[i]
            filming = video and (bool(keep_if) or rec["episode"] < n_envs * 2)
            if filming:
                frame = env.render((384, 384))
            prev_ach = {k for k, v in env._player.achievements.items() if v > 0}
            o, r, done, info = env.step(int(acts[j]))
            if gru:
                hiddens[i] = out["hiddens"][j]
            obs[i] = o
            p, daylight = env._player, env._world.daylight
            ach = {k for k, v in info["achievements"].items() if v > 0}
            rec["sleep_steps"] += p.sleeping
            rec["night_steps"] += daylight < NIGHT
            rec["night_sleep_steps"] += p.sleeping and daylight < NIGHT
            if filming:
                frames[i].append((frame, info["inventory"], ach, daylight))
            if is_ad and ach - prev_ach:
                with th.no_grad():
                    nxt = model.encode(th.from_numpy(o).to(device).permute(2, 0, 1)[None].float() / 255.0)
                states[i] = F.normalize(nxt[0] - out["latents"][j], dim=-1)
            if not done:
                next_active.append(i)
                continue
            log = p._damage_log
            dmg = {}
            for src, d, _ in log:
                dmg[src] = dmg.get(src, 0) + d
            open_n, zombies = surroundings(env)
            died = info["discount"] == 0
            rec.update(run=run, length=env._step, died=bool(died),
                       cause=(log[-1][0] if log else "unknown") if died else "timeout",
                       asleep_at_death=bool(log[-1][2]) if (died and log) else False,
                       daylight=round(float(daylight), 3), night=bool(daylight < NIGHT),
                       inventory=info["inventory"], open_neighbours=open_n, zombies_near=zombies,
                       damage=dmg, hits_asleep=sum(1 for _, _, a in log if a),
                       achievements=sorted(ach))
            with open(out_path, "a") as f:
                f.write(json.dumps(rec, default=_json) + "\n")
            hit = [a for a in keep_if if a in ach]
            if filming and hit:  # highlight mode: keep every episode that reached a wanted achievement
                kept[hit[-1]] = kept.get(hit[-1], 0) + 1
                name = f"highlight_{hit[-1]}_{kept[hit[-1]]}.mp4"
                write_death_video(os.path.join(vid_dir, name), frames[i], rec, run)
                with open(os.path.join(vid_dir, "highlights.jsonl"), "a") as f:
                    f.write(json.dumps({"file": name, **{k: rec[k] for k in
                            ("cause", "night", "length", "achievements", "seed")}}, default=_json) + "\n")
            elif filming and not keep_if and kept.get(rec["cause"], 0) < videos_per_cause:
                kept[rec["cause"]] = kept.get(rec["cause"], 0) + 1
                slug = rec["cause"].replace(" ", "_").replace("(", "").replace(")", "").replace("+", "-")
                name = f"death_{slug}_{kept[rec['cause']]}.mp4"
                write_death_video(os.path.join(vid_dir, name), frames[i], rec, run)
                with open(os.path.join(vid_dir, "deaths.jsonl"), "a") as f:
                    f.write(json.dumps({"file": name, **{k: rec[k] for k in
                            ("cause", "night", "length", "achievements", "seed")}}, default=_json) + "\n")
            done_n += 1
            if started < episodes:
                start(i)
                next_active.append(i)
        active = next_active
    print(f"{run}: {done_n} episodes in {time.time() - t0:.0f} s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--envs", type=int, default=32)
    ap.add_argument("--video_runs", nargs="*", default=[])
    ap.add_argument("--videos_per_cause", type=int, default=2)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--ckpt", default="agent-e250.pt", help="checkpoint file in runs/<run>/checkpoints/")
    ap.add_argument("--keep_if", nargs="*", default=[],
                    help="highlight mode: film every episode, keep those with any of these achievements")
    ap.add_argument("--out_dir", default=None, help="where the per-episode jsonl goes (default results/deaths)")
    args = ap.parse_args()
    sys.path[:0] = [AD, ROOT, os.path.join(AD, "upstream")]
    import gymnasium
    sys.modules.setdefault("gym", gymnasium)
    import torch as th
    th.set_num_threads(2)
    install_hooks()
    for run in args.runs:
        diagnose(run, args.episodes, args.envs, run in args.video_runs, args.videos_per_cause,
                 th.device(args.device), args.seed, args.ckpt, tuple(args.keep_if),
                 os.path.join(ROOT, args.out_dir) if args.out_dir else None)


if __name__ == "__main__":
    main()
