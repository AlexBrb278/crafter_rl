"""Multiprocess vector env for the original (numpy) Crafter.

Replaces stable-baselines3's SubprocVecEnv + VecMonitor without depending on gym/SB3,
and works with Windows' `spawn` start method. Workers import only numpy + crafter.

Semantics match SubprocVecEnv + VecMonitor:
  * auto-reset: on `done` the returned obs is the first obs of the next episode,
    while `info` still describes the terminal step (achievements, episode stats);
  * `info["episode"] = {"r": return, "l": length}` on the terminal step.

Extra: `step(actions, render_idx=i, render_size=S)` also returns a high-resolution
frame of env i (used by `common.video.ClipRecorder` to make clips).
"""
import multiprocessing as mp

import numpy as np


def _worker(remote, parent_remote, seeds, env_kwargs):
    parent_remote.close()
    import crafter  # imported here so the parent process never pays for it

    envs = [crafter.Env(seed=int(s), **env_kwargs) for s in seeds]
    ep_ret = np.zeros(len(envs), np.float64)
    ep_len = np.zeros(len(envs), np.int64)
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == "reset":
                ep_ret[:] = 0
                ep_len[:] = 0
                remote.send(np.stack([e.reset() for e in envs]))
            elif cmd == "step":
                actions, render_local, render_size = data
                obs, rews, dones, infos, frame = [], [], [], [], None
                for i, (env, a) in enumerate(zip(envs, actions)):
                    if render_local == i:
                        # Frame *before* the step, so the clip shows what the agent saw.
                        frame = env.render((render_size, render_size))
                    o, r, d, info = env.step(int(a))
                    ep_ret[i] += r
                    ep_len[i] += 1
                    small = {
                        "achievements": info["achievements"],
                        "inventory": info["inventory"],
                        "discount": info["discount"],
                        "t": int(ep_len[i]),        # step within the episode
                        "ret": float(ep_ret[i]),    # return so far
                    }
                    if d:
                        small["episode"] = {"r": float(ep_ret[i]), "l": int(ep_len[i])}
                        ep_ret[i] = 0
                        ep_len[i] = 0
                        o = env.reset()
                    obs.append(o)
                    rews.append(r)
                    dones.append(d)
                    infos.append(small)
                remote.send((np.stack(obs), np.array(rews, np.float32),
                             np.array(dones, bool), infos, frame))
            elif cmd == "close":
                break
            else:
                raise ValueError(cmd)
    except (KeyboardInterrupt, EOFError, BrokenPipeError):
        pass
    finally:
        remote.close()


class CrafterVecEnv:
    num_actions = 17
    obs_shape = (64, 64, 3)

    def __init__(self, seeds, envs_per_worker=1, env_kwargs=None):
        self.num_envs = len(seeds)
        assert self.num_envs % envs_per_worker == 0
        self.envs_per_worker = envs_per_worker
        ctx = mp.get_context("spawn")
        n_workers = self.num_envs // envs_per_worker
        self.remotes, self.procs = [], []
        for w in range(n_workers):
            parent, child = ctx.Pipe()
            chunk = seeds[w * envs_per_worker:(w + 1) * envs_per_worker]
            p = ctx.Process(target=_worker, args=(child, parent, list(chunk), env_kwargs or {}),
                            daemon=True)
            p.start()
            child.close()
            self.remotes.append(parent)
            self.procs.append(p)
        self.closed = False

    def reset(self):
        for r in self.remotes:
            r.send(("reset", None))
        return np.concatenate([r.recv() for r in self.remotes])

    def step(self, actions, render_idx=None, render_size=384):
        actions = np.asarray(actions).reshape(self.num_envs)
        k = self.envs_per_worker
        for w, r in enumerate(self.remotes):
            local = None
            if render_idx is not None and render_idx // k == w:
                local = render_idx % k
            r.send(("step", (actions[w * k:(w + 1) * k], local, render_size)))
        results = [r.recv() for r in self.remotes]
        obs = np.concatenate([res[0] for res in results])
        rews = np.concatenate([res[1] for res in results])
        dones = np.concatenate([res[2] for res in results])
        infos = [info for res in results for info in res[3]]
        frames = [res[4] for res in results if res[4] is not None]
        return obs, rews, dones, infos, (frames[0] if frames else None)

    def close(self):
        if self.closed:
            return
        for r in self.remotes:
            try:
                r.send(("close", None))
            except (BrokenPipeError, OSError):
                pass
        for p in self.procs:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
        self.closed = True
