"""Short MP4 clips of the agent while it trains.

`ClipRecorder` starts a clip at random times (gaps uniform in 0.5-1.5 x `mean_gap` env
steps), follows one randomly chosen env for `clip_len` steps, overlays a small HUD
(training step, episode step, return, unlocked achievements) and writes
`<out_dir>/clip_XXX_stepYYYYYYY.mp4` plus a line in `<out_dir>/index.jsonl`.
"""
import json
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HUD_H = 48


def _font(size):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


class HUD:
    def __init__(self, width):
        self.width = width
        self.font = _font(14)
        self.small = _font(12)
        self.flash = []  # [(text, frames_left)]

    def draw(self, frame, line1, line2, new_achievements=()):
        for a in new_achievements:
            self.flash.append(("+ " + a.replace("_", " "), 45))
        img = Image.new("RGB", (self.width, frame.shape[0] + HUD_H), (20, 20, 24))
        img.paste(Image.fromarray(frame), (0, HUD_H))
        d = ImageDraw.Draw(img)
        d.text((6, 4), line1, fill=(235, 235, 235), font=self.font)
        d.text((6, 25), line2, fill=(170, 200, 255), font=self.small)
        y = HUD_H + 6
        for text, _ in self.flash[-4:]:
            w = d.textlength(text, font=self.font)
            d.rectangle((self.width - w - 14, y - 2, self.width - 4, y + 18), fill=(0, 0, 0))
            d.text((self.width - w - 9, y), text, fill=(255, 220, 90), font=self.font)
            y += 22
        self.flash = [(t, n - 1) for t, n in self.flash if n > 1]
        return np.asarray(img)


def write_mp4(path, frames, fps):
    import imageio.v2 as imageio
    with imageio.get_writer(path, fps=fps, codec="libx264", quality=7,
                            macro_block_size=16, ffmpeg_log_level="error") as w:
        for f in frames:
            w.append_data(f)


class ClipRecorder:
    def __init__(self, out_dir, num_envs, clip_len=400, mean_gap=40_000, first_at=2_000,
                 fps=15, size=384, seed=0, label=""):
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.num_envs = num_envs
        self.clip_len = clip_len
        self.mean_gap = mean_gap
        self.fps = fps
        self.size = size
        self.label = label
        self.rng = np.random.default_rng(seed + 12345)
        self.next_start = first_at
        self.active_env = None
        self.frames = []
        self.n_clips = 0
        self._meta = None
        self._hud = None
        self._prev_ach = None

    def maybe_start(self, global_step):
        if self.active_env is None and global_step >= self.next_start:
            self.active_env = int(self.rng.integers(self.num_envs))
            self.frames = []
            self._hud = HUD(self.size)
            self._prev_ach = None
            self._meta = {"start_step": int(global_step), "env": self.active_env,
                          "unlocked": [], "episodes_ended": 0, "reward": 0.0}

    def add(self, frame, reward, done, info, global_step):
        """frame: the (size, size, 3) render of the active env *before* this step."""
        if self.active_env is None or frame is None:
            return
        ach = {k for k, v in info["achievements"].items() if v > 0}
        new = sorted(ach - self._prev_ach) if self._prev_ach is not None else []
        self._prev_ach = set() if done else ach
        self._meta["unlocked"] += new
        self._meta["reward"] += float(reward)
        inv = info["inventory"]
        line1 = (f"{self.label}  train step {global_step:,}   env #{self.active_env}   "
                 f"ep t={info['t']}   return {info['ret']:.1f}")
        line2 = (f"health {inv['health']}  food {inv['food']}  drink {inv['drink']}  "
                 f"energy {inv['energy']}   achievements this episode: {len(ach)}/22")
        self.frames.append(self._hud.draw(frame, line1, line2, new))
        if done:
            self._meta["episodes_ended"] += 1
            self._hud.flash.append(("EPISODE OVER" if info["discount"] == 1 else "DIED", 30))
        if len(self.frames) >= self.clip_len:
            self._save(global_step)

    def _save(self, global_step):
        name = f"clip_{self.n_clips:03d}_step{self._meta['start_step']:07d}.mp4"
        write_mp4(os.path.join(self.out_dir, name), self.frames, self.fps)
        self._meta.update(file=name, end_step=int(global_step), frames=len(self.frames))
        with open(os.path.join(self.out_dir, "index.jsonl"), "a") as f:
            f.write(json.dumps(self._meta) + "\n")
        self.n_clips += 1
        self.active_env = None
        self.frames = []
        # Uniform in [0.5, 1.5] x mean: random timing, but no multi-hour droughts
        # (an exponential gap once skipped 190k steps).
        self.next_start = global_step + int(self.rng.uniform(0.5, 1.5) * self.mean_gap)
