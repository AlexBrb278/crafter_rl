"""A top-down map of what the agent has seen, built from its own 64x64 frames and actions only.

Odometry (no game internals): a move action (1-4) either moves the player one tile or is blocked. If it
moved, the world part of frame t+1 is frame t shifted by one 7x7-px tile. Both hypotheses are scored by a
trimmed mean of per-tile pixel error. On uniform ground the scores tie; then a walkability table decides,
learned online from the clear moves this map has seen (key: the colour of the tile stepped onto). A
sleeping player (grey view) never moves, except on the step it wakes up, which is compared in luminance.
Checked offline against the true position in methods/08_world_model/pixel_odometry.py.

Memory: every world tile in view is written at its world position as a 2x2-px thumbnail (7x7 area means),
except while asleep. `render()` returns an egocentric 32x32-tile window (64x64x3 uint8, the agent at the
centre, never-seen tiles black): half of Crafter's 64x64 world around the agent.
"""
import numpy as np

MOVES = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}
TIE = 0.5  # score gap (mean abs pixel error, 0-255 scale) below which the pictures cannot tell
R = 72     # map half-size in tiles; positions are relative to the episode start
WIN = 32   # rendered window in tiles
PX = 2     # rendered pixels per tile
_EDGES = (0, 4, 7)  # 7 px -> 2 thumbnail px


def tiles(frame):  # (64, 64, C) -> (9 x, 7 y, 49 C) world tiles
    C = frame.shape[-1]
    w = frame[:49, :63].astype(np.float32).reshape(7, 7, 9, 7, C).transpose(2, 0, 1, 3, 4)
    return w.reshape(9, 7, -1)


def asleep(frame):
    """A sleeping player sees the world in grey (engine.py _sleep: colour removed, then a dark blue tint),
    and the game ignores its actions until energy is full."""
    w = frame[:49, :63].astype(np.float32)
    return np.abs(w[..., 0] - w[..., 1]).mean() < 1.0


def luminance(frame):  # (64, 64, 1); a sleeping view is grey L tinted to 0.5 L (+ blue), so red * 2 undoes it
    f = frame.astype(np.float32)
    if asleep(frame):
        return 2 * f[..., :1]
    return (f[..., 0] * 0.299 + f[..., 1] * 0.587 + f[..., 2] * 0.114)[..., None]


def shift_err(a, b, d):
    dx, dy = d  # if the player moved by d, b[i, j] == a[i + dx, j + dy] on the overlap
    A = a[max(0, dx):9 + min(0, dx), max(0, dy):7 + min(0, dy)]
    B = b[max(0, -dx):9 + min(0, -dx), max(0, -dy):7 + min(0, -dy)]
    e = np.sort(np.abs(A - B).mean(-1).ravel())
    return e[:int(len(e) * 0.85)].mean()  # drop the 15% worst tiles: creatures, the player sprite


def step_scores(prev, cur, a):
    """(kind, score if moved, score if stayed); kind None (not a move, or asleep throughout), "awake", or
    "wake" (grey frame, then colour: the player wakes up and its action runs; compared in luminance)."""
    if a not in MOVES:
        return None, 0.0, 0.0
    s0, s1 = asleep(prev), asleep(cur)
    if s0 and s1:
        return None, 0.0, 0.0
    if s0:
        A, B = tiles(luminance(prev)), tiles(luminance(cur))
        return "wake", shift_err(A, B, MOVES[a]), shift_err(A, B, (0, 0))
    A, B = tiles(prev), tiles(cur)
    return "awake", shift_err(A, B, MOVES[a]), shift_err(A, B, (0, 0))


def thumbs(frame):  # (64, 64, 3) -> (7 y, 9 x, 2, 2, 3) uint8 thumbnails of the world tiles
    w = frame[:49, :63].astype(np.float32).reshape(7, 7, 9, 7, 3).transpose(0, 2, 1, 3, 4)  # y, x, py, px, c
    out = np.empty((7, 9, 2, 2, 3), np.float32)
    for i in range(2):
        for j in range(2):
            out[:, :, i, j] = w[:, :, _EDGES[i]:_EDGES[i + 1], _EDGES[j]:_EDGES[j + 1]].mean((2, 3))
    return out.round().astype(np.uint8)


class PixelMap:
    def __init__(self):
        self.walk = {}  # tile colour key -> [moves that succeeded, clear moves]; kept across episodes
        self.reset_episode()

    def reset_episode(self):
        self.img = np.zeros((2 * R + 1, 2 * R + 1, 2, 2, 3), np.uint8)  # [y, x]
        self.pos = np.zeros(2, np.int64)  # (x, y)
        self.prev = None

    @staticmethod
    def _key(frame, a):
        dx, dy = MOVES[a]
        x, y = 4 + dx, 3 + dy
        c = frame[y * 7:(y + 1) * 7, x * 7:(x + 1) * 7].reshape(-1, 3).mean(0)
        return tuple((c // 16).astype(int))

    def _write(self, frame):
        if asleep(frame):
            return
        x0, y0 = self.pos[0] - 4 + R, self.pos[1] - 3 + R
        if 0 <= x0 and x0 + 9 <= 2 * R + 1 and 0 <= y0 and y0 + 7 <= 2 * R + 1:
            self.img[y0:y0 + 7, x0:x0 + 9] = thumbs(frame)

    def observe(self, frame, action=None):
        """Feed the new frame (and the action that led to it; None at the start of an episode).
        Returns the estimated move (0 = stayed)."""
        moved = 0
        if self.prev is not None and action is not None:
            a = int(action)
            kind, e_m, e_s = step_scores(self.prev, frame, a)
            if kind is not None:
                key = self._key(self.prev, a) if kind == "awake" else None
                if abs(e_m - e_s) >= TIE:
                    moved = a if e_m < e_s else 0
                    if key is not None:
                        s = self.walk.setdefault(key, [0, 0])
                        s[0] += moved != 0
                        s[1] += 1
                else:
                    s = self.walk.get(key) if key is not None else None
                    moved = a if s is None or (s[0] + 1) / (s[1] + 2) >= 0.5 else 0
            if moved:
                self.pos += MOVES[moved]
                self.pos.clip(-R + 16, R - 16, out=self.pos)  # stays inside the array (never reached in practice)
        self.prev = frame
        self._write(frame)
        return moved

    def render(self):
        """Egocentric WIN x WIN tile window around the agent -> (64, 64, 3) uint8; agent tile at (16, 16)."""
        x0, y0 = self.pos[0] - WIN // 2 + R, self.pos[1] - WIN // 2 + R
        win = self.img[y0:y0 + WIN, x0:x0 + WIN]  # (WIN y, WIN x, 2, 2, 3)
        return win.transpose(0, 2, 1, 3, 4).reshape(WIN * PX, WIN * PX, 3)


def video_panel(map_img, k):
    """For videos only (not the agent's input): the map upscaled k times, the agent's tile outlined."""
    big = np.kron(map_img, np.ones((k, k, 1), np.uint8))
    c, w = WIN // 2 * PX * k, PX * k
    big[c - 1:c + w + 1, [c - 1, c + w]] = 255
    big[[c - 1, c + w], c - 1:c + w + 1] = 255
    return big
