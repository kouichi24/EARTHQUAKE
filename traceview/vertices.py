"""Peak / valley extraction along the pen path (not along x: TR-05)."""
from dataclasses import dataclass, asdict

import numpy as np


@dataclass
class Vertex:
    idx: int            # index into the band path
    x: float
    y: float
    kind: str           # "peak" (smaller y = higher) | "valley"
    source: str = "auto"       # auto | manual | retrace
    turn: str = "bend"         # end (dead-end reversal) | bend (smooth turn)
    revisit: int = 0           # how many extra times the pen path passes this tip
    note: str = ""

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def zigzag(y, prominence):
    """Indices of alternating local extrema of y whose swing exceeds `prominence`."""
    n = len(y)
    if n < 3:
        return []
    ext = []                     # (index, kind) kind +1 = local max of y, -1 = local min
    hi_i = lo_i = 0
    hi, lo = y[0], y[0]
    direction = 0
    for i in range(1, n):
        v = y[i]
        if direction >= 0 and v > hi:
            hi, hi_i = v, i
        if direction <= 0 and v < lo:
            lo, lo_i = v, i
        if direction == 0:
            if hi - lo >= prominence:
                direction = 1 if hi_i > lo_i else -1
                if direction == 1:
                    ext.append((lo_i, -1))
                    lo, lo_i = v, i
                else:
                    ext.append((hi_i, 1))
                    hi, hi_i = v, i
                hi, lo = v, v
                hi_i = lo_i = i
        elif direction == 1:        # currently rising: waiting for a drop from hi
            if hi - v >= prominence:
                ext.append((hi_i, 1))
                direction = -1
                lo, lo_i = v, i
        else:                       # falling: waiting for a rise from lo
            if v - lo >= prominence:
                ext.append((lo_i, -1))
                direction = 1
                hi, hi_i = v, i
    if direction == 1:
        ext.append((hi_i, 1))
    elif direction == -1:
        ext.append((lo_i, -1))
    return ext


def extract_vertices(path, reversals, prominence, ink=None, merge_px=4, window=6):
    """Vertices of a pen path. Tips are refined against the ink mask so the skeleton's
    shortening of thick line ends does not lose height."""
    if len(path) < 3:
        return []
    y = path[:, 1].astype(np.float32)
    ext = zigzag(y, prominence)
    rev = np.array(sorted(reversals), dtype=np.int64) if len(reversals) else np.zeros(0, np.int64)
    out = []
    for i, k in ext:
        x, yy = float(path[i, 0]), float(path[i, 1])
        kind = "valley" if k == 1 else "peak"        # larger y = lower on the sheet = valley
        if ink is not None:
            x, yy = _refine_tip(ink, x, yy, kind, window)
        turn = "bend"
        if len(rev) and np.min(np.abs(rev - i)) <= 3:
            turn = "end"
        out.append(Vertex(int(i), x, yy, kind, "auto", turn))
    # The pen path may pass the same tip again while backing out of a branch.
    kept = []
    for v in out:
        dup = None
        for u in kept[-8:]:
            if u.kind == v.kind and abs(u.x - v.x) <= merge_px and abs(u.y - v.y) <= merge_px:
                dup = u
                break
        if dup is not None:
            dup.revisit += 1
        else:
            kept.append(v)
    return kept


def _refine_tip(ink, x, y, kind, w):
    h, wd = ink.shape
    xa, xb = int(max(0, x - w)), int(min(wd, x + w + 1))
    if kind == "peak":
        ya, yb = int(max(0, y - 2 * w)), int(min(h, y + 1))
    else:
        ya, yb = int(max(0, y)), int(min(h, y + 2 * w + 1))
    sub = ink[ya:yb, xa:xb]
    ys, xs = np.nonzero(sub)
    if len(ys) == 0:
        return x, y
    j = np.argmin(ys) if kind == "peak" else np.argmax(ys)
    return float(xs[j] + xa), float(ys[j] + ya)
