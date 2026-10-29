"""Ink extraction: paper area, ruler/reference-line/annotation exclusion (IN-03, IN-04)."""
from dataclasses import dataclass

import cv2
import numpy as np
from skimage.measure import label, regionprops


@dataclass
class Prepared:
    paper: tuple            # (x0, y0, x1, y1) in work px
    ink: np.ndarray         # bool mask of candidate waveform ink (ruler/lines/annotations removed)
    ink_raw: np.ndarray     # bool mask before reference-line removal
    ref_lines: np.ndarray   # bool mask of removed reference lines
    ref_zone: np.ndarray    # bool mask within +-3 px of every fitted reference line (also where kept)


def find_paper(rgb):
    """Bounding box of the beige paper; the grey ruler strips (low saturation) fall outside."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    sat = cv2.blur(hsv[..., 1], (15, 15)).astype(np.float32)
    h, w = sat.shape
    core = sat[h // 5: 4 * h // 5, w // 5: 4 * w // 5]
    thr = max(8.0, float(np.median(core)) * 0.7)
    col = sat[h // 5: 4 * h // 5].mean(axis=0)
    row = sat[:, w // 5: 4 * w // 5].mean(axis=1)

    def span(p, n):
        ok = np.where(p > thr)[0]
        if len(ok) == 0:
            return 0, n
        # first/last positions of the long run of "paper" values around the centre
        c = n // 2
        a = c
        while a > 0 and p[a - 1] > thr:
            a -= 1
        b = c
        while b < n - 1 and p[b + 1] > thr:
            b += 1
        return a, b + 1

    x0, x1 = span(col, w)
    y0, y1 = span(row, h)
    m = max(6, int(0.01 * w))  # stay clear of the ruler edge and the paper-edge shadow
    return (x0 + m, y0 + m, x1 - m, y1 - m)


def prepare(rgb, params):
    h, w = rgb.shape[:2]
    paper = find_paper(rgb)
    x0, y0, x1, y1 = paper
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    # Background normalisation: paper stains and shading are divided out.
    k = max(31, (min(h, w) // 40) | 1)
    bg = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    bg = cv2.GaussianBlur(bg, (0, 0), k / 3)
    norm = gray / np.maximum(bg, 1.0)
    from skimage.filters import apply_hysteresis_threshold
    # Hysteresis: faint parts of a thin pen line are kept when they touch dark ink.
    ink = apply_hysteresis_threshold(-norm, -params.ink_weak_threshold, -params.ink_threshold)
    # Coloured annotations (orange / blue / pink): dark ink is nearly neutral.
    r, g, b = [rgb[..., i].astype(np.int16) for i in range(3)]
    colored = (np.abs(r - b) > 38) | (np.abs(g - b) > 38) & (gray < 150)
    ink &= ~colored
    area = np.zeros_like(ink)
    area[y0:y1, x0:x1] = True
    ink &= area
    ink_raw = ink.copy()

    # Reference lines: long, thin, perfectly straight horizontal runs.
    L = max(25, int(0.3 * params.ref_line_min_len_frac * (x1 - x0)))   # short seed runs; lines are completed by line fitting
    ink_u8 = ink.astype(np.uint8)
    horiz = cv2.morphologyEx(ink_u8, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (L, 1)))
    # keep only thin horizontal runs (thick wiggle bursts are not removed)
    thick = cv2.morphologyEx(horiz, cv2.MORPH_OPEN,
                             cv2.getStructuringElement(cv2.MORPH_RECT, (1, params.ref_line_max_thick)))
    ref = (horiz > 0) & ~(thick > 0)
    ref = cv2.dilate(ref.astype(np.uint8), np.ones((3, 1), np.uint8)) > 0
    ref, ref_zone = _extend_ref_lines(ink, ref, params, (x0, x1))
    ink = ink & ~ref

    # Small specks away from everything are dirt/noise.
    ink = _drop_small(ink, params.min_component_area)
    # A spike that crosses a reference line loses its pixels inside the erased band: restore them
    # when ink exists above and below (vertical closing, applied only inside the erased band).
    band = cv2.dilate(ref.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_RECT, (3, 9))) > 0
    closed = cv2.morphologyEx(ink.astype(np.uint8), cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_RECT, (3, 25))) > 0
    ink = ink | (closed & band)
    ink = _fill_small_holes(ink, params.hole_fill_area)
    return Prepared(paper=paper, ink=ink, ink_raw=ink_raw, ref_lines=ref, ref_zone=ref_zone)


def _drop_small(mask, min_area):
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
    return keep[lab]


def _fill_small_holes(mask, max_area):
    inv = (~mask).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
    small = np.zeros(n, bool)
    small[1:] = stats[1:, cv2.CC_STAT_AREA] <= max_area
    out = mask | small[lab]
    return out


def _vertical_run_len(mask):
    """For every ink pixel the length of the vertical run it belongs to."""
    h, w = mask.shape
    down = np.zeros((h, w), np.int32)
    up = np.zeros((h, w), np.int32)
    m = mask.astype(np.int32)
    for y in range(h):
        down[y] = (down[y - 1] + 1) * m[y] if y else m[y]
    for y in range(h - 1, -1, -1):
        up[y] = (up[y + 1] + 1) * m[y] if y < h - 1 else m[y]
    return (down + up - 1) * m


def _extend_ref_lines(ink, ref, params, xs):
    """Fit each reference line from its detected segments and erase it along its whole length.

    Only thin ink crossing the fitted line is erased, so spikes that cross a reference
    line keep their pixels (IN-04: do not lose the real peaks).
    """
    x0, x1 = xs
    seg = cv2.dilate(ref.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_RECT, (301, 5)))
    n, lab = cv2.connectedComponents(seg, connectivity=8)
    zone = np.zeros(ink.shape, bool)
    if n <= 1:
        return ref, zone
    runs = _vertical_run_len(ink)
    thin = ink & (runs <= params.ref_line_max_thick + 3)
    out = ref.copy()
    h, w = ink.shape
    xx = np.arange(x0, x1)
    fits = []
    for i in range(1, n):
        ys, xs_ = np.nonzero((lab == i) & ref)
        if len(xs_) < 200 or xs_.max() - xs_.min() < params.ref_line_min_len_frac * (x1 - x0):
            continue
        vx, vy, cx, cy = cv2.fitLine(np.stack([xs_, ys], 1).astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01).ravel()
        if abs(vx) < 1e-3 or abs(vy / vx) > 0.02:   # reference lines are (almost) horizontal
            continue
        yl = cy + (xx - cx) * vy / vx
        # coverage: a true reference line is inked (or crossed by a spike) along nearly its whole
        # length; a flat stretch of waveform is not.
        cov = np.zeros(len(xx), bool)
        for dy in range(-2, 3):
            yy = np.clip(np.round(yl).astype(int) + dy, 0, h - 1)
            cov |= ink[yy, xx]
        fits.append((float(np.mean(yl)), yl, float(cov.mean())))
    # Reference lines come as a bundle around one almost fully inked straight line. Other
    # straight-looking runs (a flat stretch of the waveform) lie outside that bundle.
    bundle = max(12, int(0.0055 * w))
    anchors = [f[0] for f in fits if f[2] >= 0.9]
    for ym, yl, cv in fits:
        if anchors:
            if min(abs(ym - a) for a in anchors) > bundle:
                continue
        elif cv < 0.6:
            continue
        # Where other ink runs right next to the line (the waveform itself grazing a reference
        # line) the line is kept: erasing there would cut the waveform.
        y0i = np.round(yl).astype(int)
        near = np.zeros(len(xx), bool)
        for dy in range(5, 9):
            near |= ink[np.clip(y0i - dy, 0, h - 1), xx] | ink[np.clip(y0i + dy, 0, h - 1), xx]
        for dy in range(-3, 4):
            yy = np.clip(y0i + dy, 0, h - 1)
            zone[yy, xx] = True
            sel = thin[yy, xx] & ~near
            out[yy[sel], xx[sel]] = True
    return out, zone
