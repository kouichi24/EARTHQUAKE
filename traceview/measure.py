"""Measurements (MS-01..04). Image coordinates: origin top-left, y downwards.
Signed amplitude a = y0 - y (above the zero line is positive)."""
from dataclasses import dataclass, field


@dataclass
class Calibration:
    mm_per_px_x: float = 0.0        # 0 = not calibrated (values stay in px). Units: per ORIGINAL image px
    mm_per_px_y: float = 0.0
    seconds_per_mm: float = 0.0     # paper speed (s/mm); 0 = unset
    amplitude_factor: float = 0.0   # physical units per mm of amplitude; 0 = unset
    amplitude_unit: str = ""
    seconds_per_px_x: float = 0.0   # time per ORIGINAL px along x (set from two time ticks); 0 = unset
    arm_length_mm: float = 0.0      # pen arm length (arc correction); 0 = no correction
    arc_sign: int = 1               # +1 / -1: which way the pen tip is displaced along x at large amplitude
    note: str = ""

    def to_dict(self):
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, d):
        return cls(**{k: v for k, v in (d or {}).items() if k in cls.__dataclass_fields__})

    def seconds_per_orig_px(self):
        """Time per original px along x: from two time ticks, else from paper speed x mm-per-px; 0 if unknown."""
        if self.seconds_per_px_x > 0:
            return self.seconds_per_px_x
        if self.seconds_per_mm > 0 and self.mm_per_px_x > 0:
            return self.seconds_per_mm * self.mm_per_px_x
        return 0.0

    @property
    def calibrated_y(self):
        return self.mm_per_px_y > 0

    def unit_y(self):
        if self.amplitude_factor > 0 and self.mm_per_px_y > 0:
            return self.amplitude_unit or "unit"
        return "mm" if self.mm_per_px_y > 0 else "px"

    def to_y_units(self, px_orig):
        """Convert a length in original px along y to the reported unit."""
        if self.mm_per_px_y > 0:
            mm = px_orig * self.mm_per_px_y
            return mm * self.amplitude_factor if self.amplitude_factor > 0 else mm
        return px_orig


def compute_measure(vertices, y0, y_top_line, y_bottom_line, scale, cal):
    """Values are reported in original px (scale = work/orig) unless calibrated.

    Auto values come from vertices; line values from the (possibly hand-adjusted) lines.
    """
    inv = 1.0 / scale if scale else 1.0
    res = {}
    sure = [v for v in vertices if not getattr(v, "suspect", False)]
    res["n_suspect"] = len(vertices) - len(sure)
    vertices = sure
    if vertices:
        top = min(vertices, key=lambda v: v.y)
        bot = max(vertices, key=lambda v: v.y)
        a_max = (y0 - top.y) * inv
        a_min = (y0 - bot.y) * inv
        res.update(
            max_value=cal.to_y_units(a_max), max_x=top.x * inv, max_y=top.y * inv,
            min_value=cal.to_y_units(a_min), min_x=bot.x * inv, min_y=bot.y * inv,
            up_amp=cal.to_y_units(max(0.0, a_max)), down_amp=cal.to_y_units(max(0.0, -a_min)),
            span=cal.to_y_units(a_max - a_min),
        )
        ys = [v.y for v in vertices]
        diffs = [abs(ys[i + 1] - ys[i]) for i in range(len(ys) - 1)]
        if diffs:
            k = max(range(len(diffs)), key=lambda i: diffs[i])
            res.update(max_step=cal.to_y_units(diffs[k] * inv), max_step_from=k, max_step_to=k + 1)
    if y_top_line is not None and y_bottom_line is not None:
        up = (y0 - y_top_line) * inv
        dn = (y0 - y_bottom_line) * inv
        res.update(line_up=cal.to_y_units(max(0.0, up)), line_down=cal.to_y_units(max(0.0, -dn)),
                   line_span=cal.to_y_units(up - dn))
    return res


def arc_corrected_x_mm(x_mm, amp_mm, arm_length_mm, sign=1):
    """Time-direction position after removing the arc of a lever pen.

    A pen on an arm of length R that swings by amplitude a (mm, along the paper's y) is displaced
    along x by R - sqrt(R^2 - a^2). The sign depends on which side the pivot is on.
    Returns x_mm unchanged when R <= 0 or |a| >= R (outside the model).
    """
    import math
    if arm_length_mm <= 0 or abs(amp_mm) >= arm_length_mm:
        return x_mm
    return x_mm - sign * (arm_length_mm - math.sqrt(arm_length_mm ** 2 - amp_mm ** 2))


@dataclass
class SectionSettings:
    """How waves are counted inside a section (user-adjustable)."""
    min_swing_px: float = 20.0      # ignore wiggles smaller than this (original px, peak to neighbouring valley)
    count_mode: str = "peak"        # "peak" | "valley" | "both" (average of peaks and valleys)

    def to_dict(self):
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, d):
        return cls(**{k: v for k, v in (d or {}).items() if k in cls.__dataclass_fields__})


COUNT_MODES = {"peak": "山の数", "valley": "谷の数", "both": "山と谷の平均"}


def section_stats(band, x0, x1, scale, cal, settings):
    """Statistics of one section [x0, x1] (work px) of a band.

    Counts and periods use only vertices that are not doubtful (not on a stretch that leaves the
    band's territory, not on an estimated link). Positions are returned in original px.
    """
    inv = 1.0 / scale if scale else 1.0
    if x1 < x0:
        x0, x1 = x1, x0
    vs_all = [v for v in band.vertices if x0 <= v.x <= x1]
    vs = [v for v in vs_all if not getattr(v, "suspect", False) and not band.is_inferred_index(v.idx)]
    y0 = band.line_zero
    res = dict(x0=x0 * inv, x1=x1 * inv, length_px=(x1 - x0) * inv, n_vertices=len(vs),
               n_excluded=len(vs_all) - len(vs))
    res["length_mm"] = res["length_px"] * cal.mm_per_px_x if cal.mm_per_px_x > 0 else None
    sp = cal.seconds_per_orig_px()
    res["length_s"] = res["length_px"] * sp if sp > 0 else None
    if vs:
        top = min(vs, key=lambda v: v.y)
        bot = max(vs, key=lambda v: v.y)
        res.update(max_value=cal.to_y_units((y0 - top.y) * inv), max_x=top.x * inv, max_y=top.y * inv,
                   min_value=cal.to_y_units((y0 - bot.y) * inv), min_x=bot.x * inv, min_y=bot.y * inv,
                   span=cal.to_y_units((bot.y - top.y) * inv))
    # swing of each vertex against its neighbours along the pen path (sorted by path order)
    ordered = sorted(vs_all, key=lambda v: v.idx)
    keep = []
    thr = settings.min_swing_px * scale          # work px
    for i, v in enumerate(ordered):
        if getattr(v, "suspect", False) or band.is_inferred_index(v.idx):
            continue
        nb = []
        if i > 0:
            nb.append(abs(v.y - ordered[i - 1].y))
        if i + 1 < len(ordered):
            nb.append(abs(v.y - ordered[i + 1].y))
        swing = max(nb) if nb else 0.0
        if swing >= thr:
            keep.append(v)
    peaks = sorted((v for v in keep if v.kind == "peak"), key=lambda v: v.x)
    valleys = sorted((v for v in keep if v.kind == "valley"), key=lambda v: v.x)
    res["peaks"], res["valleys"] = len(peaks), len(valleys)
    mode = settings.count_mode
    res["waves"] = (len(peaks) if mode == "peak" else len(valleys) if mode == "valley"
                    else (len(peaks) + len(valleys)) / 2.0)
    ref = peaks if mode in ("peak", "both") else valleys
    if mode == "both" and len(peaks) < 2 <= len(valleys):
        ref = valleys
    xs = [v.x for v in ref]
    if len(xs) >= 2:
        d = [(b - a) * inv for a, b in zip(xs, xs[1:])]
        mean = (xs[-1] - xs[0]) * inv / (len(xs) - 1)
        res.update(period_px=mean, period_min_px=min(d), period_max_px=max(d))
        res["period_mm"] = mean * cal.mm_per_px_x if cal.mm_per_px_x > 0 else None
        res["period_s"] = mean * sp if sp > 0 else None
        res["freq_hz"] = 1.0 / res["period_s"] if res["period_s"] else None
    else:
        res.update(period_px=None, period_min_px=None, period_max_px=None, period_mm=None, period_s=None,
                   freq_hz=None)
    # the same period measured on the arc-corrected positions, reported next to the raw one
    res.update(period_arc_px=None, period_arc_mm=None, period_arc_s=None, freq_arc_hz=None)
    if arc_active(cal) and len(ref) >= 2:
        cx = sorted(corrected_x_work(v.x, v.y, y0, scale, cal) for v in ref)
        m = (cx[-1] - cx[0]) * inv / (len(cx) - 1)
        res["period_arc_px"] = m
        res["period_arc_mm"] = m * cal.mm_per_px_x
        res["period_arc_s"] = m * sp if sp > 0 else None
        res["freq_arc_hz"] = 1.0 / res["period_arc_s"] if res["period_arc_s"] else None
    # doubts that fall inside the section (MS-04)
    inside = [i for i in band.issues if x0 <= i.x <= x1]
    res["issues"] = {k: sum(1 for i in inside if i.kind == k) for k in ("ambiguous", "excursion", "untracked_end")}
    res["beyond_trace"] = (not band.complete) and x1 > band.end_x
    return res


def arc_active(cal):
    """Arc correction needs the arm length and the X and Y calibration."""
    return cal.arm_length_mm > 0 and cal.mm_per_px_x > 0 and cal.mm_per_px_y > 0


def corrected_x_work(x, y, y0, scale, cal):
    """Arc-corrected x of a point (all in work px). Unchanged when the correction is not active."""
    if not arc_active(cal):
        return x
    inv = 1.0 / scale
    amp_mm = (y0 - y) * inv * cal.mm_per_px_y
    x_mm = arc_corrected_x_mm(x * inv * cal.mm_per_px_x, amp_mm, cal.arm_length_mm, cal.arc_sign)
    return x_mm / cal.mm_per_px_x * scale


def corrected_path(path, y0, scale, cal):
    """Whole pen path with the arc correction applied (float array, work px)."""
    import numpy as np
    out = path.astype(np.float64).copy()
    if not arc_active(cal) or len(out) == 0:
        return out
    inv = 1.0 / scale
    amp_mm = (y0 - out[:, 1]) * inv * cal.mm_per_px_y
    r = cal.arm_length_mm
    ok = np.abs(amp_mm) < r
    shift = np.zeros(len(out))
    shift[ok] = r - np.sqrt(r * r - amp_mm[ok] ** 2)
    out[:, 0] = out[:, 0] - cal.arc_sign * shift / cal.mm_per_px_x * scale
    return out
