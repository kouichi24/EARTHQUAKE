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
    arm_length_mm: float = 0.0      # pen arm length (arc correction); 0 = no correction
    arc_sign: int = 1               # +1 / -1: which way the pen tip is displaced along x at large amplitude
    note: str = ""

    def to_dict(self):
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, d):
        return cls(**{k: v for k, v in (d or {}).items() if k in cls.__dataclass_fields__})

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
