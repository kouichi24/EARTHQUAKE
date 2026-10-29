from dataclasses import dataclass, asdict, field


@dataclass
class Params:
    """Tunable analysis parameters (all stored in the project file)."""

    work_max_side: int = 4800          # longest side of the analysis image [px]
    ink_threshold: float = 0.48        # pen core: background-normalised gray below this seeds ink
    ink_weak_threshold: float = 0.64   # fainter ink connected to pen core is kept (hysteresis)
    min_component_area: int = 40       # drop ink specks smaller than this [px]
    ref_line_min_len_frac: float = 0.12  # horizontal straight runs longer than this fraction of width are reference lines
    ref_line_max_thick: int = 9        # ...and thinner than this [px]
    hole_fill_area: int = 60           # holes smaller than this are filled before thinning [px]
    spur_min_len: int = 6              # skeleton spurs shorter than this are ignored [px]
    tip_min_prominence: float = 6.0    # minimum y prominence of a vertex [px]
    origin_strip_frac: float = 0.07    # width of the strip searched for trace origins (fraction of paper width)
    origin_min_gap_frac: float = 0.06  # minimum vertical gap between bands (fraction of paper height)
    snap_radius: int = 14              # radius used to snap an edited vertex to the image line [px]
    ambiguity_margin: float = 0.25     # score gap below which a junction is reported as ambiguous
    undo_limit: int = 50

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        known = {k: v for k, v in (d or {}).items() if k in cls.__dataclass_fields__}
        return cls(**known)
