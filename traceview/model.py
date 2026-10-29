"""Data model: bands, issues, undo snapshots."""
import copy
from dataclasses import dataclass, field

import numpy as np

from .vertices import Vertex

BAND_NAMES = ["上段", "中段", "下段"]
BAND_COLORS = ["#e6194b", "#1b9e3b", "#2c6fd6", "#c77700", "#8e44ad"]


@dataclass
class Issue:
    kind: str       # ambiguous | untracked_end | no_origin | inferred | pending | retrace_failed
    x: float
    y: float
    msg: str

    def to_dict(self):
        return dict(kind=self.kind, x=self.x, y=self.y, msg=self.msg)


@dataclass
class Band:
    index: int
    name: str
    color: str
    origin: tuple = (0, 0)           # (x, y) in work px
    y0: float = 0.0
    path: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), np.int32))
    inferred: list = field(default_factory=list)       # (start, end) path index spans
    vertices: list = field(default_factory=list)
    manual_points: list = field(default_factory=list)  # dicts {x, y, kind: moved|added}
    pending: list = field(default_factory=list)        # dicts {type, ox, oy, x, y}
    line_zero: float = 0.0
    line_top: float = 0.0
    line_bottom: float = 0.0
    line_top_manual: bool = False
    line_bottom_manual: bool = False
    end_x: float = 0.0
    complete: bool = False           # trace reached the end of the paper
    confirmed: bool = False          # user confirmed the reading (MS-04)
    issues: list = field(default_factory=list)
    undo: list = field(default_factory=list)
    redo: list = field(default_factory=list)
    initial: dict = None

    # -- snapshots -------------------------------------------------------------
    def snapshot(self):
        return dict(
            path=self.path.copy(), inferred=list(self.inferred),
            vertices=[Vertex.from_dict(v.to_dict()) for v in self.vertices],
            manual_points=copy.deepcopy(self.manual_points), pending=copy.deepcopy(self.pending),
            line_zero=self.line_zero, line_top=self.line_top, line_bottom=self.line_bottom,
            line_top_manual=self.line_top_manual, line_bottom_manual=self.line_bottom_manual,
            end_x=self.end_x, complete=self.complete, confirmed=self.confirmed,
            issues=copy.deepcopy(self.issues), y0=self.y0,
        )

    def restore(self, snap):
        self.path = snap["path"].copy()
        self.inferred = list(snap["inferred"])
        self.vertices = [Vertex.from_dict(v.to_dict()) for v in snap["vertices"]]
        self.manual_points = copy.deepcopy(snap["manual_points"])
        self.pending = copy.deepcopy(snap["pending"])
        for k in ("line_zero", "line_top", "line_bottom", "line_top_manual", "line_bottom_manual",
                  "end_x", "complete", "confirmed", "y0"):
            setattr(self, k, snap[k])
        self.issues = copy.deepcopy(snap["issues"])

    def push_undo(self, limit):
        self.undo.append(self.snapshot())
        if len(self.undo) > limit:
            self.undo.pop(0)
        self.redo.clear()

    def do_undo(self):
        if not self.undo:
            return False
        self.redo.append(self.snapshot())
        self.restore(self.undo.pop())
        return True

    def do_redo(self):
        if not self.redo:
            return False
        self.undo.append(self.snapshot())
        self.restore(self.redo.pop())
        return True

    def reset_to_initial(self):
        if self.initial is None:
            return False
        self.push_undo(10**6)
        self.restore(self.initial)
        return True

    def is_inferred_index(self, i):
        return any(a <= i <= b for a, b in self.inferred)
