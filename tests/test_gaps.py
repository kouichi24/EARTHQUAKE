"""No line is invented: broken traces are joined only where the image shows a faint trace."""
import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tests.synth import make_sheet  # noqa: E402
from traceview.config import Params  # noqa: E402
from traceview.engine import Engine  # noqa: E402

PAPER = np.array([205, 195, 170], np.float32)      # RGB of the synthetic paper
INK = np.array([50, 60, 55], np.float32)


def _sheet(tmp_path, name, faint, params=None):
    """Band 0 (baseline y=250) loses its pen line between x=130 and x=190, either completely
    (faint=False) or leaving a faint trace (faint=True)."""
    rgb, _ = make_sheet()
    rgb = rgb.copy()
    x0, x1, by = 130, 190, 250
    rgb[by - 20:by + 21, x0:x1] = PAPER.astype(np.uint8)            # wipe the line
    if faint:
        faint_col = (0.55 * PAPER + 0.45 * INK).astype(np.uint8)
        cv2.line(rgb, (x0, by), (x1, by), tuple(int(c) for c in faint_col), 3)
    p = tmp_path / f"{name}.png"
    cv2.imwrite(str(p), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    e = Engine(params)
    e.open(str(p))
    e.analyze()
    return e


def _in_gap(b):
    inside = (b.path[:, 0] >= 135) & (b.path[:, 0] <= 185)
    return b.path[inside]


def test_no_line_is_drawn_across_a_gap_without_any_trace(tmp_path):
    e = _sheet(tmp_path, "empty", faint=False)
    b = e.bands[0]
    assert len(_in_gap(b)) == 0                      # nothing invented inside the gap
    assert not b.complete and b.end_x < 135          # the trace stops where the line stops
    assert not b.inferred
    assert any(i.kind == "untracked_end" for i in b.issues)
    # the other bands are unaffected
    assert all(bd.complete for bd in e.bands[1:])


def test_faint_trace_is_followed_and_marked_as_estimated(tmp_path):
    e = _sheet(tmp_path, "faint", faint=True)
    b = e.bands[0]
    assert b.complete and b.end_x > 2200
    assert len(_in_gap(b)) > 0
    # the part that crosses the gap is marked as estimated (not as a sure trace)
    marked = any(a <= j <= c for j in np.flatnonzero((b.path[:, 0] >= 140) & (b.path[:, 0] <= 180))
                 for a, c in b.inferred)
    assert marked
    assert any(i.kind == "inferred" and "長さ" in i.msg for i in b.issues)


def test_gap_mode_off_never_joins(tmp_path):
    e = _sheet(tmp_path, "off", faint=True, params=Params(gap_mode="off"))
    b = e.bands[0]
    assert not b.inferred
    assert not b.complete


def test_clean_sheet_has_no_estimated_links(tmp_path):
    rgb, _ = make_sheet()
    p = tmp_path / "clean.png"
    cv2.imwrite(str(p), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    e = Engine()
    e.open(str(p))
    e.analyze()
    assert all(not b.inferred for b in e.bands)
    assert all(b.complete for b in e.bands)
