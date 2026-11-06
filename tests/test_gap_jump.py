"""A point can be taken across a gap to the line that lies on the extension of the broken line."""
import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tests.synth import make_sheet  # noqa: E402
from traceview.engine import Engine, EngineError  # noqa: E402

PAPER = np.array([205, 195, 170], np.uint8)


def _gap_engine(tmp_path, width=60, x0=130):
    """Band 0 (baseline y=250) has its pen line wiped out between x0 and x0+width: the trace stops at the gap."""
    rgb, _ = make_sheet()
    rgb = rgb.copy()
    rgb[230:271, x0:x0 + width] = PAPER
    p = tmp_path / "gap.png"
    cv2.imwrite(str(p), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    e = Engine()
    e.open(str(p))
    e.analyze()
    return e


def test_trace_stops_at_the_gap(tmp_path):
    e = _gap_engine(tmp_path)
    b = e.bands[0]
    assert not b.complete and b.end_x < 140


def test_dragging_the_end_across_the_gap_lands_on_the_line_beyond(tmp_path):
    e = _gap_engine(tmp_path)
    b = e.bands[0]
    end = b.path[-1]
    lc = e.line_constraint(float(end[0]), float(end[1]))
    assert lc is not None and lc.ok
    near = lc.snap(float(end[0]) - 10, float(end[1]))               # mouse still on the connected line
    assert lc.last_bridge is None and near[0] <= end[0] + 2
    far = lc.snap(215.0, 250.0)                                       # mouse over the line beyond the gap
    assert lc.last_bridge is not None
    assert far[0] >= 190 and abs(far[1] - 250) < 12                   # lands on the far piece, on the extension
    assert e.graph.skel[int(round(far[1])), int(round(far[0]))]       # and it is a pixel of the image's line
    (ex, ey), (px, py) = lc.last_bridge
    assert abs(ex - end[0]) < 6 and px > ex                           # the bridge starts at the end of the line


def test_no_jump_when_nothing_lies_on_the_extension(tmp_path):
    e = _gap_engine(tmp_path)
    b = e.bands[0]
    end = b.path[-1]
    lc = e.line_constraint(float(end[0]), float(end[1]), jump=20)      # a reach too short to get over the gap
    lc.snap(215.0, 250.0)
    assert lc.last_bridge is None


def test_extend_continues_the_trace_beyond_the_gap_and_marks_the_bridge(tmp_path):
    e = _gap_engine(tmp_path)
    b = e.bands[0]
    n = len(b.path)
    e.set_pending_extend(0, 205.0, 250.0)
    pv = e.preview_pending(0, "tail")
    assert len(pv["new"]) > n and b.pending                            # a preview, nothing applied yet
    ok, msg = e.apply_pending(0)
    assert ok and "延ば" in msg
    assert b.complete and b.end_x > 2200                               # the trace goes on to the end of the sheet
    assert b.inferred                                                  # the bridge is marked as an estimated link
    a, c = b.inferred[0]
    assert 120 <= b.path[a][0] <= 140 and 185 <= b.path[c][0] <= 200   # it spans the wiped-out stretch
    assert any(i.kind == "inferred" for i in b.issues)
    assert e.undo(0) and not e.bands[0].complete and len(b.path) == n


def test_a_gap_too_wide_is_refused(tmp_path):
    e = _gap_engine(tmp_path, width=700, x0=130)                       # far beyond the reach of a bridge
    b = e.bands[0]
    before = b.path.copy()
    e.set_pending_extend(0, 880.0, 250.0)
    ok, msg = e.apply_pending(0)
    assert not ok and np.array_equal(b.path, before)


def test_bridged_route_between_two_pieces(tmp_path):
    e = _gap_engine(tmp_path)
    b = e.bands[0]
    a = tuple(b.path[-1])
    target, _ = e._snap(230, 250)
    route, bridge = e._route_bridged(a, target)
    assert route is not None and bridge is not None
    i0, i1 = bridge
    assert 0 < i0 < i1 < len(route) and route[i1][0] - route[i0][0] >= 40


def test_a_real_tip_does_not_jump_to_a_neighbouring_band(tmp_path):
    rgb, _ = make_sheet()
    p = tmp_path / "tip.png"
    cv2.imwrite(str(p), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    e = Engine()
    e.open(str(p))
    e.analyze()
    b = e.bands[0]
    for v in sorted(b.vertices, key=lambda v: -v.y)[:5] + sorted(b.vertices, key=lambda v: v.y)[:5]:   # tips pointing down / up
        for band in (None, 0):
            lc = e.line_constraint(v.x, v.y, band=band)
            if lc is None:
                continue
            sx, sy = lc.snap(v.x, 600.0)                                   # the mouse is over band 1's line
            assert lc.last_bridge is None or abs(lc.last_bridge[1][1] - 250) < 150
            assert abs(sy - 250) < 200                                      # never ends up on band 1


def test_the_edge_of_the_reach_window_is_not_treated_as_a_gap(tmp_path):
    rgb, _ = make_sheet()
    p = tmp_path / "w.png"
    cv2.imwrite(str(p), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    e = Engine()
    e.open(str(p))
    e.analyze()
    b = e.bands[0]
    v = b.vertices[len(b.vertices) // 2]
    lc = e.line_constraint(v.x, v.y, limit=40)                              # a small window cuts the line short
    lc.snap(v.x + 150, v.y)
    assert lc.last_bridge is None
