"""Dragging a vertex keeps it on the image's line."""
import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tests.synth import make_sheet  # noqa: E402
from traceview.engine import Engine  # noqa: E402


@pytest.fixture(scope="module")
def eng(tmp_path_factory):
    rgb, _ = make_sheet()
    p = tmp_path_factory.mktemp("fl") / "s.png"
    cv2.imwrite(str(p), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    e = Engine()
    e.open(str(p))
    e.analyze()
    return e


def test_snapped_points_are_always_on_the_line(eng):
    b = eng.bands[0]
    v = max(b.vertices, key=lambda v: -v.y if 400 < v.x < 1500 else 1e9)   # a tall peak
    lc = eng.line_constraint(v.x, v.y)
    assert lc is not None and lc.ok
    rng = np.random.default_rng(0)
    for _ in range(200):
        tx, ty = v.x + rng.normal(0, 60), v.y + rng.normal(0, 60)           # a mouse wandering around the point
        sx, sy = lc.snap(tx, ty)
        assert eng.graph.skel[int(sy), int(sx)]                              # always a pixel of the image's line


def test_stays_on_connected_line_not_on_a_neighbour_band(eng):
    b0, b1 = eng.bands[0], eng.bands[1]
    v = b0.vertices[len(b0.vertices) // 2]
    lc = eng.line_constraint(v.x, v.y, limit=100)
    # the mouse is dragged far down onto band 2's line: the point must not leave band 1's connected line
    sx, sy = lc.snap(v.x, b1.origin[1])
    assert abs(sy - b0.origin[1]) < 200
    assert np.hypot(sx - v.x, sy - v.y) <= 100 * 1.5                        # within reach along the line


def test_snap_moves_with_the_mouse_along_the_line(eng):
    b = eng.bands[0]
    xs = [v.x for v in b.vertices if 300 < v.x < 1000]
    v = b.vertices[len(b.vertices) // 3]
    lc = eng.line_constraint(v.x, v.y, limit=200)
    near = lc.snap(v.x + 25, v.y)
    far = lc.snap(v.x + 120, v.y)
    assert far[0] >= near[0] - 1                                            # follows the mouse to the right


def test_no_constraint_without_a_graph():
    e = Engine()
    assert e.line_constraint(10, 10) is None


def test_no_constraint_far_from_any_line(eng):
    assert eng.line_constraint(5, 5) is None                                  # nothing there: free movement
