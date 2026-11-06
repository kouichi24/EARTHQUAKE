"""Deleting a vertex (Delete key): it disappears, can be undone, and stays deleted."""
import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tests.synth import make_sheet  # noqa: E402
from traceview.engine import Engine, EngineError  # noqa: E402
from traceview.project import load_project, save_project  # noqa: E402
from traceview.verify import reversals_from_path  # noqa: E402


@pytest.fixture()
def eng(tmp_path):
    rgb, _ = make_sheet()
    p = tmp_path / "s.png"
    cv2.imwrite(str(p), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    e = Engine()
    e.open(str(p))
    e.analyze()
    return e


def _spur_vertex(b, min_x=0):
    """A vertex at the tip of a branch that the pen went into and came back from."""
    rp = set(reversals_from_path(b.path))
    for i, v in enumerate(b.vertices):
        if v.x > min_x and any(abs(v.idx - r) <= 2 for r in rp) and abs(v.y - b.line_zero) > 40:
            return i, v
    raise AssertionError("no spur vertex in the test sheet")


def test_deleting_a_tip_removes_exactly_that_vertex_and_leaves_the_line(eng):
    b = eng.bands[0]
    i, v = _spur_vertex(b)
    n_v, path = len(b.vertices), b.path.copy()
    names = {(round(x.x), round(x.y), x.kind) for x in b.vertices}
    msg = eng.delete_vertex(0, i)
    assert "線は変えていません" in msg
    assert len(b.vertices) == n_v - 1
    assert names - {(round(x.x), round(x.y), x.kind) for x in b.vertices} == {(round(v.x), round(v.y), v.kind)}
    assert not any(abs(x.x - v.x) < 6 and abs(x.y - v.y) < 6 for x in b.vertices)   # nothing new pops up in its place
    assert np.array_equal(b.path, path) and b.complete


def test_a_deleted_vertex_is_left_out_of_the_measurements(eng):
    b = eng.bands[0]
    top = min(range(len(b.vertices)), key=lambda i: b.vertices[i].y)       # the highest peak
    before = eng.measures(b)["max_value"]
    eng.delete_vertex(0, top)
    assert eng.measures(b)["max_value"] < before


def test_undo_brings_the_vertex_back(eng):
    b = eng.bands[0]
    i, _v = _spur_vertex(b)
    n_v, n_p = len(b.vertices), len(b.path)
    eng.delete_vertex(0, i)
    assert eng.undo(0)
    assert len(b.vertices) == n_v and len(b.path) == n_p and not b.deleted_points
    assert eng.redo(0) and len(b.vertices) == n_v - 1


def test_deleted_vertex_stays_deleted_after_the_line_is_traced_again(eng):
    b = eng.bands[0]
    i, v = _spur_vertex(b, min_x=700)
    eng.delete_vertex(0, i)
    # another edit re-traces the band from a vertex further left: the deleted vertex must not come back
    k = next(j for j, x in enumerate(b.vertices) if x.x < v.x - 200 and x.idx > 20)
    x = b.vertices[k]
    eng.set_pending_move(0, k, x.x + 1, x.y)
    ok, msg = eng.apply_pending(0)
    assert ok, msg
    assert not any(abs(q.x - v.x) < 6 and abs(q.y - v.y) < 6 for q in b.vertices)


def test_saved_in_the_work_file(eng, tmp_path):
    b = eng.bands[0]
    i, v = _spur_vertex(b)
    eng.delete_vertex(0, i)
    f = str(tmp_path / "w.trproj")
    save_project(eng, f)
    e2 = Engine()
    load_project(e2, f)
    assert e2.bands[0].deleted_points == b.deleted_points
    assert len(e2.bands[0].vertices) == len(b.vertices)


def test_nothing_selected_is_reported(eng):
    with pytest.raises(EngineError):
        eng.delete_vertex(0, -1)
    with pytest.raises(EngineError):
        eng.delete_vertex(0, 10**6)
