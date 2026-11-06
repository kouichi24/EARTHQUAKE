"""Edit mode: local moves, previews, erasing part of the line."""
import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tests.synth import make_sheet  # noqa: E402
from traceview.engine import Engine, EngineError  # noqa: E402
from traceview.project import load_project, save_project  # noqa: E402
from traceview.verify import find_problems  # noqa: E402


@pytest.fixture()
def eng(tmp_path):
    rgb, _ = make_sheet()
    p = tmp_path / "s.png"
    cv2.imwrite(str(p), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    e = Engine()
    e.open(str(p))
    e.analyze()
    return e


def _mid_vertex(b):
    k = len(b.vertices) // 2
    return k, b.vertices[k]


def test_local_move_changes_only_the_stretch_between_neighbours(eng):
    b = eng.bands[0]
    k, v = _mid_vertex(b)
    before = b.path.copy()
    vk = v.idx
    start, end = vk - 60, vk + 60                                       # the re-routed stretch is at most this long
    eng.set_pending_move(0, k, v.x + 2, v.y + 2)
    ok, msg = eng.apply_pending(0, mode="local")
    assert ok, msg
    assert "前後" in msg
    # everything before the previous vertex and after the next vertex is exactly as it was
    assert np.array_equal(b.path[:start], before[:start])
    assert np.array_equal(b.path[-(len(before) - end - 1):], before[end + 1:])
    assert any(x.source == "manual" for x in b.vertices)


def test_tail_mode_still_re_traces_from_the_vertex(eng):
    b = eng.bands[0]
    k, v = _mid_vertex(b)
    eng.set_pending_move(0, k, v.x + 2, v.y + 2)
    ok, msg = eng.apply_pending(0, mode="tail")
    assert ok and "そこから先を再追跡" in msg


def test_preview_changes_nothing_and_reports_how_far_the_change_reaches(eng):
    b = eng.bands[0]
    k, v = _mid_vertex(b)
    before = b.path.copy()
    eng.set_pending_move(0, k, v.x + 3, v.y)
    pv_local = eng.preview_pending(0, "local")
    pv_tail = eng.preview_pending(0, "tail")
    assert np.array_equal(b.path, before) and b.pending                # nothing was applied
    assert pv_local["changed_old"] < 121 / eng.scale and pv_local["changed_new"] < 200 / eng.scale   # a short stretch only
    assert pv_local["new"] is not None and pv_local["first_changed"] is not None


def test_failed_local_move_keeps_everything(eng):
    b = eng.bands[0]
    k, v = _mid_vertex(b)
    before = b.path.copy()
    eng.set_pending_move(0, k, v.x + 500, v.y + 500)
    ok, msg = eng.apply_pending(0, mode="local")
    assert not ok and np.array_equal(b.path, before)
    eng.cancel_pending(0)


def test_erase_removes_the_line_inside_the_rectangle_and_marks_the_break(eng):
    b = eng.bands[0]
    n = len(b.path)
    x0, x1 = 600, 760
    n_removed = eng.erase_rect(0, x0, 0, x1, 400)
    assert n_removed > 0 and len(b.path) == n - n_removed
    assert not np.any((b.path[:, 0] >= x0) & (b.path[:, 0] <= x1))
    assert b.cuts and b.complete                                      # one break in the middle; the end is still reached
    assert not any(x0 <= v.x <= x1 for v in b.vertices)
    assert any(i.kind == "erased" for i in b.issues)
    assert "jump" not in [f.kind for f in find_problems(eng, b)]      # an erased stretch is not reported as a jump
    # vertices outside are untouched in number terms: only those inside went away
    assert all(0 <= v.idx < len(b.path) for v in b.vertices)


def test_erased_stretch_stays_erased_after_the_line_is_traced_again(eng):
    b = eng.bands[0]
    eng.erase_rect(0, 900, 0, 1000, 400)
    k = next(j for j, v in enumerate(b.vertices) if v.x < 800 and v.idx > 30)
    v = b.vertices[k]
    eng.set_pending_move(0, k, v.x + 1, v.y)
    ok, msg = eng.apply_pending(0, mode="tail")                      # the tail is traced again through the erased place
    assert ok, msg
    assert not np.any((b.path[:, 0] >= 900) & (b.path[:, 0] <= 1000))
    assert b.cuts


def test_erase_is_undoable_and_saved(eng, tmp_path):
    b = eng.bands[0]
    n = len(b.path)
    eng.erase_rect(0, 600, 0, 760, 400)
    f = str(tmp_path / "w.trproj")
    save_project(eng, f)
    e2 = Engine()
    load_project(e2, f)
    assert e2.bands[0].erased == b.erased and e2.bands[0].cuts == b.cuts
    assert eng.undo(0) and len(b.path) == n and not b.erased and not b.cuts


def test_erasing_an_empty_place_is_refused(eng):
    with pytest.raises(EngineError):
        eng.erase_rect(0, 5, 5, 20, 20)
    assert not eng.bands[0].erased
