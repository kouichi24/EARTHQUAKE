"""Letters touching the trace can be fenced off; the line is then traced without them."""
import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tests.synth import make_sheet  # noqa: E402
from traceview.engine import Engine, EngineError  # noqa: E402
from traceview.project import load_project, save_project  # noqa: E402

RECT = (1480, 205, 1600, 244)       # work px around the letters (above the baseline y=250 of band 0)


def _sheet_with_text(tmp_path):
    rgb, truth = make_sheet()
    rgb = rgb.copy()
    # thick dark letters standing on the pen line of band 0 so that they are joined to it
    cv2.putText(rgb, "ABC", (1490, 251), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (50, 60, 55), 5, cv2.LINE_AA)
    p = tmp_path / "text.png"
    cv2.imwrite(str(p), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    e = Engine()
    e.open(str(p))
    e.analyze()
    return e, truth


def _inside(path, r=RECT):
    return int(((path[:, 0] >= r[0]) & (path[:, 0] <= r[2]) & (path[:, 1] >= r[1]) & (path[:, 1] <= r[3])).sum())


def _recall(band, peaks, tol=6):
    hit = sum(any(v.kind == k and abs(v.x - x) <= tol and abs(v.y - y) <= tol for v in band.vertices)
              for x, y, k in peaks)
    return hit / len(peaks)


def test_letters_pull_the_trace_off_the_line_until_excluded(tmp_path):
    e, truth = _sheet_with_text(tmp_path)
    assert _inside(e.bands[0].path) > 0          # the trace wanders into the letters
    affected = e.add_exclusion(*RECT)
    assert 0 in affected
    b = e.bands[0]
    assert _inside(b.path) == 0                  # nothing of the letters is traced any more
    assert b.complete
    assert _recall(b, truth[0]["peaks"]) >= 0.95
    assert not any(RECT[0] <= v.x <= RECT[2] and RECT[1] <= v.y <= RECT[3] for v in b.vertices)


def test_other_bands_are_not_touched(tmp_path):
    e, _ = _sheet_with_text(tmp_path)
    before = [b.path.copy() for b in e.bands]
    e.add_exclusion(*RECT)
    for i in (1, 2):
        assert np.array_equal(e.bands[i].path, before[i])


def test_exclusion_is_undoable_and_removable(tmp_path):
    e, _ = _sheet_with_text(tmp_path)
    n = len(e.bands[0].path)
    e.add_exclusion(*RECT)
    assert e.undo(0)
    assert len(e.bands[0].path) == n             # the earlier line is back
    e.redo(0)
    e.remove_exclusion(0)
    assert not e.exclusions
    assert _inside(e.bands[0].path) > 0          # without the fence the letters are traced again


def test_too_small_rectangle_rejected(tmp_path):
    e, _ = _sheet_with_text(tmp_path)
    with pytest.raises(EngineError):
        e.add_exclusion(10, 10, 12, 12)


def test_exclusions_saved_and_reapplied(tmp_path):
    e, _ = _sheet_with_text(tmp_path)
    e.add_exclusion(*RECT)
    f = str(tmp_path / "w.trproj")
    save_project(e, f)
    e2 = Engine()
    load_project(e2, f)
    assert len(e2.exclusions) == 1
    e2.ensure_graph()
    assert not e2.prepared.ink[RECT[1]:RECT[3], RECT[0]:RECT[2]].any()      # fence applied when the graph is built again
