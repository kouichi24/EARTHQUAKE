"""The user fixes a band's vertical range, checks that lines reach it, and traces the band again inside it."""
import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tests.synth import BASELINES, make_sheet  # noqa: E402
from traceview.engine import Engine, EngineError  # noqa: E402
from traceview.project import load_project, save_project  # noqa: E402


@pytest.fixture()
def eng(tmp_path):
    rgb, truth = make_sheet()
    p = tmp_path / "s.png"
    cv2.imwrite(str(p), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    e = Engine()
    e.open(str(p))
    e.analyze()
    return e


def test_check_reports_whether_the_line_reaches_the_range(eng):
    b = eng.bands[0]
    lo, hi = float(b.path[:, 1].min()), float(b.path[:, 1].max())
    ok = eng.check_range(0, lo - 5, hi + 5)
    assert ok["traced"] and ok["origin_inside"] and ok["reaches_top"] and ok["reaches_bottom"]
    assert ok["outside_px"] == 0
    too_wide = eng.check_range(0, lo - 150, hi + 150)               # nothing goes that far
    assert not too_wide["reaches_top"] and not too_wide["reaches_bottom"]
    cut = eng.check_range(0, BASELINES[0] - 40, BASELINES[0] + 40)   # the range cuts off the tall spikes
    assert cut["outside_px"] > 0


def test_retrace_stays_inside_the_range_and_keeps_the_other_bands(eng):
    others = [b.path.copy() for b in eng.bands[1:]]
    top, bot = BASELINES[0] - 60, BASELINES[0] + 60
    nb = eng.retrace_in_range(0, top, bot)
    assert eng.band_ranges[0] == (top, bot)
    # the range is a 'at least up to here' guide: a spike may poke out of it, but a line that lies outside is not used
    inside = (nb.path[:, 1] >= top - 12) & (nb.path[:, 1] <= bot + 12)
    assert inside.mean() > 0.9
    assert nb.path[:, 1].min() >= top - 130 and nb.path[:, 1].max() <= bot + 130
    assert nb.complete
    for a, c in zip(others, eng.bands[1:]):
        assert np.array_equal(a, c.path)


def test_range_without_the_origin_is_refused_and_nothing_changes(eng):
    before = eng.bands[0].path.copy()
    with pytest.raises(EngineError):
        eng.retrace_in_range(0, BASELINES[0] + 20, BASELINES[0] + 200)
    with pytest.raises(EngineError):
        eng.retrace_in_range(0, 240, 250)
    assert np.array_equal(eng.bands[0].path, before) and 0 not in eng.band_ranges


def test_range_retrace_is_undoable_and_range_can_be_cleared(eng):
    n = len(eng.bands[0].path)
    eng.retrace_in_range(0, BASELINES[0] - 60, BASELINES[0] + 60)
    assert eng.undo(0)
    assert len(eng.bands[0].path) == n
    eng.redo(0)
    eng.clear_range(0)
    assert 0 not in eng.band_ranges
    assert eng.bands[0].complete


def test_ranges_are_saved(eng, tmp_path):
    eng.retrace_in_range(1, BASELINES[1] - 80, BASELINES[1] + 80)
    f = str(tmp_path / "w.trproj")
    save_project(eng, f)
    e2 = Engine()
    load_project(e2, f)
    assert e2.band_ranges == {1: (BASELINES[1] - 80, BASELINES[1] + 80)}
