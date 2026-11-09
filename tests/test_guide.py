"""A rough route drawn by the user steers the trace: the line that follows it is preferred, far lines are not entered."""
import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tests.synth import make_sheet  # noqa: E402
from traceview.engine import Engine, EngineError  # noqa: E402
from traceview.project import load_project, save_project  # noqa: E402


def _sheet(tmp_path, top=190, name="g"):
    """Band 0 has a decoy: a box (a loop) from the baseline up to y=`top` over x 100..200, in the quiet part of the sheet."""
    rgb, truth = make_sheet()
    rgb = rgb.copy()
    wave = [[x, top + (4 if (x // 10) % 2 else -4)] for x in range(100, 201, 10)]      # slightly wavy: not taken for a ruler line
    box = np.array([[100, 250]] + wave + [[200, 250]], np.int32)
    cv2.polylines(rgb, [box.reshape(-1, 1, 2)], False, (50, 60, 55), 3, cv2.LINE_AA)
    p = tmp_path / f"{name}.png"
    cv2.imwrite(str(p), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    e = Engine()
    e.open(str(p))
    e.analyze(origins=[(87, 250), (87, 600), (87, 950)])      # (a tall decoy must not be taken for a band's start)
    return e, truth


def _in(path, x0, x1, y0, y1):
    return int(((path[:, 0] >= x0) & (path[:, 0] <= x1) & (path[:, 1] >= y0) & (path[:, 1] <= y1)).sum())


def test_guide_along_the_baseline_keeps_the_trace_off_the_decoy(tmp_path):
    e, _ = _sheet(tmp_path)
    nb = e.retrace_with_guide(0, [(75, 250), (215, 250)], width=15)
    assert nb.guide and nb.complete
    assert _in(nb.path, 110, 190, 183, 197) == 0                      # the decoy's top edge is not traced
    assert _in(nb.path, 110, 190, 244, 256) > 50                      # the baseline is


def test_guide_over_the_decoy_makes_the_trace_take_it(tmp_path):
    e, _ = _sheet(tmp_path)
    nb = e.retrace_with_guide(0, [(75, 250), (100, 215), (150, 190), (200, 215), (215, 250)], width=15)
    assert _in(nb.path, 110, 190, 183, 197) > 50                      # now the loop is the route
    assert _in(nb.path, 115, 185, 244, 256) == 0                      # and the straight baseline in between is not
    assert nb.complete


def test_a_line_far_from_the_route_is_not_entered(tmp_path):
    e, _ = _sheet(tmp_path, top=60, name="far")                        # a tall decoy, 190 px above the baseline
    nb = e.retrace_with_guide(0, [(75, 250), (215, 250)], width=15)
    assert _in(nb.path, 105, 195, 55, 70) == 0                         # its top edge is blocked
    assert nb.complete


def test_a_guide_over_part_of_the_band_does_not_disturb_the_rest(tmp_path):
    e, truth = _sheet(tmp_path)
    nb = e.retrace_with_guide(0, [(75, 250), (215, 250)], width=15)
    right = [pk for pk in truth[0]["peaks"] if pk[0] > 300]
    hit = sum(any(v.kind == k and abs(v.x - x) <= 6 and abs(v.y - y) <= 6 for v in nb.vertices) for x, y, k in right)
    assert hit / len(right) >= 0.95                                    # the spikes beyond the guide are all still reached


def test_spikes_that_stick_out_of_the_route_are_still_visited(tmp_path):
    e, truth = _sheet(tmp_path)
    # a guide along the baseline of the whole band: the tall spikes (up to 120 px away) must not be lost
    nb = e.retrace_with_guide(0, [(75, 250), (2300, 250)], width=20)
    hit = sum(any(v.kind == k and abs(v.x - x) <= 6 and abs(v.y - y) <= 6 for v in nb.vertices)
              for x, y, k in truth[0]["peaks"])
    assert hit / len(truth[0]["peaks"]) >= 0.95
    assert nb.complete


def test_guide_is_undoable_removable_and_saved(tmp_path):
    e, _ = _sheet(tmp_path)
    n = len(e.bands[0].path)
    e.retrace_with_guide(0, [(75, 250), (215, 250)], width=15)
    f = str(tmp_path / "w.trproj")
    save_project(e, f)
    e2 = Engine()
    load_project(e2, f)
    assert e2.bands[0].guide == [[75.0, 250.0], [215.0, 250.0]] and e2.bands[0].guide_width == 15.0
    assert e.undo(0) and not e.bands[0].guide and len(e.bands[0].path) == n
    e.redo(0)
    e.clear_guide(0)
    assert not e.bands[0].guide and e.bands[0].complete


def test_other_bands_are_not_touched_and_too_few_points_are_refused(tmp_path):
    e, _ = _sheet(tmp_path)
    before = [b.path.copy() for b in e.bands[1:]]
    e.retrace_with_guide(0, [(75, 250), (215, 250)], width=15)
    for a, b in zip(before, e.bands[1:]):
        assert np.array_equal(a, b.path)
    with pytest.raises(EngineError):
        e.retrace_with_guide(0, [(75, 250)])


def test_window_flow_draw_the_route_and_trace_along_it(tmp_path):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QPointF
    from PySide6.QtWidgets import QApplication
    from traceview.gui.main_window import MainWindow
    app = QApplication.instance() or QApplication([])
    e, _ = _sheet(tmp_path, name="gui")
    w = MainWindow(sync_mode=True)
    w.show()
    w.engine = e
    w.rebuild_scene()
    b = e.bands[0]
    w.band_list.setCurrentRow(0)
    w.action_apply_guide()                                              # nothing drawn: told, nothing happens
    assert not b.guide and "道筋を描いて" in w.last_info
    w.action_start_guide()
    assert w.tool == "guide" and w.view.click_tool
    for x, y in ((75, 250), (100, 215), (150, 190), (200, 215), (215, 250)):
        w.on_scene_clicked(QPointF(x, y))
    assert len(w.guide_pts) == 5 and w.guide_items
    w.guide_undo_point()
    assert len(w.guide_pts) == 4
    w.on_scene_clicked(QPointF(215, 250))
    w.guide_width_spin.setValue(15)
    w.action_apply_guide()
    nb = e.bands[0]
    assert nb.guide and nb.guide_width == 15.0 and not w.guide_pts and w.tool == "select"
    assert _in(nb.path, 110, 190, 183, 197) > 50                         # it took the route that was drawn
    assert "設定済み" in w.guide_label.text()
    # Esc while drawing throws the unfinished route away
    w.action_start_guide()
    w.on_scene_clicked(QPointF(300, 250))
    w.action_cancel_pending()
    assert not w.guide_pts
    w.action_clear_guide()
    assert not e.bands[0].guide
    w.close()
