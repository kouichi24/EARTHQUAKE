import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import cv2  # noqa: E402
import pytest  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication  # noqa: E402
from tests.synth import make_sheet  # noqa: E402
from traceview.gui.main_window import MainWindow  # noqa: E402


def test_step_by_step_analysis_in_the_window(tmp_path):
    app = QApplication.instance() or QApplication([])
    rgb, _ = make_sheet()
    p = tmp_path / "s.png"
    cv2.imwrite(str(p), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    w = MainWindow(sync_mode=True)
    w.show()
    w.open_file(str(p))
    w.cb_step.setChecked(True)
    w.action_analyze()
    assert [b.traced for b in w.engine.bands] == [True, False, False]
    assert w.a_next_band.isEnabled() and "残り2段" in w.a_next_band.text()
    w.action_next_band()
    assert [b.traced for b in w.engine.bands] == [True, True, False]
    assert w.sel_band == 1 and "残り1段" in w.a_next_band.text()
    w.action_next_band()
    assert not w.a_next_band.isEnabled()
    assert all(len(b.path) > 100 for b in w.engine.bands)
    # after correcting band 0 the bands below can be traced again
    w.sel_band = 0
    w._update_step_actions()
    assert w.a_follow.isEnabled()
    w.action_retrace_following()
    assert all(b.complete for b in w.engine.bands)
    w.close()


def test_confirm_tab_flow_range_check_retrace_confirm_next(tmp_path):
    from PySide6.QtCore import QPointF
    from tests.synth import BASELINES
    app = QApplication.instance() or QApplication([])
    rgb, _ = make_sheet()
    p = tmp_path / "c.png"
    cv2.imwrite(str(p), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    w = MainWindow(sync_mode=True)
    w.show()
    w.open_file(str(p))
    w.cb_step.setChecked(True)
    w.action_analyze()
    assert w.tabs.tabText(w.tabs.currentIndex()) == "段の確認"             # the guide opens by itself
    assert "上段" in w.conf_title.text() and "1/3段目" in w.conf_title.text()
    # ① drag the vertical range  ② the check appears by itself
    w.set_tool("brange")
    w._section_drag_done(QPointF(0, BASELINES[0] - 100), QPointF(0, BASELINES[0] + 100))
    assert w.sel_band in w.pending_range
    assert "起点" in w.conf_check_label.text() and "上端" in w.conf_check_label.text()
    assert "引き直せます" in w.conf_check_label.text()
    assert w.engine.band_ranges == {}                                    # only chosen so far
    # ③ trace again inside the range
    w.action_retrace_range()
    assert w.engine.band_ranges[0] == (BASELINES[0] - 100, BASELINES[0] + 100)
    assert "この範囲で解析済み" in w.conf_range_label.text()
    # a range that does not contain the start of the line is refused and nothing changes
    before = w.engine.bands[0].path.copy()
    w.set_pending_range(BASELINES[0] + 40, BASELINES[0] + 300)
    w.action_retrace_range()
    assert (w.engine.bands[0].path == before).all() and w.engine.band_ranges[0][0] == BASELINES[0] - 100
    # ⑤ confirm and go to the next band: it is traced now
    w.pending_range.clear()
    w.action_confirm_and_next()
    assert w.engine.bands[0].confirmed and w.engine.bands[1].traced
    assert w.sel_band == 1
    w.close()


def test_delete_key_removes_the_selected_vertex_and_says_so_when_nothing_is_selected(tmp_path):
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QKeySequence
    app = QApplication.instance() or QApplication([])
    rgb, _ = make_sheet()
    p = tmp_path / "d.png"
    cv2.imwrite(str(p), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    w = MainWindow(sync_mode=True)
    w.show()
    w.open_file(str(p))
    w.action_analyze()
    keys = [s.toString() for s in w.a_delete.shortcuts()]
    assert "Del" in keys and "Backspace" in keys                      # both keys are bound
    b = w.engine.bands[0]
    n = len(b.vertices)
    # nothing selected: nothing is deleted and the user is told what to do
    w.sel_vertex = None
    w.action_delete_selected()
    assert len(b.vertices) == n and "選ばれていません" in w.last_info
    # select a vertex on the canvas (as a click does) and press Delete
    w.on_vertex_selected(0, 5, from_canvas=True)
    assert w.sel_vertex == 5
    w.a_delete.trigger()                                              # what Delete / Backspace do
    assert len(b.vertices) == n - 1 and w.sel_vertex is None and "削除しました" in w.last_info
    assert w.sel_ring is None
    # the list that has the focus wins: a selected reading point is deleted instead of a vertex
    w.engine.read_point(0, float(b.path[300][0]), float(b.path[300][1]))
    w.update_read_table()
    w.read_table.selectRow(0)
    w.on_vertex_selected(0, 3, from_canvas=True)
    n2 = len(b.vertices)
    w.action_delete_selected(focus=w.read_table)
    assert not w.engine.readouts and len(b.vertices) == n2
    # undo brings the vertex back
    assert w.engine.undo(0) and len(b.vertices) == n
    w.close()
