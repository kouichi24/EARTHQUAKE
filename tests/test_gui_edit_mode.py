"""The point edit mode in the window: managed selection, one point moves, preview before apply, erase part of the line."""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
pytest.importorskip("PySide6")
from PySide6.QtCore import QPointF, Qt  # noqa: E402
from PySide6.QtWidgets import QApplication, QGraphicsItem  # noqa: E402
from tests.synth import make_sheet  # noqa: E402
from traceview.gui.main_window import MainWindow  # noqa: E402


@pytest.fixture()
def win(tmp_path):
    app = QApplication.instance() or QApplication([])
    rgb, _ = make_sheet()
    p = tmp_path / "e.png"
    cv2.imwrite(str(p), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    w = MainWindow(sync_mode=True)
    w.show()
    w.open_file(str(p))
    w.action_analyze()
    w.set_tool("select")
    yield w
    w.close()


def test_vertex_items_are_not_qt_selectable_so_no_group_move(win):
    items = win.band_items[0]["vertices"]
    assert items and all(not (it.flags() & QGraphicsItem.ItemIsSelectable) for it in items)


def test_click_selects_one_ctrl_adds_and_rings_are_drawn(win):
    win.on_vertex_selected(0, 4, from_canvas=True)
    win.on_vertex_selected(0, 9, from_canvas=True)                  # a plain click replaces the selection
    assert win.sel_set == [(0, 9)] and len(win.sel_rings) == 1
    win.on_vertex_selected(0, 12, from_canvas=True, additive=True)
    win.on_vertex_selected(0, 15, from_canvas=True, additive=True)
    assert win.sel_set == [(0, 9), (0, 12), (0, 15)] and len(win.sel_rings) == 3
    assert "3点" in win.sel_title.text() and win.sel_list.count() == 3
    win.on_vertex_selected(0, 12, from_canvas=True, additive=True)  # toggles off
    assert (0, 12) not in win.sel_set
    win.action_cancel_pending()                                      # Esc with nothing pending clears the selection
    assert win.sel_set == [] and win.sel_rings == []


def test_frame_select(win):
    b = win.engine.bands[0]
    xs = sorted(v.x for v in b.vertices)
    lo, hi = xs[10] - 1, xs[20] + 1
    win.select_in_rect(QPointF(lo, 0), QPointF(hi, 500))
    expect = [i for i, v in enumerate(b.vertices) if lo <= v.x <= hi]
    assert [vi for _bi, vi in win.sel_set] == expect and len(expect) >= 10


def test_dragging_moves_only_the_grabbed_point_and_shows_a_preview(win):
    b = win.engine.bands[0]
    k = len(b.vertices) // 2
    others = [(0, k - 3), (0, k + 3)]
    win.sel_set = others + [(0, k)]
    v = b.vertices[k]
    before = b.path.copy()
    win.on_vertex_dragged(0, k, v.x + 3, v.y + 2)
    assert len(b.pending) == 1 and b.pending[0]["vi"] == k          # one pending edit, for the grabbed point only
    assert np.array_equal(b.path, before)                            # not applied yet
    assert win.edit_preview_items and "プレビュー" in win.preview_label.text()
    assert win.pend_title.text() == "未適用の修正: 1件"
    # apply: only a short stretch around the vertex changes
    win.action_apply()
    assert not b.pending and not win.edit_preview_items
    same = (b.path[:len(before)] == before[:len(b.path)]).all(axis=1) if len(b.path) else np.array([])
    first_diff = int(np.argmin(same)) if not same.all() else len(same)
    assert first_diff > 0
    assert len(b.path) < len(before) + 400                           # the line did not change on a large scale


def test_cancel_with_escape_removes_the_pending_edit_and_the_preview(win):
    b = win.engine.bands[0]
    k = len(b.vertices) // 2
    v = b.vertices[k]
    win.on_vertex_dragged(0, k, v.x + 3, v.y)
    assert b.pending and win.edit_preview_items
    win.action_cancel_pending()
    assert not b.pending and not win.edit_preview_items


def test_an_impossible_move_is_refused_with_a_reason_and_nothing_stays_pending(win):
    b = win.engine.bands[0]
    k = len(b.vertices) // 2
    v = b.vertices[k]
    before = b.path.copy()
    win.on_vertex_dragged(0, k, v.x + 500, v.y + 500)
    assert not b.pending and np.array_equal(b.path, before)
    assert "できません" in win.last_info


def test_delete_removes_all_selected_points_in_one_undo_step(win):
    b = win.engine.bands[0]
    n = len(b.vertices)
    win.sel_set = [(0, 3), (0, 6), (0, 9)]
    win.sel_vertex = 9
    win.action_delete_selected()
    assert len(b.vertices) == n - 3 and not win.sel_set
    assert win.engine.undo(0) and len(b.vertices) == n               # one Ctrl+Z brings all three back


def test_erase_part_of_the_line_preview_confirm_cancel(win):
    b = win.engine.bands[0]
    n = len(b.path)
    win.set_tool("erase")
    win.set_pending_erase(600, 0, 760, 400)
    assert win.erase_pending is not None and win.erase_items
    win.action_cancel_pending()                                      # Esc = do not erase
    assert win.erase_pending is None and not win.erase_items and len(b.path) == n
    win.set_pending_erase(600, 0, 760, 400)
    win.action_apply_erase()                                          # "消す"
    assert len(b.path) < n and b.cuts and b.erased
    assert not np.any((b.path[:, 0] >= 600) & (b.path[:, 0] <= 760))
    assert win.band_items[0]["path"] is not None                     # the gap is drawn, the rest stays
    win.set_pending_erase(5, 5, 20, 20)                              # nothing there: told, not erased
    assert win.erase_pending is None and "ありません" in win.last_info
