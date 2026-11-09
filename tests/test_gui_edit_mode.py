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


def test_selection_survives_a_refused_move_and_a_cancel(win):
    b = win.engine.bands[0]
    win.sel_set = []
    for k in (5, 9, 14):
        win.on_vertex_selected(0, k, from_canvas=True, additive=True)
    assert len(win.sel_set) == 3
    v = b.vertices[9]
    win.on_vertex_dragged(0, 9, v.x + 500, v.y + 500)              # refused: nothing changes
    assert len(win.sel_set) == 3 and len(win.sel_rings) == 3       # the selection is still there
    win.on_vertex_dragged(0, 9, v.x + 3, v.y)
    win.action_cancel_pending()                                      # cancel the pending edit: still selected
    assert len(win.sel_set) == 3


def test_end_handle_takes_the_trace_across_a_gap_with_preview_and_enter(tmp_path):
    from tests.test_gap_jump import _gap_engine
    from traceview.gui.view import EndHandleItem
    app = QApplication.instance() or QApplication([])
    e = _gap_engine(tmp_path)
    w = MainWindow(sync_mode=True)
    w.show()
    w.engine = e
    w.rebuild_scene()
    w.set_tool("select")
    h = w.band_items[0]["end"]
    assert isinstance(h, EndHandleItem) and h.flags() & QGraphicsItem.ItemIsMovable
    # drag the end across the gap: the item lands on the far line and a dashed link is shown
    h._lc = w.make_line_constraint(h.pos().x(), h.pos().y())
    h._dragging = True
    h.setPos(QPointF(215.0, 250.0))
    assert w.bridge_item is not None
    h._dragging = False
    w.show_drag_bridge(None)
    assert w.bridge_item is None
    w.on_end_dragged(0, h.pos().x(), h.pos().y())
    b = e.bands[0]
    assert b.pending and w.edit_preview_items and not b.complete        # preview only
    w.view.applyRequested.emit()                                          # Enter in the image area
    assert not b.pending and b.complete and b.inferred
    w.close()


def test_other_bands_are_locked_while_one_band_is_edited(win):
    from PySide6.QtCore import Qt as _Qt
    assert win.lock_others and win.cb_lock.isChecked()
    win.sel_band = 0
    win.band_list.setCurrentRow(0)
    for bi in (0, 1, 2):
        d = win.band_items[bi]
        active = bi == 0
        for it in d["vertices"][:20] + [d["end"]]:
            assert (it.acceptedMouseButtons() != _Qt.NoButton) == active           # the other bands ignore the mouse
            assert bool(it.flags() & QGraphicsItem.ItemIsMovable) == active
            assert (it.opacity() == 1.0) == active                                  # ... and are shown dimmed
        for ln in d["lines"].values():
            assert bool(ln.flags() & QGraphicsItem.ItemIsMovable) == active
    assert "ロック中" in win.lock_label.text() and "上段" in win.lock_label.text()
    # a click on a locked band's vertex does nothing
    win.on_vertex_selected(1, 5, from_canvas=True)
    assert win.sel_band == 0 and (1, 5) not in win.sel_set
    # choosing another band moves the lock to it
    win.band_list.setCurrentRow(1)
    assert win.sel_band == 1
    assert win.band_items[1]["vertices"][0].acceptedMouseButtons() != _Qt.NoButton
    assert win.band_items[0]["vertices"][0].acceptedMouseButtons() == _Qt.NoButton
    # unlocking makes every band editable again
    win.cb_lock.setChecked(False)
    assert all(it.acceptedMouseButtons() != _Qt.NoButton for bi in (0, 1, 2) for it in win.band_items[bi]["vertices"][:20])
    assert "ロックなし" in win.lock_label.text() or "編集できます" in win.lock_label.text()
    win.cb_lock.setChecked(True)
    win.band_list.setCurrentRow(0)


def test_editing_the_red_band_does_not_change_the_other_bands(win):
    win.band_list.setCurrentRow(0)
    others = [(win.engine.bands[i].path.copy(), len(win.engine.bands[i].vertices)) for i in (1, 2)]
    b = win.engine.bands[0]
    k = len(b.vertices) // 2
    v = b.vertices[k]
    win.on_vertex_dragged(0, k, v.x + 3, v.y)
    win.action_apply()
    win.delete_after = None
    win.sel_set = [(0, 4)]
    win.sel_vertex = 4
    win.action_delete_selected()
    win.set_pending_erase(900, 0, 1000, 400)
    win.action_apply_erase()
    for (path, nv), i in zip(others, (1, 2)):
        assert np.array_equal(win.engine.bands[i].path, path) and len(win.engine.bands[i].vertices) == nv


def test_exclusion_over_two_bands_re_traces_only_the_edited_one(win):
    from tests.synth import BASELINES
    win.band_list.setCurrentRow(0)
    before1 = win.engine.bands[1].path.copy()
    win.add_exclusion_rect(700, BASELINES[0] - 20, 760, BASELINES[1] + 20)     # holds pieces of the red AND the green line
    assert np.array_equal(win.engine.bands[1].path, before1)                    # green stays as it was
    assert any(i.kind == "exclusion_pending" for i in win.engine.bands[1].issues)
