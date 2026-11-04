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


@pytest.fixture(scope="module")
def win(tmp_path_factory):
    app = QApplication.instance() or QApplication([])
    rgb, _ = make_sheet()
    p = tmp_path_factory.mktemp("gui") / "s.png"
    cv2.imwrite(str(p), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    w = MainWindow(sync_mode=True)
    w.show()
    w.open_file(str(p))
    w.action_analyze()
    w.engine.add_section(0, 300, 900)
    w.engine.read_point(0, float(w.engine.bands[0].path[500][0]), float(w.engine.bands[0].path[500][1]))
    w.view.center_on_point(600, 250, zoom=1.0)        # zoomed in so that vertices are shown too
    w._build_all_items()
    yield w
    w.close()


def _overlay_items(w):
    out = []
    for d in w.band_items.values():
        for key in ("path", "inferred", "vertices", "lines", "issues", "pending", "sections", "arc", "reads"):
            v = d.get(key)
            if isinstance(v, dict):
                v = list(v.values())
            if v is None:
                continue
            out += v if isinstance(v, list) else [v]
    return out


def test_hide_all_hides_every_overlay_but_keeps_the_data(win):
    items = _overlay_items(win)
    assert len(items) > 50
    n_bands, n_vertices = len(win.engine.bands), sum(len(b.vertices) for b in win.engine.bands)
    assert any(it.isVisible() for it in items)
    win.a_hide_all.setChecked(True)
    win.apply_visibility()
    assert not any(it.isVisible() for it in items)
    assert win.pix_item.isVisible()                       # the plain image stays
    assert "表示に戻す" in win.a_hide_all.text()
    # nothing was deleted
    assert len(win.engine.bands) == n_bands
    assert sum(len(b.vertices) for b in win.engine.bands) == n_vertices
    assert win.engine.bands[0].sections and win.engine.readouts


def test_toggle_back_restores_the_overlay(win):
    win.a_hide_all.setChecked(True)
    win.apply_visibility()
    win.a_hide_all.setChecked(False)
    win.apply_visibility()
    items = _overlay_items(win)
    assert sum(it.isVisible() for it in items) > 50
    assert "全部消す" in win.a_hide_all.text()


def test_stays_hidden_after_a_new_edit_redraws_items(win):
    win.a_hide_all.setChecked(True)
    win.apply_visibility()
    win._build_band_items(0)
    win.apply_visibility()
    assert not any(it.isVisible() for it in _overlay_items(win))
    win.a_hide_all.setChecked(False)
    win.apply_visibility()


def test_dragged_vertex_item_stays_on_the_line(win):
    from PySide6.QtCore import QPointF
    d = win.band_items[0]["vertices"]
    item = d[len(d) // 2]
    x, y = item.pos().x(), item.pos().y()
    win.follow_line = True
    item._lc = win.make_line_constraint(x, y)
    assert item._lc is not None
    item._dragging = True
    item.setPos(QPointF(x + 30, y + 45))                  # the mouse is dragged off the line
    item._dragging = False
    item._lc = None
    p = item.pos()
    assert win.engine.graph.skel[int(round(p.y())), int(round(p.x()))]     # the vertex is still on the line
    item.setPos(QPointF(x, y))


def test_display_uses_light_pyramid_but_analysis_keeps_full_image(win):
    from traceview.gui.view import PyramidItem
    item = win.pix_item
    assert isinstance(item, PyramidItem)
    sizes = [pm.width() for pm in item.pixmaps]
    assert sizes[0] == win.engine.work.shape[1]                       # level 0 = the full analysis image
    assert all(sizes[k + 1] <= sizes[k] // 2 + 1 for k in range(len(sizes) - 1))
    assert PyramidItem.level_for_scale(1.5) == 0                      # zoomed in: full detail
    assert PyramidItem.level_for_scale(0.3) >= 1                      # zoomed out: light copy
    assert PyramidItem.level_for_scale(0.05) == PyramidItem.LEVELS - 1
    # drawing works at both zooms
    win.view.center_on_point(600, 250, zoom=1.5)
    win.grab()
    assert item.last_level == 0
    win.view.fit_all()
    win.grab()
    assert item.last_level >= 1
