"""Selection stays tied to the vertex that was picked, is clearly shown, and Delete removes exactly that vertex."""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import cv2  # noqa: E402
import pytest  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
pytest.importorskip("PySide6")
from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402
from tests.synth import make_sheet  # noqa: E402
from traceview.gui.main_window import MainWindow  # noqa: E402


@pytest.fixture()
def win(tmp_path):
    app = QApplication.instance() or QApplication([])
    rgb, _ = make_sheet()
    p = tmp_path / "s.png"
    cv2.imwrite(str(p), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    w = MainWindow(sync_mode=True)
    w.show()
    w.open_file(str(p))
    w.action_analyze()
    w.set_tool("select")
    yield w
    w.close()


def test_stale_selection_deletes_the_picked_vertex(win):
    b = win.engine.bands[0]
    win.on_vertex_selected(0, 8, from_canvas=True)
    target = (b.vertices[8].x, b.vertices[8].y)
    win.engine.delete_vertices(0, [3])            # list shifts behind the selection
    n = len(b.vertices)
    win.action_delete_selected()
    assert len(b.vertices) == n - 1
    assert all(abs(v.x - target[0]) > 1.5 or abs(v.y - target[1]) > 1.5 for v in b.vertices)


def test_delete_key_in_view_and_hud(win):
    b = win.engine.bands[0]
    assert "選択なし" in win.sel_hud.text()
    win.on_vertex_selected(0, 10, from_canvas=True)
    assert "1点" in win.sel_hud.text()
    n = len(b.vertices)
    QTest.keyClick(win.view, Qt.Key_Delete)
    assert len(b.vertices) == n - 1
    assert "選択なし" in win.sel_hud.text()


def test_escape_clears_selection(win):
    win.on_vertex_selected(0, 5, from_canvas=True)
    QTest.keyClick(win.view, Qt.Key_Escape)
    assert win.sel_set == []
