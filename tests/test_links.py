"""Estimated (yellow) links can be confirmed or released one by one."""
import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tests.synth import make_sheet  # noqa: E402
from traceview.engine import Engine  # noqa: E402

PAPER = np.array([205, 195, 170], np.uint8)
INK = np.array([50, 60, 55], np.float32)


@pytest.fixture()
def eng(tmp_path):
    rgb, _ = make_sheet()
    rgb = rgb.copy()
    x0, x1, by = 130, 190, 250
    rgb[by - 20:by + 21, x0:x1] = PAPER
    fc = (0.55 * PAPER.astype(np.float32) + 0.45 * INK).astype(np.uint8)
    cv2.line(rgb, (x0, by), (x1, by), tuple(int(c) for c in fc), 3)
    p = tmp_path / "l.png"
    cv2.imwrite(str(p), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    e = Engine()
    e.open(str(p))
    e.analyze()
    return e


def test_accept_one_link(eng):
    ls = eng.links(0)
    assert ls and not ls[0]["accepted"]
    n_issue = sum(1 for i in eng.bands[0].issues if i.kind == "inferred")
    assert n_issue >= 1
    eng.accept_link(0, 0)
    assert eng.links(0)[0]["accepted"]
    assert sum(1 for i in eng.bands[0].issues if i.kind == "inferred") < n_issue   # no longer needs checking


def test_release_breaks_the_line_and_undo_restores(eng):
    b = eng.bands[0]
    n = len(b.path)
    eng.release_link(0, 0)
    assert len(b.path) < n and b.cuts                      # a gap is left where the link was
    assert not eng.links(0)                                # no estimated link left on that stretch
    b.restore(b.undo.pop())
    assert len(b.path) == n and eng.links(0)


def test_gui_yellow_link_menu(eng, tmp_path, monkeypatch):
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QPoint
    from PySide6.QtWidgets import QApplication
    from traceview.gui.main_window import MainWindow
    from traceview.gui.view import LinkItem
    QApplication.instance() or QApplication([])
    p = tmp_path / "w.png"
    rgb = eng.work
    cv2.imwrite(str(p), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    w = MainWindow(sync_mode=True)
    w.show()
    w.open_file(str(p))
    w.action_analyze()
    items = [it for it in w.scene.items() if isinstance(it, LinkItem)]
    assert items                                           # yellow links are clickable items
    bi, k = items[0].band_i, items[0].link_k
    monkeypatch.setattr(w, "_ask_link_action", lambda bi_, pos: "ok")
    n = len(w.engine.links(bi))
    w.link_menu(bi, k, QPoint(0, 0))
    assert w.engine.links(bi)[k]["accepted"]
    assert len([it for it in w.scene.items() if isinstance(it, LinkItem) and it.band_i == bi and it.link_k == k]) == 0
    assert len(w.engine.links(bi)) == n
    w.close()
