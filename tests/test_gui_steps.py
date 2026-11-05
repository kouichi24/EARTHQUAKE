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
