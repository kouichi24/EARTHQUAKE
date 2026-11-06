"""Things that are not part of the graph can be fenced off BEFORE the analysis, and the program suggests them."""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tests.synth import BASELINES, make_sheet  # noqa: E402
from traceview.engine import Engine  # noqa: E402


def _sheet(tmp_path, stain=True, text=True, joined_text=False):
    rgb, truth = make_sheet()
    rgb = rgb.copy()
    rng = np.random.default_rng(3)
    if stain:                                     # a dark, dense stain below the last band (y 1066-1104)
        for _ in range(900):
            x, y = int(rng.integers(300, 900)), int(rng.integers(1066, 1104))
            cv2.circle(rgb, (x, y), int(rng.integers(3, 7)), (60, 65, 55), -1)
    if text:                                      # writing between the bands (outside every band's own territory)
        for k, word in enumerate(("ABC", "DEF", "GHI")):
            cv2.putText(rgb, word, (1500 + 85 * k, 430), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (50, 60, 55), 3, cv2.LINE_AA)
    if joined_text:                               # letters standing on band 0's pen line
        cv2.putText(rgb, "XYZ", (1490, 251), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (50, 60, 55), 5, cv2.LINE_AA)
    p = tmp_path / "x.png"
    cv2.imwrite(str(p), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    e = Engine()
    e.open(str(p))
    return e, truth


def _overlaps(c, rect):
    return not (c["x1"] < rect[0] or c["x0"] > rect[2] or c["y1"] < rect[1] or c["y0"] > rect[3])


def test_stain_and_writing_are_suggested_and_the_waveform_is_not(tmp_path):
    e, _ = _sheet(tmp_path)
    cands = e.suggest_exclusions()
    kinds = {c["kind"] for c in cands}
    assert "汚れ" in kinds and "文字・書き込み" in kinds
    stain = [c for c in cands if c["kind"] == "汚れ"]
    assert any(_overlaps(c, (300, 1066, 900, 1104)) for c in stain)
    text = [c for c in cands if c["kind"] == "文字・書き込み"]
    assert any(_overlaps(c, (1500, 400, 1760, 440)) for c in text)
    # nothing that is suggested touches a band's baseline territory
    for c in cands:
        for by in BASELINES:
            assert not (c["y0"] < by + 20 and c["y1"] > by - 20), (c, by)


def test_a_clean_sheet_gives_no_suggestions(tmp_path):
    e, _ = _sheet(tmp_path, stain=False, text=False)
    assert e.suggest_exclusions() == []


def test_exclusion_added_before_the_analysis_is_applied_by_it(tmp_path):
    e, truth = _sheet(tmp_path, stain=False, text=False, joined_text=True)
    assert e.add_exclusion(1480, 205, 1600, 244) == []              # not analysed yet: remembered only, quick
    assert e.graph is None and not e.bands and len(e.exclusions) == 1
    e.analyze()
    b = e.bands[0]
    inside = ((b.path[:, 0] >= 1480) & (b.path[:, 0] <= 1600) & (b.path[:, 1] >= 205) & (b.path[:, 1] <= 244)).sum()
    assert inside == 0 and b.complete                               # the letters never entered the trace


def test_several_suggestions_are_added_in_one_go(tmp_path):
    e, _ = _sheet(tmp_path)
    cands = e.suggest_exclusions()
    e.add_exclusions(cands)
    assert len(e.exclusions) == len(cands)
    e.analyze()
    assert len(e.bands) == 3 and all(b.complete for b in e.bands)
    # the suggestions are not offered again once they are excluded
    assert e.suggest_exclusions() == []


def test_window_flow_find_check_add_before_analysis(tmp_path):
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication
    from traceview.gui.main_window import MainWindow
    app = QApplication.instance() or QApplication([])
    e, _ = _sheet(tmp_path)
    w = MainWindow(sync_mode=True)
    w.show()
    w.open_file(e.src_path)
    assert w.t_excl.isEnabled()                                      # the fence tool works before the analysis
    w.action_find_candidates()
    assert w.cand and w.cand_list.count() == len(w.cand) and w.cand_items
    w.action_add_candidates()                                        # nothing checked: told, nothing added
    assert not w.engine.exclusions
    w._check_all_candidates(True)
    w.action_add_candidates()
    assert w.engine.exclusions and not w.cand and not w.engine.bands
    assert "自動解析" in w.state["解析"]
    w.action_analyze()
    assert len(w.engine.bands) == 3
    w.close()
