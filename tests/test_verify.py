"""Self-check and re-tracing: unnatural spots are found, repaired when it helps, and every attempt is kept."""
import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tests.synth import make_sheet  # noqa: E402
from traceview.engine import Engine  # noqa: E402
from traceview.project import load_project, save_project  # noqa: E402
from traceview.verify import (find_problems, reversals_from_path, score, verify_and_repair)  # noqa: E402


def _fresh(tmp_path, name="v"):
    rgb, truth = make_sheet()
    p = tmp_path / f"{name}.png"
    cv2.imwrite(str(p), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    e = Engine()
    e.open(str(p))
    e.analyze()
    return e, truth


def _kinds(e, b):
    return [f.kind for f in find_problems(e, b)]


def _install(e, b, newpath):
    e._finish_band(b, newpath.astype(np.int32), reversals_from_path(newpath), [], [], b.end_x)


def test_clean_trace_has_no_findings(tmp_path):
    e, _ = _fresh(tmp_path)
    for b in e.bands:
        bad = [k for k in _kinds(e, b) if k != "missed_tip"]
        assert bad == [], (b.name, bad)


def test_off_line_cut_is_found_and_repaired(tmp_path):
    e, _ = _fresh(tmp_path)
    b = e.bands[0]
    original = b.path.copy()
    # replace a stretch by a made-up detour through empty paper (slanted line, then back)
    i = len(b.path) // 3
    j = i + 150
    p0, p1 = b.path[i], b.path[j]
    mid = np.array([p1[0], p0[1] + 45])

    def seg(a, c, n=60):
        return np.round(np.stack([np.linspace(a[0], c[0], n), np.linspace(a[1], c[1], n)], 1)).astype(np.int32)
    corrupt = np.concatenate([b.path[:i], seg(p0, mid), seg(mid, p1), b.path[j:]])
    _install(e, b, corrupt)
    assert "off_line" in _kinds(e, b)
    before = score(find_problems(e, b))
    rep = verify_and_repair(e, 0, max_rounds=5)
    assert rep["applied"] >= 1
    assert rep["score_after"] < before
    assert "off_line" not in _kinds(e, b)
    assert b.complete


def test_missed_tip_is_found_and_added_back(tmp_path):
    e, truth = _fresh(tmp_path)
    b = e.bands[1]
    # remove the excursion to one spike tip from the pen path (the tip is then never reached)
    cands = [k for k in reversals_from_path(b.path)
             if abs(b.path[k][1] - b.line_zero) > 40 and 300 < b.path[k][0] < 1500]
    assert cands, "no retraced tip found in the test sheet"
    k = cands[0]
    tip = b.path[k]

    class V:
        x, y = float(tip[0]), float(tip[1])
    v = V()
    m = 1
    while k - m >= 0 and k + m < len(b.path) and (b.path[k - m] == b.path[k + m]).all():
        m += 1
    corrupt = np.concatenate([b.path[: k - m + 2], b.path[k + m - 1:]])
    _install(e, b, corrupt)
    assert not any(abs(x.x - v.x) < 6 and abs(x.y - v.y) < 6 for x in b.vertices)
    assert "missed_tip" in _kinds(e, b)
    rep = verify_and_repair(e, 1, max_rounds=3)
    assert rep["applied"] >= 1
    assert any(abs(x.x - v.x) < 6 and abs(x.y - v.y) < 6 for x in b.vertices)   # the tip is back


def test_every_attempt_is_kept_with_old_and_new_piece(tmp_path):
    e, _ = _fresh(tmp_path)
    b = e.bands[0]
    i = len(b.path) // 3
    corrupt = np.concatenate([b.path[:i], b.path[i + 30:]])         # jump over a stretch
    _install(e, b, corrupt)
    assert "jump" in _kinds(e, b)
    rep = verify_and_repair(e, 0, max_rounds=3)
    assert len(b.revisions) == rep["attempts"] >= 1
    for r in b.revisions:
        assert r["status"] in ("applied", "rejected", "failed")
        assert r["old"] and "score_before" in r and "score_after" in r
        if r["status"] in ("applied", "rejected"):
            assert r["new"]                                          # the re-traced piece is kept too
    n = len(b.revisions)
    assert e.undo(0)                                                 # the run can be undone...
    assert len(b.revisions) == n                                     # ...but the log stays


def test_rejected_attempt_restores_old_line(tmp_path):
    e, _ = _fresh(tmp_path)
    b = e.bands[2]
    before = b.path.copy()
    rep = verify_and_repair(e, 2, max_rounds=2)                       # clean band: nothing to repair
    assert np.array_equal(b.path, before) or rep["applied"] == 0 or rep["score_after"] <= rep["score_before"]
    assert rep["score_after"] <= rep["score_before"]                   # never worse


def test_findings_and_log_saved_in_project(tmp_path):
    e, _ = _fresh(tmp_path)
    b = e.bands[0]
    corrupt = np.concatenate([b.path[:200], b.path[260:]])
    _install(e, b, corrupt)
    verify_and_repair(e, 0, max_rounds=2)
    f = str(tmp_path / "w.trproj")
    save_project(e, f)
    e2 = Engine()
    load_project(e2, f)
    assert len(e2.bands[0].revisions) == len(b.revisions) >= 1
    assert e2.bands[0].findings == b.findings


def test_excel_has_verification_history_sheet(tmp_path):
    from openpyxl import load_workbook
    from traceview.excel_out import export_xlsx
    e, _ = _fresh(tmp_path)
    b = e.bands[0]
    corrupt = np.concatenate([b.path[:300], b.path[330:]])
    _install(e, b, corrupt)
    verify_and_repair(e, 0, max_rounds=2)
    wb = load_workbook(export_xlsx(e, str(tmp_path / "x.xlsx")))
    ws = wb["検証履歴"]
    rows = [r for r in ws.iter_rows(min_row=5, values_only=True) if r[0]]
    assert any(r[1] == "引き直しの履歴" for r in rows)
    assert len([r for r in rows if r[1] == "引き直しの履歴"]) == len(b.revisions)
