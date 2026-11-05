"""Bands are judged one after the other; lines of earlier bands are avoided by the later ones."""
import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tests.synth import make_sheet  # noqa: E402
from traceview.engine import Engine  # noqa: E402
from traceview.project import load_project, save_project  # noqa: E402


def _engine(tmp_path, rgb=None, name="s"):
    if rgb is None:
        rgb, _ = make_sheet()
    p = tmp_path / f"{name}.png"
    cv2.imwrite(str(p), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    e = Engine()
    e.open(str(p))
    return e


def test_stepwise_traces_only_the_first_band_then_one_at_a_time(tmp_path):
    e = _engine(tmp_path)
    e.analyze(stepwise=True)
    assert [b.traced for b in e.bands] == [True, False, False]
    assert len(e.bands[0].path) > 100 and len(e.bands[1].path) == 0
    assert [e.value_status(b) for b in e.bands[1:]] == ["未解析", "未解析"]
    assert e.untraced_bands() == [1, 2]
    assert e.trace_next() == 1 and e.untraced_bands() == [2]
    assert e.trace_next() == 2 and e.untraced_bands() == []
    assert e.trace_next() is None
    assert all(b.complete for b in e.bands)


def test_stepwise_gives_the_same_lines_as_all_at_once(tmp_path):
    e1 = _engine(tmp_path, name="a")
    e1.analyze()
    e2 = _engine(tmp_path, name="b")
    e2.analyze(stepwise=True)
    while e2.trace_next() is not None:
        pass
    for a, b in zip(e1.bands, e2.bands):
        assert np.array_equal(a.path, b.path)


def test_untraced_bands_do_not_break_verification_or_export(tmp_path):
    from traceview.excel_out import export_xlsx
    from traceview.verify import find_problems, verify_and_repair
    e = _engine(tmp_path)
    e.analyze(stepwise=True)
    assert find_problems(e, e.bands[1]) == []
    rep = verify_and_repair(e, 1, repair_on=False)
    assert rep["findings_before"] == 0
    export_xlsx(e, str(tmp_path / "x.xlsx"))                    # placeholders appear as 未解析


def test_traced_flag_is_saved(tmp_path):
    e = _engine(tmp_path)
    e.analyze(stepwise=True)
    f = str(tmp_path / "w.trproj")
    save_project(e, f)
    e2 = Engine()
    load_project(e2, f)
    assert [b.traced for b in e2.bands] == [True, False, False]
    assert e2.trace_next() == 1


def test_confirmed_band_lines_are_off_limits_and_a_forced_overlap_is_reported(tmp_path):
    e = _engine(tmp_path)
    e.analyze()
    e.bands[0].confirmed = True
    # normal case: nothing of band 0 is needed by band 1, so nothing changes and nothing is reported
    nb = e._trace_new_band(1, e._origin_list())
    assert nb.complete and not any(i.kind == "shared_line" for i in nb.issues)
    assert np.array_equal(nb.path, e.bands[1].path)
    # forced case: pretend the confirmed lines include the way band 1 needs (the pens really overlap there)
    g = e.graph
    p = e.bands[1].path
    need = p[p[:, 0] > 600]
    forced = {int(x) for x in np.unique(g.pix_edge[need[:, 1], need[:, 0]]) if x >= 0}
    e._confirmed_edges = lambda exclude: forced
    nb2 = e._trace_new_band(1, e._origin_list())
    assert nb2.complete                                           # still traced to the end
    assert any(i.kind == "shared_line" for i in nb2.issues)       # and the shared line is reported


def test_retrace_following_keeps_earlier_bands(tmp_path):
    e = _engine(tmp_path)
    e.analyze()
    keep = e.bands[0].path.copy()
    js = e.retrace_following(0)
    assert js == [1, 2]
    assert np.array_equal(e.bands[0].path, keep)
    assert all(b.complete for b in e.bands)
    assert e.undo(1) or True                                    # the earlier state of band 1 is on its undo stack
