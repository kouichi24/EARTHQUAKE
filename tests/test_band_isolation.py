"""While one band (e.g. the red top band) is edited, the other bands' lines must stay exactly as they are."""
import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tests.synth import BASELINES, make_sheet  # noqa: E402
from traceview.engine import Engine  # noqa: E402
from traceview.verify import verify_and_repair  # noqa: E402


def _engine(tmp_path, rgb=None, name="iso"):
    if rgb is None:
        rgb, _ = make_sheet()
    p = tmp_path / f"{name}.png"
    cv2.imwrite(str(p), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    e = Engine()
    e.open(str(p))
    e.analyze()
    return e


def _state(e, bands):
    """What identifies a band's line: path, vertices, estimated links, erased parts."""
    out = []
    for i in bands:
        b = e.bands[i]
        out.append((b.path.tobytes(), [(round(v.x, 2), round(v.y, 2), v.kind) for v in b.vertices], list(b.inferred),
                    list(b.cuts), len(b.erased), len(b.deleted_points), b.confirmed))
    return out


@pytest.fixture()
def eng(tmp_path):
    return _engine(tmp_path)


def test_move_local_and_tail_leave_other_bands_alone(eng):
    for mode in ("local", "tail"):
        before = _state(eng, (1, 2))
        k = len(eng.bands[0].vertices) // 2
        v = eng.bands[0].vertices[k]
        eng.set_pending_move(0, k, v.x + 2, v.y)
        ok, msg = eng.apply_pending(0, mode=mode)
        assert ok, msg
        assert _state(eng, (1, 2)) == before, mode


def test_add_extend_delete_erase_verify_range_leave_other_bands_alone(eng):
    before = _state(eng, (1, 2))
    b = eng.bands[0]
    # delete several vertices
    eng.delete_vertices(0, [3, 6, 9])
    # erase a stretch of the line
    eng.erase_rect(0, 900, 0, 1000, 400)
    # check and re-trace inside a range
    eng.retrace_in_range(0, BASELINES[0] - 80, BASELINES[0] + 80)
    # the self-check and its repairs
    verify_and_repair(eng, 0, max_rounds=2)
    assert _state(eng, (1, 2)) == before


def test_exclusion_for_one_band_does_not_re_trace_the_others(eng):
    # a rectangle that holds a piece of the red line AND a piece of the green line
    rect = (700, BASELINES[0] - 20, 760, BASELINES[1] + 20)
    assert set(eng.exclusion_conflicts(rect, 0)) >= {1}                # the other band is in the way
    before = _state(eng, (1, 2))
    affected = eng.add_exclusion(*rect, only_band=0)
    assert affected == [0]
    assert _state(eng, (1, 2)) == before                               # their lines are untouched
    assert 1 in eng.untouched_by_exclusion
    assert any(i.kind == "exclusion_pending" for i in eng.bands[1].issues)   # but they are told, so nothing is silent


def test_exclusion_without_a_band_limit_still_re_traces_every_touched_band(eng):
    rect = (700, BASELINES[0] - 20, 760, BASELINES[1] + 20)
    affected = eng.add_exclusion(*rect)
    assert set(affected) >= {0, 1}


def _two_close_lines(tmp_path):
    """Band 1 runs 12 px under band 0's baseline for a stretch, so the two lines are neighbours."""
    rgb, _ = make_sheet()
    rgb = rgb.copy()
    wave = [[x, 262 + (4 if (x // 20) % 2 else -4)] for x in range(900, 1301, 20)]      # a slightly wavy neighbour line
    pts = np.array([[900, 600]] + wave + [[1300, 600]], np.int32)
    cv2.polylines(rgb, [pts.reshape(-1, 1, 2)], False, (50, 60, 55), 3, cv2.LINE_AA)
    e = _engine(tmp_path, rgb, "close")
    # let band 1's recorded line really run along that neighbouring stretch (as it does when the pen overlaps)
    mask = np.zeros(e.graph.skel.shape, np.uint8)
    wave_pts = np.array(wave, np.int32).reshape(-1, 1, 2)
    cv2.polylines(mask, [wave_pts], False, 1, 1)                       # exactly the neighbour line, 1 px thick
    ys, xs = np.nonzero(mask)
    near_skel = cv2.dilate(e.graph.skel.astype(np.uint8), np.ones((3, 3), np.uint8))[ys, xs] > 0
    assert near_skel.mean() > 0.9                                      # it exists in the image (not removed as a reference line)
    line = np.stack([xs[near_skel], ys[near_skel]], 1).astype(np.int32)
    e.bands[1].path = np.concatenate([e.bands[1].path, line])
    return e


def test_a_point_does_not_snap_onto_a_neighbouring_bands_line(tmp_path):
    e = _two_close_lines(tmp_path)
    mask = e._other_band_mask(0)
    ys0, xs0 = np.nonzero(mask[255:272, 905:1295])
    assert mask is not None and len(ys0) > 600                          # band 1's line is marked as "not mine" for band 0
    e._tracing_band = 0
    P, _off = e._snap(1100, 258)                                        # nearer to band 1's wavy line than to band 0's baseline
    assert P is not None and not mask[P[1], P[0]]                       # it does NOT land on band 1's line


def test_a_dragged_point_cannot_reach_a_neighbouring_bands_line(tmp_path):
    e = _two_close_lines(tmp_path)
    b0 = e.bands[0]
    v = min((v for v in b0.vertices if 950 < v.x < 1250), key=lambda v: abs(v.y - 250), default=None)
    assert v is not None
    lc = e.line_constraint(v.x, v.y, band=0)
    assert lc is not None
    mask = e._other_band_mask(0)
    ys, xs = lc.pts[:, 1].astype(int), lc.pts[:, 0].astype(int)
    assert not mask[ys, xs].any()                                       # nothing reachable lies on band 1's line
    sx, sy = lc.snap(v.x, 262.0)                                        # the mouse is on band 1's line
    assert not mask[int(sy), int(sx)]


def test_manual_routes_avoid_the_other_bands_line(tmp_path):
    e = _two_close_lines(tmp_path)
    e._tracing_band = 0
    a, _ = e._snap(850, 250)
    b, _ = e._snap(1350, 250)
    route = e._route(a, b)
    assert route is not None
    mask = e._other_band_mask(0)
    on_other = int(mask[route[:, 1], route[:, 0]].sum())
    assert on_other <= 0.10 * len(route)                                # the route keeps to band 0's own line (crossings aside)
