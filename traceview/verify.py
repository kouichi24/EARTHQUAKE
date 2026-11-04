"""Self-check of the traced line and automatic re-tracing of doubtful spots.

verify  : look only at the traced line (and at what the image shows next to it) and list spots that look unnatural
repair  : re-trace one spot (another way through the skeleton, or a missed tip added)
loop    : verify -> repair -> verify again, keep a change only when the check gets better, up to N rounds

Nothing is thrown away: every attempt (kept or rejected) is written to band.revisions with the old and the
new piece of line, so it can be looked at afterwards.
"""
import time
from dataclasses import dataclass

import cv2
import numpy as np
from scipy.spatial import cKDTree

from .model import Issue

KINDS = {
    "off_line": "画像の線から外れた区間",
    "jump": "経路の飛び",
    "repeat": "同じ区間を何度も往復",
    "excursion": "他の段の領域へ入り込む区間",
    "missed_tip": "先端の取りこぼし",
    "untracked_end": "末端の未追跡",
}
REPAIRABLE = ("off_line", "jump", "repeat", "excursion", "missed_tip")


@dataclass
class Finding:
    kind: str
    i0: int            # path index range (inclusive); for missed_tip: index of the nearest path point
    i1: int
    x: float
    y: float
    weight: float
    msg: str

    def to_dict(self):
        return dict(kind=self.kind, i0=self.i0, i1=self.i1, x=self.x, y=self.y, weight=self.weight, msg=self.msg)

    def key(self):
        return (self.kind, int(self.x // 25), int(self.y // 25))


def _runs(mask, min_len):
    """(start, end) inclusive index runs of True values of at least min_len."""
    if not mask.any():
        return []
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1
    return [(int(a), int(b)) for a, b in zip(starts, ends) if b - a + 1 >= min_len]


def reversals_from_path(path):
    """Indices where the pen turns back over the pixels it came by (tips that were retraced)."""
    if len(path) < 3:
        return []
    same = np.all(path[:-2] == path[2:], axis=1)
    return [int(i) + 1 for i in np.flatnonzero(same)]


def find_problems(engine, band):
    """Unnatural spots of the traced line of one band."""
    path = band.path
    out = []
    if len(path) < 3:
        return out
    if getattr(engine, "_ink_dil", None) is None or engine._ink_dil.shape != engine.ink.shape:
        engine._ink_dil = cv2.dilate(engine.ink.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    inferred = np.zeros(len(path), bool)
    for a, c in band.inferred:
        inferred[a:c + 1] = True
    # 1. runs of the line that are not on the ink
    on = engine._ink_dil[path[:, 1], path[:, 0]]
    for a, c in _runs(~on & ~inferred, 8):
        k = (a + c) // 2
        out.append(Finding("off_line", a, c, float(path[k][0]), float(path[k][1]), np.ceil((c - a + 1) / 10.0),
                           f"画像の線がない所を通る区間（長さ約{(c - a + 1) / engine.scale:.0f}px）"))
    # 2. jumps between neighbouring points
    step = np.hypot(*(path[1:] - path[:-1]).T.astype(float))
    jump = (step > 3.5) & ~inferred[:-1] & ~inferred[1:]
    for i in np.flatnonzero(jump):
        out.append(Finding("jump", int(i), int(i) + 1, float(path[i][0]), float(path[i][1]), 2.0,
                           f"隣の点との距離が大きい（約{step[i] / engine.scale:.0f}px）"))
    # 3. stretches walked very many times (the walk itself returns over earlier pixels, so only extreme cases count)
    w = engine.ink.shape[1]
    key = path[:, 1].astype(np.int64) * w + path[:, 0]
    _u, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    for a, c in _runs(cnt[inv] >= 7, 25):
        k = (a + c) // 2
        out.append(Finding("repeat", a, c, float(path[k][0]), float(path[k][1]), np.ceil((c - a + 1) / 25.0),
                           f"同じ区間を何度も往復（長さ約{(c - a + 1) / engine.scale:.0f}px）"))
    # 4. stretches that leave the band's own territory
    origins = [b.origin for b in engine.bands]
    if len(origins) >= 2 and band.index < len(origins):
        corr, _z = engine._zone(band.index, origins)
        for a, c in _runs((path[:, 1] < corr[2]) | (path[:, 1] > corr[3]), 30):
            seg = path[a:c + 1]
            k = int(np.argmax(np.abs(seg[:, 1] - band.origin[1])))
            out.append(Finding("excursion", a, c, float(seg[k][0]), float(seg[k][1]), 3.0,
                               "隣の段の領域へ入り込む区間"))
    # 5. tips of the image's skeleton that the pen path never reached
    if engine.graph is not None:
        out += _missed_tips(engine, band, path)
    # 6. end of the trace
    if not band.complete:
        out.append(Finding("untracked_end", len(path) - 1, len(path) - 1, float(path[-1][0]), float(path[-1][1]), 5.0,
                           "ここから右が未追跡"))
    return out


def _missed_tips(engine, band, path):
    g = engine.graph
    origins = [b.origin for b in engine.bands]
    if len(origins) >= 2 and band.index < len(origins):
        corr, _z = engine._zone(band.index, origins)
        lo, hi = corr[0], corr[1]
    else:
        lo, hi = band.line_zero - 300, band.line_zero + 300
    tree = cKDTree(path.astype(float))
    ref_zone = engine.prepared.ref_zone
    res = []
    for n in g.nodes:
        if n.kind != "end" or not n.edges or not (lo <= n.y <= hi):
            continue
        e = g.edges[n.edges[0]]
        if e.inferred or e.length < 18:
            continue
        nid = g.nodes.index(n) if False else None
        tip = np.array([n.x, n.y])
        d_tip, _ = tree.query(tip)
        if d_tip <= 3.5:
            continue                                   # the pen did go there
        # the base of the spur (other end of its edge) must lie on the pen path: the tip belongs to this line
        base = e.pix[-1] if (e.pix[0] == np.round(tip)).all() else e.pix[0]
        d_base, j = tree.query(base.astype(float))
        if d_base > 3.5:
            continue
        if ref_zone[int(n.y), int(n.x)]:
            continue
        res.append(Finding("missed_tip", int(j), int(j), float(n.x), float(n.y), 1.0,
                           f"画像の先端に経路が届いていない（枝の長さ約{e.length / engine.scale:.0f}px）"))
        if len(res) >= 400:
            break
    return res


def score(findings):
    return float(sum(f.weight for f in findings))


# ------------------------------------------------------------------ repair
def _edges_of(engine, pts):
    g = engine.graph
    pts = np.asarray(pts, np.int64).reshape(-1, 2)
    e = np.unique(g.pix_edge[pts[:, 1], pts[:, 0]])
    return {int(x) for x in e if x >= 0}


def _new_band_state(engine, band, newpath, inferred, end_x):
    """Install a new path into `band` (vertices, issues, lines are recomputed)."""
    engine._finish_band(band, newpath.astype(np.int32), reversals_from_path(newpath), [], inferred, end_x)


def repair(engine, band, f):
    """Re-trace one spot. Returns True when the band was changed (the caller decides to keep it)."""
    path = band.path
    if f.kind == "missed_tip":
        return _repair_missed_tip(engine, band, f)
    # anchor: the last vertex before the spot (the line up to there is kept)
    before = [v.idx for v in band.vertices if v.idx < f.i0 - 2]
    a = before[-1] if before else 0
    a = min(a, f.i0 - 1) if f.i0 > 0 else 0
    prefix = path[: a + 1]
    a_snap, _ = engine._snap(prefix[-1][0], prefix[-1][1])
    if a_snap is None:
        return False
    d = (prefix[-1] - prefix[max(0, len(prefix) - 12)]).astype(float)
    nd = np.hypot(*d)
    d = d / nd if nd else np.array([1.0, 0.0])
    node, edge, partial = engine._continue_from(a_snap, d)
    if node is None:
        return False
    g = engine.graph
    blocked = _edges_of(engine, prefix) | _edges_of(engine, path[f.i0:f.i1 + 1])
    blocked.discard(edge)
    origins = [b.origin for b in engine.bands]
    corr, zone = engine._zone(band.index, origins)
    tail, rev, amb, inf, end_x = engine._trace_from(node, edge, corr, zone, blocked=blocked, oy=band.origin[1])
    if len(tail) == 0:
        return False
    head = np.concatenate([prefix, partial]) if len(partial) else prefix
    newpath = np.concatenate([head, tail]).astype(np.int32)
    shift = len(head)
    # inferred spans of the kept prefix stay, those of the new tail move
    keep_inf = [(x, y) for x, y in band.inferred if y <= a]
    inf2 = keep_inf + [(x + shift, y + shift) for x, y in inf]
    _new_band_state(engine, band, newpath, inf2, end_x)
    return True


def _repair_missed_tip(engine, band, f):
    P, _off = engine._snap(f.x, f.y)
    if P is None:
        return False
    path = band.path
    j = f.i0
    a_snap, _ = engine._snap(path[j][0], path[j][1])
    if a_snap is None:
        return False
    route = engine._route(a_snap, P)
    if route is None or len(route) < 2:
        return False
    out, back = route[1:], route[::-1][1:]
    newpath = np.concatenate([path[: j + 1], out, back, path[j + 1:]]).astype(np.int32)
    shift = len(out) + len(back)
    inf = [(x + (shift if x > j else 0), y + (shift if y > j else 0)) for x, y in band.inferred]
    _new_band_state(engine, band, newpath, inf, band.end_x)
    return True


# -------------------------------------------------------------------- loop
def _segment(path, i0, i1, pad=8):
    a, c = max(0, i0 - pad), min(len(path), i1 + pad + 1)
    seg = path[a:c]
    if len(seg) > 120:
        seg = seg[np.linspace(0, len(seg) - 1, 120).astype(int)]
    return seg.astype(int).tolist()


def _locate(new_path, f):
    """Index range of the new line around the place of the finding (for showing the new piece)."""
    d = np.hypot(new_path[:, 0] - f.x, new_path[:, 1] - f.y)
    k = int(np.argmin(d))
    return max(0, k - 40), min(len(new_path) - 1, k + 40)


def verify_and_repair(engine, bi, max_rounds=5, max_attempts=40, progress=lambda *a: None, repair_on=True):
    """Check one band and (optionally) re-trace unnatural spots until the check no longer improves.

    Returns a report dict. All attempts are appended to band.revisions (never removed).
    """
    engine.ensure_graph(progress)
    b = engine.bands[bi]
    t0 = time.time()
    findings = find_problems(engine, b)
    start_score = score(findings)
    report = dict(band=b.name, rounds=0, attempts=0, applied=0, rejected=0, failed=0,
                  score_before=start_score, score_after=start_score, findings_before=len(findings))
    if not repair_on:
        b.findings = [f.to_dict() for f in findings]
        return report
    b.push_undo(engine.params.undo_limit)          # the whole run can be undone in one step
    run_id = len(b.revisions) and (max(r["run"] for r in b.revisions) + 1) or 1
    tried = set()
    cur = start_score
    for rnd in range(1, max_rounds + 1):
        cand = [f for f in findings if f.kind in REPAIRABLE and f.key() not in tried]
        if not cand:
            break
        cand.sort(key=lambda f: -f.weight)
        improved = False
        report["rounds"] = rnd
        for f in cand:
            if report["attempts"] >= max_attempts:
                break
            progress(f"{b.name}: {rnd}回目の引き直し（{report['attempts'] + 1}件目）", min(0.95, report["attempts"] / max_attempts))
            tried.add(f.key())
            snap = b.snapshot()
            old_seg = _segment(b.path, f.i0, f.i1)
            report["attempts"] += 1
            try:
                ok = repair(engine, b, f)
            except Exception:                         # a failed attempt never damages the band
                ok = False
            rec = dict(run=run_id, round=rnd, kind=f.kind, x=f.x, y=f.y, msg=f.msg, old=old_seg, new=[],
                       status="failed", score_before=cur, score_after=cur)
            if not ok:
                b.restore(snap)
                report["failed"] += 1
                b.revisions.append(rec)
                continue
            new_findings = find_problems(engine, b)
            s_new = score(new_findings)
            a, c = _locate(b.path, f)
            rec["new"] = _segment(b.path, a, c, pad=0)
            rec["score_after"] = s_new
            if s_new < cur - 1e-9:
                rec["status"] = "applied"
                report["applied"] += 1
                cur, findings, improved = s_new, new_findings, True
                b.revisions.append(rec)
                break                                  # indices moved: take a fresh list for the next pass
            rec["status"] = "rejected"                 # not better: put the old line back, keep the record
            report["rejected"] += 1
            b.restore(snap)
            b.revisions.append(rec)
        if not improved:
            break
    findings = find_problems(engine, b)
    b.findings = [f.to_dict() for f in findings]
    report["score_after"] = score(findings)
    report["findings_after"] = len(findings)
    report["seconds"] = time.time() - t0
    for f in findings:
        if f.kind in REPAIRABLE:
            b.issues.append(Issue("verify", f.x, f.y, f"検証: {f.msg}"))
    return report
