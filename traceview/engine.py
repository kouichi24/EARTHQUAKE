"""Analysis engine: open -> detect bands -> trace -> vertices -> edit/retrace (IN, TR, ED)."""
import os

import cv2
import numpy as np
from scipy.signal import find_peaks
from skimage.graph import route_through_array

from . import __version__
from .config import Params
from .graph import add_virtual_edges, bridge_gaps, build_graph, skeleton_of
from .imageio import file_digest, load_image, make_work_image, page_count
from .measure import Calibration
from .model import BAND_COLORS, BAND_NAMES, Band, Issue
from .preprocess import prepare
from .tracer import Walker
from .vertices import Vertex, extract_vertices


class EngineError(Exception):
    pass


def _noop(*_a, **_k):
    pass


class Engine:
    def __init__(self, params=None):
        self.params = params or Params()
        self.src_path = None
        self.page = 0
        self.digest = ""
        self.rgb_orig = None
        self.work = None
        self.scale = 1.0
        self.prepared = None
        self.graph = None
        self.bands = []
        self.calibration = Calibration()
        self.orig_size = (0, 0)

    # ------------------------------------------------------------------ open
    def open(self, path, page=0, progress=_noop):
        if not os.path.exists(path):
            raise EngineError(f"ファイルが見つかりません: {path}")
        progress("読込中", 0.05)
        self.src_path = os.path.abspath(path)
        self.page = page
        self.digest = file_digest(path)
        self.rgb_orig = load_image(path, page, self.params.work_max_side)
        h, w = self.rgb_orig.shape[:2]
        self.orig_size = (w, h)
        self.work, self.scale = make_work_image(self.rgb_orig, self.params.work_max_side)
        self.prepared = None
        self.graph = None
        self.bands = []
        progress("読込完了", 1.0)

    @property
    def pages(self):
        return page_count(self.src_path) if self.src_path else 0

    # --------------------------------------------------------------- analyze
    def ensure_graph(self, progress=_noop):
        if self.graph is not None:
            return
        if self.work is None:
            raise EngineError("画像が読み込まれていません")
        progress("画像を前処理中", 0.1)
        self.prepared = prepare(self.work, self.params)
        progress("線を細線化中", 0.35)
        sk = skeleton_of(self.prepared.ink)
        g = build_graph(sk, self.params.spur_min_len)
        progress("途切れを補修中", 0.6)
        ink2, _ = bridge_gaps(self.prepared.ink, g, max_gap=20, cos_min=0.6, max_vy=1.1)
        g = build_graph(skeleton_of(ink2), self.params.spur_min_len)
        add_virtual_edges(g)
        self.graph = g
        self.ink = ink2

    def detect_origins(self):
        """Origin (start of trace at the left) of each band, found from the quiet pen line."""
        P = self.prepared
        x0, y0, x1, y1 = P.paper
        gap = max(10, int(self.params.origin_min_gap_frac * (y1 - y0)))
        found = []
        for frac in (0.03, 0.05):     # widen the strip when a band starts late
            strip_w = max(40, int(frac * (x1 - x0)))
            strip = P.ink[:, x0:x0 + strip_w]
            prof = cv2.GaussianBlur(strip.sum(1).astype(np.float32).reshape(-1, 1), (1, 0), 5).ravel() / strip.shape[1]
            pk, _ = find_peaks(prof, height=0.08, distance=gap)
            for p in pk:
                if all(abs(int(p) - q) > gap for q, _h in found):
                    found.append((int(p), float(prof[p])))
        found.sort(key=lambda t: -t[1])
        pk = sorted(p for p, _h in found[:5])
        strip_w = max(40, int(0.05 * (x1 - x0)))
        origins = []
        for p in pk:
            rows = P.ink[max(0, p - 8):p + 9, x0:x0 + 4 * strip_w]
            ys, xs = np.nonzero(rows)
            if len(xs) == 0:
                continue
            xmin = xs.min() + x0
            sel = xs + x0 < xmin + 60
            origins.append((int(xmin), int(np.median(ys[sel]) + max(0, p - 8))))
        return origins

    def analyze(self, origins=None, progress=_noop):
        self.ensure_graph(progress)
        progress("三段の起点を検出中", 0.7)
        if origins is None:
            origins = self.detect_origins()
        self.bands = []
        for i, (ox, oy) in enumerate(origins):
            progress(f"{BAND_NAMES[i] if i < 3 else i + 1}を追跡中", 0.72 + 0.25 * i / max(1, len(origins)))
            self.bands.append(self._trace_new_band(i, origins))
        progress("完了", 1.0)
        return self.bands

    # ----------------------------------------------------------------- helpers
    def _zone(self, i, origins):
        oy = origins[i][1]
        H = self.work.shape[0]
        up = oy - origins[i - 1][1] if i > 0 else (origins[i + 1][1] - oy if i + 1 < len(origins) else 0.2 * H)
        dn = origins[i + 1][1] - oy if i + 1 < len(origins) else up
        corr = (oy - 0.8 * up, oy + 0.8 * dn, oy - 1.0 * up, oy + 1.0 * dn)
        zone = (oy - 0.35 * up, oy + 0.35 * dn)
        return corr, zone

    def _start_node(self, ox, oy):
        g = self.graph
        best, bx = None, 1e18
        for j, n in enumerate(g.nodes):
            if n.kind == "end" and n.edges and abs(n.y - oy) < 25 and abs(n.x - ox) < 80 and n.x < bx:
                best, bx = j, n.x
        if best is not None:
            return best
        best, bd = None, 60
        for j, n in enumerate(g.nodes):
            if n.edges:
                d = np.hypot(n.x - ox, n.y - oy)
                if d < bd:
                    best, bd = j, d
        return best

    def _trace_from(self, node, edge, corr, zone, blocked=None, oy=0, prefix=None):
        """Walk, cut at the right-most point of the band zone, and bridge to further fragments."""
        g = self.graph
        W = self.work.shape[1]
        x_right = self.prepared.paper[2]
        walker = Walker(g, self.params, corr, ref_zone=self.prepared.ref_zone)
        pieces, rev, amb, inf = [], [], [], []
        used_nodes = set()
        blocked = set(blocked or ())
        offset = 0
        cur_node, cur_edge = node, edge
        end_x = 0.0
        for _ in range(60):
            res = walker.walk(cur_node, cur_edge, blocked=blocked)
            res = walker.truncate_at_best_end(res, zone[0], zone[1])
            if len(res.path) == 0:
                break
            pieces.append(res.path)
            rev += [offset + r for r in res.reversals]
            amb += [(offset + a[0], a[1], a[2]) for a in res.ambiguous]
            inf += [(offset + a, offset + b) for a, b in res.inferred]
            offset += len(res.path)
            end_x = res.end_x
            # edges used by what we kept must not be walked again
            pix = res.path
            eids = np.unique(g.pix_edge[pix[:, 1], pix[:, 0]])
            blocked |= {int(e) for e in eids if e >= 0}
            if end_x >= x_right - 0.03 * W:
                break
            end_pt = pix[-1]
            best, bd = None, 700
            for j, n in enumerate(g.nodes):
                if n.kind != "end" or not n.edges or j in used_nodes:
                    continue
                if n.x <= end_x + 5 or not (zone[0] <= n.y <= zone[1]):
                    continue
                if n.edges[0] in blocked:
                    continue
                d = np.hypot(n.x - end_pt[0], 3.0 * (n.y - end_pt[1]))
                if d < bd:
                    best, bd = j, d
            if best is None:
                break
            n = g.nodes[best]
            used_nodes.add(best)
            k = int(max(2, np.hypot(n.x - end_pt[0], n.y - end_pt[1])))
            xs = np.round(np.linspace(end_pt[0], n.x, k)).astype(np.int32)
            ys = np.round(np.linspace(end_pt[1], n.y, k)).astype(np.int32)
            bridge = np.stack([xs, ys], 1)[1:]
            pieces.append(bridge)
            inf.append((offset, offset + len(bridge) - 1))
            offset += len(bridge)
            cur_node, cur_edge = best, n.edges[0]
        path = np.concatenate(pieces).astype(np.int32) if pieces else np.zeros((0, 2), np.int32)
        return path, rev, amb, inf, end_x

    def _finish_band(self, b, path, rev, amb, inf, end_x, keep_manual=True):
        b.path, b.inferred, b.end_x = path, inf, end_x
        b.vertices = extract_vertices(path, rev, self.params.tip_min_prominence, self.ink)
        self._mark_manual(b)
        self._mark_suspect(b)
        W = self.work.shape[1]
        b.complete = end_x >= self.prepared.paper[2] - 0.03 * W
        b.issues = []
        for idx, x, y in amb[:200]:
            b.issues.append(Issue("ambiguous", x, y, "交差・接触で所属を判断しにくい箇所"))
        for a, c in inf:
            b.issues.append(Issue("inferred", float(path[a][0]), float(path[a][1]),
                                  "画像の途切れを推定でつないだ区間"))
        # Stretches that leave the band's own territory (arcs, dirt, other bands) need a human look.
        origins = [bd.origin for bd in self.bands] if self.bands else [b.origin]
        if b.index < len(origins) and len(origins) >= 2 and len(path):
            _corr, _zone = self._zone(b.index, origins)
            out = (path[:, 1] < _corr[2]) | (path[:, 1] > _corr[3])
            runs = np.flatnonzero(np.diff(np.concatenate([[0], out.astype(np.int8), [0]])))
            for a_, c_ in zip(runs[::2], runs[1::2]):
                if c_ - a_ >= 30:
                    seg = path[a_:c_]
                    k = int(np.argmax(np.abs(seg[:, 1] - b.origin[1])))
                    b.issues.append(Issue("excursion", float(seg[k][0]), float(seg[k][1]),
                                          "隣の段の領域まで伸びる経路（所属要確認）"))
        if not b.complete and len(path):
            b.issues.append(Issue("untracked_end", float(path[-1][0]), float(path[-1][1]),
                                  "ここから右が未追跡（末端まで到達していません）"))
        if len(path):
            first = path[: min(len(path), 80)]
            b.y0 = float(np.median(first[:, 1]))
        if not b.line_top_manual or not b.line_bottom_manual or True:
            self.reset_lines(b, only_auto=True)

    def _mark_suspect(self, b):
        """Vertices on stretches that leave the band's territory are candidates only (MS-04)."""
        origins = [bd.origin for bd in self.bands] if self.bands else [b.origin]
        if b.index >= len(origins) or len(origins) < 2 or not b.vertices:
            return
        corr, _zone = self._zone(b.index, origins)
        for v in b.vertices:
            if v.y < corr[2] or v.y > corr[3]:
                v.suspect = True
                v.note = "他の段の領域へ伸びる区間上（所属要確認）"

    def _mark_manual(self, b, radius=8):
        for mp in b.manual_points:
            best, bd = None, radius
            for v in b.vertices:
                d = np.hypot(v.x - mp["x"], v.y - mp["y"])
                if d <= bd:
                    best, bd = v, d
            if best is not None:
                best.source = "manual" if mp["kind"] in ("moved", "added") else "retrace"

    def reset_lines(self, b, only_auto=False):
        vs = b.vertices
        if not (only_auto and b.line_zero_manual):
            b.line_zero = b.y0
            b.line_zero_manual = False
        if not (only_auto and b.line_top_manual):
            b.line_top = min((v.y for v in vs), default=b.y0)
            b.line_top_manual = False
        if not (only_auto and b.line_bottom_manual):
            b.line_bottom = max((v.y for v in vs), default=b.y0)
            b.line_bottom_manual = False

    def _trace_new_band(self, i, origins):
        ox, oy = origins[i]
        b = Band(i, BAND_NAMES[i] if i < 3 else f"波形{i + 1}", BAND_COLORS[i % len(BAND_COLORS)], (ox, oy), float(oy))
        corr, zone = self._zone(i, origins)
        node = self._start_node(ox, oy)
        if node is None:
            b.issues.append(Issue("no_origin", ox, oy, "起点付近に線が見つかりません。起点を指定してください"))
            b.line_zero = b.line_top = b.line_bottom = float(oy)
            b.initial = b.snapshot()
            return b
        path, rev, amb, inf, end_x = self._trace_from(node, self.graph.nodes[node].edges[0], corr, zone, oy=oy)
        self._finish_band(b, path, rev, amb, inf, end_x)
        b.initial = b.snapshot()
        return b

    def retrace_band_from_origin(self, i, ox, oy):
        """IN-05: manual origin for a band whose automatic start failed."""
        origins = [bd.origin for bd in self.bands]
        origins[i] = (ox, oy)
        nb = self._trace_new_band(i, origins)
        old = self.bands[i]
        nb.undo, nb.redo = old.undo, old.redo
        nb.push_undo(self.params.undo_limit)
        self.bands[i] = nb
        return nb

    # ---------------------------------------------------------------- routing
    def _route(self, a, b, pad=150):
        """Shortest way along the skeleton from pixel a to pixel b (x, y). None if impossible."""
        g = self.graph
        xa, ya = int(a[0]), int(a[1])
        xb, yb = int(b[0]), int(b[1])
        h, w = g.skel.shape
        x0, x1 = max(0, min(xa, xb) - pad), min(w, max(xa, xb) + pad + 1)
        y0, y1 = max(0, min(ya, yb) - pad), min(h, max(ya, yb) + pad + 1)
        sub = g.skel[y0:y1, x0:x1]
        cost = np.where(sub, 1.0, 1e4)
        try:
            idx, c = route_through_array(cost, (ya - y0, xa - x0), (yb - y0, xb - x0),
                                         fully_connected=True, geometric=True)
        except Exception:
            return None
        if c >= 1e4:
            return None
        r = np.array([(x + x0, y + y0) for y, x in idx], np.int32)
        return r

    def _snap(self, x, y):
        """Nearest skeleton pixel (ED-04). Returns ((x, y), distance) or (None, None)."""
        g = self.graph
        r = self.params.snap_radius
        h, w = g.skel.shape
        xa, xb = max(0, int(x) - r), min(w, int(x) + r + 1)
        ya, yb = max(0, int(y) - r), min(h, int(y) + r + 1)
        ys, xs = np.nonzero(g.skel[ya:yb, xa:xb])
        if len(xs) == 0:
            return None, None
        d = np.hypot(xs + xa - x, ys + ya - y)
        j = int(np.argmin(d))
        if d[j] > r:
            return None, None
        return (int(xs[j] + xa), int(ys[j] + ya)), float(d[j])

    # ------------------------------------------------------------ edit (ED-*)
    def set_pending_move(self, bi, vi, x, y):
        b = self.bands[bi]
        v = b.vertices[vi]
        b.pending = [p for p in b.pending if not (p["type"] == "move" and p["vi"] == vi)]
        b.pending.append(dict(type="move", vi=vi, ox=v.x, oy=v.y, kind=v.kind, x=float(x), y=float(y)))

    def set_pending_add(self, bi, x, y):
        self.bands[bi].pending.append(dict(type="add", x=float(x), y=float(y)))

    def cancel_pending(self, bi):
        self.bands[bi].pending = []

    def has_pending(self):
        return any(b.pending for b in self.bands)

    def apply_pending(self, bi, progress=_noop):
        """Apply all pending edits of one band. Returns (ok, message). On failure nothing changes."""
        b = self.bands[bi]
        if not b.pending:
            return True, "適用する修正はありません"
        self.ensure_graph(progress)
        snap_before = b.snapshot()
        msgs = []
        try:
            moves = sorted([p for p in b.pending if p["type"] == "move"], key=lambda p: p["vi"])
            adds = [p for p in b.pending if p["type"] == "add"]
            work_b = Band(b.index, b.name, b.color)
            work_b.restore(snap_before)
            work_b.pending = []
            for p in moves:
                ok, msg = self._apply_move(work_b, p)
                if not ok:
                    raise EngineError(msg)
                msgs.append(msg)
            for p in adds:
                ok, msg = self._apply_add(work_b, p)
                if not ok:
                    raise EngineError(msg)
                msgs.append(msg)
        except EngineError as e:
            b.restore(snap_before)
            b.issues.append(Issue("retrace_failed", b.pending[0]["x"], b.pending[0]["y"], str(e)))
            return False, str(e)
        b.push_undo(self.params.undo_limit)
        keep = (b.undo, b.redo)
        b.restore(work_b.snapshot())
        b.pending = []
        b.confirmed = False
        return True, " / ".join(msgs)

    def _find_vertex(self, b, ox, oy, kind):
        best, bd = None, 12.0
        for i, v in enumerate(b.vertices):
            d = np.hypot(v.x - ox, v.y - oy)
            if d < bd:
                best, bd = i, d
        return best

    def _apply_move(self, b, p):
        k = self._find_vertex(b, p["ox"], p["oy"], p["kind"])
        if k is None:
            return False, "移動対象の頂点が見つかりません"
        P, off = self._snap(p["x"], p["y"])
        if P is None:
            return False, f"指定位置の近く({self.params.snap_radius}px以内)に画像の線がありません。修正前の結果を保持します"
        vs = b.vertices
        start_idx = vs[k - 1].idx if k > 0 else 0
        prefix = b.path[: start_idx + 1]
        a = prefix[-1]
        a_snap, _ = self._snap(a[0], a[1])
        if a_snap is None:
            return False, "再計算の開始位置が画像の線上にありません"
        route = self._route(a_snap, P)
        if route is None:
            return False, "指定位置へ至る線が見つかりません（途切れている可能性があります）。修正前の結果を保持します"
        g = self.graph
        # where does the pen go after the tip?
        d = (route[-1] - route[max(0, len(route) - 12)]).astype(float)
        nd = np.hypot(*d)
        d = d / nd if nd else np.array([1.0, 0.0])
        node, edge, partial = self._continue_from(P, d)
        if node is None:
            return False, "指定位置から先の線を特定できません"
        oy = b.origin[1]
        origins = [bd.origin for bd in self.bands]
        corr, zone = self._zone(b.index, origins)
        blocked = set()
        for part in (prefix, route):
            e_ids = np.unique(g.pix_edge[part[:, 1], part[:, 0]])
            blocked |= {int(e) for e in e_ids if e >= 0}
        if edge in blocked:
            blocked.discard(edge)
        tail, rev, amb, inf, end_x = self._trace_from(node, edge, corr, zone, blocked=blocked, oy=oy)
        head = np.concatenate([prefix, route[1:], partial]) if len(partial) else np.concatenate([prefix, route[1:]])
        newpath = np.concatenate([head, tail]).astype(np.int32) if len(tail) else head
        shift = len(head)
        inf2 = [(x + shift, y + shift) for x, y in inf]
        prev_manual = [mp for mp in b.manual_points if np.hypot(mp["x"] - p["ox"], mp["y"] - p["oy"]) > 1]
        prev_manual.append(dict(x=float(P[0]), y=float(P[1]), kind="moved"))
        b.manual_points = prev_manual
        rev2 = [r + shift for r in rev]
        self._finish_band(b, newpath, rev2, [(a_ + shift, x_, y_) for a_, x_, y_ in amb], inf2, end_x)
        offtxt = f"（画像の線へ{off:.1f}px補正）" if off and off > 0.5 else ""
        return True, f"頂点を移動し、そこから先を再追跡しました{offtxt}"

    def _continue_from(self, P, d):
        """Graph node / edge / pixels to continue with after arriving at skeleton pixel P heading d."""
        g = self.graph
        x, y = P
        nid = int(g.pix_node[y, x])
        if nid >= 0:
            node = g.nodes[nid]
            best, bs = None, -9
            for e in node.edges:
                ed = g.edges[e]
                pix = ed.pix if ed.a == nid else ed.pix[::-1]
                v = (pix[min(12, len(pix) - 1)] - pix[0]).astype(float)
                nv = np.hypot(*v)
                s = float(v @ d) / nv if nv else -9
                if s > bs:
                    best, bs = e, s
            return nid, best, np.zeros((0, 2), np.int32)
        eid = int(g.pix_edge[y, x])
        if eid < 0:
            return None, None, None
        e = g.edges[eid]
        match = np.where((e.pix[:, 0] == x) & (e.pix[:, 1] == y))[0]
        if len(match) == 0:
            return None, None, None
        i = int(match[0])
        fwd_v = (e.pix[min(len(e.pix) - 1, i + 12)] - e.pix[max(0, i - 0)]).astype(float)
        bwd_v = (e.pix[max(0, i - 12)] - e.pix[i]).astype(float)
        if float(fwd_v @ d) >= float(bwd_v @ d):
            far, partial = e.b, e.pix[i + 1:]
        else:
            far, partial = e.a, e.pix[:i][::-1]
        return far, eid, np.asarray(partial, np.int32).reshape(-1, 2)

    def _apply_add(self, b, p):
        P, off = self._snap(p["x"], p["y"])
        if P is None:
            return False, f"指定位置の近く({self.params.snap_radius}px以内)に画像の線がありません。追加できません"
        path = b.path
        if len(path) == 0:
            return False, "追跡結果がないため頂点を追加できません"
        d = np.hypot(path[:, 0] - P[0], path[:, 1] - P[1])
        j = int(np.argmin(d))
        if d[j] > 200:
            return False, "追加位置が波形の経路から離れすぎています"
        a_snap, _ = self._snap(path[j][0], path[j][1])
        if a_snap is None:
            return False, "追加位置の近くで経路が画像の線上にありません"
        route = self._route(a_snap, P)
        if route is None:
            return False, "追加位置へ至る線が見つかりません"
        out = route[1:]
        back = route[::-1][1:]
        newpath = np.concatenate([path[: j + 1], out, back, path[j + 1:]]).astype(np.int32)
        shift = 2 * len(out) - 1 + 1
        inf = []
        for a_, c_ in b.inferred:
            inf.append((a_ + (shift if a_ > j else 0), c_ + (shift if c_ > j else 0)))
        # reversals: the new tip is a dead-end turn
        rev = [j + len(out)]
        b.manual_points.append(dict(x=float(P[0]), y=float(P[1]), kind="added"))
        old_v = b.vertices
        self._finish_band(b, newpath, rev, [], inf, b.end_x)
        offtxt = f"（画像の線へ{off:.1f}px補正）" if off and off > 0.5 else ""
        return True, f"頂点を追加し、先端まで到達して折り返す経路を再計算しました{offtxt}"

    # ------------------------------------------------------------ undo / redo
    def undo(self, bi):
        return self.bands[bi].do_undo()

    def redo(self, bi):
        return self.bands[bi].do_redo()

    def reset_band(self, bi):
        return self.bands[bi].reset_to_initial()

    # --------------------------------------------------------------- measure
    def measures(self, b):
        from .measure import compute_measure
        return compute_measure(b.vertices, b.line_zero, b.line_top, b.line_bottom, self.scale, self.calibration)

    def is_determinate(self, b):
        """MS-04: values of a band count as determined only when traced to the end, no open issue
        and confirmed by the user."""
        blocking = [i for i in b.issues if i.kind in ("untracked_end", "no_origin", "retrace_failed")]
        return b.complete and not blocking and not b.pending and b.confirmed

    def value_status(self, b):
        if b.pending:
            return "未適用の修正あり"
        if self.is_determinate(b):
            return "確認済み"
        if not b.complete:
            return "候補（末端まで未追跡）"
        return "候補（未確認）"
