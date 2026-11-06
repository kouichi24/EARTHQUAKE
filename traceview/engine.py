"""Analysis engine: open -> detect bands -> trace -> vertices -> edit/retrace (IN, TR, ED)."""
import os

import cv2
import numpy as np
from scipy.signal import find_peaks
from scipy.spatial import cKDTree
from skimage.graph import route_through_array

from . import __version__
from .config import Params
from .graph import add_virtual_edges, bridge_gaps, build_graph, skeleton_of
from .imageio import file_digest, load_image, make_work_image, page_count
from .measure import Calibration, SectionSettings, arc_active, corrected_x_work, section_stats
from .model import BAND_COLORS, BAND_NAMES, Band, Issue
from .preprocess import prepare
from .tracer import Walker
from .vertices import Vertex, extract_vertices


class LineConstraint:
    """Keeps a dragged point ON the image's line.

    The point can sit on skeleton pixels reachable from the start pixel along the line within `limit` px, so it never
    jumps to a neighbouring line that is not connected. In addition it may jump ACROSS A GAP: where the line ends and
    another piece of line lies on its extension (in the direction the line was heading, within `jump` px), the point
    can be taken over to that piece. snap() then sets `last_bridge` = (end of the line, landing point)."""

    def __init__(self, skel, start_xy, limit=150, jump=160, cone_cos=0.7, y_range=None):
        from skimage.graph import MCP_Geometric
        import cv2 as _cv2
        h, w = skel.shape
        x, y = int(round(start_xy[0])), int(round(start_xy[1]))
        x0, x1 = max(0, x - limit), min(w, x + limit + 1)
        y0, y1 = max(0, y - limit), min(h, y + limit + 1)
        sub = skel[y0:y1, x0:x1]
        ys, xs = np.nonzero(sub)
        self.ok = False
        self.last_bridge = None
        self.jump_tree = None
        if len(xs) == 0:
            return
        d = np.hypot(xs + x0 - x, ys + y0 - y)
        j = int(np.argmin(d))
        if d[j] > 8:
            return
        costs = np.where(sub, 1.0, np.inf)
        cum, _tb = MCP_Geometric(costs, fully_connected=True).find_costs([(int(ys[j]), int(xs[j]))])
        reach = np.isfinite(cum) & (cum <= limit)
        ry, rx = np.nonzero(reach)
        self.pts = np.stack([rx + x0, ry + y0], 1).astype(float)
        self.tree = cKDTree(self.pts)
        self.start = (int(xs[j] + x0), int(ys[j] + y0))
        self.ok = len(self.pts) > 0
        if not self.ok or jump <= 0:
            return
        # ---- ends of the reachable piece and what lies on their extension
        # ends of the LINE (one neighbour in the whole skeleton, so the edge of the window is not an end),
        # and not the points where the reach simply stops (the line goes on there)
        pad = skel[max(0, y0 - 1):y1 + 1, max(0, x0 - 1):x1 + 1].astype(np.uint8)
        nbp = _cv2.filter2D(pad, -1, np.ones((3, 3), np.float32), borderType=_cv2.BORDER_CONSTANT) - pad
        oy_, ox_ = y0 - max(0, y0 - 1), x0 - max(0, x0 - 1)
        nb = nbp[oy_:oy_ + sub.shape[0], ox_:ox_ + sub.shape[1]]
        end_y, end_x = np.nonzero(reach & sub & (nb == 1) & (cum <= limit - 4))
        if len(end_x) == 0:
            return
        R = max(jump, limit)                           # (the window must contain the reach window)
        X0, X1 = max(0, x - R), min(w, x + R + 1)
        Y0, Y1 = max(0, y - R), min(h, y + R + 1)
        sub2 = skel[Y0:Y1, X0:X1]
        reach2 = np.zeros(sub2.shape, np.uint8)
        reach2[y0 - Y0:y1 - Y0, x0 - X0:x1 - X0] = reach.astype(np.uint8)
        far = _cv2.distanceTransform((reach2 == 0).astype(np.uint8), _cv2.DIST_L2, 3) > 6      # not the same piece
        cy, cx = np.nonzero(sub2 & far)
        if len(cx) == 0:
            return
        cpts = np.stack([cx + X0, cy + Y0], 1).astype(float)
        if y_range is not None:                       # only lines inside this band's own territory
            cpts = cpts[(cpts[:, 1] >= y_range[0]) & (cpts[:, 1] <= y_range[1])]
            if len(cpts) == 0:
                return
        keep_pts, keep_end = [], []
        for ex, ey in zip(end_x + x0, end_y + y0):
            near = self.pts[np.hypot(self.pts[:, 0] - ex, self.pts[:, 1] - ey) <= 16]
            hd = np.array([ex, ey], float) - near.mean(axis=0)
            nh = np.hypot(*hd)
            if nh < 1e-6:
                continue
            hd /= nh
            v = cpts - np.array([ex, ey], float)
            dist = np.hypot(v[:, 0], v[:, 1])
            ok = (dist >= 4) & (dist <= jump) & ((v @ hd) / np.maximum(dist, 1e-6) >= cone_cos)
            if ok.any():
                keep_pts.append(cpts[ok])
                keep_end.append(np.tile([ex, ey], (int(ok.sum()), 1)))
        if keep_pts:
            self.jump_pts = np.concatenate(keep_pts)
            self.jump_end = np.concatenate(keep_end)
            self.jump_tree = cKDTree(self.jump_pts)

    def snap(self, x, y):
        """Nearest allowed point to (x, y): on the connected line, or on a line beyond a gap on its extension."""
        self.last_bridge = None
        if not self.ok:
            return x, y
        d1, k1 = self.tree.query([x, y])
        p1 = (float(self.pts[k1][0]), float(self.pts[k1][1]))
        if self.jump_tree is not None:
            d2, k2 = self.jump_tree.query([x, y])
            if d2 < 30 and d2 < d1 - 8:             # the mouse is clearly nearer to the line beyond the gap
                p2 = (float(self.jump_pts[k2][0]), float(self.jump_pts[k2][1]))
                self.last_bridge = ((float(self.jump_end[k2][0]), float(self.jump_end[k2][1])), p2)
                return p2
        return p1


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
        self.profile_key = "generic"
        self.section_settings = SectionSettings()
        self.readouts = []          # points read from the trace: dicts {band, x, y} (work px)
        self.exclusions = []        # rectangles whose ink is ignored (text, stains): dicts {x0, y0, x1, y1} (work px)
        self.band_ranges = {}       # band index -> (y_top, y_bottom) in work px chosen by the user (the band's territory)
        self._tracing_band = -1     # index of the band being traced (its own lines are not 'occupied')
        self._origins_hint = None   # origins of all bands while they are being traced one after the other

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
        if self.work is not self.rgb_orig:
            self.rgb_orig = None       # the analysis image is enough; the original file stays untouched on disk (lighter in memory)
        self.prepared = None
        self.graph = None
        self.bands = []
        progress("読込完了", 1.0)

    # ------------------------------------------------------- profiles / compare
    def clone_for_profile(self, profile):
        """A new engine on the same loaded image, analysed with the settings of `profile`."""
        e = Engine(profile.apply(self.params))
        e.profile_key = profile.key
        e.src_path, e.page, e.digest = self.src_path, self.page, self.digest
        e.rgb_orig, e.work, e.scale, e.orig_size = self.rgb_orig, self.work, self.scale, self.orig_size
        e.calibration = Calibration.from_dict(self.calibration.to_dict())
        e.exclusions = [dict(r) for r in self.exclusions]
        return e

    def quality_summary(self):
        """Rough indicators of how cleanly the trace came out. NOT a measure of correctness."""
        right = self.prepared.paper[2] if self.prepared else self.work.shape[1]
        inv = 1.0 / self.scale
        done = sum(1 for b in self.bands if b.complete)
        untracked = sum(max(0.0, right - b.end_x) for b in self.bands) * inv
        n = lambda kind: sum(1 for b in self.bands for i in b.issues if i.kind == kind)  # noqa: E731
        return dict(bands=len(self.bands), complete=done, untracked_px=untracked,
                    ambiguous=n("ambiguous"), excursion=n("excursion"), inferred=n("inferred"),
                    no_origin=n("no_origin"), vertices=sum(len(b.vertices) for b in self.bands))

    @staticmethod
    def quality_sort_key(q):
        """Smaller is better: all three bands found, reached the end, fewer doubtful spots."""
        return (-(q["bands"] >= 3), -q["complete"], q["untracked_px"] // 50, q["excursion"], q["ambiguous"])

    def overlay_png(self, max_w=900):
        from .excel_out import overlay_image
        return overlay_image(self, max_w, with_original=True, draw_vertices=False, draw_lines=False)

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
        self._apply_exclusions()
        progress("線を細線化中", 0.35)
        sk = skeleton_of(self.prepared.ink)
        g = build_graph(sk, self.params.spur_min_len)
        progress("途切れを補修中", 0.6)
        if self.params.gap_mode == "off":
            ink2 = self.prepared.ink
        else:
            ink2, _ = bridge_gaps(self.prepared.ink, g, max_gap=20, cos_min=0.6, max_vy=1.1, evidence=self.gap_evidence)
            g = build_graph(skeleton_of(ink2), self.params.spur_min_len)
            add_virtual_edges(g, max_gap=self.params.virtual_max_gap, max_dy_frac=0.25, evidence=self.gap_evidence)
        self.graph = g
        self.ink = ink2

    def gap_evidence(self, p0, p1):
        """True when the image shows a (faint) trace along the straight link p0-p1.

        Estimated links are only allowed where the paper is darker than its background along most of
        the link; across clean paper nothing is drawn (TR-06: no line is invented).
        """
        nm = self.prepared.norm_min if self.prepared is not None else None
        if nm is None:
            return False
        n = int(max(2, np.hypot(p1[0] - p0[0], p1[1] - p0[1])))
        xs = np.clip(np.round(np.linspace(p0[0], p1[0], n)).astype(int), 0, nm.shape[1] - 1)
        ys = np.clip(np.round(np.linspace(p0[1], p1[1], n)).astype(int), 0, nm.shape[0] - 1)
        return float(np.mean(nm[ys, xs] < self.params.gap_evidence_thr)) >= self.params.gap_evidence_min

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

    def analyze(self, origins=None, progress=_noop, stepwise=False):
        """Trace the bands one after the other, top band first. Lines already taken by an earlier band are
        avoided by the later ones. With stepwise=True only the first band is traced; trace_next() does the rest
        so that each band can be checked (and corrected) before the next one is judged."""
        self.ensure_graph(progress)
        progress("三段の起点を検出中", 0.7)
        if origins is None:
            origins = self.detect_origins()
        self._origins_hint = list(origins)
        self.band_ranges = {}
        self.bands = []
        for i, (ox, oy) in enumerate(origins):
            name = BAND_NAMES[i] if i < 3 else f"波形{i + 1}"
            if stepwise and i > 0:
                self.bands.append(self._placeholder_band(i, ox, oy))
                continue
            progress(f"{name}を追跡中（{i + 1}/{len(origins)}段目）", 0.72 + 0.25 * i / max(1, len(origins)))
            self.bands.append(self._trace_new_band(i, origins))
        progress("完了", 1.0)
        return self.bands

    def _placeholder_band(self, i, ox, oy):
        b = Band(i, BAND_NAMES[i] if i < 3 else f"波形{i + 1}", BAND_COLORS[i % len(BAND_COLORS)], (ox, oy), float(oy))
        b.traced = False
        b.line_zero = b.line_top = b.line_bottom = float(oy)
        b.issues.append(Issue("not_traced", ox, oy, "まだ解析していません（「次の段を解析」で解析します）"))
        b.initial = b.snapshot()
        return b

    def untraced_bands(self):
        return [i for i, b in enumerate(self.bands) if not b.traced]

    def trace_next(self, progress=_noop):
        """Trace the next band that has not been traced yet, avoiding the lines the earlier bands use."""
        idx = self.untraced_bands()
        if not idx:
            return None
        i = idx[0]
        self.ensure_graph(progress)
        progress(f"{self.bands[i].name}を追跡中", 0.5)
        origins = self._origin_list()
        nb = self._trace_new_band(i, origins)
        nb.sections, nb.revisions, nb.findings = self.bands[i].sections, self.bands[i].revisions, self.bands[i].findings
        nb.traced = True
        self.bands[i] = nb
        return i

    def retrace_following(self, i, progress=_noop):
        """After band i was corrected, trace the bands below it again (each one avoiding the lines taken before)."""
        self.ensure_graph(progress)
        origins = self._origin_list()
        done = []
        for j in range(i + 1, len(self.bands)):
            progress(f"{self.bands[j].name}を引き直し中", 0.3 + 0.6 * (j - i) / max(1, len(self.bands) - i))
            nb = self._trace_new_band(j, origins)
            self._replace_band(j, nb)
            self.bands[j].traced = True
            done.append(j)
        return done

    def _origin_list(self):
        """Origins of all bands (also those not traced yet)."""
        base = list(self._origins_hint) if self._origins_hint else []
        out = []
        for i in range(max(len(base), len(self.bands))):
            if i < len(self.bands):
                out.append(self.bands[i].origin)
            else:
                out.append(base[i])
        return out

    # ------------------------------------------- vertical range of a band (user's check)
    def check_range(self, bi, top, bot):
        """Does the line really go as far as the chosen top / bottom? What would the range cut off?

        Looks at the current trace and at the image (tips of lines near the two borders that belong to the trace).
        """
        top, bot = float(min(top, bot)), float(max(top, bot))
        b = self.bands[bi]
        res = dict(top=top, bottom=bot, traced=b.traced and len(b.path) > 0, origin_inside=top <= b.origin[1] <= bot,
                   trace_top=None, trace_bottom=None, reaches_top=False, reaches_bottom=False,
                   outside_px=0.0, tips_top=0, tips_bottom=0, tips_total=0)
        if not res["traced"]:
            return res
        p = b.path
        res["trace_top"], res["trace_bottom"] = float(p[:, 1].min()), float(p[:, 1].max())
        margin = 12.0
        res["reaches_top"] = res["trace_top"] <= top + margin
        res["reaches_bottom"] = res["trace_bottom"] >= bot - margin
        out = (p[:, 1] < top) | (p[:, 1] > bot)
        res["outside_px"] = float(out.sum()) / self.scale
        if self.graph is not None:
            g = self.graph
            tree = cKDTree(p.astype(float))
            for n in g.nodes:
                if n.kind != "end" or not n.edges or not (top - 25 <= n.y <= bot + 25):
                    continue
                e = g.edges[n.edges[0]]
                if e.inferred or e.length < 12:
                    continue
                base = e.pix[-1] if (e.pix[0] == np.round([n.x, n.y])).all() else e.pix[0]
                if tree.query(base.astype(float))[0] > 3.5:
                    continue                      # not part of this band's line
                res["tips_total"] += 1
                if abs(n.y - top) <= 25:
                    res["tips_top"] += 1
                if abs(n.y - bot) <= 25:
                    res["tips_bottom"] += 1
        return res

    def retrace_in_range(self, bi, top, bot, progress=_noop):
        """Trace the band again inside the vertical range [top, bottom] only (undoable)."""
        top, bot = float(min(top, bot)), float(max(top, bot))
        b = self.bands[bi]
        if bot - top < 20:
            raise EngineError("範囲が狭すぎます")
        if not (top <= b.origin[1] <= bot):
            raise EngineError("範囲に、この段の起点（左端の線）が入っていません。範囲を広げてください")
        self.ensure_graph(progress)
        old = self.band_ranges.get(bi)
        self.band_ranges[bi] = (top, bot)
        try:
            nb = self._trace_new_band(bi, self._origin_list())
        except Exception:
            if old is None:
                self.band_ranges.pop(bi, None)
            else:
                self.band_ranges[bi] = old
            raise
        nb.traced = True
        return self._replace_band(bi, nb)

    def clear_range(self, bi, progress=_noop):
        """Go back to the automatic range of the band and trace it again."""
        if bi not in self.band_ranges:
            return self.bands[bi]
        self.ensure_graph(progress)
        del self.band_ranges[bi]
        nb = self._trace_new_band(bi, self._origin_list())
        nb.traced = True
        return self._replace_band(bi, nb)

    def _confirmed_edges(self, exclude):
        """Edges of bands the user has confirmed: other bands must not use them (a hard rule, unlike 'occupied')."""
        g = self.graph
        out = set()
        if g is None:
            return out
        for j, b in enumerate(self.bands):
            if j == exclude or not b.confirmed or len(b.path) == 0:
                continue
            e = np.unique(g.pix_edge[b.path[:, 1], b.path[:, 0]])
            out |= {int(x) for x in e if x >= 0}
        return out

    def _occupied_edges(self, exclude):
        """Graph edges that the lines of the other bands run along (avoided, not forbidden)."""
        g = self.graph
        occ = set()
        if g is None:
            return occ
        for j, b in enumerate(self.bands):
            if j == exclude or len(b.path) == 0:
                continue
            e = np.unique(g.pix_edge[b.path[:, 1], b.path[:, 0]])
            occ |= {int(x) for x in e if x >= 0}
        return occ

    # ----------------------------------------------------------------- helpers
    def _zone(self, i, origins):
        rng = self.band_ranges.get(i)
        if rng:                    # the user fixed the vertical range of this band
            top, bot = rng
            return (top, bot, top - 10, bot + 10), (top, bot)
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

    def _trace_from(self, node, edge, corr, zone, blocked=None, oy=0, prefix=None, use_confirmed=True):
        """Walk, cut at the right-most point of the band zone, and bridge to further fragments."""
        g = self.graph
        W = self.work.shape[1]
        x_right = self.prepared.paper[2]
        walker = Walker(g, self.params, corr, ref_zone=self.prepared.ref_zone,
                        occupied=self._occupied_edges(exclude=self._tracing_band))
        pieces, rev, amb, inf = [], [], [], []
        confirmed_edges = self._confirmed_edges(self._tracing_band) if use_confirmed else set()
        used_nodes = set()
        blocked = set(blocked or ()) | confirmed_edges
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
            if self.params.gap_mode == "off":
                break
            end_pt = pix[-1]
            cands = []
            for j, n in enumerate(g.nodes):
                if n.kind != "end" or not n.edges or j in used_nodes:
                    continue
                if n.x <= end_x + 5 or not (zone[0] <= n.y <= zone[1]):
                    continue
                if n.edges[0] in blocked:
                    continue
                # a broken pen line continues about horizontally: steep links are not accepted
                if abs(n.y - end_pt[1]) > 0.3 * max(1.0, n.x - end_pt[0]) + 8:
                    continue
                d = np.hypot(n.x - end_pt[0], 3.0 * (n.y - end_pt[1]))
                if d < self.params.jump_max_gap:
                    cands.append((d, j))
            best = None
            for d, j in sorted(cands):
                if self.gap_evidence((end_pt[0], end_pt[1]), (g.nodes[j].x, g.nodes[j].y)):
                    best = j
                    break
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

    def _finish_band(self, b, path, rev, amb, inf, end_x, keep_manual=True, recompute_vertices=True):
        b.path, b.inferred, b.end_x = path, inf, end_x
        if recompute_vertices:
            b.vertices = extract_vertices(path, rev, self.params.tip_min_prominence, self.ink)
            self._apply_deleted(b)
            path, inf = self._apply_erased(b, path, inf)       # erased stretches stay erased after a re-trace
            end_x = b.end_x
            self._mark_manual(b)
            self._mark_suspect(b)
        W = self.work.shape[1]
        b.complete = end_x >= self.prepared.paper[2] - 0.03 * W
        b.issues = []
        for idx, x, y in amb[:200]:
            b.issues.append(Issue("ambiguous", x, y, "交差・接触で所属を判断しにくい箇所"))
        for a, c in inf:
            b.issues.append(Issue("inferred", float(path[a][0]), float(path[a][1]),
                                  f"画像の薄い線の痕跡をたどって推定でつないだ区間（長さ約{(c - a + 1) / self.scale:.0f}px）"))
        # Stretches that leave the band's own territory (arcs, dirt, other bands) need a human look.
        origins = self._origin_list() or [b.origin]
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
        for j in b.cuts:
            if 0 <= j < len(path):
                b.issues.append(Issue("erased", float(path[j][0]), float(path[j][1]),
                                      "手動で消した区間（ここで線が途切れています）"))
        if not b.complete and len(path):
            b.issues.append(Issue("untracked_end", float(path[-1][0]), float(path[-1][1]),
                                  "ここから右が未追跡（末端まで到達していません）"))
        if len(path):
            first = path[: min(len(path), 80)]
            b.y0 = float(np.median(first[:, 1]))
        if not b.line_top_manual or not b.line_bottom_manual or True:
            self.reset_lines(b, only_auto=True)

    def _apply_erased(self, b, path, inf):
        """Take the path points inside the user's erase rectangles out of the line (a gap is left there).
        Vertices on erased points are dropped, indices are renumbered, and the places where the line is broken are
        remembered in b.cuts. Returns the new path and the new estimated-link spans."""
        old_cuts = list(b.cuts)
        b.cuts = []
        if not b.erased or len(path) == 0:
            return path, inf
        x, y = path[:, 0], path[:, 1]
        inside = np.zeros(len(path), bool)
        for r in b.erased:
            inside |= (x >= r["x0"]) & (x <= r["x1"]) & (y >= r["y0"]) & (y <= r["y1"])
        if not inside.any():
            b.cuts = old_cuts
            return path, inf
        keep = ~inside
        newidx = np.cumsum(keep) - 1
        kept_old = np.flatnonzero(keep)
        newpath = path[keep]
        cuts = {int(g) for g in np.flatnonzero(np.diff(kept_old) > 1)}
        for j in old_cuts:                                   # breaks that were already there
            if j + 1 < len(keep) and keep[j] and keep[j + 1] and newidx[j + 1] - newidx[j] == 1:
                cuts.add(int(newidx[j]))
        b.cuts = sorted(cuts)
        inf2 = [(int(newidx[a]), int(newidx[c])) for a, c in inf
                if a < len(keep) and c < len(keep) and keep[a] and keep[c] and not inside[a:c + 1].any()]
        verts = []
        for v in b.vertices:
            in_rect = any(r["x0"] <= v.x <= r["x1"] and r["y0"] <= v.y <= r["y1"] for r in b.erased)
            if v.idx < len(keep) and keep[v.idx] and not in_rect:   # (a tip refined against the ink may sit just across the edge)
                v.idx = int(newidx[v.idx])
                verts.append(v)
        b.vertices = verts
        b.path, b.inferred = newpath, inf2
        if len(newpath):
            b.end_x = float(newpath[:, 0].max())
        return newpath, inf2

    def preview_erase(self, bi, rect):
        """Which path points a rectangle would erase (nothing is changed)."""
        b = self.bands[bi]
        p = b.path
        if len(p) == 0:
            return np.zeros(0, bool)
        return (p[:, 0] >= rect["x0"]) & (p[:, 0] <= rect["x1"]) & (p[:, 1] >= rect["y0"]) & (p[:, 1] <= rect["y1"])

    def erase_rect(self, bi, x0, y0, x1, y1):
        """Erase the traced line of a band inside a rectangle (undoable). The vertices of the rest stay as they are."""
        b = self.bands[bi]
        rect = dict(x0=float(min(x0, x1)), y0=float(min(y0, y1)), x1=float(max(x0, x1)), y1=float(max(y0, y1)))
        n = int(self.preview_erase(bi, rect).sum())
        if n == 0:
            raise EngineError("囲んだ範囲に、この段のトレース線がありません")
        b.push_undo(self.params.undo_limit)
        b.erased.append(rect)
        path, inf = self._apply_erased(b, b.path, b.inferred)
        self._finish_band(b, path, [], [], inf, b.end_x, recompute_vertices=False)
        b.confirmed = False
        return n

    def _apply_deleted(self, b, radius=7.0):
        """Vertices the user deleted stay deleted when the line is traced again."""
        if not b.deleted_points:
            return
        keep = []
        for v in b.vertices:
            if any(np.hypot(v.x - d["x"], v.y - d["y"]) <= radius for d in b.deleted_points):
                continue
            keep.append(v)
        b.vertices = keep

    def delete_vertex(self, bi, vi):
        """Delete one vertex (undoable): it is taken out of the vertex list and out of all calculations
        (maximum / minimum, wave count, period). The line itself is not changed; to drop part of the line
        itself, fence it off with an exclusion range. A deleted vertex stays deleted when the line is traced again."""
        b = self.bands[bi]
        if not (0 <= vi < len(b.vertices)):
            raise EngineError("削除する頂点が選ばれていません")
        v = b.vertices[vi]
        b.push_undo(self.params.undo_limit)
        b.deleted_points.append(dict(x=float(v.x), y=float(v.y)))
        b.vertices = [x for i, x in enumerate(b.vertices) if i != vi]
        b.confirmed = False
        return "頂点を削除しました（線は変えていません）"

    def delete_vertices(self, bi, vis):
        """Delete several vertices in ONE undoable step (see delete_vertex)."""
        b = self.bands[bi]
        vis = sorted({int(i) for i in vis if 0 <= int(i) < len(b.vertices)})
        if not vis:
            raise EngineError("削除する頂点が選ばれていません")
        b.push_undo(self.params.undo_limit)
        for i in vis:
            v = b.vertices[i]
            b.deleted_points.append(dict(x=float(v.x), y=float(v.y)))
        b.vertices = [x for i, x in enumerate(b.vertices) if i not in set(vis)]
        b.confirmed = False
        return f"頂点を{len(vis)}個削除しました（線は変えていません）"

    def _mark_suspect(self, b):
        """Vertices on stretches that leave the band's territory are candidates only (MS-04)."""
        origins = self._origin_list() or [b.origin]
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
        self._tracing_band = i
        ox, oy = origins[i]
        b = Band(i, BAND_NAMES[i] if i < 3 else f"波形{i + 1}", BAND_COLORS[i % len(BAND_COLORS)], (ox, oy), float(oy))
        corr, zone = self._zone(i, origins)
        node = self._start_node(ox, oy)
        if node is None:
            b.issues.append(Issue("no_origin", ox, oy, "起点付近に線が見つかりません。起点を指定してください"))
            b.line_zero = b.line_top = b.line_bottom = float(oy)
            b.initial = b.snapshot()
            return b
        first = self.graph.nodes[node].edges[0]
        path, rev, amb, inf, end_x = self._trace_from(node, first, corr, zone, oy=oy)
        shared = False
        if self._confirmed_edges(i):
            W = self.work.shape[1]
            if end_x < self.prepared.paper[2] - 0.03 * W:
                # The confirmed bands' lines are off limits, but this band cannot get to the end without them
                # (the pens really overlap there): allow the shared line and say so.
                p2 = self._trace_from(node, first, corr, zone, oy=oy, use_confirmed=False)
                if p2[4] > end_x + 50:
                    path, rev, amb, inf, end_x = p2
                    shared = True
        self._finish_band(b, path, rev, amb, inf, end_x)
        if shared:
            x, y = b.path[-1] if len(b.path) else b.origin
            b.issues.append(Issue("shared_line", float(x), float(y),
                                  "確認済みの段と線が重なっています（そこを通らないと末端まで追えないため共有しました）"))
        b.initial = b.snapshot()
        return b

    def _replace_band(self, i, nb):
        """Swap in a freshly traced band; the old state goes onto the undo stack, logs and sections are kept."""
        old = self.bands[i]
        old.push_undo(self.params.undo_limit)
        nb.undo, nb.redo = old.undo, old.redo
        nb.sections, nb.revisions, nb.findings = old.sections, old.revisions, old.findings
        nb.confirmed = False
        self.bands[i] = nb
        return nb

    def retrace_band_from_origin(self, i, ox, oy):
        """IN-05: manual origin for a band whose automatic start failed."""
        origins = [bd.origin for bd in self.bands]
        origins[i] = (ox, oy)
        return self._replace_band(i, self._trace_new_band(i, origins))

    # -------------------------------------------------------------- exclusions
    def _apply_exclusions(self):
        h, w = self.prepared.ink.shape
        for r in self.exclusions:
            x0, x1 = max(0, int(r["x0"])), min(w, int(r["x1"]) + 1)
            y0, y1 = max(0, int(r["y0"])), min(h, int(r["y1"]) + 1)
            self.prepared.ink[y0:y1, x0:x1] = False

    def _bands_touching(self, rect, margin=6):
        x0, y0, x1, y1 = rect["x0"] - margin, rect["y0"] - margin, rect["x1"] + margin, rect["y1"] + margin
        hit = []
        for i, b in enumerate(self.bands):
            p = b.path
            if len(p) and np.any((p[:, 0] >= x0) & (p[:, 0] <= x1) & (p[:, 1] >= y0) & (p[:, 1] <= y1)):
                hit.append(i)
        return hit

    def _retrace_bands(self, idx, progress):
        origins = [bd.origin for bd in self.bands]
        for k, i in enumerate(idx):
            progress(f"{self.bands[i].name}を引き直し中", 0.5 + 0.45 * k / max(1, len(idx)))
            self._replace_band(i, self._trace_new_band(i, origins))

    def add_exclusion(self, x0, y0, x1, y1, progress=_noop):
        """Ignore everything inside the rectangle (e.g. letters touching the trace) and re-trace the bands
        whose line ran through it. Other bands and their edits are not touched. Before the first analysis the
        rectangle is just remembered and applied when the analysis runs."""
        rect = dict(x0=float(min(x0, x1)), y0=float(min(y0, y1)), x1=float(max(x0, x1)), y1=float(max(y0, y1)))
        return self.add_exclusions([rect], progress)

    def add_exclusions(self, rects, progress=_noop):
        rects = [dict(x0=float(min(r["x0"], r["x1"])), y0=float(min(r["y0"], r["y1"])),
                      x1=float(max(r["x0"], r["x1"])), y1=float(max(r["y0"], r["y1"]))) for r in rects]
        for r in rects:
            if r["x1"] - r["x0"] < 4 or r["y1"] - r["y0"] < 4:
                raise EngineError("除外範囲が小さすぎます")
        if not self.bands:                       # not analysed yet: remember only, the analysis applies them
            self.exclusions += rects
            self.graph = None
            self.prepared = None
            return []
        affected = sorted({i for r in rects for i in self._bands_touching(r)})
        self.exclusions += rects
        self.graph = None
        self.ensure_graph(progress)
        self._retrace_bands(affected, progress)
        return affected

    def suggest_exclusions(self, progress=_noop):
        """Find places that are probably NOT part of the graph: stains and writing/letters outside the bands.

        Heuristics, only suggestions: (1) dense dark blobs (stains), (2) clusters of small marks (letters, notes).
        Anything near a band's baseline is never suggested (that is where the waveform is). Areas that are already
        excluded are not suggested again."""
        progress("除外の候補を探しています", 0.2)
        if self.prepared is None:
            self.prepared = prepare(self.work, self.params)
            self._apply_exclusions()
        ink = self.prepared.ink
        H, W = ink.shape
        origins = self.detect_origins()
        ys = [o[1] for o in origins]
        gap = float(np.median(np.diff(ys))) if len(ys) >= 2 else 0.2 * H
        zones = [(oy - 0.3 * gap, oy + 0.3 * gap) for oy in ys]
        out = []
        # ink that is connected to a band's own territory belongs to the graph, e.g. a tall spike that sticks out of it
        n_all, lab_all = cv2.connectedComponents(ink.astype(np.uint8), connectivity=8)
        zone_labels = set()
        for zt, zb in zones:
            zone_labels |= set(np.unique(lab_all[max(0, int(zt)):int(zb), :]).tolist())
        zone_labels.discard(0)
        zl = np.zeros(n_all + 1, bool)
        for k in zone_labels:
            zl[k] = True

        def belongs_to_graph(rect):
            x0, y0, x1, y1 = rect
            sub = lab_all[y0:y1, x0:x1]
            nz = sub > 0
            if not nz.any():
                return False
            frac = float(zl[sub[nz]].mean())
            dist = min((max(0, zt - y1, y0 - zb) for zt, zb in zones), default=1e9)   # 0 when they overlap in y
            return dist <= 25 and frac >= 0.4

        def clip(rect):
            """Cut away the parts of a rectangle that lie in a band's own territory."""
            x0, y0, x1, y1 = rect
            for zt, zb in zones:
                if y1 > zt and y0 < zb:
                    up, down = (zt - y0), (y1 - zb)
                    if up >= down and up > 0:
                        y1 = zt
                    elif down > 0:
                        y0 = zb
                    else:
                        return None
            if (x1 - x0) > 8 and (y1 - y0) > 8:
                return int(x0), int(y0), int(x1), int(y1)
            return None

        # (1) stains: big dense dark areas
        dens = cv2.boxFilter(ink.astype(np.float32), -1, (41, 41))
        mask = (dens > 0.28).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)))
        n, lab, st, _c = cv2.connectedComponentsWithStats(mask, connectivity=8)
        stains = []
        for i in range(1, n):
            x, y, w, h, area = st[i]
            if area < 4000 or int(ink[y:y + h, x:x + w].sum()) < 2500:
                continue
            r = clip((x, y, x + w, y + h))
            ink_n = int(ink[r[1]:r[3], r[0]:r[2]].sum()) if r is not None else 0
            if r is not None and (r[2] - r[0]) * (r[3] - r[1]) >= 3000 and ink_n >= 1500 \
                    and ink_n / float((r[2] - r[0]) * (r[3] - r[1])) >= 0.2:       # a stain is dense; a spike sticking out is not
                stains.append(r)
                out.append(dict(x0=float(r[0]), y0=float(r[1]), x1=float(r[2]), y1=float(r[3]), kind="汚れ",
                                n=int(ink[r[1]:r[3], r[0]:r[2]].sum())))
        # (2) writing: clusters of small separate marks away from the bands and from the stains
        progress("文字・書き込みを探しています", 0.6)
        n, lab, st, cen = cv2.connectedComponentsWithStats(ink.astype(np.uint8), connectivity=8)
        canvas = np.zeros(ink.shape, np.uint8)
        sizes = {}
        for i in range(1, n):
            x, y, w, h, area = st[i]
            if not (20 <= area <= 2500) or max(w, h) > 120:
                continue
            cy = y + h / 2
            if any(zt <= cy <= zb for zt, zb in zones):
                continue
            if any(sx0 <= x + w / 2 <= sx1 and sy0 <= cy <= sy1 for sx0, sy0, sx1, sy1 in stains):
                continue
            cv2.rectangle(canvas, (max(0, x - 30), max(0, y - 30)), (min(W - 1, x + w + 30), min(H - 1, y + h + 30)), 1, -1)
            sizes[i] = (x, y, w, h)
        m2, lab2, st2, _c2 = cv2.connectedComponentsWithStats(canvas, connectivity=8)
        for j in range(1, m2):
            members = [v for i, v in sizes.items() if lab2[v[1] + v[3] // 2, v[0] + v[2] // 2] == j]
            if len(members) < 3:
                continue
            xs0 = min(v[0] for v in members) - 6
            ys0 = min(v[1] for v in members) - 6
            xs1 = max(v[0] + v[2] for v in members) + 6
            ys1 = max(v[1] + v[3] for v in members) + 6
            r = clip((max(0, xs0), max(0, ys0), min(W, xs1), min(H, ys1)))
            if r is not None and not belongs_to_graph(r):
                out.append(dict(x0=float(r[0]), y0=float(r[1]), x1=float(r[2]), y1=float(r[3]), kind="文字・書き込み",
                                n=len(members)))
        # the paper's edge and the rulers are outside the paper box already; keep suggestions inside it
        px0, py0, px1, py1 = self.prepared.paper
        out = [o for o in out if o["x0"] >= px0 - 5 and o["x1"] <= px1 + 5 and o["y0"] >= py0 - 5 and o["y1"] <= py1 + 5]
        out.sort(key=lambda o: -(o["x1"] - o["x0"]) * (o["y1"] - o["y0"]))
        progress("完了", 1.0)
        return out

    def remove_exclusion(self, k, progress=_noop):
        if not (0 <= k < len(self.exclusions)):
            return []
        rect = self.exclusions.pop(k)
        origins = [bd.origin for bd in self.bands]
        affected = []
        for i in range(len(self.bands)):
            corr, _z = self._zone(i, origins) if len(origins) >= 2 else ((rect["y0"], rect["y1"], rect["y0"], rect["y1"]),) * 2
            if rect["y1"] >= corr[0] - 20 and rect["y0"] <= corr[1] + 20:
                affected.append(i)
        self.graph = None
        self.ensure_graph(progress)
        self._retrace_bands(affected, progress)
        return affected

    def line_constraint(self, x, y, limit=150, jump=None, band=None):
        """Constraint that keeps a dragged vertex on the image's line, with a jump across gaps (None while the graph is not built)."""
        if self.graph is None:
            return None
        y_range = None
        if band is not None and 0 <= band < len(self.bands):
            origins = self._origin_list()
            if len(origins) >= 2 or band in self.band_ranges:
                corr, _z = self._zone(band, origins)
                y_range = (corr[2], corr[3])
        lc = LineConstraint(self.graph.skel, (x, y), limit, self.params.drag_jump_px if jump is None else jump, y_range=y_range)
        return lc if lc.ok else None

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

    def _route_bridged(self, a, b, max_gap=None, pad=60):
        """Route a -> b along the skeleton; if they lie on pieces of line that are not connected, cross the gap with
        a straight bridge between the two pieces' closest pixels. Returns (route, bridge) where bridge is None or
        the index pair (i0, i1) of the bridge inside the route, or (None, None) if it cannot be done."""
        from skimage.graph import MCP_Geometric
        max_gap = max_gap or (self.params.drag_jump_px + 40)
        r = self._route(a, b)
        if r is not None:
            return r, None
        g = self.graph
        h, w = g.skel.shape
        xa, ya, xb, yb = int(a[0]), int(a[1]), int(b[0]), int(b[1])
        x0, x1 = max(0, min(xa, xb) - pad - max_gap // 2), min(w, max(xa, xb) + pad + max_gap // 2 + 1)
        y0, y1 = max(0, min(ya, yb) - pad - max_gap // 2), min(h, max(ya, yb) + pad + max_gap // 2 + 1)
        sub = g.skel[y0:y1, x0:x1]
        if not (sub[ya - y0, xa - x0] and sub[yb - y0, xb - x0]):
            return None, None
        costs = np.where(sub, 1.0, np.inf)
        ca, _t = MCP_Geometric(costs, fully_connected=True).find_costs([(ya - y0, xa - x0)])
        cb, _t = MCP_Geometric(costs, fully_connected=True).find_costs([(yb - y0, xb - x0)])
        ay, ax = np.nonzero(np.isfinite(ca))
        by_, bx = np.nonzero(np.isfinite(cb))
        if len(ax) == 0 or len(bx) == 0:
            return None, None
        pa = np.stack([ax + x0, ay + y0], 1).astype(float)
        pb = np.stack([bx + x0, by_ + y0], 1).astype(float)
        d, k = cKDTree(pb).query(pa)
        i = int(np.argmin(d))
        if d[i] > max_gap:
            return None, None
        ra, rb = pa[i].astype(int), pb[k[i]].astype(int)
        r1 = self._route(tuple(a), tuple(ra))
        r2 = self._route(tuple(rb), tuple(b))
        if r1 is None or r2 is None:
            return None, None
        n = int(max(2, np.hypot(*(rb - ra))))
        line = np.stack([np.round(np.linspace(ra[0], rb[0], n)), np.round(np.linspace(ra[1], rb[1], n))], 1).astype(np.int32)
        route = np.concatenate([r1, line[1:], r2[1:]]).astype(np.int32)
        return route, (len(r1) - 1, len(r1) - 1 + len(line) - 1)

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

    def _compute_pending(self, b, mode="tail"):
        """Work out the result of all pending edits of a band on a COPY. Raises EngineError if one cannot be done.

        mode "tail" : a moved vertex re-traces the line from that vertex to the end
        mode "local": a moved vertex only changes the line between the vertices before and after it
        """
        snap_before = b.snapshot()
        msgs = []
        moves = sorted([p for p in b.pending if p["type"] == "move"], key=lambda p: p["vi"])
        adds = [p for p in b.pending if p["type"] == "add"]
        work_b = Band(b.index, b.name, b.color)
        work_b.restore(snap_before)
        work_b.pending = []
        for p in moves:
            ok, msg = self._apply_move(work_b, p, local=(mode == "local"))
            if not ok:
                raise EngineError(msg)
            msgs.append(msg)
        for p in adds:
            ok, msg = self._apply_add(work_b, p)
            if not ok:
                raise EngineError(msg)
            msgs.append(msg)
        for p in [q for q in b.pending if q["type"] == "extend"]:
            ok, msg = self._apply_extend(work_b, p)
            if not ok:
                raise EngineError(msg)
            msgs.append(msg)
        return work_b, msgs

    def preview_pending(self, bi, mode="local", progress=_noop):
        """Show what the pending edits would do (nothing is changed): old and new path, how much of the line changes."""
        b = self.bands[bi]
        self.ensure_graph(progress)
        work_b, msgs = self._compute_pending(b, mode)
        old, new = b.path, work_b.path
        n = min(len(old), len(new))
        same_head = 0
        if n:
            diff = np.flatnonzero(np.any(old[:n] != new[:n], axis=1))
            same_head = int(diff[0]) if len(diff) else n
        # length of the changed stretch: compare the tails the same way
        same_tail = 0
        if n:
            diff = np.flatnonzero(np.any(old[::-1][:n] != new[::-1][:n], axis=1))
            same_tail = int(diff[0]) if len(diff) else n
        changed_old = max(0, len(old) - same_head - same_tail)
        changed_new = max(0, len(new) - same_head - same_tail)
        return dict(old=old, new=new, messages=msgs,
                    range_old=(same_head, max(same_head, len(old) - same_tail)), range_new=(same_head, max(same_head, len(new) - same_tail)), changed_old=changed_old / self.scale, changed_new=changed_new / self.scale,
                    vertices_before=len(b.vertices), vertices_after=len(work_b.vertices),
                    first_changed=(float(new[same_head][0]), float(new[same_head][1])) if same_head < len(new) else None)

    def apply_pending(self, bi, progress=_noop, mode="tail"):
        """Apply all pending edits of one band. Returns (ok, message). On failure nothing changes."""
        b = self.bands[bi]
        if not b.pending:
            return True, "適用する修正はありません"
        self.ensure_graph(progress)
        snap_before = b.snapshot()
        try:
            work_b, msgs = self._compute_pending(b, mode)
        except EngineError as e:
            b.restore(snap_before)
            b.issues.append(Issue("retrace_failed", b.pending[0]["x"], b.pending[0]["y"], str(e)))
            return False, str(e)
        b.push_undo(self.params.undo_limit)
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

    def _apply_move(self, b, p, local=False):
        k = self._find_vertex(b, p["ox"], p["oy"], p["kind"])
        if k is None:
            return False, "移動対象の頂点が見つかりません"
        P, off = self._snap(p["x"], p["y"])
        if P is None:
            return False, f"指定位置の近く({self.params.snap_radius}px以内)に画像の線がありません。修正前の結果を保持します"
        vs = b.vertices
        if local and k + 1 < len(vs):
            return self._apply_move_local(b, p, k, P, off)
        start_idx = vs[k - 1].idx if k > 0 else 0
        prefix = b.path[: start_idx + 1]
        a = prefix[-1]
        a_snap, _ = self._snap(a[0], a[1])
        if a_snap is None:
            return False, "再計算の開始位置が画像の線上にありません"
        route, bridge = self._route_bridged(a_snap, P)
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
        self._tracing_band = b.index
        tail, rev, amb, inf, end_x = self._trace_from(node, edge, corr, zone, blocked=blocked, oy=oy)
        head = np.concatenate([prefix, route[1:], partial]) if len(partial) else np.concatenate([prefix, route[1:]])
        newpath = np.concatenate([head, tail]).astype(np.int32) if len(tail) else head
        shift = len(head)
        inf2 = [(x, y) for x, y in b.inferred if y <= start_idx] + [(x + shift, y + shift) for x, y in inf]
        if bridge is not None:
            inf2.append((len(prefix) - 1 + bridge[0], len(prefix) - 1 + bridge[1]))      # the gap crossed by hand
        prev_manual = [mp for mp in b.manual_points if np.hypot(mp["x"] - p["ox"], mp["y"] - p["oy"]) > 1]
        prev_manual.append(dict(x=float(P[0]), y=float(P[1]), kind="moved"))
        b.manual_points = prev_manual
        rev2 = [r + shift for r in rev]
        self._finish_band(b, newpath, rev2, [(a_ + shift, x_, y_) for a_, x_, y_ in amb], inf2, end_x)
        offtxt = f"（画像の線へ{off:.1f}px補正）" if off and off > 0.5 else ""
        bridged = "（途切れの部分は、手で線をつなぎました）" if bridge is not None else ""
        return True, f"頂点を移動し、そこから先を再追跡しました{bridged}{offtxt}"

    def _apply_move_local(self, b, p, k, P, off):
        """Move one vertex and re-route ONLY the stretch between the vertex before and the vertex after it.
        The rest of the line is not touched."""
        from .verify import reversals_from_path
        vs, path = b.vertices, b.path
        prev_idx = vs[k - 1].idx if k > 0 else 0
        next_idx = vs[k + 1].idx
        vk = vs[k].idx
        if not (prev_idx < vk < next_idx) or next_idx >= len(path):
            return False, "前後の頂点の位置が分かりません。「この点から先を引き直す」で試してください"
        # only a short stretch of the line around the vertex is re-routed (at most 60 path points to each side)
        start_idx = vk - max(3, min(60, (vk - prev_idx) // 2))
        end_idx = vk + max(3, min(60, (next_idx - vk) // 2))
        start_idx, end_idx = max(prev_idx, start_idx), min(next_idx, end_idx)
        a_snap, _ = self._snap(path[start_idx][0], path[start_idx][1])
        n_snap, _ = self._snap(path[end_idx][0], path[end_idx][1])
        if a_snap is None or n_snap is None:
            return False, "前後の点が画像の線の上にありません"
        r1, br1 = self._route_bridged(a_snap, P)
        r2, br2 = self._route_bridged(P, n_snap)
        if r1 is None or r2 is None:
            return False, "指定位置と前後の点を結ぶ線が見つかりません（途切れている可能性があります）。修正前の結果を保持します"
        seg = np.concatenate([r1, r2[1:]]).astype(np.int32)
        newpath = np.concatenate([path[:start_idx], seg, path[end_idx + 1:]]).astype(np.int32)
        shift = len(seg) - (end_idx - start_idx + 1)
        inf = []
        for a_, c_ in b.inferred:
            if c_ < start_idx:
                inf.append((a_, c_))
            elif a_ > end_idx:
                inf.append((a_ + shift, c_ + shift))
        if br1 is not None:
            inf.append((start_idx + br1[0], start_idx + br1[1]))
        if br2 is not None:
            off2 = start_idx + len(r1) - 1
            inf.append((off2 + br2[0], off2 + br2[1]))
        prev_manual = [mp for mp in b.manual_points if np.hypot(mp["x"] - p["ox"], mp["y"] - p["oy"]) > 1]
        prev_manual.append(dict(x=float(P[0]), y=float(P[1]), kind="moved"))
        b.manual_points = prev_manual
        self._finish_band(b, newpath, reversals_from_path(newpath), [], inf, b.end_x)
        offtxt = f"（画像の線へ{off:.1f}px補正）" if off and off > 0.5 else ""
        bridged = "（途切れの部分は、手で線をつなぎました）" if (br1 is not None or br2 is not None) else ""
        return True, f"頂点を移動しました。前後の頂点の間だけを引き直しました（他の線は変えていません）{bridged}{offtxt}"

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

    def set_pending_extend(self, bi, x, y):
        """Take the end of the trace to (x, y) - e.g. across a gap to the line on its extension - and go on from there."""
        b = self.bands[bi]
        b.pending = [q for q in b.pending if q["type"] != "extend"]
        b.pending.append(dict(type="extend", x=float(x), y=float(y)))

    def _apply_extend(self, b, p):
        P, off = self._snap(p["x"], p["y"])
        if P is None:
            return False, f"指定位置の近く({self.params.snap_radius}px以内)に画像の線がありません"
        path = b.path
        if len(path) == 0:
            return False, "トレース線がありません"
        a_snap, _ = self._snap(path[-1][0], path[-1][1])
        if a_snap is None:
            return False, "トレースの末端が画像の線の上にありません"
        route, bridge = self._route_bridged(a_snap, P)
        if route is None:
            return False, "その位置へ至る線が見つかりません（離れすぎているか、線の延長上にありません）"
        d = (route[-1] - route[max(0, len(route) - 12)]).astype(float)
        nd = np.hypot(*d)
        d = d / nd if nd else np.array([1.0, 0.0])
        node, edge, partial = self._continue_from(P, d)
        g = self.graph
        origins = self._origin_list()
        corr, zone = self._zone(b.index, origins)
        blocked = set()
        for part in (path, route):
            ids = np.unique(g.pix_edge[part[:, 1], part[:, 0]])
            blocked |= {int(e) for e in ids if e >= 0}
        tail = np.zeros((0, 2), np.int32)
        rev, amb, inf, end_x = [], [], [], float(max(path[:, 0].max(), P[0]))
        if node is not None:
            blocked.discard(edge)
            self._tracing_band = b.index
            tail, rev, amb, inf, end_x = self._trace_from(node, edge, corr, zone, blocked=blocked, oy=b.origin[1])
        L = len(path)
        head = np.concatenate([path, route[1:]] + ([partial] if len(partial) else []))
        newpath = np.concatenate([head, tail]).astype(np.int32) if len(tail) else head.astype(np.int32)
        shift = len(head)
        inf2 = list(b.inferred) + [(x + shift, y + shift) for x, y in inf]
        if bridge is not None:
            inf2.append((L - 1 + bridge[0], L - 1 + bridge[1]))
        end_x = max(end_x, float(newpath[:, 0].max()))
        b.manual_points.append(dict(x=float(P[0]), y=float(P[1]), kind="added"))
        from .verify import reversals_from_path
        self._finish_band(b, newpath, reversals_from_path(newpath), [], inf2, end_x)
        bridged = "（途切れを、手で線をつないで越えました）" if bridge is not None else ""
        return True, f"トレースの末端を指定の位置まで延ばし、そこから先を追跡しました{bridged}"

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

    # ------------------------------------------------------------ read values
    def read_point(self, bi, x, y, max_dist=40):
        """Snap to the nearest point of the traced line of band `bi` and remember it. Returns the row or None."""
        b = self.bands[bi]
        if len(b.path) == 0:
            return None
        d = np.hypot(b.path[:, 0] - x, b.path[:, 1] - y)
        j = int(np.argmin(d))
        if d[j] > max_dist:
            return None
        rd = dict(band=bi, x=float(b.path[j][0]), y=float(b.path[j][1]))
        self.readouts.append(rd)
        return self.readout_row(rd)

    def readout_row(self, rd):
        """Raw and arc-corrected values of one read point (positions in original px)."""
        b = self.bands[rd["band"]]
        cal, inv = self.calibration, 1.0 / self.scale
        sp = cal.seconds_per_orig_px()
        amp_px = (b.line_zero - rd["y"]) * inv
        row = dict(band=b.name, x=rd["x"] * inv, y=rd["y"] * inv, amp=cal.to_y_units(amp_px),
                   x_mm=rd["x"] * inv * cal.mm_per_px_x if cal.mm_per_px_x > 0 else None,
                   t_s=(rd["x"] - b.origin[0]) * inv * sp if sp > 0 else None,
                   x_arc_mm=None, t_arc_s=None)
        if arc_active(cal):
            xc = corrected_x_work(rd["x"], rd["y"], b.line_zero, self.scale, cal)
            row["x_arc_mm"] = xc * inv * cal.mm_per_px_x
            if sp > 0:
                row["t_arc_s"] = (xc - b.origin[0]) * inv * sp
        return row

    def remove_readout(self, i):
        if 0 <= i < len(self.readouts):
            del self.readouts[i]
            return True
        return False

    # ----------------------------------------------------------- sections
    def add_section(self, bi, x0_work, x1_work, name=None):
        """Add a measuring section (a range along x) to a band. Undoable."""
        b = self.bands[bi]
        lo, hi = sorted((float(x0_work), float(x1_work)))
        if hi - lo < 5:
            raise EngineError("区間が短すぎます")
        b.push_undo(self.params.undo_limit)
        b.sections.append(dict(name=name or f"区間{len(b.sections) + 1}", x0=lo, x1=hi))
        b.confirmed = False
        return b.sections[-1]

    def remove_section(self, bi, si):
        b = self.bands[bi]
        if 0 <= si < len(b.sections):
            b.push_undo(self.params.undo_limit)
            del b.sections[si]
            return True
        return False

    def section_result(self, bi, si):
        b = self.bands[bi]
        sec = b.sections[si]
        r = section_stats(b, sec["x0"], sec["x1"], self.scale, self.calibration, self.section_settings)
        r["name"] = sec["name"]
        r["band"] = b.name
        r["status"] = self.value_status(b)
        return r

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
        if not b.traced:
            return "未解析"
        if b.pending:
            return "未適用の修正あり"
        if self.is_determinate(b):
            return "確認済み"
        if not b.complete:
            return "候補（末端まで未追跡）"
        return "候補（未確認）"
