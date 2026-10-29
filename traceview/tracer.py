"""Pen-path walker over the skeleton graph (TR-01..TR-07).

The walker follows a stroke from the origin towards the right. It always goes to the
end of a branch (a spike tip), reverses there and comes back over the same pixels
(TR-02/03). At junctions it prefers an unvisited branch that continues in the same
direction; undecidable junctions are reported (TR-04).
"""
from dataclasses import dataclass, field

import numpy as np

K_HEAD = 14  # pixels used to estimate a heading


@dataclass
class TraceResult:
    path: np.ndarray                       # (N, 2) int32 (x, y) in pen order
    reversals: list = field(default_factory=list)   # indices where the walker turned at a dead end
    ambiguous: list = field(default_factory=list)   # (path_index, x, y) undecidable junctions
    end_reason: str = ""                   # "end" | "stuck" | "limit"
    end_x: float = 0.0
    inferred: list = field(default_factory=list)   # (start_idx, end_idx) spans that bridge image gaps


def _heading(pix, from_start, k=K_HEAD):
    """Unit vector of the first (or last) k pixels of an edge, pointing along the walk."""
    if len(pix) < 2:
        return np.zeros(2)
    if from_start:
        a, b = pix[0], pix[min(k, len(pix) - 1)]
    else:
        a, b = pix[-1], pix[max(len(pix) - 1 - k, 0)]
    v = (b - a).astype(float)
    n = np.hypot(*v)
    return v / n if n else v


class Walker:
    """Directed-edge walk: every skeleton edge is traversed at most once per direction."""

    def __init__(self, graph, params, corridor=None, spur_max=220, ref_zone=None):
        self.g = graph
        self.ref_zone = ref_zone
        self._refpen = {}
        self.p = params
        self.corridor = corridor   # (y_lo_soft, y_hi_soft, y_lo_hard, y_hi_hard) or None
        self.spur_max = spur_max

    def _score(self, eid, nid, in_head, visited):
        g = self.g
        e = g.edges[eid]
        out_flag = e.a == nid
        pix = e.pix if out_flag else e.pix[::-1]
        h = _heading(pix, True)
        s = float(np.dot(in_head, h)) if np.any(in_head) else float(h[0])
        s += 0.15 * h[0]
        far_id = g.other(eid, nid)
        far = g.nodes[far_id]
        if (eid, not out_flag) not in visited:
            s += 0.5                                   # untouched in both directions
        if far.kind == "end" and e.length <= self.spur_max:
            s += 0.9                                   # visit spike tips before moving on
        s -= self._ref_penalty(eid)
        if self.corridor is not None:
            lo_s, hi_s, lo_h, hi_h = self.corridor
            if far.y < lo_s or far.y > hi_s:
                s -= 0.4
            if far.y < lo_h or far.y > hi_h:
                s -= 0.8
        return s

    def _outside_edges(self):
        """Edges that lie in another band's territory are never entered (TR-01: no hopping to a neighbour)."""
        if self.corridor is None:
            return ()
        if getattr(self, "_outside", None) is None:
            lo, hi = self.corridor[0] - 20, self.corridor[1] + 20
            out = set()
            for i, e in enumerate(self.g.edges):
                if e.inferred:
                    continue
                y = e.pix[:, 1]
                if np.median(y) < lo or np.median(y) > hi:
                    out.add(i)
            self._outside = out
        return self._outside

    def _ref_penalty(self, eid):
        """Edges that run along a reference line are only taken when nothing else is possible."""
        if self.ref_zone is None:
            return 0.0
        v = self._refpen.get(eid)
        if v is None:
            pix = self.g.edges[eid].pix
            frac = float(self.ref_zone[pix[:, 1], pix[:, 0]].mean())
            v = 2.0 if frac > 0.6 and len(pix) > 12 else 0.0
            self._refpen[eid] = v
        return v

    def walk(self, start_node, first_edge, max_steps=None, blocked=None):
        """Depth-first pen walk. At a dead end (or when every way on is used) the walker
        goes back over the pixels it came by and continues at the last open junction."""
        g = self.g
        total = sum(e.length for e in g.edges) + 1
        max_steps = max_steps or 4 * total + 10
        visited = set()
        for b in self._outside_edges():
            visited.add((b, True))
            visited.add((b, False))
        for b in (blocked or ()):
            visited.add((b, True))
            visited.add((b, False))
        path = []
        node_arrivals = []          # (path_index, node_id)
        reversals, ambiguous = [], []
        stack = []                  # (eid, fwd, path pixels, from_node)
        inferred = []
        nid, eid = start_node, first_edge
        steps = 0
        reason = "exhausted"
        while steps < max_steps:
            e = g.edges[eid]
            fwd = e.a == nid
            visited.add((eid, fwd))
            pix = e.pix if fwd else e.pix[::-1]
            full_pix = pix
            if path and tuple(path[-1]) == tuple(pix[0]):
                pix = pix[1:]
            i0 = len(path)
            path.extend(map(tuple, pix))
            if e.inferred:
                inferred.append((i0, len(path) - 1))
            stack.append((eid, fwd, full_pix, nid))
            steps += len(pix)
            prev_nid = nid
            nid = e.b if fwd else e.a
            node = g.nodes[nid]
            node_arrivals.append((len(path) - 1, nid))
            in_head = -_heading(e.pix if fwd else e.pix[::-1], False)
            while True:
                cands = [c for c in node.edges
                         if (c, g.edges[c].a == nid) not in visited]
                # a loop edge that starts and ends here is skipped (cannot be oriented)
                cands = [c for c in cands if g.edges[c].a != g.edges[c].b]
                if cands:
                    break
                if node.kind == "end":
                    reversals.append(len(path) - 1)
                if not stack:
                    break
                # go back over the edge we arrived by
                beid, bfwd, bpix, bfrom = stack.pop()
                back = [tuple(q) for q in bpix[::-1][1:]]
                if g.edges[beid].inferred:
                    inferred.append((len(path), len(path) + len(back)))
                path.extend(back)
                steps += len(back)
                nid = bfrom
                node = g.nodes[nid]
                node_arrivals.append((len(path) - 1, nid))
                visited.add((beid, not bfwd))
                if stack:
                    peid, pfwd, _, _ = stack[-1]
                    pe = g.edges[peid]
                    in_head = -_heading(pe.pix if pfwd else pe.pix[::-1], False)
                if steps >= max_steps:
                    break
            else:
                pass
            if not cands:
                reason = "exhausted"
                break
            scored = sorted(((self._score(c, nid, in_head, visited), c) for c in cands), reverse=True)
            if len(scored) > 1 and scored[0][0] - scored[1][0] < self.p.ambiguity_margin:
                ambiguous.append((len(path) - 1, node.x, node.y))
            eid = scored[0][1]
        else:
            reason = "limit"
        arr = np.array(path, np.int32).reshape(-1, 2)
        res = TraceResult(arr, reversals, ambiguous, reason, float(arr[-1, 0]) if len(arr) else 0.0)
        res.node_arrivals = node_arrivals
        res.inferred = inferred
        return res

    def truncate_at_best_end(self, res, y_lo, y_hi):
        """Cut the walk where it first reaches its right-most point inside the band zone."""
        path = res.path
        if len(path) == 0:
            return res
        zone = (path[:, 1] >= y_lo) & (path[:, 1] <= y_hi)
        if not zone.any():
            return res
        xmax = path[zone, 0].max()
        idx = int(np.flatnonzero(zone & (path[:, 0] >= xmax - 1))[0])
        keep = idx + 1
        out = TraceResult(path[:keep], [r for r in res.reversals if r < keep],
                          [a for a in res.ambiguous if a[0] < keep], "end", float(path[idx, 0]))
        out.node_arrivals = [a for a in res.node_arrivals if a[0] < keep]
        out.inferred = [(a, min(b, keep - 1)) for a, b in res.inferred if a < keep]
        return out


def nearest_node_edge(graph, x, y, max_dist=60, prefer_end=True):
    """Nearest graph node (end points preferred) to (x, y) and one of its edges."""
    best, bd = None, 1e18
    for i, n in enumerate(graph.nodes):
        if not n.edges:
            continue
        d = np.hypot(n.x - x, n.y - y)
        if prefer_end and n.kind == "end":
            d *= 0.6
        if d < bd:
            bd, best = d, i
    if best is None or bd > max_dist * 1.0 + 1e-9 and bd / 0.6 > max_dist:
        return None, None
    return best, graph.nodes[best].edges[0]
