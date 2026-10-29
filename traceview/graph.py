"""Skeleton -> graph (nodes = end points / junction clusters, edges = pixel chains)."""
from dataclasses import dataclass, field

import cv2
import numpy as np
from skimage.morphology import skeletonize

_NB = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]  # circular order (dy, dx)


@dataclass
class Edge:
    a: int
    b: int
    pix: np.ndarray            # (N, 2) int32 (x, y), ordered from node a to node b
    visits: int = 0
    inferred: bool = False      # virtual edge bridging a gap (estimated connection)

    @property
    def length(self):
        return len(self.pix)


@dataclass
class Node:
    x: float
    y: float
    edges: list = field(default_factory=list)   # edge ids
    kind: str = "junction"                      # "end" | "junction"


class Graph:
    def __init__(self):
        self.nodes = []
        self.edges = []
        self.skel = None
        self.pix_edge = None   # (H, W) int32: edge id of interior pixels, -1 elsewhere
        self.pix_node = None   # (H, W) int32: node id of node pixels, -1 elsewhere

    def other(self, eid, nid):
        e = self.edges[eid]
        return e.b if e.a == nid else e.a


def _crossing_number(skel):
    p = np.pad(skel.astype(np.uint8), 1)
    h, w = skel.shape
    nb = [p[1 + dy:1 + dy + h, 1 + dx:1 + dx + w] for dy, dx in _NB]
    cn = np.zeros(skel.shape, np.int16)
    for i in range(8):
        cn += ((nb[i] == 0) & (nb[(i + 1) % 8] == 1))
    cnt = sum(n.astype(np.int16) for n in nb)
    return cn, cnt


def skeleton_of(ink):
    return skeletonize(ink)


def build_graph(skel, spur_min_len=6, prune_passes=2):
    skel = skel.copy()
    g = None
    for _ in range(prune_passes + 1):
        g = _build(skel)
        removed = False
        for e in g.edges:
            na, nb = g.nodes[e.a], g.nodes[e.b]
            dead = (na.kind == "end") != (nb.kind == "end")
            if dead and e.length < spur_min_len:
                pts = e.pix
                # keep the junction-side pixel so the remaining line stays connected
                keep = 0 if na.kind == "junction" else len(pts) - 1
                for i, (x, y) in enumerate(pts):
                    if i != keep:
                        skel[y, x] = False
                removed = True
        if not removed:
            break
    if removed:
        g = _build(skel)
    return g


def _build(skel):
    h, w = skel.shape
    cn, cnt = _crossing_number(skel)
    end = skel & (cnt == 1)
    junc = skel & ((cn >= 3) | (cnt >= 3) & (cn >= 3))
    g = Graph()
    g.skel = skel
    g.pix_node = np.full((h, w), -1, np.int32)
    g.pix_edge = np.full((h, w), -1, np.int32)

    # junction clusters
    n_j, lab = cv2.connectedComponents(junc.astype(np.uint8), connectivity=8)
    if n_j > 1:
        ys, xs = np.nonzero(lab)
        order = np.argsort(lab[ys, xs])
        ys, xs = ys[order], xs[order]
        ids = lab[ys, xs]
        bounds = np.flatnonzero(np.diff(ids)) + 1
        for yy, xx in zip(np.split(ys, bounds), np.split(xs, bounds)):
            nid = len(g.nodes)
            g.nodes.append(Node(float(xx.mean()), float(yy.mean()), [], "junction"))
            g.pix_node[yy, xx] = nid
    ys, xs = np.nonzero(end)
    for y, x in zip(ys, xs):
        nid = len(g.nodes)
        g.nodes.append(Node(float(x), float(y), [], "end"))
        g.pix_node[y, x] = nid

    def neighbours(y, x):
        for dy, dx in _NB:
            yy, xx = y + dy, x + dx
            if 0 <= yy < h and 0 <= xx < w and skel[yy, xx]:
                yield yy, xx

    node_pix = np.argwhere(g.pix_node >= 0)
    for y, x in node_pix:
        nid = g.pix_node[y, x]
        for yy, xx in neighbours(y, x):
            if g.pix_node[yy, xx] == nid:
                continue
            if g.pix_edge[yy, xx] >= 0:
                continue
            if g.pix_node[yy, xx] >= 0:
                # two adjacent nodes: direct edge, add once
                other = g.pix_node[yy, xx]
                if nid < other:
                    eid = len(g.edges)
                    g.edges.append(Edge(nid, other, np.array([(x, y), (xx, yy)], np.int32)))
                    g.nodes[nid].edges.append(eid)
                    g.nodes[other].edges.append(eid)
                continue
            chain = [(x, y)]
            prev = (y, x)
            cur = (yy, xx)
            end_node = -1
            while True:
                chain.append((cur[1], cur[0]))
                g.pix_edge[cur] = len(g.edges)
                nxt = None
                nbrs = [q for q in neighbours(*cur) if q != prev]
                # a node pixel next to the current pixel always wins over a staircase neighbour
                for ny, nx in nbrs:
                    if g.pix_node[ny, nx] >= 0 and not (g.pix_node[ny, nx] == nid and len(chain) < 4):
                        end_node = g.pix_node[ny, nx]
                        nxt = (ny, nx)
                        break
                if nxt is None:
                    for ny, nx in nbrs:
                        if g.pix_node[ny, nx] >= 0 or g.pix_edge[ny, nx] >= 0:
                            continue
                        if max(abs(ny - prev[0]), abs(nx - prev[1])) <= 1:
                            continue  # staircase neighbour, already adjacent to previous
                        nxt = (ny, nx)
                        break
                if nxt is None:
                    break
                if end_node >= 0:
                    chain.append((nxt[1], nxt[0]))
                    break
                prev, cur = cur, nxt
            if end_node < 0:
                # dangling chain without node (should be rare): close on itself as an end node
                end_node = len(g.nodes)
                g.nodes.append(Node(float(chain[-1][0]), float(chain[-1][1]), [], "end"))
            eid = len(g.edges)
            g.edges.append(Edge(int(nid), int(end_node), np.array(chain, np.int32)))
            g.nodes[nid].edges.append(eid)
            g.nodes[end_node].edges.append(eid)
    return g


def bridge_gaps(ink, g, max_gap=40, cos_min=0.92, thickness=3, max_vy=0.45):
    """Join broken thin lines: end points are linked to a skeleton pixel straight ahead.

    Returns (ink_with_bridges, bridge_mask). Bridge pixels are later reported as
    estimated connections (TR-06).
    """
    h, w = ink.shape
    bridge = np.zeros(ink.shape, np.uint8)
    for nid, n in enumerate(g.nodes):
        if n.kind != "end" or not n.edges:
            continue
        e = g.edges[n.edges[0]]
        pts = e.pix if e.a == nid else e.pix[::-1]     # starts at the end node
        k = min(len(pts) - 1, 14)
        if k < 4:
            continue
        v = (pts[0] - pts[k]).astype(float)           # pointing out of the line
        nv = np.hypot(*v)
        if nv == 0:
            continue
        v /= nv
        if abs(v[1]) > max_vy:      # spike tips point up/down: never bridge those
            continue
        x, y = int(n.x), int(n.y)
        xa, xb = max(0, x - max_gap), min(w, x + max_gap + 1)
        ya, yb = max(0, y - max_gap), min(h, y + max_gap + 1)
        yy, xx = np.nonzero(g.skel[ya:yb, xa:xb])
        if len(yy) == 0:
            continue
        yy = yy + ya
        xx = xx + xa
        own = g.pix_edge[yy, xx] == n.edges[0]
        d = np.stack([xx - x, yy - y], 1).astype(float)
        dist = np.hypot(d[:, 0], d[:, 1])
        ok = (~own) & (dist > 2) & (dist <= max_gap)
        ok &= (d @ v) / np.maximum(dist, 1e-6) >= cos_min
        if not ok.any():
            continue
        i = np.flatnonzero(ok)[np.argmin(dist[ok])]
        cv2.line(bridge, (x, y), (int(xx[i]), int(yy[i])), 1, thickness)
    out = ink | (bridge > 0)
    return out, bridge > 0


def add_virtual_edges(g, max_gap=150, cos_out=0.8, cos_in=0.5, max_dy_frac=0.45):
    """Link broken thin lines: an end point that looks rightwards is joined to the nearest
    end point ahead (an estimated connection, TR-06). Spike tips point up/down and never qualify."""
    ends = []
    for nid, n in enumerate(g.nodes):
        if n.kind != "end" or not n.edges:
            continue
        e = g.edges[n.edges[0]]
        pts = e.pix if e.a == nid else e.pix[::-1]
        k = min(len(pts) - 1, 14)
        if k < 4:
            continue
        v = (pts[0] - pts[k]).astype(float)
        nv = np.hypot(*v)
        if nv == 0:
            continue
        ends.append((nid, v / nv))
    if not ends:
        return 0
    xy = np.array([[g.nodes[i].x, g.nodes[i].y] for i, _ in ends])
    added = 0
    used = set()
    for ai, (a, va) in enumerate(ends):
        if abs(va[1]) > 0.5 or va[0] <= 0:       # only rightwards-looking thin line ends
            continue
        d = xy - xy[ai]
        dist = np.hypot(d[:, 0], d[:, 1])
        best, bd = None, 1e18
        for bi, (b, vb) in enumerate(ends):
            if bi == ai or dist[bi] < 2 or dist[bi] > max_gap or b in used:
                continue
            u = d[bi] / dist[bi]
            if float(u @ va) < cos_out:
                continue
            if abs(d[bi, 1]) > max_dy_frac * max(dist[bi], 1):
                continue
            if float(-u @ vb) < cos_in:           # b must open towards a
                continue
            if g.nodes[b].edges and g.edges[g.nodes[b].edges[0]] is g.edges[g.nodes[a].edges[0]]:
                continue
            if dist[bi] < bd:
                bd, best = dist[bi], bi
        if best is None:
            continue
        b = ends[best][0]
        n_pts = int(max(2, bd))
        xs = np.round(np.linspace(g.nodes[a].x, g.nodes[b].x, n_pts)).astype(np.int32)
        ys = np.round(np.linspace(g.nodes[a].y, g.nodes[b].y, n_pts)).astype(np.int32)
        eid = len(g.edges)
        g.edges.append(Edge(a, b, np.stack([xs, ys], 1), inferred=True))
        g.nodes[a].edges.append(eid)
        g.nodes[b].edges.append(eid)
        used.add(a)
        used.add(b)
        added += 1
    return added
