"""A separate window that draws the lines again from the numbers of the slices only (no image)."""
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush, QColor, QPainterPath, QPen
from PySide6.QtWidgets import (QCheckBox, QGraphicsScene, QGraphicsSimpleTextItem, QHBoxLayout, QLabel, QListWidget,
                               QListWidgetItem, QMainWindow, QVBoxLayout, QWidget)

from .view import ImageView, make_path_item

PALETTE = ["#e91e63", "#2e7d32", "#1565c0", "#ef6c00", "#6a1b9a", "#00838f"]


class SliceWindow(QMainWindow):
    """Left: list of slices. Right: the lines drawn from the numbers of the chosen slices (all of them when none chosen)."""

    def __init__(self, rows, unit="px", step=0.0, parent=None, colors=None):
        super().__init__(parent)
        self.setWindowTitle("分割データからの再描画")
        self.resize(1100, 700)
        self.rows, self.unit, self.step = rows, unit, step
        self.colors = colors or {}
        self.scene = QGraphicsScene(self)
        self.view = ImageView(self.scene)
        self.items = []
        # bands in order of first appearance
        seen = []
        for r in rows:
            if r["band"] not in seen:
                seen.append(r["band"])
        self.bands = seen
        self.slices = sorted({r["slice"] for r in rows})
        # index once: (band, slice) -> points in drawing order; slice -> smallest x
        self.pts = {}
        self.slice_x0 = {}
        for r in rows:
            self.pts.setdefault((r["band"], r["slice"]), []).append(r)
            self.slice_x0[r["slice"]] = min(r["x_px"], self.slice_x0.get(r["slice"], r["x_px"]))
        for v in self.pts.values():
            v.sort(key=lambda r: r["idx"])
        self.y_lo = min(r["y_px"] for r in rows)
        self.y_hi = max(r["y_px"] for r in rows)
        # ---- layout
        side = QWidget()
        sl = QVBoxLayout(side)
        self.info = QLabel("")
        self.info.setWordWrap(True)
        sl.addWidget(self.info)
        self.band_checks = {}
        for n in self.bands:
            cb = QCheckBox(n)
            cb.setChecked(True)
            cb.toggled.connect(self.redraw)
            sl.addWidget(cb)
            self.band_checks[n] = cb
        self.list = QListWidget()
        self.list.setSelectionMode(QListWidget.ExtendedSelection)
        counts = {}
        ranges = {}
        for r in rows:
            counts[r["slice"]] = counts.get(r["slice"], 0) + 1
            ranges[r["slice"]] = (r["slice_start"], r["slice_end"])
        for s in self.slices:
            a, b = ranges[s]
            it = QListWidgetItem(f"区切り{s}：{a:g}〜{b:g} {unit}（{counts[s]}点）")
            it.setData(Qt.UserRole, s)
            self.list.addItem(it)
        self.list.itemSelectionChanged.connect(self.redraw)
        _hint = QLabel("区切りを選ぶと、その区切りだけを描きます（選ばないと全体）")
        _hint.setWordWrap(True)
        sl.addWidget(_hint)
        sl.addWidget(self.list, 1)
        self.show_grid = QCheckBox("区切りの境目を表示")
        self.show_grid.setChecked(True)
        self.show_grid.toggled.connect(self.redraw)
        sl.addWidget(self.show_grid)
        root = QWidget()
        hl = QHBoxLayout(root)
        side.setMaximumWidth(340)
        hl.addWidget(side)
        hl.addWidget(self.view, 1)
        self.setCentralWidget(root)
        self.redraw()
        self.fit()

    def fit(self):
        r = self.scene.itemsBoundingRect()
        if r.width() > 0:
            self.view.fitInView(r.adjusted(-20, -20, 20, 20), Qt.KeepAspectRatio)

    def chosen_slices(self):
        sel = [i.data(Qt.UserRole) for i in self.list.selectedItems()]
        return sel or list(self.slices)

    def redraw(self, *_):
        for it in self.items:
            if it.scene() is self.scene:
                self.scene.removeItem(it)
        self.items = []
        chosen = set(self.chosen_slices())
        n_pts = 0
        for bi, name in enumerate(self.bands):
            if not self.band_checks[name].isChecked():
                continue
            col = self.colors.get(name, PALETTE[bi % len(PALETTE)])
            for s in sorted(chosen):
                rs = self.pts.get((name, s), [])
                if len(rs) < 2:
                    continue
                p = QPainterPath()
                p.moveTo(rs[0]["x_px"], rs[0]["y_px"])
                for a, r in zip(rs[:-1], rs[1:]):
                    if r["idx"] - a["idx"] > 1:                # the pen left this slice and came back: do not join across
                        p.moveTo(r["x_px"], r["y_px"])
                    else:
                        p.lineTo(r["x_px"], r["y_px"])
                it = make_path_item(p, col, 2, z=10)
                self.scene.addItem(it)
                self.items.append(it)
                n_pts += len(rs)
        if self.show_grid.isChecked():
            for s in sorted(chosen):
                x0 = self.slice_x0.get(s)
                if x0 is None:
                    continue
                y0, y1 = self.y_lo, self.y_hi
                ln = make_path_item(self._vline(x0, y0, y1), "#888888", 1, dashed=True, z=5)
                self.scene.addItem(ln)
                self.items.append(ln)
                lab = QGraphicsSimpleTextItem(f"{s}")
                lab.setFlag(QGraphicsSimpleTextItem.ItemIgnoresTransformations, True)
                lab.setBrush(QBrush(QColor("#dddddd")))
                lab.setPos(x0 + 2, y0)
                lab.setZValue(6)
                self.scene.addItem(lab)
                self.items.append(lab)
        self.info.setText(f"{len(self.bands)}段 ／ {len(self.slices)}区切り（幅 {self.step:g} {self.unit}）／ 描画中 {n_pts}点\n"
                          "数値だけから描いています（画像は使っていません）。ホイールで拡大縮小、ドラッグで移動できます。")

    @staticmethod
    def _vline(x, y0, y1):
        p = QPainterPath()
        p.moveTo(x, y0)
        p.lineTo(x, y1)
        return p
