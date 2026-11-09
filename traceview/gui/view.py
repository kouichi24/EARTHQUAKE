"""Graphics view and items for the comparison canvas (UI-02, UI-03)."""
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QPainter, QPainterPath, QPen
import cv2
import numpy as np
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (QGraphicsEllipseItem, QGraphicsItem, QGraphicsLineItem, QGraphicsPathItem,
                               QGraphicsView)


class ImageView(QGraphicsView):
    sceneClicked = Signal(QPointF)
    cursorMoved = Signal(QPointF)
    zoomChanged = Signal()
    applyRequested = Signal()
    deleteRequested = Signal()
    escapeRequested = Signal()
    dragMoved = Signal(QPointF, QPointF)
    dragFinished = Signal(QPointF, QPointF)

    def __init__(self, scene):
        super().__init__(scene)
        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setBackgroundBrush(QBrush(QColor("#2b2b2b")))
        self.setMouseTracking(True)
        self.click_tool = False      # when True a left click is reported instead of panning
        self.drag_tool = False       # when True a left drag is reported (section tool) instead of panning
        self.drag_over_items = False # when True a drag that starts on a vertex / line moves that item instead
        self._drag_start = None
        self._press_pos = None

    def wheelEvent(self, ev):
        f = 1.25 if ev.angleDelta().y() > 0 else 0.8
        self.scale(f, f)
        self.zoomChanged.emit()

    def fit_all(self):
        r = self.scene().sceneRect()
        if r.isValid():
            self.fitInView(r, Qt.KeepAspectRatio)
            self.zoomChanged.emit()

    def center_on_point(self, x, y, zoom=None):
        if zoom:
            self.resetTransform()
            self.scale(zoom, zoom)
        self.centerOn(x, y)
        self.zoomChanged.emit()

    def mousePressEvent(self, ev):
        self._press_pos = ev.position()
        if self.drag_tool and ev.button() == Qt.LeftButton:
            item = self.itemAt(ev.position().toPoint())
            if item is not None and (item.flags() & item.GraphicsItemFlag.ItemIsMovable) and self.drag_over_items:
                self.setDragMode(QGraphicsView.NoDrag)      # a vertex / line under the mouse: it handles the drag itself
                super().mousePressEvent(ev)
                return
            self.setDragMode(QGraphicsView.NoDrag)
            self._drag_start = self.mapToScene(ev.position().toPoint())
            return
        if self.click_tool and ev.button() == Qt.LeftButton:
            item = self.itemAt(ev.position().toPoint())
            if item is None or not (item.flags() & item.GraphicsItemFlag.ItemIsMovable):
                self.setDragMode(QGraphicsView.NoDrag)
        super().mousePressEvent(ev)

    def mouseReleaseEvent(self, ev):
        if self.drag_tool and self._drag_start is not None and ev.button() == Qt.LeftButton:
            a, b = self._drag_start, self.mapToScene(ev.position().toPoint())
            self._drag_start = None
            self.setDragMode(QGraphicsView.ScrollHandDrag)
            self.dragFinished.emit(a, b)
            return
        was_click = (self._press_pos is not None and ev.button() == Qt.LeftButton and
                     (ev.position() - self._press_pos).manhattanLength() < 4)
        super().mouseReleaseEvent(ev)
        if self.click_tool and was_click:
            self.sceneClicked.emit(self.mapToScene(ev.position().toPoint()))
        self.setDragMode(QGraphicsView.ScrollHandDrag)

    def keyPressEvent(self, ev):
        if ev.key() in (Qt.Key_Return, Qt.Key_Enter):       # Enter in the image area applies the pending edit
            self.applyRequested.emit()
            return
        if ev.key() in (Qt.Key_Delete, Qt.Key_Backspace):
            self.deleteRequested.emit()
            return
        if ev.key() == Qt.Key_Escape:
            self.escapeRequested.emit()
            return
        super().keyPressEvent(ev)

    def mouseMoveEvent(self, ev):
        self.cursorMoved.emit(self.mapToScene(ev.position().toPoint()))
        if self.drag_tool and self._drag_start is not None:
            self.dragMoved.emit(self._drag_start, self.mapToScene(ev.position().toPoint()))
            return
        super().mouseMoveEvent(ev)


class PyramidItem(QGraphicsItem):
    """The sheet image for display. Smaller copies (1/2, 1/4, 1/8) are made once, and the one that fits the current
    zoom is drawn, so panning and zooming stay light even for a very large scan. Analysis always uses the full image."""

    LEVELS = 4

    def __init__(self, rgb):
        super().__init__()
        self.h, self.w = rgb.shape[:2]
        self.pixmaps = []
        img = np.ascontiguousarray(rgb)
        for k in range(self.LEVELS):
            if k:
                img = cv2.resize(img, (max(1, img.shape[1] // 2), max(1, img.shape[0] // 2)), interpolation=cv2.INTER_AREA)
            qi = QImage(img.data, img.shape[1], img.shape[0], 3 * img.shape[1], QImage.Format_RGB888)
            self.pixmaps.append(QPixmap.fromImage(qi))        # copies the data
        self.last_level = 0
        self.setZValue(0)

    def boundingRect(self):
        return QRectF(0, 0, self.w, self.h)

    @staticmethod
    def level_for_scale(s, levels=LEVELS):
        """Pyramid level for a zoom of s device px per image px: the smallest copy that is still sharp enough."""
        k = 0
        while k + 1 < levels and s < 0.75 / (2 ** (k + 1)) * 2:
            k += 1
        return k

    def paint(self, painter, option, widget=None):
        s = abs(painter.worldTransform().m11())
        k = self.level_for_scale(s)
        self.last_level = k
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
        pm = self.pixmaps[k]
        painter.drawPixmap(QRectF(0, 0, self.w, self.h), pm, QRectF(0, 0, pm.width(), pm.height()))


def polyline_path(pts, step=1):
    path = QPainterPath()
    if len(pts) == 0:
        return path
    path.moveTo(float(pts[0][0]), float(pts[0][1]))
    for x, y in pts[1::step]:
        path.lineTo(float(x), float(y))
    return path


class VertexItem(QGraphicsEllipseItem):
    """Draggable vertex marker. Keeps constant size on screen."""

    def __init__(self, win, band_i, vi, x, y, color, r=4):
        super().__init__(-r, -r, 2 * r, 2 * r)
        self.win, self.band_i, self.vi = win, band_i, vi
        self.setPos(x, y)
        self.setBrush(QBrush(QColor(color)))
        self.setPen(QPen(QColor("black"), 1))
        self.setFlag(QGraphicsEllipseItem.ItemIgnoresTransformations, True)
        self.setFlag(QGraphicsEllipseItem.ItemSendsGeometryChanges, True)
        self._lc = None            # LineConstraint while dragging along the image's line
        self._dragging = False
        self.setZValue(30)
        self.setCursor(Qt.PointingHandCursor)

    def set_editable(self, on):
        self.setFlag(QGraphicsEllipseItem.ItemIsMovable, on)

    def mousePressEvent(self, ev):
        additive = bool(ev.modifiers() & (Qt.ControlModifier | Qt.ShiftModifier))
        self.win.on_vertex_selected(self.band_i, self.vi, from_canvas=True, additive=additive)
        if (self.flags() & QGraphicsEllipseItem.ItemIsMovable) and self.win.follow_line:
            self._lc = self.win.make_line_constraint(self.pos().x(), self.pos().y(), self.band_i)
        self._dragging = True
        super().mousePressEvent(ev)

    def itemChange(self, change, value):
        if change == QGraphicsEllipseItem.ItemPositionChange and self._dragging and self._lc is not None:
            sx, sy = self._lc.snap(value.x(), value.y())
            self.win.show_drag_bridge(self._lc.last_bridge)        # dashed link while the point is taken across a gap
            return QPointF(sx, sy)
        return super().itemChange(change, value)

    def mouseReleaseEvent(self, ev):
        moved = self.flags() & QGraphicsEllipseItem.ItemIsMovable
        super().mouseReleaseEvent(ev)
        self._dragging = False
        self._lc = None
        self.win.show_drag_bridge(None)
        if moved:
            self.win.on_vertex_dragged(self.band_i, self.vi, self.pos().x(), self.pos().y())


class EndHandleItem(VertexItem):
    """Handle at the end of a band's trace: dragging it takes the end across a gap to the line beyond and goes on."""

    def __init__(self, win, band_i, x, y, color):
        super().__init__(win, band_i, -1, x, y, color, r=7)
        self.setBrush(QBrush(QColor("white")))
        self.setPen(QPen(QColor(color), 3))
        self.setToolTip("トレースの末端。ドラッグして、途切れの先にある線まで連れていくと、そこから先を追跡します")
        self.setZValue(32)

    def mousePressEvent(self, ev):
        if (self.flags() & QGraphicsEllipseItem.ItemIsMovable) and self.win.follow_line:
            self._lc = self.win.make_line_constraint(self.pos().x(), self.pos().y(), self.band_i)
        self._dragging = True
        QGraphicsEllipseItem.mousePressEvent(self, ev)

    def mouseReleaseEvent(self, ev):
        moved = self.flags() & QGraphicsEllipseItem.ItemIsMovable
        QGraphicsEllipseItem.mouseReleaseEvent(self, ev)
        self._dragging = False
        self._lc = None
        self.win.show_drag_bridge(None)
        if moved:
            self.win.on_end_dragged(self.band_i, self.pos().x(), self.pos().y())


class LineItem(QGraphicsLineItem):
    """Horizontal measuring line spanning the image; can only be moved up/down (MS-01)."""

    def __init__(self, win, band_i, kind, x0, x1, y, color, dashed):
        super().__init__(x0, 0, x1, 0)
        self.win, self.band_i, self.kind = win, band_i, kind
        self._x0, self._x1 = x0, x1
        pen = QPen(QColor(color), 2)
        pen.setCosmetic(True)
        if dashed:
            pen.setStyle(Qt.DashLine)
        self.setPen(pen)
        self.setPos(0, y)
        self.setFlag(QGraphicsLineItem.ItemIsMovable, True)
        self.setFlag(QGraphicsLineItem.ItemSendsGeometryChanges, True)
        self.setZValue(20)
        self.setCursor(Qt.SizeVerCursor)
        self.setToolTip(kind)

    def itemChange(self, change, value):
        if change == QGraphicsLineItem.ItemPositionChange:
            return QPointF(0, value.y())
        return super().itemChange(change, value)

    def shape(self):
        p = QPainterPath()
        p.addRect(QRectF(self._x0, -9, self._x1 - self._x0, 18))
        return p

    def mouseReleaseEvent(self, ev):
        super().mouseReleaseEvent(ev)
        self.win.on_line_moved(self.band_i, self.kind, self.pos().y())


def make_path_item(path, color, width=2, dashed=False, z=10):
    it = QGraphicsPathItem(path)
    pen = QPen(QColor(color), width)
    pen.setCosmetic(True)
    if dashed:
        pen.setStyle(Qt.DashLine)
    it.setPen(pen)
    it.setZValue(z)
    return it
