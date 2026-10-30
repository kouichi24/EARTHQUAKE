"""Graphics view and items for the comparison canvas (UI-02, UI-03)."""
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (QGraphicsEllipseItem, QGraphicsLineItem, QGraphicsPathItem,
                               QGraphicsView)


class ImageView(QGraphicsView):
    sceneClicked = Signal(QPointF)
    cursorMoved = Signal(QPointF)
    zoomChanged = Signal()
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

    def mouseMoveEvent(self, ev):
        self.cursorMoved.emit(self.mapToScene(ev.position().toPoint()))
        if self.drag_tool and self._drag_start is not None:
            self.dragMoved.emit(self._drag_start, self.mapToScene(ev.position().toPoint()))
            return
        super().mouseMoveEvent(ev)


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
        self.setFlag(QGraphicsEllipseItem.ItemIsSelectable, True)
        self.setZValue(30)
        self.setCursor(Qt.PointingHandCursor)

    def set_editable(self, on):
        self.setFlag(QGraphicsEllipseItem.ItemIsMovable, on)

    def mousePressEvent(self, ev):
        self.win.on_vertex_selected(self.band_i, self.vi, from_canvas=True)
        super().mousePressEvent(ev)

    def mouseReleaseEvent(self, ev):
        moved = self.flags() & QGraphicsEllipseItem.ItemIsMovable
        super().mouseReleaseEvent(ev)
        if moved:
            self.win.on_vertex_dragged(self.band_i, self.vi, self.pos().x(), self.pos().y())


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
