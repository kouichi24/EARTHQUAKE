"""TraceView main window: load -> auto trace -> check against the image -> edit -> Excel (UI-01..05)."""
import os
import time
import traceback

import cv2
import numpy as np
from PySide6.QtCore import QObject, QPointF, QSettings, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QAction, QBrush, QColor, QImage, QKeySequence, QPen, QPixmap
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QCheckBox, QComboBox, QDialog,
                               QDialogButtonBox, QDockWidget, QDoubleSpinBox, QFileDialog, QFormLayout,
                               QGraphicsEllipseItem, QGraphicsPixmapItem, QGraphicsRectItem, QGraphicsScene, QGraphicsSimpleTextItem, QGroupBox,
                               QHBoxLayout, QInputDialog, QLabel, QListWidget, QListWidgetItem,
                               QMainWindow, QMessageBox, QProgressDialog, QPushButton, QSlider,
                               QSpinBox, QTableWidget, QTableWidgetItem, QTabWidget, QToolBar,
                               QVBoxLayout, QWidget)

from .. import __version__
from ..engine import Engine, EngineError
from ..excel_out import PendingEditsError, export_xlsx
from ..imageio import page_count
from ..measure import COUNT_MODES, Calibration
from ..profiles import PROFILES, get as get_profile
from ..project import SourceMismatch, load_project, save_project
from ..svg_out import export_svg
from .view import ImageView, LineItem, VertexItem, make_path_item, polyline_path

ISSUE_LABEL = {"ambiguous": "所属の曖昧さ", "untracked_end": "末端の未追跡", "inferred": "推定接続",
               "excursion": "他の段への逸脱", "no_origin": "起点未検出", "retrace_failed": "再追跡の失敗"}
FILE_FILTER = "波形画像 (*.pdf *.png *.jpg *.jpeg *.tif *.tiff *.bmp *.webp);;すべてのファイル (*)"


class Cancelled(Exception):
    pass


class Worker(QThread):
    progress = Signal(str, float)
    done = Signal(object)
    failed = Signal(str)

    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self.cancel = False

    def _cb(self, msg, frac):
        if self.cancel:
            raise Cancelled()
        self.progress.emit(msg, frac)

    def run(self):
        try:
            self.done.emit(self.fn(self._cb))
        except Cancelled:
            self.failed.emit("中止しました")
        except Exception as e:  # noqa: BLE001 - shown to the user
            if not isinstance(e, (EngineError, PendingEditsError, SourceMismatch)):
                traceback.print_exc()
            self.failed.emit(str(e))


class MainWindow(QMainWindow):
    def __init__(self, sync_mode=False):
        super().__init__()
        self.sync_mode = sync_mode          # run tasks inline (used by automated checks)
        self.engine = Engine()
        self.setWindowTitle(f"TraceView {__version__}")
        self.resize(1500, 900)
        self.scene = QGraphicsScene(self)
        self.view = ImageView(self.scene)
        self.setCentralWidget(self.view)
        self.pix_item = None
        self.band_items = {}       # bi -> dict(path, inferred[], vertices[], lines{}, issues[], pending[])
        self.sel_band = 0
        self.sel_vertex = None
        self.sel_ring = None
        self.tool = "select"
        self.calib_pts = []
        self.calib_axis = None
        self.dirty = False
        self.project_path = None
        self.worker = None
        self.auto_retrace = True
        self.replay_idx = 0
        self.replay_marker = None
        self.replay_timer = QTimer(self)
        self.replay_timer.timeout.connect(self._replay_tick)
        self.state = {"読込": "—", "解析": "—", "再追跡": "—", "保存": "—"}
        self._build_actions()
        self._build_docks()
        self._build_status()
        self.view.sceneClicked.connect(self.on_scene_clicked)
        self.view.cursorMoved.connect(self._on_cursor)
        self.view.zoomChanged.connect(self.apply_visibility)
        self.view.dragMoved.connect(self._section_drag_preview)
        self.view.dragFinished.connect(self._section_drag_done)
        self.section_preview = None
        self._update_enabled()
        self._set_state("読込", "ファイル未選択")

    # ------------------------------------------------------------------ UI build
    def _act(self, text, slot, shortcut=None, checkable=False, tip=None):
        a = QAction(text, self)
        a.triggered.connect(slot)
        if shortcut:
            a.setShortcut(QKeySequence(shortcut))
        a.setCheckable(checkable)
        if tip:
            a.setToolTip(tip)
        return a

    def _build_actions(self):
        tb = QToolBar("ファイル")
        tb.setMovable(False)
        self.addToolBar(tb)
        self.a_open = self._act("開く…", self.action_open, "Ctrl+O", tip="PDF・画像を開く")
        self.page_spin = QSpinBox()
        self.page_spin.setPrefix("ページ ")
        self.page_spin.setMinimum(1)
        self.page_spin.setMaximum(1)
        self.page_spin.setToolTip("PDF・TIFFのページ選択（変更すると読み込み直します）")
        self.page_spin.editingFinished.connect(self._page_changed)
        self.a_analyze = self._act("自動解析", self.action_analyze, "F5", tip="三段の起点と波形を自動取得")
        self.profile_combo = QComboBox()
        for pr in PROFILES:
            self.profile_combo.addItem(pr.name, pr.key)
        self.profile_combo.setToolTip("記録方式の設定。方式が分からないときは「方式を探す」で数種類を試せます")
        self.profile_combo.setMaximumWidth(280)
        self.profile_combo.activated.connect(self._profile_chosen)
        self.a_compare = self._act("方式を探す…", self.action_compare, "F6",
                                   tip="記録方式の設定を数種類試し、重ね表示の見た目で選ぶ")
        self.a_save = self._act("作業を保存", self.action_save, "Ctrl+S")
        self.a_load = self._act("作業を開く…", self.action_load_project)
        self.a_xlsx = self._act("Excel出力…", self.action_export_xlsx)
        self.a_svg = self._act("トレース出力(SVG)…", self.action_export_svg)
        for a in (self.a_open,):
            tb.addAction(a)
        tb.addWidget(self.page_spin)
        tb.addWidget(self.profile_combo)
        for a in (self.a_analyze, self.a_compare, self.a_save, self.a_load, self.a_xlsx, self.a_svg):
            tb.addAction(a)
        tb.addSeparator()
        # editing tools
        self.addToolBarBreak()
        tb2 = QToolBar("編集")
        tb2.setMovable(False)
        self.addToolBar(tb2)
        self.t_select = self._act("選択・移動", lambda: self.set_tool("select"), "S", True,
                                  "頂点をドラッグして移動／線をドラッグして測定線を調整")
        self.t_add = self._act("頂点を追加", lambda: self.set_tool("add"), "A", True,
                               "取り逃した先端をクリックして追加")
        self.t_origin = self._act("起点を指定", lambda: self.set_tool("origin"), "O", True,
                                  "自動検出に失敗した段の起点をクリックして指定")
        self.t_section = self._act("区間を指定", lambda: self.set_tool("section"), "R", True,
                                   "画像上で横にドラッグして、測定する範囲（区間）を指定")
        for a in (self.t_select, self.t_add, self.t_origin, self.t_section):
            tb2.addAction(a)
        tb2.addSeparator()
        self.a_apply = self._act("再追跡を適用", self.action_apply, "Ctrl+Return")
        self.a_cancel_pending = self._act("修正を取り消す", self.action_cancel_pending, "Esc")
        self.cb_auto = QCheckBox("移動後に自動で再追跡")
        self.cb_auto.setChecked(True)
        self.cb_auto.toggled.connect(lambda v: setattr(self, "auto_retrace", v))
        tb2.addAction(self.a_apply)
        tb2.addAction(self.a_cancel_pending)
        tb2.addWidget(self.cb_auto)
        tb2.addSeparator()
        self.a_undo = self._act("戻す", self.action_undo, "Ctrl+Z")
        self.a_redo = self._act("やり直す", self.action_redo, "Ctrl+Y")
        self.a_reset = self._act("この段を初回結果へ", self.action_reset_band)
        for a in (self.a_undo, self.a_redo, self.a_reset):
            tb2.addAction(a)
        # view toolbar
        self.addToolBarBreak()
        tb3 = QToolBar("表示")
        tb3.setMovable(False)
        self.addToolBar(tb3)
        self.cb_image = QCheckBox("元画像")
        self.cb_image.setChecked(True)
        self.cb_trace = QCheckBox("トレース")
        self.cb_trace.setChecked(True)
        self.cb_vertices = QCheckBox("頂点")
        self.cb_vertices.setChecked(True)
        self.cb_lines = QCheckBox("測定線")
        self.cb_lines.setChecked(True)
        self.cb_issues = QCheckBox("要確認印")
        self.cb_issues.setChecked(True)
        for cb in (self.cb_image, self.cb_trace, self.cb_vertices, self.cb_lines, self.cb_issues):
            cb.toggled.connect(self.apply_visibility)
            tb3.addWidget(cb)
        tb3.addSeparator()
        self.a_fit = self._act("全体表示", self.view.fit_all, "Ctrl+0")
        self.a_zoom_sel = self._act("選択頂点へ拡大", self.zoom_to_selected, "Z")
        self.a_play = self._act("再生", self.toggle_replay, "Space", True, "復元順に経路を再生")
        self.a_prev_v = self._act("◀頂点", lambda: self.step_vertex(-1), "Left")
        self.a_next_v = self._act("頂点▶", lambda: self.step_vertex(1), "Right")
        for a in (self.a_fit, self.a_zoom_sel, self.a_play, self.a_prev_v, self.a_next_v):
            tb3.addAction(a)
        self.speed = QSlider(Qt.Horizontal)
        self.speed.setRange(1, 100)
        self.speed.setValue(30)
        self.speed.setMaximumWidth(110)
        self.speed.setToolTip("再生速度")
        tb3.addWidget(self.speed)

    def _build_docks(self):
        # band list + tabs
        dock = QDockWidget("波形・測定・確認", self)
        dock.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        w = QWidget()
        lay = QVBoxLayout(w)
        self.band_list = QListWidget()
        self.band_list.setMaximumHeight(96)
        self.band_list.currentRowChanged.connect(self.on_band_selected)
        self.band_list.itemChanged.connect(lambda _i: self.apply_visibility())
        lay.addWidget(QLabel("段（色・チェックで表示切替）"))
        lay.addWidget(self.band_list)
        self.tabs = QTabWidget()
        lay.addWidget(self.tabs, 1)
        # measure tab
        mw = QWidget()
        ml = QVBoxLayout(mw)
        self.meas_table = QTableWidget(0, 3)
        self.meas_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        ml.addWidget(self.meas_table, 1)
        row = QHBoxLayout()
        self.cb_confirm = QCheckBox("この段の読取内容を確認済みにする")
        self.cb_confirm.toggled.connect(self.on_confirm_toggled)
        row.addWidget(self.cb_confirm)
        ml.addLayout(row)
        b = QPushButton("測定線を自動位置へ戻す")
        b.clicked.connect(self.action_reset_lines)
        ml.addWidget(b)
        self.meas_note = QLabel()
        self.meas_note.setWordWrap(True)
        ml.addWidget(self.meas_note)
        self.tabs.addTab(mw, "測定")
        # vertices tab
        vw = QWidget()
        vl = QVBoxLayout(vw)
        self.v_table = QTableWidget(0, 7)
        self.v_table.setHorizontalHeaderLabels(["#", "X", "Y", "山/谷", "振幅", "由来", "折返し"])
        self.v_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.v_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.v_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.v_table.itemSelectionChanged.connect(self.on_vertex_row)
        vl.addWidget(self.v_table, 1)
        g = QGroupBox("選択した頂点の座標（元画像のpx）")
        gl = QFormLayout(g)
        self.sp_x = QDoubleSpinBox()
        self.sp_y = QDoubleSpinBox()
        for sp in (self.sp_x, self.sp_y):
            sp.setRange(0, 200000)
            sp.setDecimals(1)
        gl.addRow("X", self.sp_x)
        gl.addRow("Y", self.sp_y)
        bb = QPushButton("座標を反映（修正として保留）")
        bb.clicked.connect(self.action_apply_coords)
        gl.addRow(bb)
        vl.addWidget(g)
        self.tabs.addTab(vw, "頂点")
        # sections tab
        sw = QWidget()
        sl = QVBoxLayout(sw)
        sl.addWidget(QLabel("「区間を指定」(R)で画像上を横にドラッグすると、選択中の段に区間が追加されます。"))
        self.sec_table = QTableWidget(0, 12)
        self.sec_table.setHorizontalHeaderLabels(["区間", "X開始", "X終了", "長さ", "最大", "最小", "上下幅",
                                                  "波の数", "平均周期", "周波数", "除外", "状態"])
        self.sec_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.sec_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.sec_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.sec_table.itemSelectionChanged.connect(self.on_section_row)
        sl.addWidget(self.sec_table, 1)
        gs = QGroupBox("区間を数値で追加（元画像のpx）")
        gsl = QFormLayout(gs)
        self.sec_x0 = QDoubleSpinBox()
        self.sec_x1 = QDoubleSpinBox()
        for sp in (self.sec_x0, self.sec_x1):
            sp.setRange(0, 200000)
            sp.setDecimals(0)
        gsl.addRow("X開始", self.sec_x0)
        gsl.addRow("X終了", self.sec_x1)
        self.sec_all = QCheckBox("全段に同じ範囲を追加する")
        gsl.addRow(self.sec_all)
        rowb = QHBoxLayout()
        b1 = QPushButton("区間を追加")
        b1.clicked.connect(self.action_add_section_numeric)
        b2 = QPushButton("選択した区間を削除")
        b2.clicked.connect(self.action_remove_section)
        rowb.addWidget(b1)
        rowb.addWidget(b2)
        gsl.addRow(rowb)
        sl.addWidget(gs)
        gset = QGroupBox("数え方の設定")
        gsetl = QFormLayout(gset)
        self.sec_swing = QDoubleSpinBox()
        self.sec_swing.setRange(0, 100000)
        self.sec_swing.setDecimals(1)
        self.sec_swing.setToolTip("隣の頂点との振れ幅がこれ未満の小さな揺れは、波として数えません（元画像のpx）")
        self.sec_mode = QComboBox()
        for k, v in COUNT_MODES.items():
            self.sec_mode.addItem(v, k)
        self.sec_swing.valueChanged.connect(self.on_section_settings)
        self.sec_mode.currentIndexChanged.connect(self.on_section_settings)
        gsetl.addRow("数えない小さな揺れ [px]", self.sec_swing)
        gsetl.addRow("波の数え方", self.sec_mode)
        sl.addWidget(gset)
        self.sec_note = QLabel("")
        self.sec_note.setWordWrap(True)
        sl.addWidget(self.sec_note)
        self.tabs.addTab(sw, "区間")
        # issues tab
        iw = QWidget()
        il = QVBoxLayout(iw)
        self.issue_list = QListWidget()
        self.issue_list.itemDoubleClicked.connect(self.on_issue_activated)
        self.issue_list.currentRowChanged.connect(lambda _r: None)
        il.addWidget(QLabel("ダブルクリックで該当位置へ移動"))
        il.addWidget(self.issue_list, 1)
        self.tabs.addTab(iw, "要確認")
        # calibration tab
        cw = QWidget()
        cl = QFormLayout(cw)
        self.c_x = QDoubleSpinBox()
        self.c_y = QDoubleSpinBox()
        self.c_speed = QDoubleSpinBox()
        self.c_amp = QDoubleSpinBox()
        for sp in (self.c_x, self.c_y):
            sp.setRange(0, 100000)
            sp.setDecimals(4)
        self.c_speed.setRange(0, 1000)
        self.c_speed.setDecimals(4)
        self.c_amp.setRange(0, 1e9)
        self.c_amp.setDecimals(5)
        self.c_arm = QDoubleSpinBox()
        self.c_arm.setRange(0, 100000)
        self.c_arm.setDecimals(2)
        self.c_arc = QComboBox()
        self.c_arc.addItem("+1（振幅が大きいほどXが進む向きにずれる）", 1)
        self.c_arc.addItem("−1（振幅が大きいほどXが戻る向きにずれる）", -1)
        self.c_unit = QComboBox()
        self.c_unit.setEditable(True)
        self.c_unit.addItems(["", "μm", "nm", "cm/s", "gal"])
        cl.addRow("X: 1mmあたりのpx（元画像）", self.c_x)
        cl.addRow("Y: 1mmあたりのpx（元画像）", self.c_y)
        cl.addRow("紙送り速度 [mm/s]", self.c_speed)
        cl.addRow("振幅換算係数 [単位/mm]", self.c_amp)
        cl.addRow("振幅の単位", self.c_unit)
        cl.addRow("ペンアーム長 [mm]（円弧補正。0=補正しない）", self.c_arm)
        cl.addRow("円弧のずれの向き", self.c_arc)
        bx = QPushButton("定規上の2点でX方向を設定")
        bx.clicked.connect(lambda: self.start_calibration("x"))
        by = QPushButton("定規上の2点でY方向を設定")
        by.clicked.connect(lambda: self.start_calibration("y"))
        bt = QPushButton("時刻の刻み2点で時間を設定（刻みの間隔[s]を入力）")
        bt.clicked.connect(lambda: self.start_calibration("t"))
        ba = QPushButton("換算条件を適用")
        ba.clicked.connect(self.action_apply_calibration)
        cl.addRow(bx)
        cl.addRow(by)
        cl.addRow(bt)
        self.c_tick_label = QLabel("")
        cl.addRow(self.c_tick_label)
        cl.addRow(ba)
        note = QLabel("未設定（0）のときは画像座標のpxで表示します。時間は紙送り速度が設定されたときだけ出力します。")
        note.setWordWrap(True)
        cl.addRow(note)
        self.tabs.addTab(cw, "校正")
        dock.setWidget(w)
        dock.setMinimumWidth(430)
        self.addDockWidget(Qt.RightDockWidgetArea, dock)

    def _build_status(self):
        sb = self.statusBar()
        self.lbl_state = QLabel()
        self.lbl_cursor = QLabel()
        self.lbl_review = QLabel()
        sb.addWidget(self.lbl_state, 1)
        sb.addPermanentWidget(self.lbl_review)
        sb.addPermanentWidget(self.lbl_cursor)

    # ----------------------------------------------------------------- status
    def _set_state(self, key, text):
        self.state[key] = text
        self.lbl_state.setText("　".join(f"{k}: {v}" for k, v in self.state.items()))

    def _update_review_label(self):
        bands = self.engine.bands
        if not bands:
            self.lbl_review.setText("")
            return
        done = sum(1 for b in bands if self.engine.is_determinate(b))
        pend = sum(1 for b in bands if b.pending)
        txt = f"読取確認: {done}/{len(bands)}段"
        if pend:
            txt += f"　未適用の修正あり({pend}段)"
        self.lbl_review.setText(txt)

    def _on_cursor(self, p):
        e = self.engine
        if e.src_path:
            inv = 1 / e.scale
            self.lbl_cursor.setText(f"X {p.x() * inv:.0f}  Y {p.y() * inv:.0f} px")

    def _update_enabled(self):
        has_img = self.engine.work is not None
        has_b = bool(self.engine.bands)
        self.a_analyze.setEnabled(has_img)
        self.a_compare.setEnabled(has_img)
        self.profile_combo.setEnabled(has_img)
        self._sync_profile_combo()
        for a in (self.a_save, self.a_xlsx, self.a_svg, self.a_undo, self.a_redo, self.a_reset, self.a_apply,
                  self.a_cancel_pending, self.t_add, self.t_origin, self.t_section, self.a_play, self.a_prev_v, self.a_next_v,
                  self.a_zoom_sel):
            a.setEnabled(has_b)

    # ---------------------------------------------------------------- tasks
    def run_task(self, title, fn, on_done, on_fail=None):
        if self.sync_mode:
            try:
                res = fn(lambda *_a: None)
            except Exception as e:  # noqa: BLE001
                (on_fail or self._show_error)(str(e))
                return
            on_done(res)
            return
        dlg = QProgressDialog(title, "中止", 0, 100, self)
        dlg.setWindowModality(Qt.WindowModal)
        dlg.setMinimumDuration(0)
        dlg.setAutoClose(False)
        dlg.setAutoReset(False)
        dlg.setValue(0)
        wk = Worker(fn)
        self.worker = wk
        wk.progress.connect(lambda m, f: (dlg.setLabelText(m), dlg.setValue(int(f * 100))))
        dlg.canceled.connect(lambda: setattr(wk, "cancel", True))

        def _done(res):
            dlg.close()
            on_done(res)

        def _fail(msg):
            dlg.close()
            (on_fail or self._show_error)(msg)

        wk.done.connect(_done)
        wk.failed.connect(_fail)
        wk.finished.connect(lambda: None)
        wk.start()

    def _show_error(self, msg):
        self._set_state("再追跡", "失敗")
        if self.sync_mode:
            self.last_error = msg
            return
        QMessageBox.warning(self, "TraceView", msg)

    def _info(self, msg):
        self.last_info = msg
        self.statusBar().showMessage(msg, 8000)

    # ------------------------------------------------------------ file actions
    def action_open(self):
        path, _ = QFileDialog.getOpenFileName(self, "PDF・画像を開く", self._last_dir(), FILE_FILTER)
        if path:
            self.open_file(path)

    def _last_dir(self):
        return QSettings("TraceView", "TraceView").value("last_dir", os.path.expanduser("~"))

    def _remember_dir(self, path):
        QSettings("TraceView", "TraceView").setValue("last_dir", os.path.dirname(path))

    def open_file(self, path, page=0, then=None):
        if not self._confirm_discard():
            return
        self._set_state("読込", "読込中…")
        eng = Engine()

        def job(cb):
            eng.open(path, page, cb)
            return eng

        def done(e):
            self.engine = e
            self.engine.pages_total = page_count(path)
            self.page_spin.blockSignals(True)
            self.page_spin.setMaximum(max(1, self.engine.pages_total))
            self.page_spin.setValue(page + 1)
            self.page_spin.blockSignals(False)
            self.project_path = None
            self.dirty = False
            self._remember_dir(path)
            self._set_state("読込", f"{os.path.basename(path)}（{e.orig_size[0]}×{e.orig_size[1]}px）")
            self._set_state("解析", "未実施")
            self._set_state("保存", "未保存")
            self.rebuild_scene()
            self._update_enabled()
            if then:
                then()

        self.run_task("読み込み中…", job, done, lambda m: (self._set_state("読込", "失敗"), self._show_error(m)))

    def _page_changed(self):
        e = self.engine
        if not e.src_path or self.page_spin.value() - 1 == e.page:
            return
        self.open_file(e.src_path, self.page_spin.value() - 1)

    def action_analyze(self, force=False):
        e = self.engine
        if e.work is None:
            return
        if e.bands and not force and not self._confirm("解析をやり直すと、現在のトレースと修正は初期化されます。続けますか？"):
            return
        self._set_state("解析", "解析中…")

        def job(cb):
            e.graph = None
            return e.analyze(progress=cb)

        def done(_b):
            self.dirty = True
            self._set_state("解析", f"完了（{len(e.bands)}段を検出）")
            self.sel_band, self.sel_vertex = 0, None
            self.rebuild_scene(keep_view=True)
            self._update_enabled()
            if len(e.bands) < 3:
                self._info("三段のうち一部の起点が見つかりませんでした。「起点を指定」で補助してください")

        self.run_task("解析中…", job, done, lambda m: (self._set_state("解析", "失敗"), self._show_error(m)))

    # ------------------------------------------------------------ profiles
    def _sync_profile_combo(self):
        i = self.profile_combo.findData(self.engine.profile_key)
        self.profile_combo.blockSignals(True)
        self.profile_combo.setCurrentIndex(max(0, i))
        self.profile_combo.blockSignals(False)

    def _profile_chosen(self, index):
        self.set_profile(self.profile_combo.itemData(index))

    def set_profile(self, key, reanalyze=True):
        e = self.engine
        if e.work is None or key == e.profile_key:
            return
        pr = get_profile(key)
        if e.bands and not self._confirm(f"記録方式の設定を「{pr.name}」に変えると、解析をやり直します（現在のトレースと修正は初期化）。よろしいですか？"):
            self._sync_profile_combo()
            return
        e.profile_key = key
        e.params = pr.apply(e.params)
        e.graph = None
        e.prepared = None
        e.bands = []
        self.dirty = True
        self._set_state("解析", f"設定「{pr.name}」（{pr.verified}）")
        self.rebuild_scene()
        self._update_enabled()
        if reanalyze:
            self.action_analyze(force=True)

    def action_compare(self):
        """Try every profile and let the user choose by the look of the overlay."""
        e = self.engine
        if e.work is None:
            return
        if e.bands and not self._confirm("記録方式を探すと、結果を選んだときに現在のトレースと修正は初期化されます。続けますか？"):
            return
        self._set_state("解析", "方式を探索中…")

        def job(cb):
            rows = []
            n = len(PROFILES)
            for i, pr in enumerate(PROFILES):
                c = e.clone_for_profile(pr)
                c.analyze(progress=lambda m, f, i=i: cb(f"{pr.name}（{i + 1}/{n}）: {m}", (i + f) / n))
                png, _w, _h = c.overlay_png(900)
                rows.append(dict(profile=pr, engine=c, quality=c.quality_summary(), png=png))
            rows.sort(key=lambda r: Engine.quality_sort_key(r["quality"]))
            return rows

        def done(rows):
            self.compare_results = rows
            self._set_state("解析", "方式の候補を表示中")
            if self.sync_mode:
                return
            dlg = CompareDialog(rows, self)
            if dlg.exec() == QDialog.Accepted and dlg.chosen is not None:
                self.adopt_candidate(dlg.chosen)
            else:
                self._set_state("解析", "方式の選択を中止しました")

        self.run_task("記録方式を探しています…", job, done,
                      lambda m: (self._set_state("解析", "失敗"), self._show_error(m)))

    def adopt_candidate(self, index):
        row = self.compare_results[index]
        new = row["engine"]
        old = self.engine
        new.calibration = old.calibration
        if hasattr(old, "pages_total"):
            new.pages_total = old.pages_total
        self.engine = new
        self.dirty = True
        self.sel_band, self.sel_vertex = 0, None
        self._set_state("解析", f"方式「{row['profile'].name}」を採用（{row['profile'].verified}）")
        self.rebuild_scene()
        self._update_enabled()

    def action_save(self):
        e = self.engine
        if not e.bands:
            return
        path = self.project_path
        if not path:
            base = os.path.splitext(os.path.basename(e.src_path))[0] + ".trproj"
            path, _ = QFileDialog.getSaveFileName(self, "作業ファイルを保存", os.path.join(self._last_dir(), base),
                                                  "TraceView作業ファイル (*.trproj)")
        if not path:
            return
        self.save_project_to(path)

    def save_project_to(self, path):
        try:
            save_project(self.engine, path)
        except Exception as ex:  # noqa: BLE001
            self._show_error(f"保存に失敗しました: {ex}")
            return False
        self.project_path = path
        self.dirty = False
        self._set_state("保存", time.strftime("%H:%M:%S に保存"))
        return True

    def action_load_project(self):
        path, _ = QFileDialog.getOpenFileName(self, "作業ファイルを開く", self._last_dir(),
                                              "TraceView作業ファイル (*.trproj)")
        if path:
            self.load_project_from(path)

    def load_project_from(self, path, source_path=None):
        if not self._confirm_discard():
            return
        eng = Engine()
        self._set_state("読込", "作業ファイルを読込中…")

        def job(cb):
            load_project(eng, path, source_path, cb)
            return eng

        def done(e):
            self.engine = e
            self.project_path = path
            self.dirty = False
            self.sel_band, self.sel_vertex = 0, None
            self.page_spin.blockSignals(True)
            self.page_spin.setMaximum(max(1, page_count(e.src_path)))
            self.page_spin.setValue(e.page + 1)
            self.page_spin.blockSignals(False)
            self._set_state("読込", f"{os.path.basename(e.src_path)}（作業ファイルから再開）")
            self._set_state("解析", f"保存済みの結果（{len(e.bands)}段）")
            self._set_state("保存", "読込済み")
            self._load_calibration_fields()
            self.rebuild_scene()
            self._update_enabled()

        def fail(msg):
            if "見つかりません" in msg and not self.sync_mode:
                p, _ = QFileDialog.getOpenFileName(self, "元ファイルを選択", self._last_dir(), FILE_FILTER)
                if p:
                    self.load_project_from(path, p)
                    return
            self._show_error(msg)

        self.run_task("作業ファイルを読み込み中…", job, done, fail)

    def action_export_xlsx(self):
        e = self.engine
        if not self._export_guard():
            return
        base = os.path.splitext(os.path.basename(e.src_path))[0] + "_測定結果.xlsx"
        path, _ = QFileDialog.getSaveFileName(self, "Excel出力", os.path.join(self._last_dir(), base), "Excel (*.xlsx)")
        if path:
            self.export_xlsx_to(path)

    def export_xlsx_to(self, path):
        if not self._export_guard():
            return False
        try:
            export_xlsx(self.engine, path)
        except PermissionError:
            self._show_error("出力先に書き込めません。Excelで開いている場合は閉じてから再実行してください")
            return False
        except Exception as ex:  # noqa: BLE001
            self._show_error(f"Excel出力に失敗しました: {ex}")
            return False
        self._info(f"Excelを出力しました: {path}")
        return True

    def action_export_svg(self):
        e = self.engine
        if not self._export_guard():
            return
        base = os.path.splitext(os.path.basename(e.src_path))[0] + "_trace.svg"
        path, _ = QFileDialog.getSaveFileName(self, "トレース出力(SVG)", os.path.join(self._last_dir(), base), "SVG (*.svg)")
        if not path:
            return
        with_img = QMessageBox.question(self, "TraceView", "元画像を含めますか？\n（「いいえ」でトレースのみ。トレースのみでも匿名化済みとは限りません）"
                                        ) == QMessageBox.Yes
        self.export_svg_to(path, with_img)

    def export_svg_to(self, path, with_img=False):
        if not self._export_guard():
            return False
        try:
            export_svg(self.engine, path, with_img)
        except Exception as ex:  # noqa: BLE001
            self._show_error(f"SVG出力に失敗しました: {ex}")
            return False
        self._info(f"SVGを出力しました: {path}")
        return True

    def _export_guard(self):
        """ED-07: never export while unapplied edits exist."""
        if not self.engine.bands:
            return False
        if self.engine.has_pending():
            self._show_error("未適用の頂点修正があります。「再追跡を適用」するか「修正を取り消す」を行ってから出力してください")
            return False
        return True

    def _confirm(self, text):
        if self.sync_mode:
            return True
        return QMessageBox.question(self, "TraceView", text) == QMessageBox.Yes

    def _confirm_discard(self):
        if not self.dirty and not self.engine.has_pending():
            return True
        if self.sync_mode:
            return True
        r = QMessageBox.question(self, "TraceView", "保存していない変更があります。破棄して続けますか？")
        return r == QMessageBox.Yes

    def closeEvent(self, ev):
        if self._confirm_discard():
            ev.accept()
        else:
            ev.ignore()

    # --------------------------------------------------------------- scene
    def rebuild_scene(self, keep_view=False):
        e = self.engine
        tr = self.view.transform()
        self.replay_timer.stop()
        self.a_play.setChecked(False)
        self.scene.clear()
        self.band_items.clear()
        self.sel_ring = None
        self.replay_marker = None
        self.pix_item = None
        if e.work is None:
            return
        h, w = e.work.shape[:2]
        self._img_data = np.ascontiguousarray(e.work)
        qimg = QImage(self._img_data.data, w, h, 3 * w, QImage.Format_RGB888)
        self.pix_item = QGraphicsPixmapItem(QPixmap.fromImage(qimg))
        self.pix_item.setZValue(0)
        self.scene.addItem(self.pix_item)
        self.scene.setSceneRect(0, 0, w, h)
        self.band_list.blockSignals(True)
        self.band_list.clear()
        for b in e.bands:
            it = QListWidgetItem(b.name)
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            it.setCheckState(Qt.Checked)
            it.setForeground(QBrush(QColor(b.color)))
            self.band_list.addItem(it)
        self.band_list.blockSignals(False)
        for b in e.bands:
            self._build_band_items(b.index)
        if e.bands:
            self.band_list.setCurrentRow(min(self.sel_band, len(e.bands) - 1))
        self.apply_visibility()
        self.refresh_panels()
        if keep_view and not tr.isIdentity():
            self.view.setTransform(tr)
        else:
            self.view.fit_all()
        self._update_review_label()

    def _clear_band_items(self, bi):
        d = self.band_items.pop(bi, None)
        if not d:
            return
        for it in [d["path"], *d["inferred"], *d["vertices"], *d["lines"].values(), *d["issues"], *d["pending"],
                   *d.get("sections", [])]:
            if it is not None and it.scene() is self.scene:
                self.scene.removeItem(it)

    def _build_band_items(self, bi):
        e = self.engine
        b = e.bands[bi]
        self._clear_band_items(bi)
        d = dict(path=None, inferred=[], vertices=[], lines={}, issues=[], pending=[])
        pts = b.path
        if len(pts) > 1:
            simp = cv2.approxPolyDP(pts.reshape(-1, 1, 2).astype(np.int32), 0.7, False).reshape(-1, 2)
            d["path"] = make_path_item(polyline_path(simp), b.color, 2)
            self.scene.addItem(d["path"])
            for a, c in b.inferred:
                seg = pts[a:c + 1]
                if len(seg) > 1:
                    it = make_path_item(polyline_path(seg[::max(1, len(seg) // 60)].tolist() + [seg[-1].tolist()]),
                                        "#ffd400", 3, dashed=True, z=12)
                    self.scene.addItem(it)
                    d["inferred"].append(it)
        for vi, v in enumerate(b.vertices):
            col = {"auto": b.color, "manual": "#ff8c00", "retrace": "#00bcd4"}.get(v.source, b.color)
            it = VertexItem(self, bi, vi, v.x, v.y, col)
            it.setToolTip(f"{b.name} 頂点{vi + 1}（{'山' if v.kind == 'peak' else '谷'}）")
            it.set_editable(self.tool == "select")
            self.scene.addItem(it)
            d["vertices"].append(it)
        paper = e.prepared.paper if e.prepared else (0, 0, e.work.shape[1], e.work.shape[0])
        x0, x1 = paper[0], paper[2]
        for kind, y, dashed in (("0線", b.line_zero, False), ("上線", b.line_top, True), ("下線", b.line_bottom, True)):
            it = LineItem(self, bi, kind, x0, x1, y, b.color, dashed)
            self.scene.addItem(it)
            d["lines"][kind] = it
        for iss in b.issues:
            if iss.kind == "inferred":
                continue
            r = 7
            m = QGraphicsEllipseItem(-r, -r, 2 * r, 2 * r)
            m.setPos(iss.x, iss.y)
            m.setBrush(QBrush(QColor(255, 0, 0, 90)))
            m.setPen(QPen(QColor("red"), 2))
            m.setFlag(QGraphicsEllipseItem.ItemIgnoresTransformations, True)
            m.setZValue(25)
            m.setToolTip(f"{ISSUE_LABEL.get(iss.kind, iss.kind)}: {iss.msg}")
            m.setData(0, iss.kind)
            self.scene.addItem(m)
            d["issues"].append(m)
        d["sections"] = []
        if b.sections:
            ya, yb = self._section_y_range(bi)
            for sec in b.sections:
                rc = QGraphicsRectItem(sec["x0"], ya, sec["x1"] - sec["x0"], yb - ya)
                col = QColor(b.color)
                col.setAlpha(36)
                rc.setBrush(QBrush(col))
                pen = QPen(QColor(b.color), 1, Qt.DashDotLine)
                pen.setCosmetic(True)
                rc.setPen(pen)
                rc.setZValue(6)
                rc.setToolTip(f"{b.name} {sec['name']}")
                self.scene.addItem(rc)
                lbl = QGraphicsSimpleTextItem(sec["name"])
                lbl.setBrush(QBrush(QColor(b.color)))
                lbl.setFlag(QGraphicsSimpleTextItem.ItemIgnoresTransformations, True)
                lbl.setPos(sec["x0"] + 4, ya + 4)
                lbl.setZValue(7)
                self.scene.addItem(lbl)
                d["sections"] += [rc, lbl]
        for p in b.pending:
            ox, oy = p.get("ox"), p.get("oy")
            m = QGraphicsEllipseItem(-7, -7, 14, 14)
            m.setPos(p["x"], p["y"])
            m.setBrush(QBrush(QColor(255, 140, 0, 160)))
            m.setPen(QPen(QColor("#ff6f00"), 2, Qt.DashLine))
            m.setFlag(QGraphicsEllipseItem.ItemIgnoresTransformations, True)
            m.setZValue(35)
            m.setToolTip("未適用の修正（再追跡を適用してください）")
            self.scene.addItem(m)
            d["pending"].append(m)
            if ox is not None:
                from PySide6.QtWidgets import QGraphicsLineItem
                ln = QGraphicsLineItem(ox, oy, p["x"], p["y"])
                pen = QPen(QColor("#ff6f00"), 1, Qt.DashLine)
                pen.setCosmetic(True)
                ln.setPen(pen)
                ln.setZValue(34)
                self.scene.addItem(ln)
                d["pending"].append(ln)
        self.band_items[bi] = d

    def apply_visibility(self, *_):
        if self.pix_item is not None:
            self.pix_item.setVisible(self.cb_image.isChecked())
            self.scene.setBackgroundBrush(QBrush(QColor("white"))) if not self.cb_image.isChecked() else None
        # thousands of vertex markers are unreadable when zoomed out: show them from a moderate zoom
        zoomed = self.view.transform().m11() >= 0.4
        for bi, d in self.band_items.items():
            vis_band = self.band_list.item(bi) is not None and self.band_list.item(bi).checkState() == Qt.Checked
            tr = self.cb_trace.isChecked() and vis_band
            if d["path"] is not None:
                d["path"].setVisible(tr)
            for it in d["inferred"]:
                it.setVisible(tr)
            for it in d["vertices"]:
                it.setVisible(self.cb_vertices.isChecked() and vis_band and zoomed)
            for it in d["lines"].values():
                it.setVisible(self.cb_lines.isChecked() and vis_band)
            for it in d["issues"]:
                # the many small ambiguity marks only appear once zoomed in; serious ones always show
                it.setVisible(self.cb_issues.isChecked() and vis_band and (zoomed or it.data(0) != "ambiguous"))
            for it in d["pending"]:
                it.setVisible(vis_band)
            for it in d.get("sections", []):
                it.setVisible(vis_band)

    def set_tool(self, tool):
        self.tool = tool
        for name, act in (("select", self.t_select), ("add", self.t_add), ("origin", self.t_origin),
                          ("section", self.t_section)):
            act.setChecked(name == tool)
        self.view.click_tool = tool in ("add", "origin", "calib")
        self.view.drag_tool = tool == "section"
        for d in self.band_items.values():
            for it in d["vertices"]:
                it.set_editable(tool == "select")
        hint = {"select": "頂点をドラッグして移動／測定線は上下にドラッグ", "add": "追加する頂点（先端）をクリック",
                "origin": "選択中の段の起点（波形の左端）をクリック", "calib": "1点目、2点目の順にクリック",
                "section": "測定したい範囲を横にドラッグ（選択中の段に追加）"}[tool]
        self.statusBar().showMessage(hint, 6000)

    # ------------------------------------------------------------ panels
    def current_band(self):
        if 0 <= self.sel_band < len(self.engine.bands):
            return self.engine.bands[self.sel_band]
        return None

    def on_band_selected(self, row):
        if row < 0:
            return
        self.sel_band = row
        self.sel_vertex = None
        self.refresh_panels()

    def refresh_panels(self):
        self.update_measure_table()
        self.update_vertex_table()
        self.update_issue_list()
        self.update_section_table()
        b = self.current_band()
        self.cb_confirm.blockSignals(True)
        self.cb_confirm.setChecked(bool(b and b.confirmed))
        self.cb_confirm.blockSignals(False)
        self._update_review_label()

    def update_measure_table(self):
        e = self.engine
        t = self.meas_table
        bands = e.bands
        t.setColumnCount(len(bands))
        t.setHorizontalHeaderLabels([b.name for b in bands])
        unit = e.calibration.unit_y()
        inv = 1 / e.scale
        rows = ["状態", "0線の高さY[px]", "上線の高さY[px]", "下線の高さY[px]", f"最大値[{unit}]", "最大値の位置(X,Y)",
                f"最小値[{unit}]", "最小値の位置(X,Y)", f"上振幅[{unit}]", f"下振幅[{unit}]", f"上下幅[{unit}]",
                f"線基準 上下幅[{unit}]", f"連続頂点間の最大差(参考)[{unit}]", "末端まで追跡", "要確認の数", "頂点数"]
        t.setRowCount(len(rows))
        t.setVerticalHeaderLabels(rows)

        def f(v, n=2):
            return "—" if v is None else f"{v:.{n}f}"
        for c, b in enumerate(bands):
            m = e.measures(b)
            cells = [e.value_status(b), f(b.line_zero * inv, 1) + ("（手動）" if b.line_zero_manual else ""),
                     f(b.line_top * inv, 1) + ("（手動）" if b.line_top_manual else ""),
                     f(b.line_bottom * inv, 1) + ("（手動）" if b.line_bottom_manual else ""),
                     f(m.get("max_value")), f"({m['max_x']:.0f}, {m['max_y']:.0f})" if "max_x" in m else "—",
                     f(m.get("min_value")), f"({m['min_x']:.0f}, {m['min_y']:.0f})" if "min_x" in m else "—",
                     f(m.get("up_amp")), f(m.get("down_amp")), f(m.get("span")), f(m.get("line_span")),
                     f(m.get("max_step")), "はい" if b.complete else "いいえ", str(len(b.issues)),
                     f"{len(b.vertices)}（所属要確認{m.get('n_suspect', 0)}）"]
            for r, txt in enumerate(cells):
                it = QTableWidgetItem(txt)
                if r == 0 and not e.is_determinate(b):
                    it.setForeground(QBrush(QColor("#b45f00")))
                t.setItem(r, c, it)
        t.resizeColumnsToContents()
        cal = e.calibration
        self.meas_note.setText("未校正: 値は元画像のpx。「確認済み」以外は候補で、確定値として扱えません。" if not cal.calibrated_y
                               else f"換算: Y {cal.mm_per_px_y:.5f} mm/px。「確認済み」以外は候補です。")

    def update_vertex_table(self):
        e = self.engine
        b = self.current_band()
        t = self.v_table
        t.blockSignals(True)
        t.setRowCount(0)
        if b is None:
            t.blockSignals(False)
            return
        inv = 1 / e.scale
        t.setRowCount(len(b.vertices))
        for i, v in enumerate(b.vertices):
            amp = e.calibration.to_y_units((b.line_zero - v.y) * inv)
            vals = [str(i + 1), f"{v.x * inv:.1f}", f"{v.y * inv:.1f}", "山" if v.kind == "peak" else "谷",
                    f"{amp:.2f}", {"auto": "自動", "manual": "手動", "retrace": "再追跡"}[v.source],
                    "先端で反転" if v.turn == "end" else "曲がり"]
            for c, s in enumerate(vals):
                t.setItem(i, c, QTableWidgetItem(s))
        t.resizeColumnsToContents()
        t.blockSignals(False)

    def update_issue_list(self):
        self.issue_list.clear()
        b = self.current_band()
        if b is None:
            return
        inv = 1 / self.engine.scale
        for it in b.issues:
            li = QListWidgetItem(f"[{ISSUE_LABEL.get(it.kind, it.kind)}] X{it.x * inv:.0f} Y{it.y * inv:.0f}  {it.msg}")
            li.setData(Qt.UserRole, (it.x, it.y))
            self.issue_list.addItem(li)
        for p in b.pending:
            li = QListWidgetItem(f"[未適用の修正] X{p['x'] * inv:.0f} Y{p['y'] * inv:.0f}  再追跡を適用してください")
            li.setData(Qt.UserRole, (p["x"], p["y"]))
            self.issue_list.addItem(li)

    def on_issue_activated(self, item):
        x, y = item.data(Qt.UserRole)
        self.view.center_on_point(x, y, zoom=max(0.5, self.view.transform().m11()))

    # --------------------------------------------------------- vertex handling
    def on_vertex_row(self):
        rows = self.v_table.selectionModel().selectedRows()
        if rows:
            self.on_vertex_selected(self.sel_band, rows[0].row(), from_canvas=False)

    def on_vertex_selected(self, bi, vi, from_canvas=False):
        e = self.engine
        if bi != self.sel_band:
            self.sel_band = bi
            self.band_list.blockSignals(True)
            self.band_list.setCurrentRow(bi)
            self.band_list.blockSignals(False)
            self.refresh_panels()
        b = e.bands[bi]
        if not (0 <= vi < len(b.vertices)):
            return
        self.sel_vertex = vi
        v = b.vertices[vi]
        inv = 1 / e.scale
        self.sp_x.setValue(v.x * inv)
        self.sp_y.setValue(v.y * inv)
        if self.sel_ring is not None and self.sel_ring.scene() is self.scene:
            self.scene.removeItem(self.sel_ring)
        r = 11
        ring = QGraphicsEllipseItem(-r, -r, 2 * r, 2 * r)
        ring.setPos(v.x, v.y)
        ring.setPen(QPen(QColor("#ffeb3b"), 3))
        ring.setFlag(QGraphicsEllipseItem.ItemIgnoresTransformations, True)
        ring.setZValue(40)
        self.scene.addItem(ring)
        self.sel_ring = ring
        if from_canvas:
            self.v_table.blockSignals(True)
            self.v_table.selectRow(vi)
            self.v_table.scrollToItem(self.v_table.item(vi, 0))
            self.v_table.blockSignals(False)
        else:
            self.zoom_to_selected(keep_zoom=True)

    def zoom_to_selected(self, keep_zoom=False):
        b = self.current_band()
        if b is None or self.sel_vertex is None or self.sel_vertex >= len(b.vertices):
            return
        v = b.vertices[self.sel_vertex]
        z = self.view.transform().m11()
        self.view.center_on_point(v.x, v.y, zoom=None if (keep_zoom and z > 0.4) else 0.8)

    def step_vertex(self, d):
        b = self.current_band()
        if b is None or not b.vertices:
            return
        k = 0 if self.sel_vertex is None else self.sel_vertex + d
        k = max(0, min(len(b.vertices) - 1, k))
        self.v_table.blockSignals(True)
        self.v_table.selectRow(k)
        self.v_table.blockSignals(False)
        self.on_vertex_selected(self.sel_band, k)

    def on_vertex_dragged(self, bi, vi, x, y):
        e = self.engine
        e.set_pending_move(bi, vi, x, y)
        self.dirty = True
        self._build_band_items(bi)
        self.apply_visibility()
        self.update_issue_list()
        self._update_review_label()
        if self.auto_retrace:
            self.apply_pending(bi)
        else:
            self._set_state("再追跡", "未適用の修正あり")

    def action_apply_coords(self):
        b = self.current_band()
        if b is None or self.sel_vertex is None:
            return
        s = self.engine.scale
        self.on_vertex_dragged(self.sel_band, self.sel_vertex, self.sp_x.value() * s, self.sp_y.value() * s)

    # ---------------------------------------------------------------- tools
    def on_scene_clicked(self, pt):
        e = self.engine
        b = self.current_band()
        if b is None:
            return
        if self.tool == "add":
            e.set_pending_add(self.sel_band, pt.x(), pt.y())
            self.dirty = True
            self._build_band_items(self.sel_band)
            self.apply_visibility()
            self.update_issue_list()
            if self.auto_retrace:
                self.apply_pending(self.sel_band)
            else:
                self._set_state("再追跡", "未適用の修正あり")
        elif self.tool == "origin":
            self._set_state("解析", "起点から再追跡中…")
            bi = self.sel_band

            def job(cb):
                e.ensure_graph(cb)
                return e.retrace_band_from_origin(bi, int(pt.x()), int(pt.y()))

            def done(_nb):
                self.dirty = True
                self._set_state("解析", f"{b.name}を起点指定で再追跡しました")
                self._build_band_items(bi)
                self.apply_visibility()
                self.refresh_panels()

            self.run_task("再追跡中…", job, done)
        elif self.tool == "calib":
            self.calib_pts.append((pt.x(), pt.y()))
            if len(self.calib_pts) == 2:
                self._finish_calibration()
            else:
                self.statusBar().showMessage("2点目をクリックしてください", 5000)

    def apply_pending(self, bi=None):
        e = self.engine
        bi = self.sel_band if bi is None else bi
        b = e.bands[bi]
        if not b.pending:
            return
        self._set_state("再追跡", "再追跡中…")

        def job(cb):
            return e.apply_pending(bi, cb)

        def done(res):
            ok, msg = res
            self._set_state("再追跡", "完了" if ok else "失敗（修正前の結果を保持）")
            if not ok:
                self._show_error(msg)
            else:
                self._info(msg)
                self.dirty = True
            self._build_band_items(bi)
            self.apply_visibility()
            self.sel_vertex = None
            self.refresh_panels()

        self.run_task("再追跡中…", job, done)

    def action_apply(self):
        self.apply_pending(self.sel_band)

    def action_cancel_pending(self):
        b = self.current_band()
        if b is None:
            return
        self.engine.cancel_pending(self.sel_band)
        self._build_band_items(self.sel_band)
        self.apply_visibility()
        self.refresh_panels()
        self._set_state("再追跡", "修正を取り消しました")

    def action_undo(self):
        if self.engine.undo(self.sel_band):
            self.dirty = True
            self._after_history()
        else:
            self._info("これ以上戻せません")

    def action_redo(self):
        if self.engine.redo(self.sel_band):
            self.dirty = True
            self._after_history()
        else:
            self._info("やり直せる操作がありません")

    def action_reset_band(self):
        b = self.current_band()
        if b is None:
            return
        if not self._confirm(f"{b.name}を初回の自動結果へ戻します（他の段は変更しません）。よろしいですか？"):
            return
        if self.engine.reset_band(self.sel_band):
            self.dirty = True
            self._after_history()

    def _after_history(self):
        self._build_band_items(self.sel_band)
        self.apply_visibility()
        self.sel_vertex = None
        self.refresh_panels()

    def on_line_moved(self, bi, kind, y):
        b = self.engine.bands[bi]
        b.push_undo(self.engine.params.undo_limit)
        if kind == "0線":
            b.line_zero, b.line_zero_manual = y, True
        elif kind == "上線":
            b.line_top, b.line_top_manual = y, True
        else:
            b.line_bottom, b.line_bottom_manual = y, True
        b.confirmed = False
        self.dirty = True
        self.refresh_panels()

    def action_reset_lines(self):
        b = self.current_band()
        if b is None:
            return
        b.push_undo(self.engine.params.undo_limit)
        b.line_zero_manual = b.line_top_manual = b.line_bottom_manual = False
        self.engine.reset_lines(b)
        self.dirty = True
        self._build_band_items(self.sel_band)
        self.apply_visibility()
        self.refresh_panels()

    def on_confirm_toggled(self, on):
        b = self.current_band()
        if b is None:
            return
        b.confirmed = on
        self.dirty = True
        self.update_measure_table()
        self._update_review_label()

    # ------------------------------------------------------------ sections
    def _section_y_range(self, bi):
        e = self.engine
        origins = [bd.origin for bd in e.bands]
        if len(origins) >= 2:
            corr, _z = e._zone(bi, origins)
            return corr[0], corr[1]
        b = e.bands[bi]
        return b.line_zero - 300, b.line_zero + 300

    def _section_drag_preview(self, a, b):
        bd = self.current_band()
        if bd is None:
            return
        y0, y1 = self._section_y_range(self.sel_band)
        if self.section_preview is None:
            it = QGraphicsRectItem()
            it.setBrush(QBrush(QColor(255, 235, 59, 60)))
            it.setPen(QPen(QColor("#ffeb3b"), 1, Qt.DashLine))
            it.setZValue(8)
            self.scene.addItem(it)
            self.section_preview = it
        self.section_preview.setRect(min(a.x(), b.x()), y0, abs(b.x() - a.x()), y1 - y0)

    def _section_drag_done(self, a, b):
        if self.section_preview is not None:
            self.scene.removeItem(self.section_preview)
            self.section_preview = None
        bands = range(len(self.engine.bands)) if self.sec_all.isChecked() else [self.sel_band]
        self._add_sections(bands, a.x(), b.x())

    def _add_sections(self, bands, x0, x1):
        try:
            for bi in bands:
                self.engine.add_section(bi, x0, x1)
        except EngineError as ex:
            self._show_error(str(ex))
            return
        self.dirty = True
        for bi in bands:
            self._build_band_items(bi)
        self.apply_visibility()
        self.refresh_panels()
        self.tabs.setCurrentIndex(2)
        self._info("区間を追加しました")

    def action_add_section_numeric(self):
        if not self.engine.bands:
            return
        s = self.engine.scale
        bands = range(len(self.engine.bands)) if self.sec_all.isChecked() else [self.sel_band]
        self._add_sections(bands, self.sec_x0.value() * s, self.sec_x1.value() * s)

    def action_remove_section(self):
        rows = self.sec_table.selectionModel().selectedRows()
        if rows and self.engine.remove_section(self.sel_band, rows[0].row()):
            self.dirty = True
            self._build_band_items(self.sel_band)
            self.apply_visibility()
            self.refresh_panels()

    def on_section_settings(self, *_):
        ss = self.engine.section_settings
        ss.min_swing_px = self.sec_swing.value()
        ss.count_mode = self.sec_mode.currentData()
        self.dirty = True
        self.update_section_table()

    def on_section_row(self):
        rows = self.sec_table.selectionModel().selectedRows()
        b = self.current_band()
        if rows and b is not None and rows[0].row() < len(b.sections):
            sec = b.sections[rows[0].row()]
            self.view.center_on_point((sec["x0"] + sec["x1"]) / 2, b.line_zero, zoom=None)
            self.view.centerOn((sec["x0"] + sec["x1"]) / 2, b.line_zero)

    def update_section_table(self):
        e = self.engine
        b = self.current_band()
        t = self.sec_table
        t.blockSignals(True)
        t.setRowCount(0)
        self.sec_swing.blockSignals(True)
        self.sec_mode.blockSignals(True)
        self.sec_swing.setValue(e.section_settings.min_swing_px)
        self.sec_mode.setCurrentIndex(max(0, self.sec_mode.findData(e.section_settings.count_mode)))
        self.sec_swing.blockSignals(False)
        self.sec_mode.blockSignals(False)
        if b is None:
            t.blockSignals(False)
            return
        unit = e.calibration.unit_y()
        sp = e.calibration.seconds_per_orig_px()
        t.setRowCount(len(b.sections))

        def f(v, n=1, suffix=""):
            return "—" if v is None else f"{v:.{n}f}{suffix}"
        any_doubt = False
        for i in range(len(b.sections)):
            r = e.section_result(self.sel_band, i)
            period = f(r["period_s"], 3, " s") if r["period_s"] is not None else f(r["period_px"], 1, " px")
            doubt = r["n_excluded"] or r["beyond_trace"] or r["issues"]["untracked_end"]
            any_doubt = any_doubt or bool(doubt)
            cells = [r["name"], f(r["x0"], 0), f(r["x1"], 0),
                     f(r["length_s"], 2, " s") if r["length_s"] is not None else f(r["length_px"], 0, " px"),
                     f(r.get("max_value"), 2), f(r.get("min_value"), 2), f(r.get("span"), 2),
                     f(r["waves"], 1 if r["waves"] % 1 else 0), period,
                     f(r["freq_hz"], 3, " Hz") if r["freq_hz"] is not None else "—",
                     str(r["n_excluded"]), r["status"] + ("・追跡外を含む" if r["beyond_trace"] else "")]
            for c, txt in enumerate(cells):
                t.setItem(i, c, QTableWidgetItem(txt))
        t.resizeColumnsToContents()
        t.blockSignals(False)
        self.sec_note.setText(
            f"値の単位: {unit}。時間換算: " + (f"{sp:.5f} 秒/px" if sp > 0 else "未設定（周期はpx表示。「校正」タブで設定）")
            + "。「確認済み」以外の値は候補です。" + (" 区間内に除外した頂点／追跡できていない部分があります。" if any_doubt else ""))

    # ---------------------------------------------------------- calibration
    def _load_calibration_fields(self):
        c = self.engine.calibration
        self.c_x.setValue(1 / c.mm_per_px_x if c.mm_per_px_x > 0 else 0)
        self.c_y.setValue(1 / c.mm_per_px_y if c.mm_per_px_y > 0 else 0)
        self.c_speed.setValue(1 / c.seconds_per_mm if c.seconds_per_mm > 0 else 0)
        self.c_amp.setValue(c.amplitude_factor)
        self.c_unit.setCurrentText(c.amplitude_unit)
        self.c_arm.setValue(c.arm_length_mm)
        self._show_tick_label()
        self.c_arc.setCurrentIndex(0 if c.arc_sign >= 0 else 1)

    def action_apply_calibration(self):
        c = self.engine.calibration
        c.mm_per_px_x = 1 / self.c_x.value() if self.c_x.value() > 0 else 0.0
        c.mm_per_px_y = 1 / self.c_y.value() if self.c_y.value() > 0 else 0.0
        c.seconds_per_mm = 1 / self.c_speed.value() if self.c_speed.value() > 0 else 0.0
        c.amplitude_factor = self.c_amp.value()
        c.amplitude_unit = self.c_unit.currentText()
        c.arm_length_mm = self.c_arm.value()
        c.arc_sign = int(self.c_arc.currentData())
        self.dirty = True
        self.refresh_panels()
        self._show_tick_label()
        self._info("換算条件を適用しました（作業ファイルとExcelに保存されます）")

    def _show_tick_label(self):
        c = self.engine.calibration
        sp = c.seconds_per_orig_px()
        self.c_tick_label.setText(f"現在の時間換算: {sp:.5f} 秒/px（{1 / sp:.2f} px/秒）" if sp > 0 else "現在の時間換算: 未設定（時間は出力しません）")

    def start_calibration(self, axis):
        self.calib_axis = axis
        self.calib_pts = []
        self.set_tool("calib")

    def _finish_calibration(self):
        (x1, y1), (x2, y2) = self.calib_pts
        d_work = abs(y2 - y1) if self.calib_axis == "y" else abs(x2 - x1)
        d_px = d_work / self.engine.scale
        self.set_tool("select")
        if d_px < 5:
            self._info("2点の距離が近すぎます")
            return
        if self.calib_axis == "t":
            sec, ok = (5.0, True) if self.sync_mode else QInputDialog.getDouble(
                self, "時間の設定", f"選んだ2点の刻みの間の時間 [秒]（画像上 {d_px:.1f}px）", 10.0, 0.0001, 1e7, 4)
            if ok and sec > 0:
                self.engine.calibration.seconds_per_px_x = sec / d_px
                self.dirty = True
                self.refresh_panels()
                self._info(f"時間を設定しました（{sec / d_px:.5f} 秒/px）")
            return
        mm, ok = (50.0, True) if self.sync_mode else QInputDialog.getDouble(
            self, "校正", f"2点間の実距離 [mm]（画像上 {d_px:.1f}px）", 100.0, 0.001, 100000, 3)
        if ok and mm > 0:
            (self.c_x if self.calib_axis == "x" else self.c_y).setValue(d_px / mm)
            self.action_apply_calibration()

    # --------------------------------------------------------------- replay
    def toggle_replay(self, on=None):
        b = self.current_band()
        if b is None or len(b.path) < 2:
            self.a_play.setChecked(False)
            return
        on = self.a_play.isChecked() if on is None else on
        if on:
            if self.replay_idx >= len(b.path) - 1:
                self.replay_idx = 0
            if self.replay_marker is None:
                r = 8
                m = QGraphicsEllipseItem(-r, -r, 2 * r, 2 * r)
                m.setBrush(QBrush(QColor("#00e5ff")))
                m.setPen(QPen(QColor("black"), 1))
                m.setFlag(QGraphicsEllipseItem.ItemIgnoresTransformations, True)
                m.setZValue(50)
                self.scene.addItem(m)
                self.replay_marker = m
            self.replay_timer.start(30)
        else:
            self.replay_timer.stop()

    def _replay_tick(self):
        b = self.current_band()
        if b is None:
            self.replay_timer.stop()
            return
        step = max(1, int(len(b.path) / 1200 * self.speed.value() / 10))
        self.replay_idx = min(len(b.path) - 1, self.replay_idx + step)
        x, y = b.path[self.replay_idx]
        self.replay_marker.setPos(float(x), float(y))
        self.view.centerOn(float(x), float(y))
        self.lbl_cursor.setText(f"復元順 {self.replay_idx + 1}/{len(b.path)}")
        if self.replay_idx >= len(b.path) - 1:
            self.replay_timer.stop()
            self.a_play.setChecked(False)


class CompareDialog(QDialog):
    """Candidates side by side: pick the one whose overlay looks right."""

    def __init__(self, rows, parent=None):
        super().__init__(parent)
        self.setWindowTitle("記録方式を選ぶ")
        self.resize(1500, 800)
        self.chosen = None
        root = QVBoxLayout(self)
        note = QLabel("各設定で自動解析した結果です。元画像に重なって見えるものを選んでください。"
                      "下の数値は「うまくたどれたように見えるか」の目安で、正しさの保証ではありません。")
        note.setWordWrap(True)
        root.addWidget(note)
        grid = QHBoxLayout()
        root.addLayout(grid, 1)
        for i, r in enumerate(rows):
            q = r["quality"]
            box = QGroupBox(f"{i + 1}位　{r['profile'].name}")
            lay = QVBoxLayout(box)
            img = QLabel()
            pm = QPixmap()
            pm.loadFromData(r["png"])
            img.setPixmap(pm.scaledToWidth(340, Qt.SmoothTransformation))
            lay.addWidget(img)
            info = QLabel(
                f"三段の検出: {q['bands']}段\n末端まで追跡: {q['complete']}/{q['bands']}段\n"
                f"未追跡の長さ: {q['untracked_px']:.0f}px\n要確認: 逸脱{q['excursion']}・曖昧{q['ambiguous']}・推定接続{q['inferred']}\n"
                f"頂点数: {q['vertices']}")
            lay.addWidget(info)
            ver = QLabel(f"検証状況: {r['profile'].verified}")
            if "未検証" in r["profile"].verified:
                ver.setStyleSheet("color:#b45f00")
            lay.addWidget(ver)
            desc = QLabel(r["profile"].description)
            desc.setWordWrap(True)
            lay.addWidget(desc)
            btn = QPushButton("この設定を使う")
            btn.clicked.connect(lambda _c=False, i=i: self._choose(i))
            lay.addWidget(btn)
            lay.addStretch(1)
            grid.addWidget(box)
        bb = QDialogButtonBox(QDialogButtonBox.Cancel)
        bb.rejected.connect(self.reject)
        root.addWidget(bb)

    def _choose(self, i):
        self.chosen = i
        self.accept()


def main(argv=None):
    import sys
    app = QApplication(argv or sys.argv)
    app.setApplicationName("TraceView")
    win = MainWindow()
    win.show()
    args = [a for a in (argv or sys.argv)[1:] if not a.startswith("-")]
    if args:
        p = args[0]
        if p.lower().endswith(".trproj"):
            QTimer.singleShot(0, lambda: win.load_project_from(p))
        else:
            QTimer.singleShot(0, lambda: win.open_file(p))
    return app.exec()
