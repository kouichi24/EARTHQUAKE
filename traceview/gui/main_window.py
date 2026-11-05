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
                               QScrollArea, QSpinBox, QTableWidget, QTableWidgetItem, QTabWidget, QToolBar,
                               QVBoxLayout, QWidget)

from .. import __version__
from ..engine import Engine, EngineError
from ..excel_out import PendingEditsError, export_xlsx
from ..imageio import page_count
from ..measure import COUNT_MODES, Calibration, arc_active, corrected_path, corrected_x_work
from ..profiles import PROFILES, get as get_profile
from ..project import SourceMismatch, load_project, save_project
from ..svg_out import export_svg
from ..verify import KINDS as VERIFY_KINDS, find_problems, verify_and_repair
from .view import ImageView, LineItem, PyramidItem, VertexItem, make_path_item, polyline_path

ISSUE_LABEL = {"ambiguous": "所属の曖昧さ", "untracked_end": "末端の未追跡", "inferred": "推定接続",
               "excursion": "他の段への逸脱", "no_origin": "起点未検出", "retrace_failed": "再追跡の失敗",
               "verify": "検証", "not_traced": "未解析", "shared_line": "確認済みの段と線が重なる"}
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
        self.follow_line = True
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
        self.rev_items = []
        self.excl_items = []
        self.ver_range = None
        self.ver_range_item = None
        self._update_enabled()
        self._set_state("読込", "ファイル未選択")
        if not sync_mode:
            self._restore_layout()

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
        self.cb_step = QCheckBox("一段ずつ順番に解析")
        self.cb_step.setToolTip("オン: 自動解析で、まず上段だけを解析します。確認・修正してから「次の段を解析」(F8)で、\n次の段へ進みます（確認済みの段の線は、あとの段が使いません）。オフ: 上段から順に、まとめて解析します")
        self.cb_gap = QCheckBox("途切れを推定でつなぐ")
        self.cb_gap.setChecked(True)
        self.cb_gap.setToolTip("オン: 画像に薄い線の痕跡がある所だけ推定でつなぐ（黄色の破線で表示）。\nオフ: 途切れはつながず、未追跡として示す")
        self.cb_gap.toggled.connect(self._gap_toggled)
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
        tb.addWidget(self.cb_gap)
        tb.addWidget(self.cb_step)
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
        self.t_read = self._act("値を読む", lambda: self.set_tool("read"), "V", True,
                                "トレース上をクリックして、その点の値（補正前・円弧補正後）を読み取る")
        self.t_excl = self._act("除外範囲を囲む", lambda: self.set_tool("excl"), "E", True,
                                "文字・汚れなど、線として取りたくない所を四角で囲む（その中の線は使われません）")
        for a in (self.t_select, self.t_add, self.t_origin, self.t_section, self.t_read, self.t_excl):
            tb2.addAction(a)
        tb2.addSeparator()
        self.a_verify = self._act("検証して引き直す", lambda: self.action_verify(True), "F7",
                                  tip="トレース線の不自然な所を検証し、引き直して、また検証（上限回数まで）。引き直した結果は履歴に残ります")
        self.a_next_band = self._act("次の段を解析", self.action_next_band, "F8",
                                     tip="まだ解析していない次の段を解析します（前の段の線は避けます。確認済みの段の線は使いません）")
        self.a_follow = self._act("後の段を引き直す", self.action_retrace_following, None,
                                  tip="選択中の段を直したあと、それより下の段を、直した線を避けて引き直します")
        self.a_apply = self._act("再追跡を適用", self.action_apply, "Ctrl+Return")
        self.a_cancel_pending = self._act("修正を取り消す", self.action_cancel_pending, "Esc")
        self.cb_follow = QCheckBox("線に沿って動かす")
        self.cb_follow.setChecked(True)
        self.cb_follow.setToolTip("オン: 頂点をドラッグすると、画像の線の上だけを動く（つながっていない別の線へは飛ばない）。\nオフ: 自由に動かせる")
        self.cb_follow.toggled.connect(lambda v: setattr(self, "follow_line", v))
        self.cb_auto = QCheckBox("移動後に自動で再追跡")
        self.cb_auto.setChecked(True)
        self.cb_auto.toggled.connect(lambda v: setattr(self, "auto_retrace", v))
        tb2.addAction(self.a_next_band)
        tb2.addAction(self.a_follow)
        tb2.addAction(self.a_verify)
        tb2.addAction(self.a_apply)
        tb2.addAction(self.a_cancel_pending)
        tb2.addWidget(self.cb_follow)
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
        self.cb_arc = QCheckBox("円弧補正後")
        self.cb_arc.setChecked(True)
        self.cb_arc.setToolTip("円弧補正後のトレース（点線）を重ねて表示（校正タブでアーム長とX・Y校正を設定したとき）")
        self.cb_issues = QCheckBox("要確認印")
        self.cb_issues.setChecked(True)
        for cb in (self.cb_image, self.cb_trace, self.cb_arc, self.cb_vertices, self.cb_lines, self.cb_issues):
            cb.toggled.connect(self.apply_visibility)
            tb3.addWidget(cb)
        tb3.addSeparator()
        self.a_hide_all = self._act("図形を全部消す", self.apply_visibility, "H", True,
                                    "トレース・頂点・測定線・区間・印をすべて一時的に隠し、元画像だけを表示（データは消えません）。もう一度押すと戻ります")
        self.a_fit = self._act("全体表示", self.view.fit_all, "Ctrl+0")
        self.a_zoom_sel = self._act("選択頂点へ拡大", self.zoom_to_selected, "Z")
        self.a_play = self._act("再生", self.toggle_replay, "Space", True, "復元順に経路を再生")
        self.a_prev_v = self._act("◀頂点", lambda: self.step_vertex(-1), "Left")
        self.a_next_v = self._act("頂点▶", lambda: self.step_vertex(1), "Right")
        tb3.addAction(self.a_hide_all)
        tb3.addSeparator()
        self.a_panel = self._act("右パネル", self._toggle_panel, "F4", True,
                                 "右のパネル（測定・頂点・区間など）を表示／隠す。隠すと画像エリアが最大になります")
        self.a_panel.setChecked(True)
        tb3.addAction(self.a_panel)
        tb3.addWidget(QLabel(" 画像エリアの幅 "))
        self.width_slider = QSlider(Qt.Horizontal)
        self.width_slider.setRange(30, 90)
        self.width_slider.setValue(70)
        self.width_slider.setMaximumWidth(130)
        self.width_slider.setToolTip("画像エリアの幅（ウィンドウに対する割合 %）。\n画像とパネルの境目を、マウスでドラッグして変えることもできます")
        self.width_slider.valueChanged.connect(self._width_slider_changed)
        tb3.addWidget(self.width_slider)
        self.width_label = QLabel("70%")
        tb3.addWidget(self.width_label)
        tb3.addSeparator()
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
        self.dock = dock
        dock.setObjectName("main_dock")
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
        # confirm tab (one band at a time, checked by the user)
        cw2 = QWidget()
        cl2 = QVBoxLayout(cw2)
        self.conf_title = QLabel("")
        self.conf_title.setStyleSheet("font-weight:bold;font-size:14px")
        cl2.addWidget(self.conf_title)
        cl2.addWidget(QLabel("1段ずつ、①→⑤の順に進めます。"))
        g1 = QGroupBox("① 線が通る上下の範囲を選ぶ")
        l1 = QVBoxLayout(g1)
        self.conf_range_label = QLabel("")
        l1.addWidget(self.conf_range_label)
        r1 = QHBoxLayout()
        bp = QPushButton("画面で上から下へドラッグして選ぶ")
        bp.clicked.connect(lambda: self.set_tool("brange"))
        bc = QPushButton("自動の範囲に戻す")
        bc.clicked.connect(self.action_clear_range)
        r1.addWidget(bp)
        r1.addWidget(bc)
        l1.addLayout(r1)
        f1 = QFormLayout()
        self.rng_top = QDoubleSpinBox()
        self.rng_bot = QDoubleSpinBox()
        for sp in (self.rng_top, self.rng_bot):
            sp.setRange(0, 200000)
            sp.setDecimals(0)
        f1.addRow("上端 Y [px]", self.rng_top)
        f1.addRow("下端 Y [px]", self.rng_bot)
        bn = QPushButton("数値を反映")
        bn.clicked.connect(lambda: self.set_pending_range(self.rng_top.value() * self.engine.scale,
                                                          self.rng_bot.value() * self.engine.scale))
        f1.addRow(bn)
        l1.addLayout(f1)
        cl2.addWidget(g1)
        g2 = QGroupBox("② その範囲まで線が届いているか確認する")
        l2 = QVBoxLayout(g2)
        b2 = QPushButton("範囲の線を確認")
        b2.clicked.connect(self.action_check_range)
        l2.addWidget(b2)
        self.conf_check_label = QLabel("（範囲を選ぶと、自動で確認します）")
        self.conf_check_label.setTextFormat(Qt.RichText)
        l2.addWidget(self.conf_check_label)
        cl2.addWidget(g2)
        g3 = QGroupBox("③ この範囲の中で、線を引き直す")
        l3 = QVBoxLayout(g3)
        b3 = QPushButton("この範囲で引き直す")
        b3.clicked.connect(self.action_retrace_range)
        l3.addWidget(b3)
        l3.addWidget(QLabel("範囲の外が中心の線は使われません（範囲から少し突き出る線は使われます）。戻す(Ctrl+Z)で元に戻せます。"))
        cl2.addWidget(g3)
        g4 = QGroupBox("④ 結果を見て、必要なら直す")
        l4 = QVBoxLayout(g4)
        l4.addWidget(QLabel(
            "・線の位置がずれている頂点 → 頂点を画像上でドラッグ（画像の線の上を動きます。離すと、そこから先を自動で引き直します）\n"
            "・取りこぼした先端 → 「頂点を追加」(A)で先端をクリック\n"
            "・文字・汚れを線と間違えている → 「除外範囲を囲む」(E)で囲む\n"
            "・全体を点検したい → 「検証して引き直す」(F7)"))
        r4 = QHBoxLayout()
        for text, fn in (("頂点を追加", lambda: self.set_tool("add")), ("除外範囲を囲む", lambda: self.set_tool("excl")),
                         ("検証して引き直す", lambda: self.action_verify(True))):
            bb_ = QPushButton(text)
            bb_.clicked.connect(fn)
            r4.addWidget(bb_)
        l4.addLayout(r4)
        cl2.addWidget(g4)
        g5 = QGroupBox("⑤ この段を確認済みにして、次へ")
        l5 = QVBoxLayout(g5)
        b5 = QPushButton("確認済みにして次の段へ")
        b5.clicked.connect(self.action_confirm_and_next)
        l5.addWidget(b5)
        self.conf_next_label = QLabel("")
        l5.addWidget(self.conf_next_label)
        cl2.addWidget(g5)
        cl2.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(cw2)
        self.conf_scroll = scroll
        self.tabs.insertTab(0, scroll, "段の確認")
        self.pending_range = {}
        # vertices tab
        vw = QWidget()
        vl = QVBoxLayout(vw)
        self.v_table = QTableWidget(0, 8)
        self.v_table.setHorizontalHeaderLabels(["#", "X", "Y", "山/谷", "振幅", "由来", "折返し", "補正後X[mm]"])
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
        self.sec_table = QTableWidget(0, 13)
        self.sec_table.setHorizontalHeaderLabels(["区間", "X開始", "X終了", "長さ", "最大", "最小", "上下幅",
                                                  "波の数", "平均周期", "補正後周期", "周波数", "除外", "状態"])
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
        # exclusion tab
        xw = QWidget()
        xl = QVBoxLayout(xw)
        xl.addWidget(QLabel("文字・汚れなど、線として取りたくない所を、「除外範囲を囲む」(E)で四角にドラッグして囲むと、"
                            "その中の線は使われなくなり、該当する段は引き直されます（他の段は変わりません）。"))
        self.excl_table = QTableWidget(0, 5)
        self.excl_table.setHorizontalHeaderLabels(["#", "X", "Y", "幅", "高さ"])
        self.excl_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.excl_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.excl_table.setSelectionMode(QAbstractItemView.SingleSelection)
        xl.addWidget(self.excl_table, 1)
        rx = QHBoxLayout()
        bx1 = QPushButton("選んだ除外範囲を削除")
        bx1.clicked.connect(lambda: self.action_remove_exclusion(False))
        bx2 = QPushButton("すべて削除")
        bx2.clicked.connect(lambda: self.action_remove_exclusion(True))
        rx.addWidget(bx1)
        rx.addWidget(bx2)
        xl.addLayout(rx)
        xl.addWidget(QLabel("削除すると、その範囲の線も再び使われ、該当する段は引き直されます。"
                            "引き直し前の状態は「戻す」(Ctrl+Z)で戻せます。"))
        self.tabs.addTab(xw, "除外")
        # verify tab
        vw2 = QWidget()
        vl2 = QVBoxLayout(vw2)
        vl2.addWidget(QLabel("トレース線を検証し、不自然な所を引き直します。引き直した結果は、採用・不採用を問わず下の履歴に残ります。"))
        rowv = QHBoxLayout()
        self.ver_rounds = QSpinBox()
        self.ver_rounds.setRange(1, 20)
        self.ver_rounds.setValue(5)
        self.ver_rounds.setPrefix("最大 ")
        self.ver_rounds.setSuffix(" 回")
        self.ver_target = QComboBox()
        self.ver_target.addItem("選択中の段")
        self.ver_target.setToolTip("検証・引き直しをする段。「選択中の段」は、右上の段の一覧で選んでいる段です")
        bchk = QPushButton("検証のみ")
        bchk.clicked.connect(lambda: self.action_verify(False))
        brun = QPushButton("検証して引き直す")
        brun.clicked.connect(lambda: self.action_verify(True))
        vl2.addWidget(QLabel("① 対象の段（右上の段の一覧で選ぶか、ここで選ぶ）"))
        vl2.addWidget(self.ver_target)
        vl2.addWidget(QLabel("② 範囲（省略すると、段の全体）"))
        self.ver_range_label = QLabel("範囲: 指定なし（段の全体）")
        vl2.addWidget(self.ver_range_label)
        rowr = QHBoxLayout()
        b_rng = QPushButton("画面で範囲をドラッグして指定")
        b_rng.clicked.connect(lambda: self.set_tool("vrange"))
        b_rng_clear = QPushButton("範囲を解除")
        b_rng_clear.clicked.connect(self.clear_verify_range)
        rowr.addWidget(b_rng)
        rowr.addWidget(b_rng_clear)
        vl2.addLayout(rowr)
        vl2.addWidget(QLabel("③ 実行"))
        for w_ in (self.ver_rounds, bchk, brun):
            rowv.addWidget(w_)
        vl2.addLayout(rowv)
        vl2.addWidget(QLabel("現在の不自然な箇所（ダブルクリックで移動）"))
        self.find_table = QTableWidget(0, 4)
        self.find_table.setHorizontalHeaderLabels(["種類", "X", "Y", "内容"])
        self.find_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.find_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.find_table.cellDoubleClicked.connect(self.on_finding_activated)
        vl2.addWidget(self.find_table, 1)
        vl2.addWidget(QLabel("引き直しの履歴（選ぶと、引き直し前＝灰色の点線／後＝緑の線を表示。消えません）"))
        self.rev_table = QTableWidget(0, 7)
        self.rev_table.setHorizontalHeaderLabels(["実行", "回", "種類", "X", "Y", "結果", "点数 前→後"])
        self.rev_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.rev_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.rev_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.rev_table.itemSelectionChanged.connect(self.on_revision_row)
        vl2.addWidget(self.rev_table, 1)
        self.ver_note = QLabel("")
        self.ver_note.setWordWrap(True)
        vl2.addWidget(self.ver_note)
        self.tabs.addTab(vw2, "検証")
        # read tab
        rw = QWidget()
        rl = QVBoxLayout(rw)
        rl.addWidget(QLabel("「値を読む」(V)でトレース上をクリックすると、その点の値を補正前・円弧補正後で並べて記録します。"))
        self.read_table = QTableWidget(0, 8)
        self.read_table.setHorizontalHeaderLabels(["段", "X[px]", "Y[px]", "振幅", "X[mm]", "時間[s]",
                                                   "補正後X[mm]", "補正後時間[s]"])
        self.read_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.read_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        rl.addWidget(self.read_table, 1)
        rb = QPushButton("選んだ読取点を削除")
        rb.clicked.connect(self.action_remove_readout)
        rl.addWidget(rb)
        self.read_note = QLabel("")
        self.read_note.setWordWrap(True)
        rl.addWidget(self.read_note)
        self.tabs.addTab(rw, "読取")
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
        for lab in w.findChildren(QLabel):       # long explanatory texts must wrap, or they force the panel wide
            lab.setWordWrap(True)
        dock.setWidget(w)
        dock.setMinimumWidth(240)          # narrow enough to give the image area most of the window
        self.addDockWidget(Qt.RightDockWidgetArea, dock)
        self.resizeDocks([dock], [430], Qt.Horizontal)

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
            txt = f"X {p.x() * inv:.0f}  Y {p.y() * inv:.0f} px"
            b = self.current_band()
            cal = e.calibration
            if b is not None and cal.mm_per_px_x > 0:
                txt += f"  ({p.x() * inv * cal.mm_per_px_x:.1f} mm)"
            if b is not None:
                txt += f"  振幅 {cal.to_y_units((b.line_zero - p.y()) * inv):.1f}"
            self.lbl_cursor.setText(txt)

    def _update_enabled(self):
        has_img = self.engine.work is not None
        has_b = bool(self.engine.bands)
        self.a_analyze.setEnabled(has_img)
        self.a_compare.setEnabled(has_img)
        self._update_step_actions()
        self.profile_combo.setEnabled(has_img)
        self.cb_gap.setEnabled(has_img)
        self.cb_gap.blockSignals(True)
        self.cb_gap.setChecked(self.engine.params.gap_mode != "off")
        self.cb_gap.blockSignals(False)
        self._sync_profile_combo()
        for a in (self.a_save, self.a_xlsx, self.a_svg, self.a_undo, self.a_redo, self.a_reset, self.a_apply, self.a_verify,
                  self.a_cancel_pending, self.t_add, self.t_origin, self.t_section, self.t_read, self.t_excl, self.a_play, self.a_prev_v, self.a_next_v,
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
            return e.analyze(progress=cb, stepwise=self.cb_step.isChecked())

        def done(_b):
            self.dirty = True
            self._set_state("解析", f"完了（{len(e.bands)}段を検出）")
            if self.cb_step.isChecked():
                self.tabs.setCurrentIndex(0)             # "段の確認": the steps for checking one band at a time
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

    def _gap_toggled(self, on):
        e = self.engine
        mode = "evidence" if on else "off"
        if e.work is None or e.params.gap_mode == mode:
            return
        if e.bands and not self._confirm("設定を変えると解析をやり直します（現在のトレースと修正は初期化）。よろしいですか？"):
            self._update_enabled()
            return
        e.params.gap_mode = mode
        e.graph = None
        e.prepared = None
        e.bands = []
        self.dirty = True
        self.rebuild_scene()
        self._update_enabled()
        self.action_analyze(force=True)

    # ------------------------------------------------------ one band checked by the user
    def _band_range(self, bi):
        """The range to show: the one just picked, else the one in use, else None."""
        return self.pending_range.get(bi) or self.engine.band_ranges.get(bi)

    def set_pending_range(self, y0, y1):
        if not self.engine.bands:
            return
        top, bot = min(y0, y1), max(y0, y1)
        if bot - top < 20:
            self._info("範囲が狭すぎます")
            return
        self.pending_range[self.sel_band] = (top, bot)
        self.set_tool("select")
        self._draw_range_items()
        self.update_confirm_tab()
        self.action_check_range()

    def action_check_range(self):
        e = self.engine
        rng = self._band_range(self.sel_band)
        if rng is None or not e.bands:
            self.conf_check_label.setText("先に ① で範囲を選んでください。")
            return
        r = e.check_range(self.sel_band, *rng)
        inv = 1 / e.scale
        ok, ng = "<span style='color:#2e7d32'>◯</span>", "<span style='color:#c62828'>×</span>"
        if not r["traced"]:
            self.conf_check_label.setText(f"{ng} この段は、まだ解析していません。")
            return
        lines = [f"{ok if r['origin_inside'] else ng} 起点（左端の線）は、範囲の{'中' if r['origin_inside'] else '外'}です"]
        for name, reach, ty, tips, edge in (("上端", r["reaches_top"], r["trace_top"], r["tips_top"], r["top"]),
                                            ("下端", r["reaches_bottom"], r["trace_bottom"], r["tips_bottom"], r["bottom"])):
            txt = (f"{ok if reach else ng} {name}: 範囲の{name} Y={edge * inv:.0f}px に対し、現在の線は Y={ty * inv:.0f}px まで"
                   f"{'届いています' if reach else '届いていません'}")
            if tips:
                txt += f"（{name}の近くに線の先端が{tips}本あります）"
            elif not reach:
                txt += "（{}の近くに、この段の線の先端は見つかりません。範囲が広すぎるかもしれません）".format(name)
            lines.append(txt)
        if r["outside_px"] > 0:
            lines.append(f"△ 範囲の外にある線の長さ: 約{r['outside_px']:.0f}px（引き直すと、範囲の外が中心の線は使われません）")
        good = r["origin_inside"] and (r["reaches_top"] or r["tips_top"]) and (r["reaches_bottom"] or r["tips_bottom"])
        lines.append("<b>" + ("→ この範囲で引き直せます（③へ）" if good else "→ 範囲を調整してから、引き直してください") + "</b>")
        self.conf_check_label.setText("<br>".join(lines))
        self.check_result = r

    def action_retrace_range(self):
        e = self.engine
        rng = self._band_range(self.sel_band)
        if rng is None or not e.bands:
            self._info("先に ① で範囲を選んでください")
            return
        bi = self.sel_band
        self._set_state("再追跡", "範囲内で引き直し中…")

        def job(cb):
            return e.retrace_in_range(bi, rng[0], rng[1], cb)

        def done(_nb):
            self.pending_range.pop(bi, None)
            self.dirty = True
            self._set_state("再追跡", f"{e.bands[bi].name}を範囲内で引き直しました")
            self._build_band_items(bi)
            self.apply_visibility()
            self.refresh_panels()
            self._draw_range_items()

        self.run_task("範囲内で引き直しています…", job, done,
                      lambda m: (self._set_state("再追跡", "失敗"), self._show_error(m)))

    def action_clear_range(self):
        e = self.engine
        bi = self.sel_band
        if not e.bands:
            return
        self.pending_range.pop(bi, None)
        if bi not in e.band_ranges:
            self._draw_range_items()
            self.update_confirm_tab()
            return

        def job(cb):
            return e.clear_range(bi, cb)

        def done(_nb):
            self.dirty = True
            self._build_band_items(bi)
            self.apply_visibility()
            self.refresh_panels()
            self._draw_range_items()

        self.run_task("自動の範囲で引き直しています…", job, done)

    def action_confirm_and_next(self):
        e = self.engine
        b = self.current_band()
        if b is None:
            return
        if e.is_pending(self.sel_band) if hasattr(e, "is_pending") else bool(b.pending):
            self._show_error("未適用の修正があります。再追跡を適用するか、取り消してから進めてください")
            return
        b.confirmed = True
        self.dirty = True
        self.cb_confirm.blockSignals(True)
        self.cb_confirm.setChecked(True)
        self.cb_confirm.blockSignals(False)
        if e.untraced_bands():
            self.action_next_band()
        elif self.sel_band + 1 < len(e.bands):
            self.band_list.setCurrentRow(self.sel_band + 1)
            self._info(f"{b.name}を確認済みにしました。次は{e.bands[self.sel_band].name}を確認してください")
        else:
            self._info("すべての段を確認済みにしました")
        self.refresh_panels()

    def update_confirm_tab(self):
        e = self.engine
        b = self.current_band()
        if b is None:
            self.conf_title.setText("画像を開いて、自動解析してください")
            return
        n = len(e.bands)
        self.conf_title.setText(f"いま確認する段: {b.name}（{self.sel_band + 1}/{n}段目）　状態: {e.value_status(b)}")
        rng = self._band_range(self.sel_band)
        inv = 1 / e.scale
        if rng is None:
            self.conf_range_label.setText("範囲: 自動（隣の段との中間まで）")
        else:
            using = self.sel_band in e.band_ranges and e.band_ranges[self.sel_band] == rng
            self.conf_range_label.setText(f"範囲: Y {rng[0] * inv:.0f} ～ {rng[1] * inv:.0f} px"
                                          + ("（この範囲で解析済み）" if using else "（選んだだけ。③で引き直すと反映）"))
            self.rng_top.setValue(rng[0] * inv)
            self.rng_bot.setValue(rng[1] * inv)
        left = len(e.untraced_bands())
        self.conf_next_label.setText(f"未解析の段が{left}段あります。「確認済みにして次の段へ」で、次の段を解析します。" if left
                                     else ("次の段へ進みます。" if self.sel_band + 1 < n else "これが最後の段です。"))

    def _draw_range_items(self):
        for it in getattr(self, "range_items", []):
            if it.scene() is self.scene:
                self.scene.removeItem(it)
        self.range_items = []
        e = self.engine
        if e.work is None:
            return
        W = e.work.shape[1]
        ranges = {bi: (r, True) for bi, r in e.band_ranges.items()}
        for bi, r in self.pending_range.items():
            ranges[bi] = (r, False)
        for bi, (rng, applied) in ranges.items():
            if bi >= len(e.bands):
                continue
            col = QColor(e.bands[bi].color)
            rc = QGraphicsRectItem(0, rng[0], W, rng[1] - rng[0])
            c2 = QColor(col)
            c2.setAlpha(28 if applied else 50)
            rc.setBrush(QBrush(c2))
            pen = QPen(col, 2, Qt.SolidLine if applied else Qt.DashLine)
            pen.setCosmetic(True)
            rc.setPen(pen)
            rc.setZValue(3)
            rc.setToolTip(f"{e.bands[bi].name}の線が通る範囲" + ("" if applied else "（選んだだけ。引き直すと反映）"))
            self.scene.addItem(rc)
            lbl = QGraphicsSimpleTextItem(f"{e.bands[bi].name}の範囲")
            lbl.setBrush(QBrush(col))
            lbl.setFlag(QGraphicsSimpleTextItem.ItemIgnoresTransformations, True)
            lbl.setPos(6, rng[0] + 4)
            lbl.setZValue(3)
            self.scene.addItem(lbl)
            self.range_items += [rc, lbl]
        self.apply_visibility()

    # ------------------------------------------------ one band after the other
    def _update_step_actions(self):
        e = self.engine
        self.a_next_band.setEnabled(bool(e.bands) and bool(e.untraced_bands()))
        self.a_follow.setEnabled(bool(e.bands) and self.sel_band < len(e.bands) - 1)
        left = len(e.untraced_bands()) if e.bands else 0
        self.a_next_band.setText(f"次の段を解析（残り{left}段）" if left else "次の段を解析")

    def action_next_band(self):
        e = self.engine
        if not e.untraced_bands():
            return
        self._set_state("解析", "次の段を解析中…")

        def job(cb):
            return e.trace_next(cb)

        def done(i):
            if i is None:
                return
            self.dirty = True
            self.sel_band = i
            self._set_state("解析", f"{e.bands[i].name}を解析しました（残り{len(e.untraced_bands())}段）")
            self._build_band_items(i)
            self.apply_visibility()
            self.band_list.setCurrentRow(i)
            self.refresh_panels()
            self._update_step_actions()

        self.run_task("次の段を解析しています…", job, done,
                      lambda m: (self._set_state("解析", "失敗"), self._show_error(m)))

    def action_retrace_following(self):
        e = self.engine
        i = self.sel_band
        if not e.bands or i >= len(e.bands) - 1:
            return
        if not self._confirm(f"{e.bands[i].name}より下の段を、{e.bands[i].name}の線を避けて引き直します（それらの段の修正は初期化。戻す(Ctrl+Z)で戻せます）。よろしいですか？"):
            return
        self._set_state("解析", "後の段を引き直し中…")

        def job(cb):
            return e.retrace_following(i, cb)

        def done(js):
            self.dirty = True
            self._set_state("解析", f"{len(js)}段を引き直しました")
            self._build_all_items()
            self.refresh_panels()
            self._update_step_actions()

        self.run_task("後の段を引き直しています…", job, done,
                      lambda m: (self._set_state("解析", "失敗"), self._show_error(m)))

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
            self._prepare_for_editing()

        def fail(msg):
            if "見つかりません" in msg and not self.sync_mode:
                p, _ = QFileDialog.getOpenFileName(self, "元ファイルを選択", self._last_dir(), FILE_FILTER)
                if p:
                    self.load_project_from(path, p)
                    return
            self._show_error(msg)

        self.run_task("作業ファイルを読み込み中…", job, done, fail)

    def _prepare_for_editing(self):
        """Build the image graph after a work file is loaded (needed to move vertices along the line)."""
        e = self.engine
        if e.graph is not None or not e.bands:
            return
        self._set_state("解析", "編集の準備中（線の構造を作成）…")
        self.run_task("編集の準備をしています…", lambda cb: e.ensure_graph(cb),
                      lambda _r: self._set_state("解析", f"保存済みの結果（{len(e.bands)}段）・編集の準備完了"),
                      lambda m: self._set_state("解析", "編集の準備に失敗"))

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

    # ------------------------------------------------------------ layout (width of the image area)
    def _toggle_panel(self, *_):
        self.dock.setVisible(self.a_panel.isChecked())

    def _width_slider_changed(self, v):
        """Set the image area to v % of the window by resizing the right panel."""
        self.width_label.setText(f"{v}%")
        if not self.dock.isVisible():
            return
        total = max(1, self.width())
        self.resizeDocks([self.dock], [max(240, int(total * (100 - v) / 100))], Qt.Horizontal)
        QTimer.singleShot(0, self._sync_width_slider)      # show the width that was really possible

    def _sync_width_slider(self):
        """Keep the slider in step when the border between image and panel is dragged with the mouse."""
        if not self.dock.isVisible() or self.width() <= 0:
            return
        v = int(round(100 * (1 - self.dock.width() / self.width())))
        v = max(self.width_slider.minimum(), min(self.width_slider.maximum(), v))
        self.width_slider.blockSignals(True)
        self.width_slider.setValue(v)
        self.width_slider.blockSignals(False)
        self.width_label.setText(f"{v}%")

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self._sync_width_slider()

    def _save_layout(self):
        st = QSettings("TraceView", "TraceView")
        st.setValue("geometry", self.saveGeometry())
        st.setValue("windowState", self.saveState())

    def _restore_layout(self):
        st = QSettings("TraceView", "TraceView")
        try:
            g, ws = st.value("geometry"), st.value("windowState")
            if g:
                self.restoreGeometry(g)
            if ws:
                self.restoreState(ws)
            self.a_panel.setChecked(self.dock.isVisible())
        except Exception:                      # a damaged saved layout is simply ignored
            pass

    def closeEvent(self, ev):
        if self._confirm_discard():
            self._save_layout()
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
        self.rev_items = []
        self.excl_items = []
        self.range_items = []
        self.ver_range_item = None
        self.pix_item = None
        if e.work is None:
            return
        h, w = e.work.shape[:2]
        self.pix_item = PyramidItem(e.work)        # light display copies; analysis keeps the full image
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
                   *d.get("sections", []), *d.get("arc", []), *d.get("reads", [])]:
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
        d["arc"] = []
        cal = e.calibration
        if arc_active(cal) and len(b.path) > 1:
            cp = corrected_path(b.path, b.line_zero, e.scale, cal)
            simp = cv2.approxPolyDP(cp.astype(np.float32).reshape(-1, 1, 2), 0.7, False).reshape(-1, 2)
            col = QColor(b.color).darker(180)
            it = make_path_item(polyline_path(simp), col.name(), 1, dashed=True, z=11)
            it.pen().setStyle(Qt.DotLine)
            self.scene.addItem(it)
            d["arc"].append(it)
        d["reads"] = []
        for k, rd in enumerate(e.readouts):
            if rd["band"] != bi:
                continue
            m = QGraphicsEllipseItem(-6, -6, 12, 12)
            m.setPos(rd["x"], rd["y"])
            m.setBrush(QBrush(QColor(255, 255, 255, 80)))
            m.setPen(QPen(QColor("#00acc1"), 2))
            m.setFlag(QGraphicsEllipseItem.ItemIgnoresTransformations, True)
            m.setZValue(45)
            m.setToolTip(self._format_readout(e.readout_row(rd)))
            self.scene.addItem(m)
            lbl = QGraphicsSimpleTextItem(str(k + 1))
            lbl.setBrush(QBrush(QColor("#00838f")))
            lbl.setFlag(QGraphicsSimpleTextItem.ItemIgnoresTransformations, True)
            lbl.setPos(rd["x"] + 6, rd["y"] - 18)
            lbl.setZValue(45)
            self.scene.addItem(lbl)
            d["reads"] += [m, lbl]
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
        hide = self.a_hide_all.isChecked()
        self.a_hide_all.setText("図形を表示に戻す" if hide else "図形を全部消す")
        for extra in (self.sel_ring, self.replay_marker, self.section_preview, *self.rev_items, *self.excl_items,
                      self.ver_range_item, *getattr(self, "range_items", [])):
            if extra is not None and extra.scene() is self.scene:
                extra.setVisible(not hide)
        if self.pix_item is not None:
            self.pix_item.setVisible(self.cb_image.isChecked())
            self.scene.setBackgroundBrush(QBrush(QColor("white"))) if not self.cb_image.isChecked() else None
        # thousands of vertex markers are unreadable when zoomed out: show them from a moderate zoom
        zoomed = self.view.transform().m11() >= 0.4
        for bi, d in self.band_items.items():
            vis_band = (self.band_list.item(bi) is not None and self.band_list.item(bi).checkState() == Qt.Checked
                        and not hide)
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
            for it in d.get("arc", []):
                it.setVisible(self.cb_arc.isChecked() and tr)
            for it in d.get("reads", []):
                it.setVisible(vis_band)

    def set_tool(self, tool):
        self.tool = tool
        for name, act in (("select", self.t_select), ("add", self.t_add), ("origin", self.t_origin),
                          ("section", self.t_section), ("read", self.t_read), ("excl", self.t_excl),
                          ("vrange", None), ("brange", None)):
            if act is None:
                continue
            act.setChecked(name == tool)
        self.view.click_tool = tool in ("add", "origin", "calib", "read")
        self.view.drag_tool = tool in ("section", "excl", "vrange", "brange")
        for d in self.band_items.values():
            for it in d["vertices"]:
                it.set_editable(tool == "select")
        hint = {"select": "頂点をドラッグして移動／測定線は上下にドラッグ", "add": "追加する頂点（先端）をクリック",
                "origin": "選択中の段の起点（波形の左端）をクリック", "calib": "1点目、2点目の順にクリック",
                "section": "測定したい範囲を横にドラッグ（選択中の段に追加）",
                "read": "値を読みたい点をトレース上でクリック（選択中の段）",
                "excl": "線として取りたくない所（文字など）を、四角でドラッグして囲む",
                "vrange": "検証・引き直しをする範囲を、横にドラッグして指定",
                "brange": "この段の線が通る範囲を、上から下へドラッグして指定（横の位置は関係ありません）"}[tool]
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
        if self.ver_range is not None:
            self._show_verify_range()

    def refresh_panels(self):
        self.update_measure_table()
        self.update_vertex_table()
        self.update_issue_list()
        self.update_section_table()
        self.update_read_table()
        self.update_verify_tables()
        self._update_verify_target_combo()
        self.update_exclusion_table()
        self._update_step_actions()
        self.update_confirm_tab()
        self._draw_range_items()
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
                    "先端で反転" if v.turn == "end" else "曲がり",
                    (f"{corrected_x_work(v.x, v.y, b.line_zero, e.scale, e.calibration) * inv * e.calibration.mm_per_px_x:.2f}"
                     if arc_active(e.calibration) else "—")]
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

    def make_line_constraint(self, x, y):
        """Constraint keeping a dragged vertex on the image's line; falls back to free movement without a graph."""
        lc = self.engine.line_constraint(x, y)
        if lc is None:
            self._info("線に沿う移動は、解析後（または編集の準備後）に使えます。いまは自由に動きます")
        return lc

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
        elif self.tool == "read":
            row = e.read_point(self.sel_band, pt.x(), pt.y())
            if row is None:
                self._info("近くにトレースがありません（選択中の段のトレース上をクリックしてください）")
                return
            self.dirty = True
            self._build_band_items(self.sel_band)
            self.apply_visibility()
            self.update_read_table()
            self.tabs.setCurrentIndex(self.tabs.count() - 3)
            self._info(self._format_readout(row))
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

    # ------------------------------------------------------------ exclusions
    def add_exclusion_rect(self, x0, y0, x1, y1):
        e = self.engine
        self._set_state("解析", "除外範囲を反映中…")

        def job(cb):
            return e.add_exclusion(x0, y0, x1, y1, cb)

        def done(affected):
            self.dirty = True
            self._set_state("解析", f"除外範囲を追加（引き直した段: {', '.join(e.bands[i].name for i in affected) or 'なし'}）")
            self._build_all_items()
            self.refresh_panels()
            self.tabs.setCurrentWidget(self.excl_table.parentWidget())
            self.set_tool("select")

        self.run_task("除外範囲を反映しています…", job, done,
                      lambda m: (self._set_state("解析", "失敗"), self._show_error(m)))

    def action_remove_exclusion(self, all_=False):
        e = self.engine
        rows = self.excl_table.selectionModel().selectedRows()
        if not all_ and not rows:
            return
        ks = list(range(len(e.exclusions)))[::-1] if all_ else [rows[0].row()]
        if all_ and not self._confirm("除外範囲をすべて削除し、線を引き直します。よろしいですか？"):
            return

        def job(cb):
            return [e.remove_exclusion(k, cb) for k in ks]

        def done(_r):
            self.dirty = True
            self._build_all_items()
            self.refresh_panels()
            self._set_state("解析", "除外範囲を削除しました")

        self.run_task("除外範囲を外しています…", job, done)

    def update_exclusion_table(self):
        e = self.engine
        t = self.excl_table
        t.setRowCount(len(e.exclusions))
        inv = 1 / e.scale
        for i, r in enumerate(e.exclusions):
            for c, txt in enumerate([str(i + 1), f"{r['x0'] * inv:.0f}", f"{r['y0'] * inv:.0f}",
                                     f"{(r['x1'] - r['x0']) * inv:.0f}", f"{(r['y1'] - r['y0']) * inv:.0f}"]):
                t.setItem(i, c, QTableWidgetItem(txt))
        t.resizeColumnsToContents()
        self._draw_exclusions()

    def _draw_exclusions(self):
        for it in self.excl_items:
            if it.scene() is self.scene:
                self.scene.removeItem(it)
        self.excl_items = []
        for i, r in enumerate(self.engine.exclusions):
            rc = QGraphicsRectItem(r["x0"], r["y0"], r["x1"] - r["x0"], r["y1"] - r["y0"])
            rc.setBrush(QBrush(QColor(244, 67, 54, 55), Qt.BDiagPattern))
            pen = QPen(QColor("#f44336"), 1, Qt.DashLine)
            pen.setCosmetic(True)
            rc.setPen(pen)
            rc.setZValue(5)
            rc.setToolTip(f"除外範囲 {i + 1}（この中の線は使いません）")
            self.scene.addItem(rc)
            lbl = QGraphicsSimpleTextItem(f"除外{i + 1}")
            lbl.setBrush(QBrush(QColor("#f44336")))
            lbl.setFlag(QGraphicsSimpleTextItem.ItemIgnoresTransformations, True)
            lbl.setPos(r["x0"] + 3, r["y0"] + 3)
            lbl.setZValue(5)
            self.scene.addItem(lbl)
            self.excl_items += [rc, lbl]
        self.apply_visibility()

    # ------------------------------------------------- verify: target and range
    def _verify_targets(self):
        e = self.engine
        k = self.ver_target.currentIndex()
        if k == 0:
            return [self.sel_band]
        if k == self.ver_target.count() - 1:
            return list(range(len(e.bands)))
        return [k - 1]

    def _update_verify_target_combo(self):
        names = [b.name for b in self.engine.bands]
        cur = self.ver_target.currentIndex()
        self.ver_target.blockSignals(True)
        self.ver_target.clear()
        b = self.current_band()
        self.ver_target.addItem(f"選択中の段（いま: {b.name if b else '—'}）")
        for n in names:
            self.ver_target.addItem(n)
        self.ver_target.addItem("全段")
        self.ver_target.setCurrentIndex(max(0, min(cur, self.ver_target.count() - 1)))
        self.ver_target.blockSignals(False)

    def set_verify_range(self, x0, x1):
        if abs(x1 - x0) < 5:
            return
        self.ver_range = (min(x0, x1), max(x0, x1))
        self._show_verify_range()
        self.set_tool("select")
        self.tabs.setCurrentWidget(self.find_table.parentWidget())

    def clear_verify_range(self):
        self.ver_range = None
        self._show_verify_range()

    def _show_verify_range(self):
        if self.ver_range_item is not None and self.ver_range_item.scene() is self.scene:
            self.scene.removeItem(self.ver_range_item)
        self.ver_range_item = None
        inv = 1 / self.engine.scale if self.engine.work is not None else 1
        if self.ver_range is None:
            self.ver_range_label.setText("範囲: 指定なし（段の全体）")
        else:
            x0, x1 = self.ver_range
            self.ver_range_label.setText(f"範囲: X {x0 * inv:.0f} ～ {x1 * inv:.0f} px（この範囲の不自然な所だけを引き直す）")
            y0, y1 = (self._section_y_range(self.sel_band) if self.engine.bands else (0, self.engine.work.shape[0]))
            it = QGraphicsRectItem(x0, y0, x1 - x0, y1 - y0)
            it.setBrush(QBrush(QColor(33, 150, 243, 40)))
            pen = QPen(QColor("#2196f3"), 1, Qt.DashLine)
            pen.setCosmetic(True)
            it.setPen(pen)
            it.setZValue(4)
            self.scene.addItem(it)
            self.ver_range_item = it
        self.apply_visibility()

    # --------------------------------------------------------------- verify
    def action_verify(self, repair=True):
        e = self.engine
        if not e.bands:
            return
        targets = self._verify_targets()
        xr = self.ver_range
        rounds = self.ver_rounds.value()
        self._set_state("再追跡", "検証中…")
        self.reports = []

        def job(cb):
            out = []
            for k, bi in enumerate(targets):
                out.append(verify_and_repair(e, bi, max_rounds=rounds,
                                             progress=lambda m, f, k=k: cb(m, (k + f) / len(targets)),
                                             repair_on=repair, x_range=xr))
            return out

        def done(reports):
            self.reports = reports
            self.dirty = True
            for bi in targets:
                self._build_band_items(bi)
            self.apply_visibility()
            self.refresh_panels()
            self.tabs.setCurrentWidget(self.find_table.parentWidget())
            self._set_state("再追跡", "検証完了" if not repair else "検証・引き直し完了")
            lines = [f"{r['band']}: 採用{r['applied']}・不採用{r['rejected']}・失敗{r['failed']}　"
                     f"点数 {r['score_before']:.0f}→{r['score_after']:.0f}" if repair else
                     f"{r['band']}: 不自然な箇所 {r['findings_before']}件" for r in reports]
            self._info(" ／ ".join(lines))

        self.run_task("検証しています…", job, done,
                      lambda m: (self._set_state("再追跡", "失敗"), self._show_error(m)))

    def update_verify_tables(self):
        b = self.current_band()
        ft, rt = self.find_table, self.rev_table
        ft.setRowCount(0)
        rt.blockSignals(True)
        rt.setRowCount(0)
        if b is None:
            rt.blockSignals(False)
            return
        inv = 1 / self.engine.scale
        ft.setRowCount(len(b.findings))
        for i, f in enumerate(b.findings):
            for c, txt in enumerate([VERIFY_KINDS.get(f["kind"], f["kind"]), f"{f['x'] * inv:.0f}", f"{f['y'] * inv:.0f}", f["msg"]]):
                ft.setItem(i, c, QTableWidgetItem(txt))
            ft.item(i, 0).setData(Qt.UserRole, (f["x"], f["y"]))
        ft.resizeColumnsToContents()
        status = {"applied": "採用", "rejected": "不採用（元に戻した）", "failed": "引き直せず"}
        rt.setRowCount(len(b.revisions))
        for i, r in enumerate(b.revisions):
            vals = [str(r["run"]), str(r["round"]), VERIFY_KINDS.get(r["kind"], r["kind"]), f"{r['x'] * inv:.0f}",
                    f"{r['y'] * inv:.0f}", status.get(r["status"], r["status"]),
                    f"{r['score_before']:.0f} → {r['score_after']:.0f}"]
            for c, txt in enumerate(vals):
                rt.setItem(i, c, QTableWidgetItem(txt))
        rt.resizeColumnsToContents()
        rt.blockSignals(False)
        n_app = sum(1 for r in b.revisions if r["status"] == "applied")
        self.ver_note.setText(f"{b.name}: 履歴 {len(b.revisions)}件（採用 {n_app}）。引き直しの実行全体は「戻す」(Ctrl+Z)で一度に取り消せます（履歴は残ります）。"
                              "点数は不自然さの目安で、小さいほど良いですが、正しさの保証ではありません。")

    def on_finding_activated(self, row, _col):
        x, y = self.find_table.item(row, 0).data(Qt.UserRole)
        self.view.center_on_point(x, y, zoom=max(0.8, self.view.transform().m11()))

    def _clear_rev_items(self):
        for it in self.rev_items:
            if it.scene() is self.scene:
                self.scene.removeItem(it)
        self.rev_items = []

    def on_revision_row(self):
        self._clear_rev_items()
        rows = self.rev_table.selectionModel().selectedRows()
        b = self.current_band()
        if not rows or b is None or rows[0].row() >= len(b.revisions):
            return
        r = b.revisions[rows[0].row()]
        if r["old"]:
            it = make_path_item(polyline_path(r["old"]), "#757575", 3, z=48)
            pen = it.pen()
            pen.setStyle(Qt.DotLine)
            it.setPen(pen)
            self.scene.addItem(it)
            self.rev_items.append(it)
        if r["new"]:
            it = make_path_item(polyline_path(r["new"]), "#00c853" if r["status"] == "applied" else "#ff6d00", 3, z=49)
            self.scene.addItem(it)
            self.rev_items.append(it)
        self.apply_visibility()
        self.view.center_on_point(r["x"], r["y"], zoom=max(1.0, self.view.transform().m11()))

    # --------------------------------------------------------------- readouts
    @staticmethod
    def _format_readout(r):
        txt = f"{r['band']}  X={r['x']:.0f}px  Y={r['y']:.0f}px  振幅={r['amp']:.2f}"
        if r["x_mm"] is not None:
            txt += f"  X={r['x_mm']:.2f}mm"
        if r["t_s"] is not None:
            txt += f"  時間={r['t_s']:.3f}s"
        if r["x_arc_mm"] is not None:
            txt += f"  ／補正後 X={r['x_arc_mm']:.2f}mm"
            if r["t_arc_s"] is not None:
                txt += f"  時間={r['t_arc_s']:.3f}s"
        return txt

    def update_read_table(self):
        e = self.engine
        t = self.read_table
        t.setRowCount(len(e.readouts))

        def f(v, n):
            return "—" if v is None else f"{v:.{n}f}"
        for i, rd in enumerate(e.readouts):
            r = e.readout_row(rd)
            for c, txt in enumerate([r["band"], f(r["x"], 0), f(r["y"], 0), f(r["amp"], 2), f(r["x_mm"], 2),
                                     f(r["t_s"], 3), f(r["x_arc_mm"], 2), f(r["t_arc_s"], 3)]):
                t.setItem(i, c, QTableWidgetItem(txt))
        t.resizeColumnsToContents()
        cal = e.calibration
        self.read_note.setText(
            "時間は各段の起点（左端）からの値です。" +
            ("円弧補正を表示中。" if arc_active(cal) else "円弧補正は、「校正」タブでペンアーム長とX・Y校正を設定すると有効になります。"))

    def action_remove_readout(self):
        rows = self.read_table.selectionModel().selectedRows()
        if rows and self.engine.remove_readout(rows[0].row()):
            self.dirty = True
            self._build_all_items()
            self.update_read_table()

    def _build_all_items(self):
        for bi in range(len(self.engine.bands)):
            self._build_band_items(bi)
        self.apply_visibility()

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
        if self.tool == "brange":
            y0, y1 = a.y(), b.y()
            color, pen_col = QColor(76, 175, 80, 60), "#4caf50"
        elif self.tool == "excl":
            y0, y1 = a.y(), b.y()
            color, pen_col = QColor(244, 67, 54, 70), "#f44336"
        else:
            y0, y1 = self._section_y_range(self.sel_band)
            color, pen_col = ((QColor(33, 150, 243, 60), "#2196f3") if self.tool == "vrange"
                              else (QColor(255, 235, 59, 60), "#ffeb3b"))
        if self.section_preview is None:
            it = QGraphicsRectItem()
            it.setZValue(8)
            self.scene.addItem(it)
            self.section_preview = it
        self.section_preview.setBrush(QBrush(color))
        self.section_preview.setPen(QPen(QColor(pen_col), 1, Qt.DashLine))
        if self.tool == "brange":
            self.section_preview.setRect(0, min(y0, y1), self.engine.work.shape[1], abs(y1 - y0))
        else:
            self.section_preview.setRect(min(a.x(), b.x()), min(y0, y1), abs(b.x() - a.x()), abs(y1 - y0))

    def _section_drag_done(self, a, b):
        if self.section_preview is not None:
            self.scene.removeItem(self.section_preview)
            self.section_preview = None
        if self.tool == "brange":
            self.set_pending_range(a.y(), b.y())
        elif self.tool == "excl":
            self.add_exclusion_rect(a.x(), a.y(), b.x(), b.y())
        elif self.tool == "vrange":
            self.set_verify_range(a.x(), b.x())
        else:
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
                     (f(r["period_arc_s"], 3, " s") if r["period_arc_s"] is not None else f(r["period_arc_px"], 1, " px")),
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
        self._build_all_items()
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
