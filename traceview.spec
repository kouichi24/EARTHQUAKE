# PyInstaller spec: builds a self-contained folder (dist/TraceView) that runs without Python and
# without a network connection. Build on Windows to get TraceView.exe:  build_windows.bat
# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_submodules

hidden = collect_submodules("skimage") + collect_submodules("scipy.signal") + ["scipy.special._cdflib"]

a = Analysis(
    ["app.py"],
    pathex=["."],
    binaries=[],
    datas=[],
    hiddenimports=hidden,
    hookspath=[],
    runtime_hooks=[],
    # Only QtCore / QtGui / QtWidgets are used: leave the large unused Qt modules out.
    excludes=["PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets", "PySide6.QtQml", "PySide6.QtQuick",
              "PySide6.Qt3DCore", "PySide6.QtMultimedia", "PySide6.QtPdf", "PySide6.QtCharts",
              "tkinter", "matplotlib", "pytest"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="TraceView",
    debug=False,
    strip=False,
    upx=False,
    console=False,          # no console window; use run_windows.bat to see messages
)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="TraceView")
