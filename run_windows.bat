@echo off
rem TraceView: start from source (needs Python, see docs\WINDOWS配布.md)
chcp 65001 >nul
cd /d "%~dp0"
rem Prefer the "py" launcher (uses the newest installed Python 3), else plain "python"
set PY=python
where py >nul 2>&1 && set PY=py -3
rem First start on a PC: install the libraries (needs the network once)
%PY% -c "import cv2, numpy, scipy, skimage, PySide6, pymupdf, openpyxl, PIL" >nul 2>&1
if errorlevel 1 (
  echo 必要なライブラリが入っていません。導入します（ネットワークが必要です。数分かかります）
  %PY% -m pip install -r requirements.txt
  if errorlevel 1 (
    echo.
    echo ライブラリの導入に失敗しました。ネットワークとPythonのバージョンを確認してください。
    pause
    exit /b 1
  )
)
%PY% app.py %*
if errorlevel 1 (
  echo.
  echo 起動に失敗しました。docs\WINDOWS配布.md の「困ったとき」を確認してください。
  pause
)
