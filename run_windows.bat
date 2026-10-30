@echo off
rem TraceView: start from source (needs Python + "pip install -r requirements.txt", see docs\WINDOWS配布.md)
chcp 65001 >nul
cd /d "%~dp0"
python app.py %*
if errorlevel 1 (
  echo.
  echo 起動に失敗しました。docs\WINDOWS配布.md の「困ったとき」を確認してください。
  pause
)
