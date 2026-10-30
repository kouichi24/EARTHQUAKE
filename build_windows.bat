@echo off
rem TraceView: build a distributable folder (dist\TraceView\TraceView.exe). Run on Windows.
chcp 65001 >nul
cd /d "%~dp0"
set PY=python
where py >nul 2>&1 && set PY=py -3
set PY=python
where py >nul 2>nul && set PY=py
echo [1/4] 仮想環境を作成します
%PY% -m venv .venv || goto :error
call .venv\Scripts\activate.bat || goto :error
echo [2/4] 必要なライブラリを入れます（この手順だけネットワークが必要です）
python -m pip install --upgrade pip || goto :error
pip install -r requirements-dev.txt || goto :error
echo [3/4] 自己診断（画面なし）
python app.py --selftest || goto :error
echo [4/4] exe を作成します
pyinstaller traceview.spec --noconfirm --clean || goto :error
echo.
echo 作成した exe の自己診断:
pushd dist\TraceView
start /wait "" TraceView.exe --selftest
type selftest_result.txt
popd
echo.
echo 完了: dist\TraceView フォルダごと配布してください（TraceView.exe を起動）
pause
exit /b 0
:error
echo.
echo 失敗しました。上のメッセージを確認してください。
pause
exit /b 1
