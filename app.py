"""TraceView entry point.

  python app.py                      start the desktop app
  python app.py image.pdf            start and open a PDF / image
  python app.py work.trproj          start and resume a work file
  python app.py --selftest           check this installation (no window, no network)
"""
import sys


def selftest():
    """Runs the whole pipeline on a built-in synthetic sheet. Exit code 0 = everything works.
    The result is also written to selftest_result.txt (a windowed .exe has no console)."""
    log_lines = []

    def say(*args):
        text = " ".join(str(x) for x in args)
        log_lines.append(text)
        if sys.stdout is not None:
            print(text)

    try:
        rc = _selftest_body(say)
    except Exception as ex:  # noqa: BLE001
        say("失敗:", ex)
        rc = 1
    try:
        with open("selftest_result.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(log_lines) + "\n" + ("RESULT: OK" if rc == 0 else "RESULT: NG") + "\n")
    except OSError:
        pass
    return rc


def _selftest_body(say):
    import os
    import tempfile

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    import cv2

    from traceview import __version__
    from traceview.engine import Engine
    from traceview.excel_out import export_xlsx
    from traceview.project import load_project, save_project
    from traceview.selftest_sheet import make_sheet

    say(f"TraceView {__version__} selftest")
    rgb = make_sheet()
    with tempfile.TemporaryDirectory() as d:
        img = os.path.join(d, "sheet.png")
        cv2.imwrite(img, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        e = Engine()
        e.open(img)
        e.analyze()
        assert len(e.bands) == 3, f"三段を検出できません（{len(e.bands)}段）"
        assert all(b.complete and len(b.vertices) > 20 for b in e.bands), "追跡結果が不十分です"
        say("  解析: OK（3段・頂点", [len(b.vertices) for b in e.bands], "）")
        proj = os.path.join(d, "w.trproj")
        save_project(e, proj)
        e2 = Engine()
        load_project(e2, proj)
        assert len(e2.bands) == 3
        say("  作業ファイルの保存・再開: OK")
        export_xlsx(e, os.path.join(d, "r.xlsx"))
        say("  Excel出力: OK")
        from PySide6.QtWidgets import QApplication
        from traceview.gui.main_window import MainWindow
        app = QApplication.instance() or QApplication([])
        win = MainWindow(sync_mode=True)
        win.open_file(img)
        win.action_analyze()
        assert len(win.engine.bands) == 3
        say("  画面: OK")
    say("すべて正常です")
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    from traceview.gui.main_window import main
    sys.exit(main())
