"""Slice the traced lines along x, keep every point's numbers, draw again from the numbers (window and CSV)."""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tests.synth import make_sheet  # noqa: E402
from traceview.engine import Engine  # noqa: E402
from traceview.slicing import SliceError, available_units, read_csv, save_csv, slice_rows  # noqa: E402


@pytest.fixture(scope="module")
def eng(tmp_path_factory):
    d = tmp_path_factory.mktemp("sl")
    rgb, _ = make_sheet()
    p = d / "s.png"
    cv2.imwrite(str(p), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    e = Engine()
    e.open(str(p))
    e.analyze()
    return e


def test_every_point_is_kept_and_slices_are_ordered(eng):
    rows = slice_rows(eng, 200, "px")
    assert len(rows) == sum(len(b.path) for b in eng.bands)
    for b in eng.bands:
        rs = [r for r in rows if r["band"] == b.name]
        sl = [r["slice"] for r in rs]
        assert sl == sorted(sl) and sl[0] == 1               # slices run left to right along the pen path
        assert all(r["slice_end"] - r["slice_start"] == pytest.approx(200) for r in rs)
    r0 = rows[0]
    assert r0["x_px"] == pytest.approx(eng.bands[0].path[0, 0] / eng.scale, abs=1e-6)


def test_free_step_and_units(eng):
    assert slice_rows(eng, 37.5, "px")
    assert available_units(eng) == ["px"]                    # not calibrated
    with pytest.raises(SliceError):
        slice_rows(eng, 1.0, "mm")
    with pytest.raises(SliceError):
        slice_rows(eng, 0, "px")


def test_csv_roundtrip_and_window_redraw(eng, tmp_path):
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication
    from traceview.gui.slice_window import SliceWindow
    QApplication.instance() or QApplication([])
    rows = slice_rows(eng, 300, "px")
    p = tmp_path / "x.csv"
    save_csv(p, rows, "px", 300)
    back, unit, step = read_csv(p)
    assert unit == "px" and step == 300 and len(back) == len(rows)
    assert back[10]["x_px"] == pytest.approx(rows[10]["x_px"], abs=1e-3)
    w = SliceWindow(back, unit, step)
    n_slices = len({r["slice"] for r in rows})
    lines = [i for i in w.items if hasattr(i, "path") and i.path().elementCount() > 1 and i.pen().style() == Qt.SolidLine]
    assert len(lines) >= n_slices                            # at least one line per slice, drawn from the numbers
    w.list.item(0).setSelected(True)
    assert w.chosen_slices() == [w.slices[0]]
    w.close()


def test_bad_csv_is_refused(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("a,b\n1,2\n", encoding="utf-8")
    with pytest.raises(SliceError):
        read_csv(p)


def test_main_window_slice_tab(eng, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QApplication, QFileDialog
    from traceview.gui.main_window import MainWindow
    from traceview.gui.slice_window import SliceWindow
    QApplication.instance() or QApplication([])
    rgb, _ = make_sheet()
    p = tmp_path / "w.png"
    cv2.imwrite(str(p), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    w = MainWindow(sync_mode=True)
    w.show()
    w.open_file(str(p))
    w.action_analyze()
    w.slice_step.setValue(250)
    w.action_slice_show()
    assert any(isinstance(x, SliceWindow) and x.isVisible() for x in w._slice_windows)
    out = tmp_path / "o.csv"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (str(out), "")))
    w.action_slice_save()
    assert out.exists()
    monkeypatch.setattr(QFileDialog, "getOpenFileName", staticmethod(lambda *a, **k: (str(out), "")))
    n = len(w._slice_windows)
    w.action_slice_open()
    assert len(w._slice_windows) == n + 1
    for x in w._slice_windows:
        x.close()
    w.close()


def test_xlsx_has_numbers_chart_and_slice_filter(eng, tmp_path):
    from openpyxl import load_workbook
    from traceview.slicing import write_xlsx
    rows = slice_rows(eng, 300, "px")
    p = tmp_path / "s.xlsx"
    write_xlsx(p, eng, rows, "px", 300)
    wb = load_workbook(p)
    names = [b.name for b in eng.bands]
    for n in names:
        full, light = wb[f"{n}_数値"], wb[f"{n}_グラフ"]
        n_rows = sum(1 for r in rows if r["band"] == n)
        assert full.max_row == n_rows + 1                    # every point is kept in the numbers sheet
        assert 2 < light.max_row <= full.max_row             # the chart sheet is lighter
        assert light.auto_filter.ref and len(light._charts) == 1
        assert full.cell(2, 1).value == 1                    # slice number of the first point


def test_main_excel_has_native_charts(eng, tmp_path):
    from openpyxl import load_workbook
    from traceview.excel_out import export_xlsx
    wb = load_workbook(export_xlsx(eng, str(tmp_path / "m.xlsx")))
    assert "グラフ" in wb.sheetnames and wb.sheetnames.index("グラフ") == 1
    assert len(wb["グラフ"]._charts) == len(eng.bands)       # a real Excel chart for every band, not a picture
