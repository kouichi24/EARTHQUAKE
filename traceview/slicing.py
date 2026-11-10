"""Cut the traced lines into slices along x and keep the numbers of every point (so the lines can be drawn again
from the numbers alone, in another window or from a CSV file).

Slice 1 starts where the band's trace starts (its origin). Positions are in ORIGINAL image px (the work image is a
scaled copy), plus the x position in the chosen unit and the amplitude relative to the band's zero line.
"""
import csv
import io

import numpy as np

COLUMNS = ["band", "slice", "slice_start", "slice_end", "idx", "x_px", "y_px", "x_unit", "amp"]
UNITS = ("px", "mm", "s")


class SliceError(Exception):
    pass


def px_per_unit(engine, unit):
    """Original px along x for one unit (None when the unit needs a calibration that is not there)."""
    cal = engine.calibration
    if unit == "px":
        return 1.0
    if unit == "mm":
        return 1.0 / cal.mm_per_px_x if cal.mm_per_px_x > 0 else None
    if unit == "s":
        sp = cal.seconds_per_orig_px()
        return 1.0 / sp if sp > 0 else None
    raise SliceError(f"単位が不明です: {unit}")


def available_units(engine):
    return [u for u in UNITS if px_per_unit(engine, u)]


def slice_rows(engine, step, unit="px", bands=None):
    """All points of the traced lines with their slice number. step: slice width in `unit`."""
    if not step or step <= 0:
        raise SliceError("区切りの幅は0より大きくしてください")
    ppu = px_per_unit(engine, unit)
    if ppu is None:
        raise SliceError("この単位には、校正（「校正」タブ）が必要です" if unit != "px" else "単位が不正です")
    step_px = step * ppu
    inv = 1.0 / engine.scale
    cal = engine.calibration
    rows = []
    for bi, b in enumerate(engine.bands):
        if bands is not None and bi not in bands:
            continue
        if len(b.path) == 0:
            continue
        x_orig = b.path[:, 0].astype(float) * inv
        y_orig = b.path[:, 1].astype(float) * inv
        x0 = float(b.origin[0]) * inv
        k = np.floor((x_orig - x0) / step_px).astype(int)
        k = np.maximum(k, 0)                         # a point left of the origin belongs to slice 1
        xu = (x_orig - x0) / ppu
        for i in range(len(b.path)):
            amp = cal.to_y_units((b.line_zero - b.path[i, 1]) * inv)
            rows.append(dict(band=b.name, slice=int(k[i]) + 1,
                             slice_start=float(k[i] * step), slice_end=float((k[i] + 1) * step),
                             idx=i, x_px=float(x_orig[i]), y_px=float(y_orig[i]), x_unit=float(xu[i]), amp=float(amp)))
    if not rows:
        raise SliceError("トレースした線がありません")
    return rows


def to_csv_text(rows, unit="px", step=0.0):
    out = io.StringIO()
    out.write(f"# TraceView slices unit={unit} step={step}\n")
    w = csv.writer(out, lineterminator="\n")
    w.writerow(COLUMNS)
    for r in rows:
        w.writerow([r["band"], r["slice"], f"{r['slice_start']:.6g}", f"{r['slice_end']:.6g}", r["idx"],
                    f"{r['x_px']:.3f}", f"{r['y_px']:.3f}", f"{r['x_unit']:.6g}", f"{r['amp']:.6g}"])
    return out.getvalue()


def save_csv(path, rows, unit="px", step=0.0):
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        f.write(to_csv_text(rows, unit, step))


def read_csv(path):
    """Read a CSV written by save_csv. Returns (rows, unit, step)."""
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        text = f.read()
    unit, step = "px", 0.0
    lines = text.splitlines()
    if lines and lines[0].startswith("#"):
        for tok in lines[0].split():
            if tok.startswith("unit="):
                unit = tok[5:]
            elif tok.startswith("step="):
                try:
                    step = float(tok[5:])
                except ValueError:
                    pass
        lines = lines[1:]
    rd = csv.DictReader(lines)
    if not rd.fieldnames or any(c not in rd.fieldnames for c in COLUMNS):
        raise SliceError("TraceViewが保存した分割CSVではありません")
    rows = []
    for r in rd:
        try:
            rows.append(dict(band=r["band"], slice=int(r["slice"]), slice_start=float(r["slice_start"]),
                             slice_end=float(r["slice_end"]), idx=int(r["idx"]), x_px=float(r["x_px"]),
                             y_px=float(r["y_px"]), x_unit=float(r["x_unit"]), amp=float(r["amp"])))
        except (ValueError, TypeError):
            raise SliceError("CSVの数値が読み取れません")
    if not rows:
        raise SliceError("CSVにデータがありません")
    return rows, unit, step


def _simplify_idx(x, y, eps):
    """Indices of a Douglas-Peucker simplification of the polyline (keeps the first and last point)."""
    n = len(x)
    if n <= 2:
        return list(range(n))
    keep = np.zeros(n, bool)
    keep[0] = keep[-1] = True
    stack = [(0, n - 1)]
    while stack:
        a, b = stack.pop()
        if b <= a + 1:
            continue
        px, py = x[a + 1:b], y[a + 1:b]
        dx, dy = x[b] - x[a], y[b] - y[a]
        L = np.hypot(dx, dy)
        d = np.hypot(px - x[a], py - y[a]) if L == 0 else np.abs(dy * (px - x[a]) - dx * (py - y[a])) / L
        k = int(np.argmax(d))
        if d[k] > eps:
            m = a + 1 + k
            keep[m] = True
            stack.append((a, m))
            stack.append((m, b))
    return np.flatnonzero(keep).tolist()


def write_xlsx(path, engine, rows, unit="px", step=0.0, chart_eps_px=1.0):
    """Excel file with, for every band: a sheet with all numbers, and a sheet with a lighter copy of the same line
    (for the chart: points that change the drawn line by less than `chart_eps_px` original px are left out) and a
    scatter chart (straight lines). The chart sheet has a filter on the slice number: filtering the slice shows
    only that slice in the chart."""
    from openpyxl import Workbook
    from openpyxl.chart import Reference, ScatterChart, Series
    wb = Workbook(write_only=True)
    cal = engine.calibration
    ycap = f"振幅[{cal.unit_y()}]"
    xcap = {"px": "X [px]（元画像）", "mm": "X [mm]", "s": "時間 [秒]"}.get(unit, f"X [{unit}]")
    ws0 = wb.create_sheet("説明")
    for line in ("TraceView 分割データ", f"区切りの単位: {unit}　区切りの幅: {step:g}",
                 "区切りは、各段のトレースの起点から数えます。",
                 "「○○_数値」シート: 線のすべての点（区切り番号付き）。X(px)・Y(px)は元画像のpx。",
                 "「○○_グラフ」シート: グラフ用に点を減らしたデータと散布図。Yの向きは、上が大きい値（振幅）です。",
                 "「区切り」列のフィルターで区切りを選ぶと、グラフにはその区切りだけが表示されます。",
                 "（Excelは、非表示の行をグラフに描かない設定が既定です）"):
        ws0.append([line])
    names = []
    for r in rows:
        if r["band"] not in names:
            names.append(r["band"])
    hdr = ["区切り", "点の順番", "X(px)", "Y(px)", xcap, ycap]
    for name in names:
        rs = [r for r in rows if r["band"] == name]
        rs.sort(key=lambda r: r["idx"])
        full = wb.create_sheet(f"{name}_数値"[:31])
        full.append(hdr)
        for r in rs:
            full.append([r["slice"], r["idx"], r["x_px"], r["y_px"], r["x_unit"], r["amp"]])
        full.auto_filter.ref = f"A1:F{len(rs) + 1}"
        xs = np.array([r["x_px"] for r in rs])
        ys = np.array([r["y_px"] for r in rs])
        sel = _simplify_idx(xs, ys, chart_eps_px)
        light = wb.create_sheet(f"{name}_グラフ"[:31])
        light.append(hdr)
        for i in sel:
            r = rs[i]
            light.append([r["slice"], r["idx"], r["x_px"], r["y_px"], r["x_unit"], r["amp"]])
        light.auto_filter.ref = f"A1:F{len(sel) + 1}"
        ch = ScatterChart()
        ch.title = f"{name}（{len(sel)}点／全{len(rs)}点）"
        ch.style = 13
        ch.scatterStyle = "lineMarker"
        ch.x_axis.title = xcap
        ch.y_axis.title = ycap
        ch.x_axis.delete = False
        ch.y_axis.delete = False
        ch.legend = None
        ch.height, ch.width = 12, 26
        xr = Reference(light, min_col=5, min_row=2, max_row=len(sel) + 1)
        yr = Reference(light, min_col=6, min_row=2, max_row=len(sel) + 1)
        s = Series(yr, xr, title=name)
        s.marker.symbol = "none"
        s.smooth = False
        col = next((b.color for b in engine.bands if b.name == name), "#1565c0")
        s.graphicalProperties.line.solidFill = col.lstrip("#").upper()
        s.graphicalProperties.line.width = 9525          # 0.75 pt
        ch.series.append(s)
        light.add_chart(ch, "H2")
    wb.save(path)
