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
