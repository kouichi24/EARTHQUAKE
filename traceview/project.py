"""Work file (.trproj): everything needed to resume on another day (OUT-04)."""
import gzip
import json
import os
import time

import numpy as np

from . import __version__
from .config import Params
from .imageio import file_digest
from .measure import Calibration
from .model import Band, Issue
from .vertices import Vertex

FORMAT = 1
HISTORY_KEEP = 15


class SourceMismatch(Exception):
    """The original file named in the work file is missing or different."""


def _snap_to_json(s):
    return dict(
        path=s["path"].astype(np.int32).ravel().tolist(), inferred=[list(x) for x in s["inferred"]],
        vertices=[v.to_dict() for v in s["vertices"]], manual_points=s["manual_points"], pending=s["pending"],
        line_zero=s["line_zero"], line_top=s["line_top"], line_bottom=s["line_bottom"],
        line_top_manual=s["line_top_manual"], line_bottom_manual=s["line_bottom_manual"],
        line_zero_manual=s.get("line_zero_manual", False),
        end_x=s["end_x"], complete=s["complete"], confirmed=s["confirmed"], y0=s["y0"],
        issues=[i.to_dict() if hasattr(i, "to_dict") else i for i in s["issues"]],
    )


def _snap_from_json(d):
    return dict(
        path=np.array(d["path"], np.int32).reshape(-1, 2), inferred=[tuple(x) for x in d["inferred"]],
        vertices=[Vertex.from_dict(v) for v in d["vertices"]], manual_points=d["manual_points"],
        pending=d["pending"], line_zero=d["line_zero"], line_top=d["line_top"], line_bottom=d["line_bottom"],
        line_top_manual=d["line_top_manual"], line_bottom_manual=d["line_bottom_manual"],
        line_zero_manual=d.get("line_zero_manual", False),
        end_x=d["end_x"], complete=d["complete"], confirmed=d["confirmed"], y0=d["y0"],
        issues=[Issue(**i) for i in d["issues"]],
    )


def save_project(engine, path):
    if engine.src_path is None:
        raise ValueError("画像が読み込まれていません")
    doc = dict(
        format=FORMAT, app="TraceView", app_version=__version__, saved_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        source=dict(path=engine.src_path, name=os.path.basename(engine.src_path), sha256=engine.digest,
                    page=engine.page, orig_size=list(engine.orig_size)),
        transform=dict(scale=engine.scale, rotation_deg=0.0, note="work = original * scale"),
        profile=engine.profile_key,
        params=engine.params.to_dict(), calibration=engine.calibration.to_dict(),
        bands=[],
    )
    for b in engine.bands:
        d = _snap_to_json(b.snapshot())
        d.update(index=b.index, name=b.name, color=b.color, origin=list(b.origin))
        d["initial"] = _snap_to_json(b.initial) if b.initial else None
        d["undo"] = [_snap_to_json(s) for s in b.undo[-HISTORY_KEEP:]]
        d["redo"] = [_snap_to_json(s) for s in b.redo[-HISTORY_KEEP:]]
        doc["bands"].append(d)
    tmp = path + ".tmp"
    with gzip.open(tmp, "wt", encoding="utf-8", compresslevel=5) as f:
        json.dump(doc, f, ensure_ascii=False)
    os.replace(tmp, path)       # never leave a half-written work file


def peek_project(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def load_project(engine, path, source_path=None, progress=lambda *a: None):
    doc = peek_project(path)
    if doc.get("app") != "TraceView":
        raise ValueError("TraceViewの作業ファイルではありません")
    src = source_path or doc["source"]["path"]
    if not os.path.exists(src):
        raise SourceMismatch(f"元ファイルが見つかりません: {src}")
    if file_digest(src) != doc["source"]["sha256"]:
        raise SourceMismatch("元ファイルが作業ファイルの作成時と異なります（別の原本への誤適用を防ぐため中止しました）")
    engine.params = Params.from_dict(doc.get("params"))
    engine.open(src, doc["source"]["page"], progress)
    engine.profile_key = doc.get("profile", "generic")
    engine.calibration = Calibration.from_dict(doc.get("calibration"))
    engine.bands = []
    for d in doc["bands"]:
        b = Band(d["index"], d["name"], d["color"], tuple(d["origin"]), d["y0"])
        b.restore(_snap_from_json(d))
        b.initial = _snap_from_json(d["initial"]) if d.get("initial") else None
        b.undo = [_snap_from_json(s) for s in d.get("undo", [])]
        b.redo = [_snap_from_json(s) for s in d.get("redo", [])]
        engine.bands.append(b)
    return doc
