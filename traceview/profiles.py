"""Recording-method profiles: presets of analysis settings (not an instrument database).

The recording method of a sheet is often unknown. Instead of guessing it, the app tries
several profiles and lets the user pick the one whose overlay looks right.
`verified` says whether a profile has been checked against a real sheet.
"""
from dataclasses import dataclass, field, replace

from .config import Params


@dataclass(frozen=True)
class Profile:
    key: str
    name: str
    description: str
    verified: str                      # "確認済み(B1-01)" | "未検証"
    overrides: dict = field(default_factory=dict)
    arc: bool = False                  # pen arm writes along an arc (time correction possible)

    def apply(self, base=None):
        p = replace(base or Params())
        for k, v in self.overrides.items():
            setattr(p, k, v)
        return p


PROFILES = [
    Profile("ink_lever", "① レバー式インクペン（暗い線・円弧あり）",
            "明るい紙に暗い線。大きな振れが円弧状。時刻の刻み・小さな黒い印を除外する",
            "確認済み(サンプルB1-01のみ)", dict(drop_marks=True), arc=True),
    Profile("inverted", "② 明暗反転（暗い紙に明るい線）",
            "煤書きなど、暗い地に白い線。画像を反転して解析する",
            "未検証（見本なし）", dict(invert=True, drop_marks=True), arc=True),
    Profile("thick", "③ 太い線・にじみ（感光紙・光学など）",
            "線が太い・濃淡が大きい場合。弱い線も拾い、小さな欠けを埋める",
            "未検証（見本なし）",
            dict(ink_threshold=0.55, ink_weak_threshold=0.72, hole_fill_area=140, min_component_area=80,
                 spur_min_len=10, tip_min_prominence=10.0)),
    Profile("generic", "④ 汎用（現在の標準設定）",
            "方式が不明なときの標準。刻みの除外や円弧補正はしない", "確認済み(サンプルB1-01のみ)", {}),
]
BY_KEY = {p.key: p for p in PROFILES}


def get(key):
    return BY_KEY.get(key, BY_KEY["generic"])
