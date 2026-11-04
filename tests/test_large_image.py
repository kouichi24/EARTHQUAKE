import os
import sys

import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tests.synth import make_sheet  # noqa: E402
from traceview.engine import Engine  # noqa: E402


def test_large_image_is_reduced_for_work_and_original_is_released(tmp_path):
    rgb, _ = make_sheet()
    big = cv2.resize(rgb, (7200, 3600), interpolation=cv2.INTER_CUBIC)       # larger than the 4800 px analysis size
    p = tmp_path / "big.png"
    cv2.imwrite(str(p), cv2.cvtColor(big, cv2.COLOR_RGB2BGR))
    before = os.path.getsize(p)
    e = Engine()
    e.open(str(p))
    assert e.work.shape[1] == 4800 and e.scale < 1
    assert e.orig_size == (7200, 3600)                 # positions are still reported in original pixels
    assert e.rgb_orig is None                          # the full-size copy is not kept in memory
    assert os.path.getsize(p) == before                # the original file is untouched


def test_small_image_is_used_as_is(tmp_path):
    rgb, _ = make_sheet()
    p = tmp_path / "small.png"
    cv2.imwrite(str(p), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    e = Engine()
    e.open(str(p))
    assert e.scale == 1.0 and e.work.shape[1] == 2400
