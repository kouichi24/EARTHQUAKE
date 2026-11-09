"""A steep stroke whose ink fades out (dashes, crossings under ruler lines) is followed to its tip, marked as estimated."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from traceview.engine import Engine  # noqa: E402

SAMPLE = os.path.join(os.path.dirname(__file__), "..", "samples", "sample_B1-01.pdf")


@pytest.mark.skipif(not os.path.exists(SAMPLE), reason="sample sheet not present")
def test_long_diagonal_stroke_reaches_its_tip():
    e = Engine()
    e.open(SAMPLE)
    e.analyze()
    b = e.bands[0]
    p = b.path
    near = (p[:, 0] > 2400) & (p[:, 0] < 2540)
    assert p[near][:, 1].max() >= 940            # work px: the faded tip is at about y=957
    assert b.inferred                             # the faded part is an estimated link
