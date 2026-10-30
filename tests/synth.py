"""Synthetic seismogram sheet with known peaks, used to check the pipeline without real data."""
import cv2
import numpy as np

W, H = 2400, 1200
BASELINES = [250, 600, 950]


def make_sheet(seed=1, ticks=False, invert=False):
    rng = np.random.default_rng(seed)
    img = np.zeros((H, W, 3), np.uint8)
    img[:] = (170, 195, 205)                                   # beige paper (BGR)
    img = np.clip(img + rng.normal(0, 3, img.shape), 0, 255).astype(np.uint8)
    # rulers (grey, low saturation)
    img[:, :60] = (190, 190, 190)
    img[H - 60:, :] = (190, 190, 190)
    for x in range(0, W, 20):
        cv2.line(img, (x, H - 60), (x, H - 40), (60, 60, 60), 1)
    for y in range(0, H, 20):
        cv2.line(img, (0, y), (25, y), (60, 60, 60), 1)
    ink = (55, 60, 50)
    truth = {}
    for bi, by in enumerate(BASELINES):
        # reference lines
        for dy in (-90, -78):
            cv2.line(img, (60, by + dy), (W, by + dy), ink, 2)
        pts = [(70, by)]
        peaks = []
        x = 70
        k = 0
        while x < W - 150:
            x += 25 + 3 * (k % 4)
            if x < 220:
                pts.append((x, by + (3 if k % 2 else -3)))
                k += 1
                continue
            up = 25 + (k * 37) % 95            # up-peak height 25..120
            dn = 20 + (k * 53) % 80
            wid = 5 if k % 2 == 0 else 12     # narrow (arms merge) / wide
            pts.append((x, by))
            pts.append((x + wid, by - up))      # peak
            pts.append((x + 2 * wid, by))
            pts.append((x + 3 * wid, by + dn))  # valley
            pts.append((x + 4 * wid, by))
            peaks.append((x + wid, by - up, "peak"))
            peaks.append((x + 3 * wid, by + dn, "valley"))
            x += 4 * wid
            k += 1
        pts.append((W - 80, by))
        cv2.polylines(img, [np.array(pts, np.int32).reshape(-1, 1, 2)], False, ink, 3, cv2.LINE_AA)
        truth[bi] = dict(baseline=by, peaks=peaks)
    if ticks:
        # time ticks (short vertical marks) and small solid marks just above the reference lines
        for bi, by in enumerate(BASELINES):
            for x in range(120, W - 100, 90):
                cv2.line(img, (x, by - 112), (x, by - 96), ink, 2)
            cv2.rectangle(img, (150, by - 108), (166, by - 98), ink, -1)
    # annotation in orange that must be ignored
    cv2.putText(img, "memo", (900, 420), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (60, 120, 230), 3)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if invert:
        rgb = 255 - rgb          # light line on dark paper
    return rgb, truth
