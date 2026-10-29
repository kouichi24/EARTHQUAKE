"""Single-trace export (OUT-03): SVG with one path per waveform, estimated links dashed."""
import base64

import cv2

from .excel_out import PendingEditsError


def export_svg(engine, out_path, include_image=False):
    if engine.has_pending():
        raise PendingEditsError("未適用の頂点修正があります。再追跡を適用するか取り消してから出力してください。")
    inv = 1.0 / engine.scale
    W, H = engine.orig_size
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">']
    if include_image:
        small = cv2.cvtColor(engine.work, cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 85])
        b64 = base64.b64encode(buf.tobytes()).decode()
        out.append(f'<image x="0" y="0" width="{W}" height="{H}" href="data:image/jpeg;base64,{b64}"/>')
    else:
        out.append(f'<rect width="{W}" height="{H}" fill="white"/>')
    for b in engine.bands:
        pts = b.path
        if len(pts) < 2:
            continue
        spans = sorted(b.inferred)
        cur = 0
        segs = []
        for a, c in spans:
            if a > cur:
                segs.append((cur, a, False))
            segs.append((a, c, True))
            cur = c
        if cur < len(pts) - 1:
            segs.append((cur, len(pts) - 1, False))
        for a, c, inferred in segs:
            seg = pts[a:c + 1]
            if len(seg) < 2:
                continue
            d = "M" + " L".join(f"{x * inv:.1f},{y * inv:.1f}" for x, y in seg[::1])
            dash = ' stroke-dasharray="12 8"' if inferred else ""
            out.append(f'<path d="{d}" fill="none" stroke="{b.color}" stroke-width="3"{dash} '
                       f'id="{b.name}{"-inferred" if inferred else ""}"/>')
    out.append("</svg>")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(out))
    return out_path
