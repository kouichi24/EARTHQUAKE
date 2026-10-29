"""Reading PDFs / images. The source file is never modified (IN-01, IN-02)."""
import hashlib
import os

import numpy as np

Image = None
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


def file_digest(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def pdf_page_count(path):
    import pymupdf
    with pymupdf.open(path) as doc:
        return len(doc)


def _pdf_page_array(path, page, max_side):
    import pymupdf
    with pymupdf.open(path) as doc:
        p = doc[page]
        imgs = p.get_images(full=True)
        # A scanned PDF holds one big page-sized image: use it as is (no resampling loss).
        if len(imgs) == 1:
            info = doc.extract_image(imgs[0][0])
            from PIL import Image as PILImage
            import io
            PILImage.MAX_IMAGE_PIXELS = None
            im = PILImage.open(io.BytesIO(info["image"])).convert("RGB")
            return np.asarray(im)
        # Vector content: render to a raster at a resolution that reaches max_side.
        r = p.rect
        zoom = max_side / max(r.width, r.height)
        pix = p.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)
        return np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, 3).copy()


def load_image(path, page=0, max_side=4800):
    """Return (rgb_original uint8 HxWx3). Large originals are returned untouched."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return _pdf_page_array(path, page, max_side)
    if ext in IMAGE_EXTS:
        from PIL import Image as PILImage
        PILImage.MAX_IMAGE_PIXELS = None
        im = PILImage.open(path)
        try:
            im.seek(page)
        except EOFError:
            pass
        return np.asarray(im.convert("RGB"))
    raise ValueError(f"unsupported file type: {ext}")


def page_count(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return pdf_page_count(path)
    if ext in (".tif", ".tiff"):
        from PIL import Image as PILImage
        with PILImage.open(path) as im:
            return getattr(im, "n_frames", 1)
    return 1


def make_work_image(rgb, max_side):
    """Downscale for analysis. Returns (work_rgb, scale) with work = original * scale."""
    import cv2
    h, w = rgb.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale == 1.0:
        return rgb, 1.0
    out = cv2.resize(rgb, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    return out, out.shape[1] / w
