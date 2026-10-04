"""PyMuPDF page rendering, executed in one persistent CPU subprocess."""
import io
import os
import time

import fitz
from PIL import Image, ImageOps


def ready():
    return os.getpid()


def page_count(pdf):
    with fitz.open(str(pdf)) as document:
        return document.page_count


def render_page(pdf, page, dpi):
    started = time.perf_counter()
    with fitz.open(str(pdf)) as document:
        pixmap = document[page - 1].get_pixmap(matrix=fitz.Matrix(dpi / 72.0, dpi / 72.0), alpha=False)
        Image.MAX_IMAGE_PIXELS = None
        with Image.open(io.BytesIO(pixmap.tobytes("png"))) as original:
            image = ImageOps.exif_transpose(original).convert("RGB")
            try:
                buffer = io.BytesIO()
                image.save(buffer, format="PNG")
                return buffer.getvalue(), list(image.size), time.perf_counter() - started
            finally:
                image.close()
