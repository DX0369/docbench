#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import tempfile
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from PIL import Image
from paths import TMP_ROOT


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, data: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def remove_file_if_exists(path: Path) -> None:
    if path.exists():
        path.unlink()


def remove_dir_if_exists(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)


def collect_pdf_paths(input_root: Path, limit: Optional[int] = None) -> List[Path]:
    pdfs = sorted(path for path in input_root.rglob("*.pdf") if path.is_file())
    if limit is not None:
        return pdfs[:limit]
    return pdfs


def load_pdf_paths_from_file(list_path: Path, limit: Optional[int] = None) -> List[Path]:
    pdfs: List[Path] = []
    list_dir = list_path.resolve().parent
    for line in list_path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        path = Path(text)
        if not path.is_absolute():
            path = list_dir / path
        pdfs.append(path.resolve())
    if limit is not None:
        return pdfs[:limit]
    return pdfs


def add_shard_args(parser: Any) -> None:
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Total number of document shards. Use 1 to disable sharding.",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="Zero-based shard index for the current worker.",
    )


def apply_shard_to_pdf_paths(
    pdf_paths: Sequence[Path],
    *,
    num_shards: int = 1,
    shard_index: int = 0,
) -> List[Path]:
    if num_shards <= 0:
        raise ValueError("num_shards must be positive")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError("shard_index must satisfy 0 <= shard_index < num_shards")
    if num_shards == 1:
        return list(pdf_paths)
    return [path for idx, path in enumerate(pdf_paths) if idx % num_shards == shard_index]


def get_doc_rel_dir(input_root: Path, pdf_path: Path) -> Path:
    rel = pdf_path.relative_to(input_root)
    if rel.parent.name == pdf_path.stem:
        return rel.parent
    return rel.with_suffix("")


def get_doc_output_dir(input_root: Path, output_root: Path, pdf_path: Path) -> Path:
    return output_root / get_doc_rel_dir(input_root, pdf_path)


def extract_page_number(path: Path) -> int:
    match = re.search(r"-(\d+)$", path.stem)
    if match:
        return int(match.group(1))
    match = re.search(r"(\d+)$", path.stem)
    if match:
        return int(match.group(1))
    raise ValueError(f"Unable to parse page index from: {path}")


def pdf_page_count(pdf: Path) -> int:
    with error_context("inspect_pdf"):
        proc = subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True, check=True)
        match = re.search(r"^Pages:\s+(\d+)\s*$", proc.stdout, re.MULTILINE)
        if match is None or int(match[1]) < 1:
            raise ValueError(f"Missing valid page count: {pdf}")
        return int(match[1])


@contextmanager
def render_pdf_page(pdf: Path, page: int, dpi: int, temp_root: Path):
    temp_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="page_", dir=temp_root) as temp:
        image = Path(temp) / "page.png"
        with error_context("render", page):
            subprocess.run(["pdftoppm", "-f", str(page), "-l", str(page), "-singlefile",
                            "-png", "-r", str(dpi), str(pdf), str(image.with_suffix(""))],
                           capture_output=True, text=True, check=True)
        yield image


def render_pdf_pages(
    pdf_path: Path,
    dpi: int,
    render_dir: Optional[Path] = None,
) -> Tuple[List[Path], Path, bool]:
    created_temp_dir = render_dir is None
    if render_dir is None:
        runtime_root = TMP_ROOT / "runtime"
        runtime_root.mkdir(parents=True, exist_ok=True)
        render_dir = Path(tempfile.mkdtemp(prefix="render_", dir=runtime_root))
    ensure_dir(render_dir)

    prefix = render_dir / "page"
    cmd = [
        "pdftoppm",
        "-png",
        "-r",
        str(dpi),
        str(pdf_path),
        str(prefix),
    ]
    try:
        completed = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "pdftoppm failed for "
                f"{pdf_path}\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
            )

        pages = sorted(render_dir.glob("page-*.png"), key=extract_page_number)
        if not pages:
            raise RuntimeError(f"No rendered pages were produced for {pdf_path}")
        return pages, render_dir, created_temp_dir
    except BaseException:
        cleanup_render_dir(render_dir, created_temp_dir)
        raise


def cleanup_render_dir(render_dir: Path, created_temp_dir: bool) -> None:
    if created_temp_dir and render_dir.exists():
        shutil.rmtree(render_dir, ignore_errors=True)


def get_image_size(image_path: Path) -> Tuple[int, int]:
    with Image.open(image_path) as image:
        return image.size


def concat_document_markdown(page_markdowns: Sequence[str]) -> str:
    cleaned: List[str] = []
    for page_md in page_markdowns:
        cleaned.append(page_md.rstrip("\n"))
    if not cleaned:
        return ""
    return "\n\n".join(cleaned).rstrip() + "\n"


def save_markdown_bundle(
    doc_output_dir: Path,
    page_markdowns: Sequence[str],
    document_markdown: str,
) -> None:
    pages_dir = doc_output_dir / "pages"
    ensure_dir(pages_dir)
    for page_index, page_markdown in enumerate(page_markdowns, start=1):
        page_path = pages_dir / f"page_{page_index:04d}.md"
        write_text(page_path, page_markdown)
    write_text(doc_output_dir / "document.md", document_markdown)


def build_page_records(
    page_markdowns: Sequence[str],
    page_latencies: Sequence[Any],
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for page_index, page_markdown in enumerate(page_markdowns, start=1):
        latency_missing = page_index > len(page_latencies)
        record: Dict[str, Any] = {
            "page_index": page_index,
            "markdown_path": f"pages/page_{page_index:04d}.md",
            "latency_seconds": (
                "N/A" if latency_missing else page_latencies[page_index - 1]
            ),
            "markdown_characters": len(page_markdown),
        }
        if latency_missing:
            record["latency_missing"] = True
        records.append(record)
    return records


def normalize_box_0_1000(
    box: Sequence[float],
    page_width: int,
    page_height: int,
) -> List[int]:
    if page_width <= 0 or page_height <= 0:
        raise ValueError("Page width and height must be positive")
    x1, y1, x2, y2 = box
    normalized = [
        int(round(max(0.0, min(1000.0, float(x1) * 1000.0 / page_width)))),
        int(round(max(0.0, min(1000.0, float(y1) * 1000.0 / page_height)))),
        int(round(max(0.0, min(1000.0, float(x2) * 1000.0 / page_width)))),
        int(round(max(0.0, min(1000.0, float(y2) * 1000.0 / page_height)))),
    ]
    return normalized


_HTML_IMG_RE = re.compile(r"""<img\b[^>]*\bsrc=["']([^"']+)["'][^>]*>""", re.IGNORECASE)
_MARKDOWN_IMG_RE = re.compile(r"""!\[[^\]]*\]\(([^)]+)\)""")
_BOX_NAME_RE = re.compile(
    r"""(?:^|[_-])box[_-](\d+(?:\.\d+)?)_(\d+(?:\.\d+)?)_(\d+(?:\.\d+)?)_(\d+(?:\.\d+)?)""",
    re.IGNORECASE,
)
_COORD_ARRAY_RE = re.compile(
    r"""\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]"""
)


def extract_markdown_image_markers(markdown_text: str) -> List[Dict[str, Any]]:
    markers: List[Dict[str, Any]] = []
    for match in _HTML_IMG_RE.finditer(markdown_text):
        src = match.group(1).strip()
        markers.append(
            {
                "marker_type": "html_img",
                "match_text": match.group(0),
                "src": src,
                "name": Path(src).name,
                "offset": match.start(),
            }
        )
    for match in _MARKDOWN_IMG_RE.finditer(markdown_text):
        src = match.group(1).strip()
        markers.append(
            {
                "marker_type": "markdown_image",
                "match_text": match.group(0),
                "src": src,
                "name": Path(src).name,
                "offset": match.start(),
            }
        )
    markers.sort(key=lambda item: item["offset"])
    for index, marker in enumerate(markers, start=1):
        marker["marker_index"] = index
    return markers


def parse_box_from_image_name(name: str) -> Optional[List[float]]:
    match = _BOX_NAME_RE.search(name)
    if not match:
        return None
    return [float(match.group(i)) for i in range(1, 5)]


def extract_coordinate_arrays(text: str) -> List[List[float]]:
    boxes: List[List[float]] = []
    for match in _COORD_ARRAY_RE.finditer(text):
        boxes.append([float(match.group(i)) for i in range(1, 5)])
    return boxes


def build_image_boxes_payload(
    *,
    method: str,
    pdf_path: Path,
    entries: Sequence[Dict[str, Any]],
    notes: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    return {
        "method": method,
        "pdf_path": str(pdf_path),
        "entry_count": len(entries),
        "entries": list(entries),
        "notes": list(notes or []),
    }


def build_error_payload(
    method_name: str,
    pdf_path: Path,
    extra: Optional[Dict[str, Any]] = None,
    *,
    exception: BaseException | None = None,
    stage: str = "process_document",
    page_number: int | None = None,
) -> Dict[str, Any]:
    exception = exception if exception is not None else sys.exc_info()[1]
    if exception is None:
        raise ValueError("An actual exception is required to record a document failure")
    payload: Dict[str, Any] = {
        "method": method_name,
        "pdf_path": str(pdf_path),
        "stage": getattr(exception, "gdb_stage", stage),
        "page_number": getattr(exception, "gdb_page_number", page_number),
        "error_type": type(exception).__name__,
        "message": str(exception),
        "traceback": "".join(traceback.format_exception(type(exception), exception, exception.__traceback__)),
        "runtime_unusable": runtime_unusable(exception),
    }
    # Native pipelines may wrap a worker exception; preserve its known context.
    cause = exception
    seen: set[int] = set()
    while cause is not None and id(cause) not in seen:
        seen.add(id(cause))
        for attribute, key in (("gdb_stage", "stage"), ("gdb_page_number", "page_number"),
                               ("gdb_page_numbers", "page_numbers")):
            value = getattr(cause, attribute, None)
            if value is not None:
                payload[key] = value
        response = getattr(cause, "response", None)
        if response is not None and hasattr(response, "status_code"):
            payload["http_status"] = response.status_code
            payload["response_excerpt"] = response.text[:2000]
        cause = cause.__cause__ if cause.__cause__ is not None else cause.__context__
    if extra:
        payload.update(extra)
    return payload


def append_error_payload(path: Path, payload: Dict[str, Any]) -> None:
    """Append an observed failure from a serial collector, retaining the first.

    Existing files must use this recorder's errors list; malformed or older
    records are not silently replaced. Callers must not reuse past run outputs.
    """
    errors = json.loads(path.read_text(encoding="utf-8"))["errors"] if path.exists() else []
    errors = errors + [payload]
    write_json(path, {**errors[0], "errors": errors,
                     "runtime_unusable": any(error["runtime_unusable"] for error in errors)})


def document_completed(output_dir: Path) -> bool:
    """A partial export must not hide a recorded failure."""
    return not (output_dir / "error.json").exists() and all(
        (output_dir / name).is_file()
        for name in ("document.md", "metrics.json", "image_boxes.json")
    )


@contextmanager
def error_context(stage: str, page_number: int | None = None):
    """Attach known provenance, preserving the original exception and traceback."""
    try:
        yield
    except Exception as exc:
        if not hasattr(exc, "gdb_stage"):
            exc.gdb_stage = stage
        if not hasattr(exc, "gdb_page_number"):
            exc.gdb_page_number = page_number
        raise


def runtime_unusable(exc: BaseException) -> bool:
    """Known fatal runtimes only; allocation OOM and request timeouts are not fatal."""
    seen: set[int] = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if getattr(exc, "gdb_runtime_unusable", False):
            return True
        if type(exc).__name__ in {"EngineDeadError", "AsyncEngineDeadError"}:
            return True
        message = str(exc).lower()
        if isinstance(exc, RuntimeError) and any(token in message for token in (
            "cuda error: device-side assert", "cuda error: an illegal memory access",
            "cuda error: unspecified launch failure",
        )):
            return True
        if isinstance(exc, OSError) and exc.errno in {5, 28, 30}:
            return True  # I/O error, disk full, read-only filesystem.
        exc = exc.__cause__ if exc.__cause__ is not None else exc.__context__
    return False
