#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from paths import PDF_ROOT, TMP_ROOT  # noqa: E402


DEFAULT_SOURCE_PDF = (
    PDF_ROOT
    / "industry_standard"
    / "en"
    / "industry_standard_en_rfc6263_1_4"
    / "industry_standard_en_rfc6263_1_4.pdf"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract one reproducible PDF page for smoke tests.")
    parser.add_argument("--source-pdf", type=Path, default=DEFAULT_SOURCE_PDF)
    parser.add_argument("--page", type=int, default=1)
    parser.add_argument("--run-id", default=time.strftime("smoke_%Y%m%d_%H%M%S"))
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--output-name", default="basic-1.pdf")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    source_pdf = args.source_pdf.expanduser().resolve()
    output_root = (args.output_root or TMP_ROOT / "smoke" / args.run_id / "inputs").expanduser().resolve()
    output_pdf = output_root / args.output_name
    if args.page < 1:
        raise SystemExit("--page must be at least 1")
    if not source_pdf.is_file():
        raise SystemExit(f"Source PDF not found: {source_pdf}")
    if output_pdf.exists() and not args.force:
        raise SystemExit(f"Output already exists: {output_pdf}; use --force to replace it")
    pdfseparate = shutil.which("pdfseparate")
    if pdfseparate is None:
        raise SystemExit("pdfseparate is not installed or not on PATH")

    output_root.mkdir(parents=True, exist_ok=True)
    temporary_pattern = output_root / ".smoke-page-%d.pdf"
    temporary_pdf = output_root / f".smoke-page-{args.page}.pdf"
    subprocess.run(
        [
            pdfseparate,
            "-f",
            str(args.page),
            "-l",
            str(args.page),
            str(source_pdf),
            str(temporary_pattern),
        ],
        check=True,
    )
    if not temporary_pdf.is_file():
        raise SystemExit(f"pdfseparate did not create expected output: {temporary_pdf}")
    temporary_pdf.replace(output_pdf)

    manifest = {
        "source_pdf": str(source_pdf),
        "source_page": args.page,
        "output_pdf": str(output_pdf),
        "output_size_bytes": output_pdf.stat().st_size,
        "output_sha256": sha256(output_pdf),
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[DONE   ] {output_pdf}")
    print(f"[MANIFEST] {manifest_path}")


if __name__ == "__main__":
    main()
