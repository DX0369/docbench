#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import re
import subprocess
from pathlib import Path
from typing import Dict, List


PAGE_RE = re.compile(r"^Pages:\s+(\d+)\s*$", re.MULTILINE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a stratified PDF subset for throughput tuning before running "
            "the full dataset."
        )
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        required=True,
        help="Root directory containing PDFs.",
    )
    parser.add_argument(
        "--output-list-file",
        type=Path,
        required=True,
        help="Path to the generated PDF list file.",
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        help="Optional JSON summary path. Defaults to <output-list-file>.json.",
    )
    parser.add_argument(
        "--pressure-pdf-list-file",
        type=Path,
        help=(
            "Optional PDF list file whose entries are always included in the "
            "generated subset. Relative paths are resolved relative to the list file."
        ),
    )
    parser.add_argument(
        "--subset-mode",
        type=str,
        choices=["mixed", "pressure-only", "stratified-only"],
        default="mixed",
        help=(
            "How to build the tuning subset: stratified-only uses only sampled "
            "short/medium/long docs, pressure-only uses only pressure docs, and "
            "mixed merges both."
        ),
    )
    parser.add_argument(
        "--pressure-sample-count",
        type=int,
        help=(
            "Optional cap for how many pressure PDFs to randomly keep. "
            "If omitted, all pressure PDFs are included."
        ),
    )
    parser.add_argument(
        "--pressure-sample-seed",
        type=int,
        help=(
            "Optional random seed for pressure PDF sampling. Defaults to --seed."
        ),
    )
    parser.add_argument(
        "--short-count",
        type=int,
        default=6,
        help="Number of short documents to sample.",
    )
    parser.add_argument(
        "--medium-count",
        type=int,
        default=6,
        help="Number of medium-length documents to sample.",
    )
    parser.add_argument(
        "--long-count",
        type=int,
        default=6,
        help="Number of long documents to sample.",
    )
    parser.add_argument(
        "--short-max-pages",
        type=int,
        default=2,
        help="Maximum page count for the short bucket.",
    )
    parser.add_argument(
        "--medium-max-pages",
        type=int,
        default=8,
        help="Maximum page count for the medium bucket.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducible sampling.",
    )
    return parser.parse_args()


def collect_pdf_paths(input_root: Path) -> List[Path]:
    return sorted(
        path
        for path in input_root.rglob("*.pdf")
        if path.is_file() and "_pressure_repeats" not in path.parts
    )


def load_pdf_list_file(list_path: Path) -> List[Path]:
    resolved_list_path = list_path.resolve()
    base_dir = resolved_list_path.parent
    paths: List[Path] = []
    for raw_line in resolved_list_path.read_text(encoding="utf-8").splitlines():
        text = raw_line.strip()
        if not text or text.startswith("#"):
            continue
        candidate = Path(text)
        if not candidate.is_absolute():
            candidate = base_dir / candidate
        paths.append(candidate.resolve())
    return paths


def get_pdf_page_count(pdf_path: Path) -> int:
    completed = subprocess.run(
        ["pdfinfo", str(pdf_path)],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "pdfinfo failed for "
            f"{pdf_path}\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    match = PAGE_RE.search(completed.stdout)
    if match is None:
        raise RuntimeError(f"Unable to parse page count from pdfinfo output for {pdf_path}")
    return int(match.group(1))


def bucket_name(page_count: int, short_max_pages: int, medium_max_pages: int) -> str:
    if page_count <= short_max_pages:
        return "short"
    if page_count <= medium_max_pages:
        return "medium"
    return "long"


def build_entry(
    pdf_path: Path,
    *,
    short_max_pages: int,
    medium_max_pages: int,
) -> Dict[str, object]:
    page_count = get_pdf_page_count(pdf_path)
    return {
        "pdf_path": str(pdf_path),
        "page_count": page_count,
        "bucket": bucket_name(page_count, short_max_pages, medium_max_pages),
    }


def sample_bucket(
    rng: random.Random,
    items: List[Dict[str, object]],
    requested_count: int,
) -> List[Dict[str, object]]:
    if requested_count <= 0 or not items:
        return []
    if len(items) <= requested_count:
        return list(items)
    sampled = list(items)
    rng.shuffle(sampled)
    sampled = sampled[:requested_count]
    sampled.sort(key=lambda item: str(item["pdf_path"]))
    return sampled


def merge_selected_with_pressure(
    selected: List[Dict[str, object]],
    pressure_entries: List[Dict[str, object]],
) -> List[Dict[str, object]]:
    merged: Dict[str, Dict[str, object]] = {
        str(item["pdf_path"]): dict(item) for item in selected
    }
    for entry in pressure_entries:
        merged[str(entry["pdf_path"])] = dict(entry)
    return sorted(merged.values(), key=lambda item: str(item["pdf_path"]))


def sample_pressure_entries(
    pressure_entries: List[Dict[str, object]],
    *,
    pressure_sample_count: int | None,
    pressure_sample_seed: int,
) -> List[Dict[str, object]]:
    if pressure_sample_count is None or pressure_sample_count <= 0:
        return sorted(pressure_entries, key=lambda item: str(item["pdf_path"]))
    if len(pressure_entries) <= pressure_sample_count:
        return sorted(pressure_entries, key=lambda item: str(item["pdf_path"]))
    rng = random.Random(pressure_sample_seed)
    sampled = list(pressure_entries)
    rng.shuffle(sampled)
    sampled = sampled[:pressure_sample_count]
    sampled.sort(key=lambda item: str(item["pdf_path"]))
    return sampled


def build_subset_entries(
    *,
    input_root: Path,
    short_count: int,
    medium_count: int,
    long_count: int,
    short_max_pages: int,
    medium_max_pages: int,
    seed: int,
    subset_mode: str = "mixed",
    pressure_pdf_list_file: Path | None = None,
    pressure_sample_count: int | None = None,
    pressure_sample_seed: int | None = None,
) -> Dict[str, object]:
    if short_max_pages >= medium_max_pages:
        raise ValueError("--short-max-pages must be smaller than --medium-max-pages.")

    rng = random.Random(seed)
    pdf_paths = collect_pdf_paths(input_root)
    if not pdf_paths:
        raise RuntimeError(f"No PDFs found under {input_root}")

    buckets: Dict[str, List[Dict[str, object]]] = {"short": [], "medium": [], "long": []}
    for pdf_path in pdf_paths:
        entry = build_entry(
            pdf_path,
            short_max_pages=short_max_pages,
            medium_max_pages=medium_max_pages,
        )
        buckets[str(entry["bucket"])].append(entry)

    stratified_selected = (
        sample_bucket(rng, buckets["short"], short_count)
        + sample_bucket(rng, buckets["medium"], medium_count)
        + sample_bucket(rng, buckets["long"], long_count)
    )

    pressure_entries: List[Dict[str, object]] = []
    if pressure_pdf_list_file is not None:
        for pdf_path in load_pdf_list_file(pressure_pdf_list_file):
            pressure_entries.append(
                build_entry(
                    pdf_path,
                    short_max_pages=short_max_pages,
                    medium_max_pages=medium_max_pages,
                )
            )
        pressure_entries = sample_pressure_entries(
            pressure_entries,
            pressure_sample_count=pressure_sample_count,
            pressure_sample_seed=(
                seed if pressure_sample_seed is None else pressure_sample_seed
            ),
        )
    if subset_mode == "pressure-only":
        if pressure_pdf_list_file is None:
            raise ValueError(
                "--subset-mode pressure-only requires --pressure-pdf-list-file."
            )
        selected = sorted(pressure_entries, key=lambda item: str(item["pdf_path"]))
    elif subset_mode == "stratified-only":
        selected = sorted(stratified_selected, key=lambda item: str(item["pdf_path"]))
    else:
        selected = merge_selected_with_pressure(stratified_selected, pressure_entries)

    return {
        "buckets": buckets,
        "stratified_selected": sorted(
            stratified_selected, key=lambda item: str(item["pdf_path"])
        ),
        "selected": selected,
        "pressure_entries": pressure_entries,
    }


def main() -> None:
    args = parse_args()
    args.input_root = args.input_root.resolve()
    args.output_list_file = args.output_list_file.resolve()
    if args.summary_path is None:
        args.summary_path = args.output_list_file.with_suffix(args.output_list_file.suffix + ".json")
    else:
        args.summary_path = args.summary_path.resolve()
    if args.pressure_pdf_list_file is not None:
        args.pressure_pdf_list_file = args.pressure_pdf_list_file.resolve()

    try:
        subset_data = build_subset_entries(
            input_root=args.input_root,
            short_count=args.short_count,
            medium_count=args.medium_count,
            long_count=args.long_count,
            short_max_pages=args.short_max_pages,
            medium_max_pages=args.medium_max_pages,
            seed=args.seed,
            subset_mode=args.subset_mode,
            pressure_pdf_list_file=args.pressure_pdf_list_file,
            pressure_sample_count=args.pressure_sample_count,
            pressure_sample_seed=args.pressure_sample_seed,
        )
    except (RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    buckets = subset_data["buckets"]
    stratified_selected = subset_data["stratified_selected"]
    selected = subset_data["selected"]
    pressure_entries = subset_data["pressure_entries"]

    args.output_list_file.parent.mkdir(parents=True, exist_ok=True)
    args.output_list_file.write_text(
        "".join(f"{item['pdf_path']}\n" for item in selected),
        encoding="utf-8",
    )

    args.summary_path.parent.mkdir(parents=True, exist_ok=True)
    args.summary_path.write_text(
        json.dumps(
            {
                "input_root": str(args.input_root),
                "output_list_file": str(args.output_list_file),
                "subset_mode": args.subset_mode,
                "pressure_pdf_list_file": (
                    str(args.pressure_pdf_list_file)
                    if args.pressure_pdf_list_file is not None
                    else None
                ),
                "pressure_sample_count": args.pressure_sample_count,
                "pressure_sample_seed": (
                    args.seed if args.pressure_sample_seed is None else args.pressure_sample_seed
                ),
                "seed": args.seed,
                "bucket_thresholds": {
                    "short_max_pages": args.short_max_pages,
                    "medium_max_pages": args.medium_max_pages,
                },
                "requested_counts": {
                    "short": args.short_count,
                    "medium": args.medium_count,
                    "long": args.long_count,
                },
                "available_counts": {
                    bucket: len(entries) for bucket, entries in buckets.items()
                },
                "selected_count": len(selected),
                "stratified_selected_count": len(stratified_selected),
                "pressure_available_count": (
                    len(load_pdf_list_file(args.pressure_pdf_list_file))
                    if args.pressure_pdf_list_file is not None
                    else 0
                ),
                "pressure_selected_count": len(pressure_entries),
                "selected": selected,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"[done] subset list: {args.output_list_file}")
    print(f"[done] subset summary: {args.summary_path}")
    print(
        "[done] selected docs: "
        f"short={len([x for x in selected if x['bucket'] == 'short'])}, "
        f"medium={len([x for x in selected if x['bucket'] == 'medium'])}, "
        f"long={len([x for x in selected if x['bucket'] == 'long'])}, "
        f"pressure={len(pressure_entries)}, "
        f"mode={args.subset_mode}"
    )


if __name__ == "__main__":
    main()
