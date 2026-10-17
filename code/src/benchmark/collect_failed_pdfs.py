#!/usr/bin/env python3
"""Collect failed and unprocessed inputs from a run_single_gpu aggregate summary."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Write failed and unprocessed PDF paths from run_single_gpu "
            "aggregate_summary.json into a reusable list file."
        )
    )
    parser.add_argument("--summary-path", type=Path, required=True,
                        help="aggregate_summary.json written by run_single_gpu.py.")
    parser.add_argument("--output-list-file", type=Path, required=True,
                        help="Where to write the collected PDF path list.")
    parser.add_argument("--failed-only", action="store_true",
                        help="Exclude unprocessed inputs and keep only explicit failures.")
    return parser.parse_args()


def collect_paths(summary: dict[str, Any], *, include_unprocessed: bool) -> tuple[list[str], int, int]:
    results = summary["results"]
    if not isinstance(results, list) or not results:
        raise ValueError("aggregate summary must contain a nonempty results list")
    keys = ("failed_docs", "unprocessed_docs") if include_unprocessed else ("failed_docs",)
    collected: list[str] = []
    counts = {key: 0 for key in keys}
    for trial in results:
        for key in keys:
            for item in trial[key]:
                pdf_path = item["pdf_path"]
                if not isinstance(pdf_path, str) or not pdf_path:
                    raise ValueError(f"{key} entry has invalid pdf_path: {item!r}")
                counts[key] += 1
                if pdf_path not in collected:
                    collected.append(pdf_path)
    return collected, counts["failed_docs"], counts.get("unprocessed_docs", 0)


def main() -> None:
    args = parse_args()
    summary = json.loads(args.summary_path.read_text(encoding="utf-8"))
    paths, failed, unprocessed = collect_paths(summary, include_unprocessed=not args.failed_only)
    args.output_list_file.parent.mkdir(parents=True, exist_ok=True)
    args.output_list_file.write_text("".join(f"{path}\n" for path in paths), encoding="utf-8")
    print(f"[done] {len(paths)} PDFs (failed={failed} unprocessed={unprocessed}): {args.output_list_file}")


if __name__ == "__main__":
    main()
