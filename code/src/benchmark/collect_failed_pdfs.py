#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Set

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from paths import PDF_ROOT  # noqa: E402


DEFAULT_INPUT_ROOT = PDF_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect failed PDF paths from a previous throughput method-queue run "
            "into a reusable pressure-subset list file."
        )
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        required=True,
        help="Path to method_queue_summary.json from a previous throughput run.",
    )
    parser.add_argument(
        "--output-list-file",
        type=Path,
        required=True,
        help="Where to write the collected PDF path list.",
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
        help="Dataset PDF root used to reconstruct paths from output-relative error.json files.",
    )
    parser.add_argument(
        "--methods",
        type=str,
        help="Optional comma-separated subset of methods to include.",
    )
    parser.add_argument(
        "--max-per-method",
        type=int,
        help="Optional cap for how many failed PDFs to keep per method.",
    )
    parser.add_argument(
        "--summary-output-path",
        type=Path,
        help="Optional JSON summary path. Defaults to <output-list-file>.json.",
    )
    return parser.parse_args()


def parse_methods(text: str | None) -> Set[str] | None:
    if text is None:
        return None
    methods = {item.strip() for item in text.split(",") if item.strip()}
    return methods or None


def try_get_trial_output_root(attempt: Dict[str, object]) -> Path | None:
    summary_path = attempt.get("summary_path")
    if not isinstance(summary_path, str):
        return None
    path = Path(summary_path)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    best = data.get("best_stable_config")
    if isinstance(best, dict):
        trial_output_root = best.get("trial_output_root")
        if isinstance(trial_output_root, str):
            return Path(trial_output_root)
    for trial in data.get("trials", []):
        if isinstance(trial, dict):
            trial_output_root = trial.get("trial_output_root")
            if isinstance(trial_output_root, str):
                return Path(trial_output_root)
    return None


def collect_failed_paths_from_aggregate_summary(
    aggregate_summary_path: Path,
    input_root: Path,
    method: str | None = None,
) -> List[Path]:
    try:
        data = json.loads(aggregate_summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []

    inferred_method = method or str(data.get("method", ""))
    collected: List[Path] = []
    seen: Set[Path] = set()

    for trial in data.get("results", []):
        if not isinstance(trial, dict):
            continue
        for job_status in trial.get("job_statuses", []):
            if not isinstance(job_status, dict):
                continue
            for failed_doc in job_status.get("failed_docs", []):
                if isinstance(failed_doc, str):
                    pdf_path = Path(failed_doc).resolve()
                    if pdf_path not in seen:
                        seen.add(pdf_path)
                        collected.append(pdf_path)

        trial_output_root_value = trial.get("trial_output_root")
        if not isinstance(trial_output_root_value, str):
            continue
        trial_output_root = Path(trial_output_root_value)
        method_root = trial_output_root / inferred_method
        if not method_root.exists():
            continue
        for error_json_path in sorted(method_root.rglob("error.json")):
            pdf_path = reconstruct_pdf_path(input_root, method_root, error_json_path)
            if pdf_path not in seen:
                seen.add(pdf_path)
                collected.append(pdf_path)
    return collected


def reconstruct_pdf_path(input_root: Path, method_root: Path, error_json_path: Path) -> Path:
    relative_doc_dir = error_json_path.parent.relative_to(method_root)
    pdf_name = f"{relative_doc_dir.name}.pdf"
    return (input_root / relative_doc_dir / pdf_name).resolve()


def collect_failed_paths_from_result(
    result: Dict[str, object],
    input_root: Path,
) -> List[Path]:
    collected: List[Path] = []
    seen: Set[Path] = set()
    method = str(result.get("method", ""))

    attempts = [
        attempt
        for attempt in result.get("full_attempts", [])
        if isinstance(attempt, dict)
    ]
    if not attempts:
        fallback_attempt: Dict[str, object] = {
            "summary_path": result.get("full_summary_path") or result.get("tune_summary_path")
        }
        attempts = [fallback_attempt]

    for attempt in attempts:
        summary_path_value = attempt.get("summary_path")
        if isinstance(summary_path_value, str):
            summary_path = Path(summary_path_value)
            if summary_path.exists():
                aggregate_paths = collect_failed_paths_from_aggregate_summary(
                    summary_path,
                    input_root,
                    method=method,
                )
                for pdf_path in aggregate_paths:
                    if pdf_path not in seen:
                        seen.add(pdf_path)
                        collected.append(pdf_path)
                try:
                    full_summary = json.loads(summary_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    full_summary = {}
                best = full_summary.get("best_stable_config")
                if isinstance(best, dict):
                    for job_status in best.get("job_statuses", []):
                        if not isinstance(job_status, dict):
                            continue
                        for failed_doc in job_status.get("failed_docs", []):
                            if isinstance(failed_doc, str):
                                pdf_path = Path(failed_doc).resolve()
                                if pdf_path not in seen:
                                    seen.add(pdf_path)
                                    collected.append(pdf_path)

        trial_output_root = try_get_trial_output_root(attempt)
        if trial_output_root is None or not trial_output_root.exists():
            continue
        method_root = trial_output_root / method
        if not method_root.exists():
            continue
        for error_json_path in sorted(method_root.rglob("error.json")):
            pdf_path = reconstruct_pdf_path(input_root, method_root, error_json_path)
            if pdf_path not in seen:
                seen.add(pdf_path)
                collected.append(pdf_path)

    return collected


def main() -> None:
    args = parse_args()
    args.summary_path = args.summary_path.resolve()
    args.output_list_file = args.output_list_file.resolve()
    args.input_root = args.input_root.resolve()
    if args.summary_output_path is None:
        args.summary_output_path = args.output_list_file.with_suffix(
            args.output_list_file.suffix + ".json"
        )
    else:
        args.summary_output_path = args.summary_output_path.resolve()

    methods_filter = parse_methods(args.methods)
    summary = json.loads(args.summary_path.read_text(encoding="utf-8"))

    per_method: Dict[str, List[Path]] = {}
    if isinstance(summary.get("method"), str) and isinstance(summary.get("results"), list):
        method = str(summary.get("method", ""))
        if methods_filter is None or method in methods_filter:
            failed_paths = collect_failed_paths_from_aggregate_summary(
                args.summary_path,
                args.input_root,
                method=method,
            )
            if args.max_per_method is not None:
                failed_paths = failed_paths[: args.max_per_method]
            per_method[method] = failed_paths
    else:
        for result in summary.get("results", []):
            if not isinstance(result, dict):
                continue
            method = str(result.get("method", ""))
            if methods_filter is not None and method not in methods_filter:
                continue
            failed_paths = collect_failed_paths_from_result(result, args.input_root)
            if args.max_per_method is not None:
                failed_paths = failed_paths[: args.max_per_method]
            per_method[method] = failed_paths

    ordered_paths: List[Path] = []
    seen_global: Set[Path] = set()
    for method in sorted(per_method):
        for pdf_path in per_method[method]:
            if pdf_path not in seen_global:
                seen_global.add(pdf_path)
                ordered_paths.append(pdf_path)

    args.output_list_file.parent.mkdir(parents=True, exist_ok=True)
    args.output_list_file.write_text(
        "".join(f"{path}\n" for path in ordered_paths),
        encoding="utf-8",
    )

    args.summary_output_path.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output_path.write_text(
        json.dumps(
            {
                "summary_path": str(args.summary_path),
                "input_root": str(args.input_root),
                "methods_filter": sorted(methods_filter) if methods_filter is not None else None,
                "max_per_method": args.max_per_method,
                "selected_count": len(ordered_paths),
                "per_method_counts": {
                    method: len(paths) for method, paths in sorted(per_method.items())
                },
                "selected": [str(path) for path in ordered_paths],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"[done] pressure list: {args.output_list_file}")
    print(f"[done] pressure summary: {args.summary_output_path}")


if __name__ == "__main__":
    main()
