#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from paths import OUTPUTS_FORMAL_ROOT  # noqa: E402
from common import get_doc_output_dir, document_completed, write_json, build_error_payload
from execution.trial import get_pdf_page_count


def validate_completed_page_counts(method, input_root, output_root, pdf_paths):
    counts, failures = {}, []
    for path in pdf_paths:
        output = get_doc_output_dir(input_root, output_root, path)
        if not document_completed(output):
            continue
        try:
            expected = get_pdf_page_count(path)
            metrics = json.loads((output / "metrics.json").read_text())
            if not isinstance(metrics, dict):
                raise ValueError("metrics.json must contain an object")
            if method in {"qianfan_ocr", "qianfan_ocr_think"}:
                validate_qianfan_metrics(metrics, expected)
            elif method == "unlimited_ocr_sglang":
                validate_unlimited_metrics(metrics, expected)
            elif method == "teleocr_vllm_async":
                validate_teleocr_metrics(metrics, expected)
            elif type(metrics["page_count"]) is not int or metrics["page_count"] != expected or not isinstance(metrics["pages"], list) or len(metrics["pages"]) != expected:
                raise ValueError(f"Input has {expected} pages, output reports {metrics['page_count']} pages / {len(metrics['pages'])} page records")
            counts[path] = expected
        except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
            error_path = output / "error.json"
            write_json(error_path, build_error_payload(method, path, exception=exc, stage="output_page_coverage"))
            failures.append({"pdf_path": str(path), "status": "failed", "reason": str(exc),
                             "error_json": str(error_path)})
    return counts, failures


@dataclass
class ValidationIssue:
    severity: str
    path: str
    message: str


@dataclass
class ValidationResult:
    checked_docs: int = 0
    passed_docs: int = 0
    failed_docs: int = 0
    issues: list[ValidationIssue] = field(default_factory=list)

    def add_issue(self, severity: str, path: Path, message: str) -> None:
        self.issues.append(ValidationIssue(severity, str(path), message))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate GlobalDocBench runner output bundles."
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=OUTPUTS_FORMAL_ROOT,
        help="Root containing method output directories.",
    )
    parser.add_argument(
        "--method",
        action="append",
        help="Optional method id to validate. Can be repeated.",
    )
    parser.add_argument(
        "--allow-document-only",
        action="store_true",
        help=(
            "Allow document.md without pages/page_*.md. Formal benchmark outputs "
            "should not use this; it is intended only for temporary experiments."
        ),
    )
    parser.add_argument(
        "--json-report",
        type=Path,
        help="Optional path to write a JSON validation report.",
    )
    parser.add_argument("--max-issues", type=int, default=200)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def is_doc_dir(path: Path) -> bool:
    return (path / "document.md").exists() or (path / "metrics.json").exists() or (
        path / "error.json"
    ).exists()


def validate_qianfan_metrics(metrics: dict[str, Any], expected: int) -> None:
    """Each input page must have one request and an ordered page record."""
    method = metrics["method_id"]
    if method not in {"qianfan_ocr", "qianfan_ocr_think"}:
        raise ValueError("Qianfan metadata has an unknown method ID")
    if metrics["execution_profile"] != f"{method}_vllm_page_pool" or metrics["request_mode"] != "page_pool":
        raise ValueError("Qianfan page-pool profile mismatch")
    if metrics["input_page_count"] != expected or metrics["page_count"] != expected:
        raise ValueError("Qianfan input page count mismatch")
    if metrics["output_scope"] != "document_and_pages" or metrics["page_boundary_source"] != "input_pdf":
        raise ValueError("Qianfan must export pages using input PDF boundaries")
    requests = metrics["requests"]
    indices = list(range(1, expected + 1))
    if [r["page_index"] for r in requests] != indices or any(
        r["image_count"] != 1 or r["finish_reason"] not in {"stop", "length"} for r in requests
    ):
        raise ValueError("Qianfan requires one completed single-image request per page")
    if [p["page_index"] for p in metrics["pages"]] != indices:
        raise ValueError("Incomplete Qianfan page records")


def validate_unlimited_metrics(metrics: dict[str, Any], expected: int) -> None:
    """Completed requests count input pages; output quality is recorded separately."""
    if metrics["method_id"] != "unlimited_ocr_sglang" or metrics["backend"] != "sglang":
        raise ValueError("Unlimited completion policy only applies to its SGLang adapter")
    if (type(expected) is not int or expected <= 0
            or type(metrics["input_page_count"]) is not int or type(metrics["page_count"]) is not int
            or metrics["input_page_count"] != expected or metrics["page_count"] != expected):
        raise ValueError("Unlimited completed input page count mismatch")
    if metrics.get("request_mode") == "page_pool":
        if metrics["execution_profile"] != "unlimited_ocr_sglang_page_pool":
            raise ValueError("Unlimited page-pool profile mismatch")
        if (metrics["page_boundary_source"] != "input_pdf" or metrics["page_count"] != expected
                or metrics["input_page_count"] != expected or metrics["request_completed"] is not True):
            raise ValueError("Unlimited page-pool input coverage mismatch")
        pages = metrics["pages"]
        if ([p["page_index"] for p in pages] != list(range(1, expected + 1))
                or any(p["request_completed"] is not True or p["finish_reason"] not in ("stop", "length") for p in pages)):
            raise ValueError("Unlimited page request did not complete")
        return
    if metrics["request_completed"] is not True or metrics["finish_reason"] not in ("stop", "length"):
        raise ValueError("Unlimited requires a completed request with a known stop/length reason")
    segments = metrics["output_segment_count"]
    if type(segments) is not int or segments < 0:
        raise ValueError("Unlimited output_segment_count must be a nonnegative integer")
    mismatch = segments != expected
    warnings = (["generation_length"] if metrics["finish_reason"] == "length" else [])
    if mismatch:
        warnings.append("page_count_mismatch")
    if metrics["quality_warnings"] != warnings:
        raise ValueError("Unlimited quality warnings do not match generation/segment counts")
    if mismatch:
        if (metrics["output_scope"] != "document_only" or metrics["pages"] != []
                or metrics["page_boundary_source"] != "unavailable"):
            raise ValueError("Unlimited mismatched segments must not fabricate page records")
    elif (metrics["output_scope"] != "document_and_pages"
          or metrics["page_boundary_source"] != "model_page_tokens"
          or [p["page_index"] for p in metrics["pages"]] != list(range(1, expected + 1))):
        raise ValueError("Unlimited page view metadata mismatch")


def validate_teleocr_metrics(metrics: dict[str, Any], expected: int) -> None:
    if metrics.get("method_id") != "teleocr_vllm_async":
        raise ValueError("TeleOCR method id mismatch")
    if metrics.get("execution_profile") != "teleocr_vllm_async":
        raise ValueError("TeleOCR execution profile mismatch")
    if metrics.get("backend") != "vllm-async-engine":
        raise ValueError("TeleOCR backend mismatch")
    if metrics.get("input_page_count") != expected or metrics.get("page_count") != expected:
        raise ValueError("TeleOCR input/output page count mismatch")
    if metrics.get("recognized_page_count") != expected:
        raise ValueError("TeleOCR recognized page count mismatch")
    if type(metrics.get("max_concurrency")) is not int or metrics["max_concurrency"] < 1:
        raise ValueError("TeleOCR max_concurrency must be a positive integer")


def iter_doc_dirs(method_root: Path) -> list[Path]:
    return sorted(path for path in method_root.rglob("*") if path.is_dir() and is_doc_dir(path))


def validate_doc_dir(
    doc_dir: Path,
    *,
    result: ValidationResult,
    allow_document_only: bool,
) -> bool:
    ok = True
    error_json = doc_dir / "error.json"
    if error_json.exists():
        try:
            payload = load_json(error_json)
            if not isinstance(payload, dict):
                result.add_issue("error", error_json, "error.json must contain a JSON object")
                ok = False
        except Exception as exc:  # noqa: BLE001
            result.add_issue("error", error_json, f"error.json is invalid JSON: {exc}")
            ok = False
        return ok

    document_md = doc_dir / "document.md"
    if not document_md.exists():
        result.add_issue("error", doc_dir, "missing document.md")
        ok = False

    pages_dir = doc_dir / "pages"
    page_files = sorted(pages_dir.glob("page_*.md")) if pages_dir.exists() else []

    metrics_path = doc_dir / "metrics.json"
    metrics: dict[str, Any] | None = None
    if not metrics_path.exists():
        result.add_issue("error", doc_dir, "missing metrics.json")
        ok = False
    else:
        try:
            loaded = load_json(metrics_path)
            if not isinstance(loaded, dict):
                raise ValueError("metrics.json must contain a JSON object")
            metrics = loaded
        except Exception as exc:  # noqa: BLE001
            result.add_issue("error", metrics_path, f"invalid metrics.json: {exc}")
            ok = False

    image_boxes_path = doc_dir / "image_boxes.json"
    if not image_boxes_path.exists():
        result.add_issue("error", doc_dir, "missing image_boxes.json")
        ok = False
    else:
        try:
            load_json(image_boxes_path)
        except Exception as exc:  # noqa: BLE001
            result.add_issue("error", image_boxes_path, f"invalid image_boxes.json: {exc}")
            ok = False

    qianfan = bool(metrics and metrics.get("method_id") in {"qianfan_ocr", "qianfan_ocr_think"})
    unlimited = bool(metrics and metrics.get("method_id") == "unlimited_ocr_sglang")
    unlimited_document_only = unlimited and metrics.get("output_scope") == "document_only"
    if not page_files and not (allow_document_only or unlimited_document_only):
        result.add_issue("error", doc_dir, "missing pages/page_*.md")
        ok = False
    if metrics is not None:
        page_count = metrics.get("page_count")
        pages = metrics.get("pages")
        if not isinstance(page_count, int) or page_count <= 0:
            result.add_issue("error", metrics_path, "metrics.page_count must be a positive integer")
            ok = False
        if not isinstance(pages, list):
            result.add_issue("error", metrics_path, "metrics.pages must be a list")
            ok = False
        elif isinstance(page_count, int) and len(pages) != page_count and not unlimited_document_only:
            result.add_issue(
                "error",
                metrics_path,
                f"metrics.pages length {len(pages)} != page_count {page_count}",
            )
            ok = False
        if page_files and isinstance(page_count, int) and len(page_files) != page_count:
            result.add_issue(
                "error",
                pages_dir,
                f"page markdown count {len(page_files)} != metrics.page_count {page_count}",
            )
            ok = False
        if "doc_total_seconds" not in metrics:
            result.add_issue("warning", metrics_path, "missing doc_total_seconds")
        if unlimited:
            try:
                validate_unlimited_metrics(metrics, page_count)
                if unlimited_document_only and page_files:
                    raise ValueError("Document-only Unlimited contains stale page files")
            except (KeyError, TypeError, ValueError) as exc:
                result.add_issue("error", metrics_path, f"Invalid Unlimited metadata: {exc}")
                ok = False
        if qianfan:
            try:
                validate_qianfan_metrics(metrics, page_count)
            except (KeyError, TypeError, ValueError) as exc:
                result.add_issue("error", metrics_path, f"Invalid Qianfan metadata: {exc}")
                ok = False

    return ok


def validate_method_root(
    method_root: Path,
    *,
    result: ValidationResult,
    allow_document_only: bool,
) -> None:
    doc_dirs = iter_doc_dirs(method_root)
    if not doc_dirs:
        result.add_issue("error", method_root, "no document output directories found")
        return
    for doc_dir in doc_dirs:
        result.checked_docs += 1
        document_tree_method = method_root.name == "mineru_344_2605_popo"
        if document_tree_method and (doc_dir / "error.json").exists():
            result.add_issue("error", doc_dir, "MinerU + Popo document failed; see error.json")
            result.failed_docs += 1
            continue
        if document_tree_method and not (doc_dir / "error.json").exists():
            try:
                tree = load_json(doc_dir / "document_tree.json")
                if tree["type"] != "root":
                    raise ValueError("Expected official Popo root node")
            except Exception as exc:
                result.add_issue("error", doc_dir, f"Invalid document_tree.json: {exc}")
                result.failed_docs += 1
                continue
        if validate_doc_dir(
            doc_dir,
            result=result,
            allow_document_only=allow_document_only or document_tree_method,
        ):
            result.passed_docs += 1
        else:
            result.failed_docs += 1


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    methods = args.method or [
        path.name for path in sorted(output_root.iterdir()) if path.is_dir()
    ]
    result = ValidationResult()
    for method in methods:
        validate_method_root(
            output_root / method,
            result=result,
            allow_document_only=args.allow_document_only,
        )

    payload = {
        "output_root": str(output_root),
        "methods": methods,
        "checked_docs": result.checked_docs,
        "passed_docs": result.passed_docs,
        "failed_docs": result.failed_docs,
        "issue_count": len(result.issues),
        "issues": [issue.__dict__ for issue in result.issues],
    }
    if args.json_report:
        args.json_report.parent.mkdir(parents=True, exist_ok=True)
        args.json_report.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print(
        "checked_docs={checked_docs} passed_docs={passed_docs} "
        "failed_docs={failed_docs} issue_count={issue_count}".format(**payload)
    )
    for issue in result.issues[: args.max_issues]:
        print(f"[{issue.severity}] {issue.path}: {issue.message}")
    if len(result.issues) > args.max_issues:
        print(f"... {len(result.issues) - args.max_issues} more issues omitted")
    if any(issue.severity == "error" for issue in result.issues):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
