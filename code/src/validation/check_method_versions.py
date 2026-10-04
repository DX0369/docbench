#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from execution.tasks import NATIVE_TASKS, SERVICE_TASKS as VLLM_TASKS  # noqa: E402
from config.method_versions import METHOD_CATALOG_PATH, load_method_catalog, load_method_versions  # noqa: E402
from paths import CONDA_ENV_ROOT, PROJECT_ROOT  # noqa: E402


@dataclass(frozen=True)
class AuditResult:
    method_id: str
    category: str
    status: str
    detail: str


def run_git(repo_path: Path, *args: str) -> tuple[int, str]:
    completed = subprocess.run(
        ["git", "-C", str(repo_path), *args],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return completed.returncode, completed.stdout.strip()


def add_result(
    results: list[AuditResult],
    method_id: str,
    category: str,
    status: str,
    detail: str,
) -> None:
    results.append(AuditResult(method_id, category, status, detail))


def audit() -> list[AuditResult]:
    versions = load_method_versions()
    lock_payload = load_method_catalog()
    repositories = {item["id"]: item for item in lock_payload["repositories"]}
    registered_tasks = set(VLLM_TASKS) | set(NATIVE_TASKS)
    results: list[AuditResult] = []

    for version in versions.values():
        method_id = version.method_id

        model_status = "ok" if version.model_path.is_dir() else (
            "error" if version.model_required else "expected_missing"
        )
        add_result(results, method_id, "primary_model", model_status, str(version.model_path))
        for path in version.additional_model_paths:
            status = "ok" if path.is_dir() else (
                "error" if version.model_required else "expected_missing"
            )
            add_result(results, method_id, "additional_model", status, str(path))

        if version.source_binding == "model_only":
            status = "ok" if version.source_repository_id is None and version.source_revision is None else "error"
            add_result(results, method_id, "source_binding", status, "model_only")
        elif version.source_repository_id is None:
            add_result(
                results,
                method_id,
                "source_binding",
                "error",
                f"{version.source_binding} requires source_repository_id",
            )
        elif version.source_repository_id not in repositories:
            add_result(
                results,
                method_id,
                "source",
                "error",
                f"repository id missing from lock: {version.source_repository_id}",
            )
        else:
            repository = repositories[version.source_repository_id]
            repo_path = PROJECT_ROOT / repository["path"]
            if not repo_path.is_dir():
                add_result(results, method_id, "source", "error", f"missing {repo_path}")
            else:
                code, head = run_git(repo_path, "rev-parse", "HEAD")
                expected_head = version.source_revision
                status = "ok" if code == 0 and expected_head and head == expected_head else "error"
                add_result(
                    results,
                    method_id,
                    "source_checkout",
                    status,
                    f"HEAD={head or 'unavailable'} lock={expected_head}",
                )
                if version.source_binding == "compatibility_source":
                    binding_status = "compatibility"
                    binding_detail = (
                        f"runtime={version.source_revision} historical="
                        f"{version.historical_source_revision or 'missing'}"
                    )
                elif version.source_binding == "reference_checkout":
                    binding_status = "reference"
                    binding_detail = "repository is provenance reference; runtime code is model-bundled"
                else:
                    binding_status = "ok"
                    binding_detail = "exact_checkout"
                if version.source_binding == "compatibility_source" and not version.historical_source_revision:
                    binding_status = "error"
                add_result(
                    results,
                    method_id,
                    "source_binding",
                    binding_status,
                    binding_detail,
                )
                if version.historical_source_revision:
                    code, _ = run_git(
                        repo_path,
                        "cat-file",
                        "-e",
                        f"{version.historical_source_revision}^{{commit}}",
                    )
                    add_result(
                        results,
                        method_id,
                        "historical_source_revision",
                        "ok" if code == 0 else "error",
                        version.historical_source_revision,
                    )
                code, dirty = run_git(repo_path, "status", "--porcelain")
                add_result(
                    results,
                    method_id,
                    "source_dirty",
                    "warning" if code == 0 and dirty else "ok",
                    dirty or "clean",
                )

        if version.adapter_module is None:
            adapter_status = "error" if version.benchmark_status == "active" else "pending"
            add_result(results, method_id, "adapter", adapter_status, "no adapter module")
        else:
            spec = importlib.util.find_spec(version.adapter_module)
            add_result(
                results,
                method_id,
                "adapter",
                "ok" if spec is not None else "error",
                version.adapter_module,
            )

        for env_name in version.conda_environments:
            python_path = CONDA_ENV_ROOT / env_name / "bin" / "python"
            status = "ok" if python_path.is_file() else (
                "error" if version.benchmark_status == "active" else "pending"
            )
            add_result(results, method_id, "environment", status, f"{env_name}: {python_path}")

        if version.benchmark_status == "active":
            add_result(
                results,
                method_id,
                "task_registry",
                "ok" if method_id in registered_tasks else "error",
                "registered" if method_id in registered_tasks else "active variant is not registered",
            )
        else:
            add_result(
                results,
                method_id,
                "task_registry",
                "not_active",
                version.benchmark_status,
            )

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit method version, source, model, adapter, and environment mappings.")
    parser.add_argument("--json-report", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = audit()
    for result in results:
        print(
            f"[{result.status.upper():16}] {result.method_id}: "
            f"{result.category} -> {result.detail}"
        )

    summary = {
        "manifest": str(METHOD_CATALOG_PATH),
        "error_count": sum(result.status == "error" for result in results),
        "warning_count": sum(result.status == "warning" for result in results),
        "results": [asdict(result) for result in results],
    }
    if args.json_report is not None:
        args.json_report.parent.mkdir(parents=True, exist_ok=True)
        args.json_report.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"[REPORT          ] {args.json_report}")
    if summary["error_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
