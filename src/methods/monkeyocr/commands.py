from __future__ import annotations

import argparse
from pathlib import Path

from execution.commands import conda_python
from config.method_versions import MethodVersion
from methods.monkeyocr.versions import PRO_1_2B, PRO_3B
from paths import SRC_ROOT


def _models_dir(args: argparse.Namespace, version: MethodVersion) -> Path:
    override = getattr(args, "monkey_models_dir", None)
    return Path(override) if override else version.model_path


def _build_runner(
    args: argparse.Namespace,
    gpu_id: int,
    port: int | None,
    pdf_list_file: Path,
    version: MethodVersion,
) -> list[str]:
    cmd = conda_python(version.conda_environment) + [
        str(SRC_ROOT / "methods" / "monkeyocr" / "pipeline.py"),
        "--input-root",
        args.input_root,
        "--output-root",
        str(Path(args.output_root_base) / version.method_id),
        "--method-id",
        version.method_id,
        "--method-display-name",
        version.display_name,
        "--region-concurrency",
        str(args.monkey_region_concurrency),
        "--models-dir",
        str(_models_dir(args, version)),
        "--pdf-list-file",
        str(pdf_list_file),
    ]
    if args.skip_existing:
        cmd.append("--skip-existing")
    return cmd


def _build_worker(
    args: argparse.Namespace,
    gpu_id: int,
    queue_dir: Path,
    worker_id: str,
    version: MethodVersion,
) -> list[str]:
    cmd = conda_python(version.conda_environment) + [
        str(SRC_ROOT / "methods" / "monkeyocr" / "worker.py"),
        "--input-root",
        args.input_root,
        "--output-root",
        str(Path(args.output_root_base) / version.method_id),
        "--method-id",
        version.method_id,
        "--method-display-name",
        version.display_name,
        "--queue-dir",
        str(queue_dir),
        "--worker-id",
        worker_id,
        "--region-concurrency",
        str(args.monkey_region_concurrency),
        "--models-dir",
        str(_models_dir(args, version)),
    ]
    if args.skip_existing:
        cmd.append("--skip-existing")
    return cmd


def build_pro_3b_runner(
    args: argparse.Namespace,
    gpu_id: int,
    port: int | None,
    pdf_list_file: Path,
) -> list[str]:
    return _build_runner(args, gpu_id, port, pdf_list_file, PRO_3B)


def build_pro_3b_worker(
    args: argparse.Namespace,
    gpu_id: int,
    queue_dir: Path,
    worker_id: str,
) -> list[str]:
    return _build_worker(args, gpu_id, queue_dir, worker_id, PRO_3B)


def build_pro_1_2b_runner(
    args: argparse.Namespace,
    gpu_id: int,
    port: int | None,
    pdf_list_file: Path,
) -> list[str]:
    return _build_runner(args, gpu_id, port, pdf_list_file, PRO_1_2B)


def build_pro_1_2b_worker(
    args: argparse.Namespace,
    gpu_id: int,
    queue_dir: Path,
    worker_id: str,
) -> list[str]:
    return _build_worker(args, gpu_id, queue_dir, worker_id, PRO_1_2B)
