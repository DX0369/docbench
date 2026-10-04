from __future__ import annotations

import argparse
from pathlib import Path

from config.method_versions import get_method_version
from execution.commands import conda_python, env_vllm
from paths import SRC_ROOT


QWEN35_4B = get_method_version("qwen35_4b")
GEMMA3_4B = get_method_version("gemma3_4b")


def build_qwen_runner(
    args: argparse.Namespace,
    gpu_id: int,
    port: int | None,
    pdf_list_file: Path,
) -> list[str]:
    return _build_qwen_runner(args, port, pdf_list_file, enable_think=False)


def build_qwen_think_runner(
    args: argparse.Namespace,
    gpu_id: int,
    port: int | None,
    pdf_list_file: Path,
) -> list[str]:
    return _build_qwen_runner(args, port, pdf_list_file, enable_think=True)


def _build_qwen_runner(
    args: argparse.Namespace,
    port: int | None,
    pdf_list_file: Path,
    *,
    enable_think: bool,
) -> list[str]:
    assert port is not None
    method_id = "qwen35_4b_think" if enable_think else "qwen35_4b"
    method_name = "Qwen3.5-4B-thinking" if enable_think else "Qwen3.5-4B"
    cmd = conda_python("gdb_glmocr") + [
        str(SRC_ROOT / "methods" / "generic_vlm" / "vllm.py"),
        "--input-root",
        args.input_root,
        "--output-root",
        str(Path(args.output_root_base) / method_id),
        "--server-url",
        f"http://127.0.0.1:{port}/v1",
        "--model-name",
        "qwen35-4b",
        "--method-name",
        method_name,
        "--thinking-mode",
        "enabled" if enable_think else "disabled",
        "--max-tokens",
        "12000",
        "--dpi",
        "200",
        "--pdf-list-file",
        str(pdf_list_file),
    ]
    if args.skip_existing:
        cmd.append("--skip-existing")
    return cmd


def build_qwen_service(
    args: argparse.Namespace,
    gpu_id: int,
    port: int,
) -> list[str]:
    return [
        *conda_python("gdb_glmocr"),
        str(SRC_ROOT / "methods" / "generic_vlm" / "serve.py"),
        "--patch-qwen35-warmup",
        str(QWEN35_4B.model_path),
        "--served-model-name",
        "qwen35-4b",
        "--port",
        str(port),
        "--host",
        "127.0.0.1",
        "--max-model-len",
        "16384",
        "--gpu-memory-utilization",
        "0.95",
        "--enforce-eager",
        "--reasoning-parser",
        "qwen3",
    ]


def build_gemma_runner(
    args: argparse.Namespace,
    gpu_id: int,
    port: int | None,
    pdf_list_file: Path,
) -> list[str]:
    assert port is not None
    cmd = conda_python("gdb_glmocr") + [
        str(SRC_ROOT / "methods" / "generic_vlm" / "vllm.py"),
        "--input-root",
        args.input_root,
        "--output-root",
        str(Path(args.output_root_base) / "gemma3_4b"),
        "--server-url",
        f"http://127.0.0.1:{port}/v1",
        "--model-name",
        "gemma3-4b-it",
        "--method-name",
        "Gemma-3-4B-it",
        "--max-tokens",
        "12000",
        "--dpi",
        "200",
        "--pdf-list-file",
        str(pdf_list_file),
    ]
    if args.skip_existing:
        cmd.append("--skip-existing")
    return cmd


def build_gemma_service(
    args: argparse.Namespace,
    gpu_id: int,
    port: int,
) -> list[str]:
    return [
        env_vllm("gdb_glmocr"),
        "serve",
        str(GEMMA3_4B.model_path),
        "--served-model-name",
        "gemma3-4b-it",
        "--port",
        str(port),
        "--host",
        "127.0.0.1",
        "--max-model-len",
        "16384",
        "--gpu-memory-utilization",
        "0.95",
        "--enforce-eager",
    ]
