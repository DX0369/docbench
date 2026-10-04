from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

from config.method_versions import get_method_version
from execution.commands import conda_python, env_vllm
from paths import CONDA_ENV_ROOT, SRC_ROOT


# 1.0: user-selected 0.90 overrides the official README example's 0.20.
# 1.5: official vLLM 0.18.1 serve.sh default.
# DFlash has its own profile; do not apply AR defaults to its service.
AR_GPU_MEMORY_DEFAULTS = {"hunyuanocr_10": 0.90, "hunyuanocr_15": 0.90}


def resolve_ar_gpu_memory_utilization(method_id: str, override: float | None) -> float:
    return AR_GPU_MEMORY_DEFAULTS[method_id] if override is None else override


@dataclass(frozen=True)
class HunyuanVersion:
    method_id: str
    display_name: str
    model_dir: Path
    served_model_name: str
    max_model_len: int
    max_tokens: int
    prompt_profile: str
    top_p: float
    top_k: int
    repetition_penalty: float


HUNYUAN_10_MANIFEST = get_method_version("hunyuanocr_10")
HUNYUAN_15_MANIFEST = get_method_version("hunyuanocr_15")
HUNYUAN_15_DFLASH_MANIFEST = get_method_version("hunyuanocr_15_dflash")

HUNYUAN_10 = HunyuanVersion(
    method_id="hunyuanocr_10",
    display_name=HUNYUAN_10_MANIFEST.display_name,
    model_dir=HUNYUAN_10_MANIFEST.model_path,
    served_model_name="hunyuanocr-1.0",
    max_model_len=32768,
    max_tokens=16384,
    prompt_profile="hunyuanocr_10",
    top_p=0.95,
    top_k=1,
    repetition_penalty=1.0,
)

HUNYUAN_15 = HunyuanVersion(
    method_id="hunyuanocr_15",
    display_name=HUNYUAN_15_MANIFEST.display_name,
    model_dir=HUNYUAN_15_MANIFEST.model_path,
    served_model_name="hunyuanocr-1.5",
    max_model_len=131072,
    max_tokens=32768,
    prompt_profile="hunyuanocr_15",
    top_p=1.0,
    top_k=-1,
    repetition_penalty=1.08,
)

HUNYUAN_15_DFLASH = HunyuanVersion(
    method_id="hunyuanocr_15_dflash",
    display_name=HUNYUAN_15_DFLASH_MANIFEST.display_name,
    model_dir=HUNYUAN_15_DFLASH_MANIFEST.model_path,
    served_model_name="hunyuanocr-1.5-dflash",
    max_model_len=131072,
    max_tokens=32768,
    prompt_profile="hunyuanocr_15",
    top_p=1.0,
    top_k=-1,
    repetition_penalty=1.08,
)


def _build_runner(
    args: argparse.Namespace,
    port: int | None,
    pdf_list_file: Path,
    version: HunyuanVersion,
) -> list[str]:
    assert port is not None
    return conda_python(args.hunyuan_runner_env) + [
        str(SRC_ROOT / "methods" / "hunyuanocr" / "vllm.py"),
        "--input-root",
        args.input_root,
        "--output-root",
        str(Path(args.output_root_base) / version.method_id),
        "--server-url",
        f"http://127.0.0.1:{port}/v1",
        "--model-name",
        version.served_model_name,
        "--method-id",
        version.method_id,
        "--method-display-name",
        version.display_name,
        "--prompt-profile",
        version.prompt_profile,
        "--max-tokens",
        str(version.max_tokens),
        "--top-p",
        str(version.top_p),
        "--top-k",
        str(version.top_k),
        "--repetition-penalty",
        str(version.repetition_penalty),
        "--dpi",
        "200",
        "--page-concurrency",
        str(args.hunyuan_page_concurrency),
        "--pdf-list-file",
        str(pdf_list_file),
        *(["--skip-existing"] if args.skip_existing else []),
    ]


def _build_service(
    args: argparse.Namespace,
    port: int,
    version: HunyuanVersion,
) -> list[str]:
    cmd = [
        env_vllm(args.hunyuan_service_env),
        "serve",
        str(version.model_dir),
        "--served-model-name",
        version.served_model_name,
        "--port",
        str(port),
        "--host",
        "127.0.0.1",
        "--trust-remote-code",
        "--no-enable-prefix-caching",
        "--max-model-len",
        str(args.hunyuan_vllm_max_model_len or version.max_model_len),
        "--mm-processor-cache-gb",
        "0",
        "--gpu-memory-utilization",
        str(resolve_ar_gpu_memory_utilization(version.method_id, args.hunyuan_vllm_gpu_memory_utilization)),
    ]
    if version.method_id == "hunyuanocr_15":
        cmd.extend(
            [
                "--limit-mm-per-prompt",
                '{"image":4,"video":0}',
                "--max-num-batched-tokens",
                str(
                    args.hunyuan_vllm_max_num_batched_tokens
                    or args.hunyuan_vllm_max_model_len
                    or version.max_model_len
                ),
            ]
        )
    if args.hunyuan_vllm_enforce_eager:
        cmd.append("--enforce-eager")
    return cmd


def _append_no_proxy(value: str, *hosts: str) -> str:
    entries = [item.strip() for item in value.split(",") if item.strip()]
    for host in hosts:
        if host not in entries:
            entries.append(host)
    return ",".join(entries)


def _build_dflash_service(
    args: argparse.Namespace,
    port: int,
    version: HunyuanVersion,
) -> list[str]:
    cuda_home = Path(args.hunyuan_dflash_cuda_home).expanduser().resolve()
    service_bin = CONDA_ENV_ROOT / args.hunyuan_dflash_service_env / "bin"
    existing_path = os.environ.get("PATH", "")
    service_path = f"{service_bin}:{cuda_home / 'bin'}"
    if existing_path:
        service_path = f"{service_path}:{existing_path}"
    no_proxy = _append_no_proxy(
        os.environ.get("no_proxy") or os.environ.get("NO_PROXY", ""),
        "127.0.0.1",
        "localhost",
    )
    draft_model = HUNYUAN_15_DFLASH_MANIFEST.additional_model_paths[0]
    max_model_len = args.hunyuan_dflash_max_model_len
    max_num_batched_tokens = (
        args.hunyuan_dflash_max_num_batched_tokens or max_model_len
    )
    speculative_config = (
        '{"method":"dflash","model":"'
        f"{draft_model}"
        '","num_speculative_tokens":'
        f"{args.hunyuan_dflash_num_speculative_tokens}"
        "}"
    )
    return [
        "/usr/bin/env",
        f"PATH={service_path}",
        f"CUDA_HOME={cuda_home}",
        f"FLASHINFER_NVCC={cuda_home / 'bin' / 'nvcc'}",
        f"FLASHINFER_WORKSPACE_BASE={Path(args.hunyuan_dflash_flashinfer_workspace).expanduser().resolve()}",
        f"NO_PROXY={no_proxy}",
        f"no_proxy={no_proxy}",
        "HF_HUB_OFFLINE=1",
        "TRANSFORMERS_OFFLINE=1",
        env_vllm(args.hunyuan_dflash_service_env),
        "serve",
        str(version.model_dir),
        "--served-model-name",
        version.served_model_name,
        "-tp",
        "1",
        "--limit-mm-per-prompt",
        '{"image":4,"video":0}',
        "--trust-remote-code",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--gpu-memory-utilization",
        str(args.hunyuan_dflash_gpu_memory_utilization),
        "--max-model-len",
        str(max_model_len),
        "--max-num-batched-tokens",
        str(max_num_batched_tokens),
        "--speculative-config",
        speculative_config,
    ]


def build_10_runner(
    args: argparse.Namespace,
    gpu_id: int,
    port: int | None,
    pdf_list_file: Path,
) -> list[str]:
    return _build_runner(args, port, pdf_list_file, HUNYUAN_10)


def build_15_runner(
    args: argparse.Namespace,
    gpu_id: int,
    port: int | None,
    pdf_list_file: Path,
) -> list[str]:
    return _build_runner(args, port, pdf_list_file, HUNYUAN_15)


def build_15_dflash_runner(
    args: argparse.Namespace,
    gpu_id: int,
    port: int | None,
    pdf_list_file: Path,
) -> list[str]:
    return _build_runner(args, port, pdf_list_file, HUNYUAN_15_DFLASH)


def _build_worker(args, queue_dir, worker_id, version):
    cmd = _build_runner(args, args.service_port, queue_dir, version)
    index = cmd.index("--pdf-list-file")
    del cmd[index:index + 2]
    cmd.extend(["--queue-dir", str(queue_dir), "--worker-id", worker_id])
    if getattr(args, "allow_partial_success", False):
        cmd.append("--allow-partial-success")
    return cmd


def build_10_worker(args, gpu_id, queue_dir, worker_id):
    return _build_worker(args, queue_dir, worker_id, HUNYUAN_10)


def build_15_worker(args, gpu_id, queue_dir, worker_id):
    return _build_worker(args, queue_dir, worker_id, HUNYUAN_15)


def build_15_dflash_worker(args, gpu_id, queue_dir, worker_id):
    return _build_worker(args, queue_dir, worker_id, HUNYUAN_15_DFLASH)


def build_10_service(args: argparse.Namespace, gpu_id: int, port: int) -> list[str]:
    return _build_service(args, port, HUNYUAN_10)


def build_15_service(args: argparse.Namespace, gpu_id: int, port: int) -> list[str]:
    return _build_service(args, port, HUNYUAN_15)


def build_15_dflash_service(
    args: argparse.Namespace,
    gpu_id: int,
    port: int,
) -> list[str]:
    return _build_dflash_service(args, port, HUNYUAN_15_DFLASH)
