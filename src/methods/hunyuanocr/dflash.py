from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

from paths import CONDA_ENV_ROOT, PROJECT_ROOT
from config.method_versions import get_method_version


@dataclass(frozen=True)
class DFlashProfile:
    profile_id: str
    gpu_memory_utilization: float
    max_model_len: int
    max_num_batched_tokens: int
    num_speculative_tokens: int


DFLASH_PROFILES: dict[str, DFlashProfile] = {
    "official": DFlashProfile("official", 0.85, 131072, 131072, 15),
    "rtx3090": DFlashProfile("rtx3090", 0.85, 65536, 65536, 15),
}


def add_dflash_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--hunyuan-dflash-profile",
        choices=tuple(DFLASH_PROFILES),
        default="rtx3090",
    )
    parser.add_argument("--hunyuan-dflash-service-env")
    parser.add_argument("--hunyuan-dflash-cuda-home")
    parser.add_argument("--hunyuan-dflash-flashinfer-workspace")
    parser.add_argument("--hunyuan-dflash-gpu-memory-utilization", type=float)
    parser.add_argument("--hunyuan-dflash-max-model-len", type=int)
    parser.add_argument("--hunyuan-dflash-max-num-batched-tokens", type=int)
    parser.add_argument("--hunyuan-dflash-num-speculative-tokens", type=int)


def default_cuda_home() -> Path:
    configured = os.environ.get("GDB_DFLASH_CUDA_HOME")
    if configured:
        return Path(configured).expanduser().resolve()
    return (Path.home() / "software" / "cuda-envs" / "cuda-12.8").resolve()


def default_flashinfer_workspace(profile_id: str) -> Path:
    configured = os.environ.get("GDB_DFLASH_FLASHINFER_WORKSPACE")
    if configured:
        return Path(configured).expanduser().resolve()
    return (PROJECT_ROOT / ".cache" / "flashinfer" / f"hunyuan_dflash_{profile_id}").resolve()


def resolve_dflash_args(args: object) -> None:
    profile_id = str(getattr(args, "hunyuan_dflash_profile", "rtx3090"))
    try:
        profile = DFLASH_PROFILES[profile_id]
    except KeyError as exc:
        raise ValueError(f"Unknown Hunyuan DFlash profile: {profile_id}") from exc

    defaults = {
        "hunyuan_dflash_gpu_memory_utilization": profile.gpu_memory_utilization,
        "hunyuan_dflash_max_model_len": profile.max_model_len,
        "hunyuan_dflash_max_num_batched_tokens": profile.max_num_batched_tokens,
        "hunyuan_dflash_num_speculative_tokens": profile.num_speculative_tokens,
        "hunyuan_dflash_cuda_home": default_cuda_home(),
        "hunyuan_dflash_flashinfer_workspace": default_flashinfer_workspace(profile_id),
        "hunyuan_dflash_service_env": "gdb_hunyuan_dflash",
    }
    for name, default in defaults.items():
        if getattr(args, name, None) is None:
            setattr(args, name, str(default) if isinstance(default, Path) else default)


def validate_dflash_paths(args: object, *, create_workspace: bool) -> None:
    cuda_home = Path(str(getattr(args, "hunyuan_dflash_cuda_home")))
    service_env = str(getattr(args, "hunyuan_dflash_service_env"))
    workspace = Path(str(getattr(args, "hunyuan_dflash_flashinfer_workspace")))
    missing = []
    if not (cuda_home / "bin" / "nvcc").is_file():
        missing.append(str(cuda_home / "bin" / "nvcc"))
    if not (CONDA_ENV_ROOT / service_env / "bin" / "vllm").is_file():
        missing.append(str(CONDA_ENV_ROOT / service_env / "bin" / "vllm"))
    version = get_method_version("hunyuanocr_15_dflash")
    draft_model = version.additional_model_paths[0]
    for filename in ("config.json", "model.safetensors"):
        if not (draft_model / filename).is_file():
            missing.append(str(draft_model / filename))
    if missing:
        raise FileNotFoundError("Missing DFlash runtime path(s): " + ", ".join(missing))
    if create_workspace:
        workspace.mkdir(parents=True, exist_ok=True)
