from __future__ import annotations

import os
import subprocess
from pathlib import Path

from paths import METHOD_ROOT


METHOD_REPOSITORY_DIRS: dict[str, str] = {
    "deepseek_ocr2": "deepseek_ocr2",
    "glmocr": "glmocr",
    "hunyuanocr": "hunyuanocr",
    "marker": "marker",
    "mineru": "mineru",
    "mineru_popo": "MinerU-Popo",
    "monkeyocr": "monkeyocr",
    "paddleocr": "paddleocr",
    "qianfan_vl": "qianfan_vl",
    "teleocr": "TeleOCR",
    "unlimited_ocr": "unlimited_ocr",
    "youtu": "youtu",
}

METHOD_REPOSITORY_ENVS: dict[str, str] = {
    method_id: f"GDB_{method_id.upper()}_REPO_ROOT"
    for method_id in METHOD_REPOSITORY_DIRS
}


def configured_path(env_name: str, default: Path) -> Path:
    value = os.environ.get(env_name)
    if not value:
        return default
    return Path(os.path.expandvars(value)).expanduser().resolve()


def method_repo_path(method_id: str) -> Path:
    """Resolve a method checkout with an optional per-method override."""
    try:
        directory = METHOD_REPOSITORY_DIRS[method_id]
        env_name = METHOD_REPOSITORY_ENVS[method_id]
    except KeyError as exc:
        raise KeyError(f"Unknown method repository: {method_id}") from exc
    return configured_path(env_name, METHOD_ROOT / directory)


def require_failure_patch(method_id: str, patch_name: str = "failure_propagation.patch") -> None:
    """Check the documented source patch once at runtime construction; never apply it implicitly."""
    repo = method_repo_path(method_id)
    patch = Path(__file__).parent / "methods" / method_id / "patches" / patch_name
    result = subprocess.run(
        ["git", "-C", str(repo), "apply", "--reverse", "--check", str(patch)],
        capture_output=True, text=True, check=False,
    )
    if result.returncode:
        raise RuntimeError(
            f"Required source patch {patch_name} is absent or has drifted: {repo}. "
            f"Apply the documented patch first: git -C {repo} apply {patch}\n{result.stderr}"
        )
