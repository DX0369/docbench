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
    "monkeyocrv2": "monkeyocrv2",
    "paddleocr": "paddleocr",
    "qianfan_vl": "qianfan_vl",
    "teleocr": "TeleOCR",
    "xiaomi_ocr": "Xiaomi-OCR-0",
    "wevisdoc": "WeVisDoc",
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


def patch_apply_command(repo: Path, *options: str) -> list[str]:
    """`git apply` targeting a source snapshot.

    Snapshots are plain directories inside the root repository, where `git -C <snapshot> apply`
    silently skips every file. Run from the enclosing work tree with --directory instead.
    """
    toplevel = subprocess.run(["git", "-C", str(repo), "rev-parse", "--show-toplevel"],
                              capture_output=True, text=True, check=False)
    if toplevel.returncode:
        return ["git", "-C", str(repo), "apply", *options]
    root = Path(toplevel.stdout.strip()).resolve()
    relative = repo.resolve().relative_to(root)
    directory = [] if str(relative) == "." else [f"--directory={relative}"]
    return ["git", "-C", str(root), "apply", *directory, *options]


def check_patches_applied(repo: Path, patches: list[Path]) -> None:
    """Fail unless every documented patch is present in the snapshot; never apply it implicitly."""
    # Absolute patch paths: the command may run from the enclosing work tree.
    result = subprocess.run(patch_apply_command(repo, "--reverse", "--check", *(str(p.resolve()) for p in patches)),
                            capture_output=True, text=True, check=False)
    if result.returncode:
        names = ", ".join(patch.name for patch in patches)
        raise RuntimeError(f"Required source patch {names} is absent or has drifted: {repo}\n{result.stderr}")


def require_failure_patch(method_id: str, patch_name: str = "failure_propagation.patch") -> None:
    check_patches_applied(method_repo_path(method_id),
                          [Path(__file__).parent / "methods" / method_id / "patches" / patch_name])
