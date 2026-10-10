#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from paths import CONDA_ENV_ROOT, MODEL_ROOT, PROJECT_ROOT  # noqa: E402
from config.method_versions import get_method_version  # noqa: E402
from methods.mineru.versions.v303_2509 import PROFILE as MINERU_303_2509  # noqa: E402
from methods.mineru.versions.v344_2605 import PROFILE as MINERU_344_2605  # noqa: E402


MINERU_PROFILES = {
    MINERU_303_2509.method_id: MINERU_303_2509,
    MINERU_344_2605.method_id: MINERU_344_2605,
}

ENV_BY_METHOD = {
    "jina_ocr_v1": ("gdb_hunyuan_dflash",),
    "xiaomi_ocr_0": ("gdb_glmocr",),
    "teleocr_vllm_async": ("gdb_teleocr",),
    "lightonocr2_1b": ("gdb_glmocr",),
    "wevisdoc_2b": ("gdb_glmocr",),
    "mineru_344_2605_popo": ("gdb_mineru", "gdb_mineru_popo"),
    "hunyuanocr_10": ("gdb_hunyuan",),
    "hunyuanocr_15": ("gdb_hunyuan",),
    "hunyuanocr_15_dflash": ("gdb_hunyuan", "gdb_hunyuan_dflash"),
    "qianfan_ocr": ("gdb_qianfan",),
    "qianfan_ocr_think": ("gdb_qianfan",),
    "deepseek_ocr2": ("gdb_deepseek_ocr2",),
    "paddleocr_vl_15": ("gdb_paddle", "gdb_paddle_vllm"),
    "paddleocr_vl_16": ("gdb_paddle", "gdb_paddle_vllm"),
    "glmocr": ("gdb_glmocr",),
    "monkeyocr_pro_3b": ("gdb_monkey",),
    "monkeyocr_pro_1_2b": ("gdb_monkey",),
    "monkeyocrv2_b_parsing": ("gdb_monkeyocrv2",),
    "mineru_303_2509": ("gdb_mineru_303",),
    "mineru_344_2605": ("gdb_mineru",),
    "mineru_4011_standard": ("gdb_mineru_4",),
    "youtu": ("gdb_youtu",),
    "marker": ("gdb_marker",),
    "pp_structurev3": ("gdb_paddle",),
    "unlimited_ocr_sglang": ("gdb_unlimited_ocr_sglang",),
}

DEFAULT_METHODS = tuple(ENV_BY_METHOD)


@dataclass(frozen=True)
class CheckResult:
    method: str
    item: str
    path: str
    status: str
    detail: str


def populated_directory(path: Path) -> tuple[bool, str]:
    if not path.is_dir():
        return False, "directory missing"
    try:
        next(path.iterdir())
    except StopIteration:
        return False, "directory empty"
    return True, "directory populated"


def regular_file(path: Path) -> tuple[bool, str]:
    if not path.is_file():
        return False, "file missing"
    if path.stat().st_size == 0:
        return False, "file empty"
    return True, f"{path.stat().st_size} bytes"


def add_path_check(
    results: list[CheckResult],
    method: str,
    item: str,
    path: Path,
    checker: Callable[[Path], tuple[bool, str]] = populated_directory,
) -> None:
    ok, detail = checker(path)
    results.append(
        CheckResult(
            method=method,
            item=item,
            path=str(path),
            status="ok" if ok else "missing",
            detail=detail,
        )
    )


def check_environment(results: list[CheckResult], method: str, env_name: str) -> None:
    add_path_check(
        results,
        method,
        f"conda environment {env_name}",
        CONDA_ENV_ROOT / env_name / "bin" / "python",
        regular_file,
    )


def check_marker_cache(results: list[CheckResult], method: str) -> None:
    cache_root = get_method_version("marker").model_path
    add_path_check(results, method, "Marker/Surya cache", cache_root)
    manifests = sorted(cache_root.rglob("manifest.json")) if cache_root.is_dir() else []
    if not manifests:
        results.append(
            CheckResult(method, "Marker manifests", str(cache_root), "missing", "no manifest.json")
        )
        return
    missing_files: list[str] = []
    for manifest in manifests:
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            for name in payload.get("files", []):
                if not (manifest.parent / name).is_file():
                    missing_files.append(str(manifest.parent / name))
        except (OSError, ValueError, TypeError) as exc:
            missing_files.append(f"{manifest}: {exc}")
    results.append(
        CheckResult(
            method,
            "Marker manifests",
            str(cache_root),
            "ok" if not missing_files else "missing",
            f"{len(manifests)} manifests complete"
            if not missing_files
            else f"{len(missing_files)} files invalid or missing",
        )
    )


def check_mineru_config(results: list[CheckResult], method: str, config_path: Path) -> None:
    add_path_check(results, method, "MinerU config", config_path, regular_file)
    if not config_path.is_file():
        return
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        results.append(CheckResult(method, "MinerU config JSON", str(config_path), "missing", str(exc)))
        return
    model_dirs = payload.get("models-dir", {})
    version = get_method_version(method)
    expected = {"pipeline": version.additional_model_paths[0], "vlm": version.model_path}
    for model_type, expected_path in expected.items():
        configured = Path(str(model_dirs.get(model_type, ""))).expanduser()
        status = "ok" if configured.resolve() == expected_path.resolve() else "missing"
        results.append(
            CheckResult(
                method,
                f"MinerU {model_type} config path",
                str(configured),
                status,
                f"expected {expected_path}",
            )
        )


def check_method_models(
    results: list[CheckResult],
    method: str,
    mineru_config: Path,
) -> None:
    single_directories = {
        method_id: (get_method_version(method_id).display_name, get_method_version(method_id).model_path)
        for method_id in (
            "jina_ocr_v1",
            "xiaomi_ocr_0",
            "teleocr_vllm_async",
            "lightonocr2_1b",
            "monkeyocrv2_b_parsing",
            "wevisdoc_2b",
            "hunyuanocr_10",
            "qianfan_ocr",
            "qianfan_ocr_think",
            "deepseek_ocr2",
            "youtu",
            "unlimited_ocr_sglang",
        )
    }
    if method in single_directories:
        item, path = single_directories[method]
        add_path_check(results, method, item, path)
    elif method in {"hunyuanocr_15", "hunyuanocr_15_dflash"}:
        model_dir = get_method_version(method).model_path
        add_path_check(results, method, "HunyuanOCR-1.5", model_dir)
        add_path_check(results, method, "HunyuanOCR-1.5 weights", model_dir / "model.safetensors", regular_file)
        if method == "hunyuanocr_15_dflash":
            add_path_check(results, method, "HunyuanOCR-1.5 DFlash", model_dir / "dflash")
        config_path = model_dir / "config.json"
        add_path_check(results, method, "HunyuanOCR-1.5 config", config_path, regular_file)
        if config_path.is_file():
            try:
                config = json.loads(config_path.read_text(encoding="utf-8"))
                context_length = config.get("max_position_embeddings")
                results.append(
                    CheckResult(
                        method,
                        "HunyuanOCR-1.5 context length",
                        str(config_path),
                        "ok" if context_length == 131072 else "missing",
                        f"max_position_embeddings={context_length}",
                    )
                )
            except (OSError, ValueError) as exc:
                results.append(CheckResult(method, "HunyuanOCR-1.5 config JSON", str(config_path), "missing", str(exc)))
    elif method.startswith("paddleocr_vl_"):
        version = get_method_version(method)
        add_path_check(results, method, version.display_name, version.model_path)
        add_path_check(results, method, "PaddleX cache", version.additional_model_paths[0])
    elif method == "pp_structurev3":
        add_path_check(results, method, "PaddleX cache", get_method_version(method).model_path)
    elif method == "glmocr":
        version = get_method_version("glmocr")
        add_path_check(results, method, version.display_name, version.model_path)
        add_path_check(
            results,
            method,
            "PP-DocLayoutV3 safetensors",
            version.additional_model_paths[0],
        )
    elif method.startswith("monkeyocr_"):
        monkey_models_dir = get_method_version(method).model_path
        for component in ("Structure", "Relation", "Recognition"):
            add_path_check(results, method, f"MonkeyOCR {component}", monkey_models_dir / component)
    elif method == "mineru_344_2605_popo":
        version = get_method_version(method)
        for path in (version.model_path, *version.additional_model_paths):
            add_path_check(results, method, path.name, path)
    elif method == "mineru_4011_standard":
        version = get_method_version(method)
        for path in (version.model_path, *version.additional_model_paths):
            add_path_check(results, method, path.name, path)
        marker = version.model_path / ".mineru_complete"
        ok = marker.is_file()
        results.append(CheckResult(method, "official VLM completion marker", str(marker),
                                   "ok" if ok else "missing", "present" if ok else "run official models download/verify"))
    elif method.startswith("mineru"):
        version = get_method_version(method)
        add_path_check(
            results,
            method,
            "MinerU pipeline models",
            version.additional_model_paths[0],
        )
        add_path_check(
            results,
            method,
            version.display_name,
            version.model_path,
        )
        check_mineru_config(results, method, mineru_config)
    elif method == "marker":
        check_marker_cache(results, method)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check smoke-test environments and local model assets.")
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    parser.add_argument("--skip-envs", action="store_true")
    parser.add_argument("--popo-vllm-python", type=Path,
                        help="Explicit Popo service interpreter; no verified default environment yet.")
    parser.add_argument(
        "--mineru-config",
        type=Path,
        help="Override the generated runtime config when checking one MinerU variant.",
    )
    parser.add_argument("--json-report", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    unknown = [method for method in methods if method not in ENV_BY_METHOD]
    if unknown:
        raise SystemExit(f"Unknown methods: {', '.join(unknown)}")

    results: list[CheckResult] = []
    for method in methods:
        if not args.skip_envs:
            for env_name in ENV_BY_METHOD[method]:
                check_environment(results, method, env_name)
            if method == "mineru_344_2605_popo":
                if args.popo_vllm_python is None:
                    results.append(CheckResult(method, "Popo vLLM service interpreter", "", "missing",
                        "set --popo-vllm-python; old Transformers smoke does not verify this backend"))
                else:
                    add_path_check(results, method, "Popo vLLM interpreter (GPU compatibility unverified)",
                                   args.popo_vllm_python, regular_file)
        if args.mineru_config is not None:
            mineru_config = args.mineru_config.resolve()
        elif method in MINERU_PROFILES:
            mineru_config = MINERU_PROFILES[method].runtime_config_path
        else:
            mineru_config = Path()
        check_method_models(results, method, mineru_config)

    for result in results:
        print(f"[{result.status.upper():7}] {result.method}: {result.item} -> {result.path} ({result.detail})")

    report = {
        "project_root": str(PROJECT_ROOT),
        "model_root": str(MODEL_ROOT),
        "methods": methods,
        "ready": all(result.status == "ok" for result in results),
        "checks": [asdict(result) for result in results],
    }
    if args.json_report is not None:
        args.json_report.parent.mkdir(parents=True, exist_ok=True)
        args.json_report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"[REPORT ] {args.json_report}")
    if not report["ready"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
