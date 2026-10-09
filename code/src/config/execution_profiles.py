from __future__ import annotations

from dataclasses import dataclass
import argparse
import json
from pathlib import Path
from typing import Literal

from config.throughput_pressure import PressureKind


ExecutionClass = Literal[
    "exclusive_model_service",
    "colocated_service_pipeline",
    "integrated_parser_runtime",
    "replicated_native_workers",
]


@dataclass(frozen=True)
class ExecutionProfile:
    profile_id: str
    method_id: str
    execution_class: ExecutionClass
    pressure_kind: PressureKind
    gpu_components: tuple[str, ...]
    replicated_components: tuple[str, ...] = ()
    profile_role: Literal["legacy", "recommended"] = "recommended"
    parent_profile: str | None = None


EXECUTION_PROFILES: tuple[ExecutionProfile, ...] = (
    ExecutionProfile(
        "jina_ocr_v1_vllm_fastmtp_page_pool", "jina_ocr_v1", "integrated_parser_runtime",
        "jina_page_pool", ("Jina-OCR-v1 resident AsyncLLM with FastMTP K=3",),
    ),
    ExecutionProfile(
        "xiaomi_ocr_0_vllm_page_pool", "xiaomi_ocr_0", "exclusive_model_service",
        "xiaomi_page_pool", ("Xiaomi-OCR-0 vLLM service",),
    ),
    ExecutionProfile(
        "teleocr_vllm_async", "teleocr_vllm_async", "integrated_parser_runtime",
        "teleocr_request_concurrency", ("TeleOCR resident AsyncLLM layout/recognition model",),
    ),
    ExecutionProfile(
        "lightonocr2_vllm_page_pool", "lightonocr2_1b", "exclusive_model_service",
        "lighton_page_pool", ("LightOnOCR-2-1B vLLM service",),
    ),
    ExecutionProfile(
        "wevisdoc_2b_vllm_page_pool", "wevisdoc_2b", "exclusive_model_service",
        "wevisdoc_page_pool", ("WeVisDoc-2B vLLM service",),
    ),
    *(
        ExecutionProfile(f"{method}_vllm_async_multifile", method, "integrated_parser_runtime",
                         "monkey_region_concurrency", ("MonkeyOCR model stack",))
        for method in ("monkeyocr_pro_1_2b", "monkeyocr_pro_3b")
    ),
    *(
        ExecutionProfile(profile + "_page_pool", method, "exclusive_model_service",
                         "hunyuan_page_pool", components, parent_profile=profile)
        for method, profile, components in (
            ("hunyuanocr_10", "hunyuanocr_10_vllm_ar", ("HunyuanOCR-1.0",)),
            ("hunyuanocr_15", "hunyuanocr_15_vllm_ar", ("HunyuanOCR-1.5 AR",)),
            ("hunyuanocr_15_dflash", "hunyuanocr_15_vllm_dflash",
             ("HunyuanOCR-1.5 AR target", "HunyuanOCR-1.5 DFlash draft")),
        )
    ),
    *(
        ExecutionProfile(
            f"{method}_vllm_page_pool", method, "exclusive_model_service",
            "qianfan_page_pool",
            ("Qianfan-OCR",),
        )
        for method in ("qianfan_ocr", "qianfan_ocr_think")
    ),
    ExecutionProfile(
        "mineru_344_2605_popo_vllm_services", "mineru_344_2605_popo",
        "colocated_service_pipeline", "service_concurrency",
        ("MinerU /tasks async hybrid service", "Popo vLLM service"),
        replicated_components=("CPU document API clients (no model replicas)",),
    ),
    ExecutionProfile(
        "hunyuanocr_10_vllm_ar",
        "hunyuanocr_10",
        "exclusive_model_service",
        "service_concurrency",
        ("HunyuanOCR-1.0",),
        profile_role="legacy",
    ),
    ExecutionProfile(
        "hunyuanocr_15_vllm_ar",
        "hunyuanocr_15",
        "exclusive_model_service",
        "service_concurrency",
        ("HunyuanOCR-1.5 AR",),
        profile_role="legacy",
    ),
    ExecutionProfile(
        "hunyuanocr_15_vllm_dflash",
        "hunyuanocr_15_dflash",
        "exclusive_model_service",
        "service_concurrency",
        ("HunyuanOCR-1.5 AR target", "HunyuanOCR-1.5 DFlash draft"),
        profile_role="legacy",
    ),
    ExecutionProfile(
        "deepseek_ocr2_page_pool",
        "deepseek_ocr2",
        "exclusive_model_service",
        "deepseek_page_pool",
        ("DeepSeek-OCR-2",),
        profile_role="recommended",
    ),
    ExecutionProfile(
        "paddleocr_vl_15_multi_pdf_queue",
        "paddleocr_vl_15",
        "colocated_service_pipeline",
        "paddle_internal_workers",
        ("PaddleOCR-VL-1.5", "PP-DocLayoutV3"),
        ("PP-DocLayoutV3",),
        profile_role="recommended",
    ),
    ExecutionProfile(
        "paddleocr_vl_16_multi_pdf_queue",
        "paddleocr_vl_16",
        "colocated_service_pipeline",
        "paddle_internal_workers",
        ("PaddleOCR-VL-1.6", "PP-DocLayoutV3"),
        ("PP-DocLayoutV3",),
        profile_role="recommended",
    ),
    ExecutionProfile(
        "glmocr_multi_pdf_regions",
        "glmocr",
        "colocated_service_pipeline",
        "glm_internal_workers",
        ("GLM-OCR", "PP-DocLayoutV3"),
        ("PP-DocLayoutV3",),
        profile_role="recommended",
    ),
    ExecutionProfile(
        "mineru_303_2509_api",
        "mineru_303_2509",
        "integrated_parser_runtime",
        "mineru_api_tasks",
        ("MinerU 3.0.3 parser runtime",),
    ),
    ExecutionProfile(
        "mineru_344_2605_api",
        "mineru_344_2605",
        "integrated_parser_runtime",
        "mineru_api_tasks",
        ("MinerU 3.4.x parser runtime",),
    ),
    ExecutionProfile(
        "mineru_4011_standard_v1_api", "mineru_4011_standard",
        "integrated_parser_runtime", "mineru_v1_jobs",
        ("MinerU 4.0.11 Torch models and shared AsyncLLM",),
        ("CPU-only document SDK clients",),
    ),
    ExecutionProfile(
        "youtu_native",
        "youtu",
        "integrated_parser_runtime",
        "youtu_query_batch",
        ("One resident Youtu HF parser; text-region query grouping",),
        profile_role="recommended",
    ),
    ExecutionProfile(
        "pp_structurev3_page_prefetch",
        "pp_structurev3",
        "integrated_parser_runtime",
        "pp_page_batch",
        ("One PP-StructureV3 model stack; bounded CPU prefetch feeding native page batches",),
    ),
    ExecutionProfile(
        "marker_shared_workers",
        "marker",
        "integrated_parser_runtime",
        "marker_shared_workers",
        ("Marker shared model stack",),
        ("Per-process inference caches and intermediate tensors",),
    ),
    *(ExecutionProfile(profile, "unlimited_ocr_sglang", "exclusive_model_service", kind,
                       ("Unlimited-OCR SGLang service",))
      for profile, kind in (("unlimited_ocr_sglang_service", "service_concurrency"),
                            ("unlimited_ocr_sglang_page_pool", "unlimited_page_pool"))),
)


EXECUTION_PROFILE_BY_ID = {
    profile.profile_id: profile for profile in EXECUTION_PROFILES
}

DEFAULT_PROFILE_BY_METHOD: dict[str, str] = {
    "jina_ocr_v1": "jina_ocr_v1_vllm_fastmtp_page_pool",
    "xiaomi_ocr_0": "xiaomi_ocr_0_vllm_page_pool",
    "teleocr_vllm_async": "teleocr_vllm_async",
    "lightonocr2_1b": "lightonocr2_vllm_page_pool",
    "wevisdoc_2b": "wevisdoc_2b_vllm_page_pool",
    "mineru_344_2605_popo": "mineru_344_2605_popo_vllm_services",
    "deepseek_ocr2": "deepseek_ocr2_page_pool",
    "glmocr": "glmocr_multi_pdf_regions",
    "hunyuanocr_10": "hunyuanocr_10_vllm_ar_page_pool",
    "hunyuanocr_15": "hunyuanocr_15_vllm_ar_page_pool",
    "hunyuanocr_15_dflash": "hunyuanocr_15_vllm_dflash_page_pool",
    "marker": "marker_shared_workers",
    "unlimited_ocr_sglang": "unlimited_ocr_sglang_service",
    "mineru_303_2509": "mineru_303_2509_api",
    "mineru_344_2605": "mineru_344_2605_api",
    "mineru_4011_standard": "mineru_4011_standard_v1_api",
    "monkeyocr_pro_3b": "monkeyocr_pro_3b_vllm_async_multifile",
    "monkeyocr_pro_1_2b": "monkeyocr_pro_1_2b_vllm_async_multifile",
    "paddleocr_vl_15": "paddleocr_vl_15_multi_pdf_queue",
    "paddleocr_vl_16": "paddleocr_vl_16_multi_pdf_queue",
    "pp_structurev3": "pp_structurev3_page_prefetch",
    "qianfan_ocr": "qianfan_ocr_vllm_page_pool",
    "qianfan_ocr_think": "qianfan_ocr_think_vllm_page_pool",
    "youtu": "youtu_native",
}


def profiles_for_method(method_id: str) -> tuple[ExecutionProfile, ...]:
    return tuple(
        profile for profile in EXECUTION_PROFILES if profile.method_id == method_id
    )


def resolve_execution_profile(
    method_id: str,
    *,
    execution_mode: str | None = None,
    unlimited_request_mode: str = "joint",
) -> ExecutionProfile:
    if method_id == "unlimited_ocr_sglang":
        if unlimited_request_mode not in ("joint", "page_pool"):
            raise ValueError("Unlimited request mode must be joint or page_pool")
        profile_id = ("unlimited_ocr_sglang_service" if unlimited_request_mode == "joint"
                      else "unlimited_ocr_sglang_page_pool")
    elif method_id == "pp_structurev3":
        if execution_mode not in (None, "page_batch"):
            raise ValueError("PP document route is retired; use single-pipeline page prefetch")
        profile_id = "pp_structurev3_page_prefetch"
    elif method_id == "deepseek_ocr2":
        profile_id = "deepseek_ocr2_page_pool"
    elif method_id in {"qianfan_ocr", "qianfan_ocr_think"}:
        profile_id = f"{method_id}_vllm_page_pool"
    elif method_id == "mineru_344_2605_popo":
        if execution_mode not in (None, "vllm_services"):
            raise ValueError("Popo cold/Transformers execution is archived; use shared vLLM services")
        profile_id = "mineru_344_2605_popo_vllm_services"
    elif method_id.startswith("mineru") and execution_mode == "direct":
        profile_id = f"{method_id}_direct"
    else:
        try:
            profile_id = DEFAULT_PROFILE_BY_METHOD[method_id]
        except KeyError as exc:
            raise KeyError(f"No default execution profile for method: {method_id}") from exc
    return EXECUTION_PROFILE_BY_ID[profile_id]


# Non-P options only. Both throughput entrypoints use this same declaration;
# per-method profiles cannot accumulate unrelated methods' options.
RUNTIME_ARGUMENTS = {
    "jina": [
        ("jina_env", str, "gdb_hunyuan_dflash"),
        ("jina_cuda_home", str, str(Path.home() / "software/cuda-envs/cuda-12.8")),
        ("jina_flashinfer_workspace", str, None),
        ("jina_model_path", str, None), ("jina_dpi", int, 200),
        ("jina_max_tokens", int, 4096), ("jina_max_model_len", int, 32768),
    ],
    "xiaomi": [
        ("xiaomi_env", str, "gdb_glmocr"),
        ("xiaomi_model_path", str, None),
        ("xiaomi_dpi", int, 200),
        ("xiaomi_max_tokens", int, 16384),
        ("xiaomi_max_model_len", int, 32768),
    ],
    "teleocr": [("teleocr_env", str, "gdb_teleocr"),
        ("teleocr_model_path", str, None), ("teleocr_backend", str, "vllm-async-engine"),
        ("teleocr_layout_mode", str, "Detection"), ("teleocr_max_model_len", int, 16384),
        ("teleocr_gpu_memory_utilization", float, .95), ("teleocr_pdf_tools", str, "pypdfium2"),
        ("teleocr_pdf_tools_workers", int, 4)],
    "lightonocr2": [
        ("lightonocr2_env", str, "gdb_glmocr"),
        ("lightonocr2_model_path", str, None),
        ("lightonocr2_dpi", int, 200),
        ("lightonocr2_max_image_size", int, 1540),
        ("lightonocr2_max_tokens", int, 4096),
        ("lightonocr2_temperature", float, 0.2),
        ("lightonocr2_top_p", float, 0.9),
    ],
    "wevisdoc": [
        ("wevisdoc_env", str, "gdb_glmocr"),
        ("wevisdoc_model_path", str, None),
        ("wevisdoc_dpi", int, 200),
    ],
    "hunyuan_common": [("hunyuan_runner_env", str, "gdb_hunyuan")],
    "hunyuan_ar": [("hunyuan_service_env", str, "gdb_hunyuan"),
        ("hunyuan_vllm_gpu_memory_utilization", float, None),
        ("hunyuan_vllm_max_model_len", int, None),
        ("hunyuan_vllm_max_num_batched_tokens", int, 65536),
        ("hunyuan_vllm_enforce_eager", bool, False)],
    "qianfan": [("qianfan_service_env", str, "gdb_qianfan"),
        ("qianfan_execution_mode", str, "auto")],
    "deepseek": [("deepseek_service_env", str, "gdb_deepseek_ocr2"),
        ("deepseek_max_model_len", int, 8192), ("deepseek_max_tokens", int, 8192),
        ("deepseek_gpu_memory_utilization", float, .90), ("deepseek_preprocess_workers", int, 64),
        ("deepseek_engine_iteration_timeout_seconds", int, 600)],
    "paddle": [("paddle_service_env", str, "gdb_paddle_vllm"),
        ("paddle_vllm_gpu_memory_utilization", float, .55), ("paddle_vllm_max_model_len", int, 16384),
        ("paddle_vllm_enforce_eager", bool, False)],
    "glm": [("glm_vllm_gpu_memory_utilization", float, .7), ("glm_layout_model_dir", str, None)],
    "monkey": [("monkey_models_dir", str, None)],
    "mineru": [("mineru_vllm_gpu_memory_utilization", float, .55),
        ("mineru_parse_method", str, "auto"), ("mineru_lang", str, "auto"),
        ("mineru_disable_formula", bool, False), ("mineru_disable_table", bool, False),
        ("mineru_api_processing_window_size", int, None), ("mineru_api_task_retention_seconds", int, 3600),
        ("mineru_task_timeout_seconds", float, 3600.), ("mineru_task_poll_seconds", float, 1.),
        ("mineru_http_timeout_seconds", float, None), ("mineru_hybrid_batch_ratio", int, None),
        ("mineru_pdf_render_threads", int, None), ("mineru_intra_op_num_threads", int, None),
        ("mineru_inter_op_num_threads", int, None)],
    "youtu": [("youtu_env", str, "gdb_youtu"), ("youtu_model_path", str, None)],
    "pp": [("pp_device", str, "gpu")],
}
DFLASH_KEYS = tuple("hunyuan_dflash_" + key for key in (
    "profile", "service_env", "cuda_home", "flashinfer_workspace", "gpu_memory_utilization",
    "max_model_len", "max_num_batched_tokens", "num_speculative_tokens"))


def add_runtime_arguments(parser):
    from config.method_versions import get_method_version
    from methods.hunyuanocr.dflash import add_dflash_args
    from methods.unlimited_ocr.commands import add_runtime_args as unlimited_args
    from methods.pp_structurev3.commands import add_runtime_args as pp_args
    choices = {"qianfan_execution_mode": ("auto", "eager"),
               "teleocr_backend": ("vllm-async-engine",), "teleocr_layout_mode": ("Detection",),
               "mineru_parse_method": ("auto", "txt", "ocr"), "pp_device": ("gpu", "cpu")}
    for specs in RUNTIME_ARGUMENTS.values():
        for key, typ, default in specs:
            if key == "glm_layout_model_dir":
                default = str(get_method_version("glmocr").additional_model_paths[0])
            options = {"action": "store_true"} if typ is bool else {"type": typ}
            if key in choices:
                options["choices"] = choices[key]
            parser.add_argument("--" + key.replace("_", "-"), default=default, **options)
    add_dflash_args(parser)
    unlimited_args(parser, include_pressure=False)
    pp_args(parser, include_batch_scale=False)


def runtime_keys(method):
    if method.startswith("hunyuanocr"):
        groups = ("hunyuan_common",) if method.endswith("dflash") else ("hunyuan_common", "hunyuan_ar")
    elif method.startswith("qianfan"):
        groups = ("qianfan",)
    elif method.startswith("paddleocr"):
        groups = ("paddle",)
    elif method.startswith("monkeyocr"):
        groups = ("monkey",)
    elif method in ("mineru_303_2509", "mineru_344_2605"):
        groups = ("mineru",)
    elif method == "mineru_4011_standard":
        # P is recorded in pressure_mapping; all other runtime settings are
        # fixed official defaults, not the 3.x flags or memory budget.
        groups = ()
    else:
        groups = {"deepseek_ocr2": ("deepseek",), "glmocr": ("glm",), "youtu": ("youtu",),
                  "pp_structurev3": ("pp",), "marker": (), "unlimited_ocr_sglang": (),
                  "teleocr_vllm_async": ("teleocr",),
                  "xiaomi_ocr_0": ("xiaomi",), "jina_ocr_v1": ("jina",),
                  "lightonocr2_1b": ("lightonocr2",),
                  "wevisdoc_2b": ("wevisdoc",)}[method]
    keys = tuple(key for group in groups for key, _, _ in RUNTIME_ARGUMENTS[group])
    if method == "hunyuanocr_15_dflash":
        keys += DFLASH_KEYS
    if method == "unlimited_ocr_sglang":
        from methods.unlimited_ocr.commands import RUNTIME_KEYS
        keys += RUNTIME_KEYS
    if method == "pp_structurev3":
        keys += ("pp_render_dpi",)
    return keys


def runtime_options(args, method):
    return {key: str(value) if isinstance(value := getattr(args, key), Path) else value
            for key in runtime_keys(method)}


def runtime_cli_args(args, method):
    flags = []
    for key, value in runtime_options(args, method).items():
        flag = "--" + key.replace("_", "-")
        if value is True:
            flags.append(flag)
        elif value is not None and value is not False:
            flags.extend((flag, str(value)))
    return flags


def resolve_runtime_defaults(args, method):
    if method == "jina_ocr_v1":
        from config.method_versions import get_method_version
        args.jina_model_path = str(Path(args.jina_model_path).expanduser().resolve()
            if args.jina_model_path else get_method_version(method).model_path)
        if args.jina_dpi < 1 or args.jina_max_tokens < 1 or args.jina_max_model_len <= args.jina_max_tokens:
            raise ValueError("Jina requires positive DPI/tokens and context greater than generation limit")
    elif method == "xiaomi_ocr_0":
        from config.method_versions import get_method_version
        args.xiaomi_model_path = str(Path(args.xiaomi_model_path).expanduser().resolve()
            if args.xiaomi_model_path else get_method_version(method).model_path)
        if args.xiaomi_dpi < 1 or args.xiaomi_max_tokens < 1:
            raise ValueError("Xiaomi DPI and generation limit must be positive")
        if args.xiaomi_max_model_len <= args.xiaomi_max_tokens:
            raise ValueError("Xiaomi context must exceed the generation limit to leave room for input")
    elif method == "teleocr_vllm_async":
        from config.method_versions import get_method_version
        args.teleocr_model_path = str(Path(args.teleocr_model_path).expanduser().resolve()
            if args.teleocr_model_path else get_method_version(method).model_path)
    elif method == "lightonocr2_1b":
        from config.method_versions import get_method_version
        args.lightonocr2_model_path = str(Path(args.lightonocr2_model_path).expanduser().resolve()
            if args.lightonocr2_model_path else get_method_version(method).model_path)
    elif method == "wevisdoc_2b":
        from config.method_versions import get_method_version
        args.wevisdoc_model_path = str(Path(args.wevisdoc_model_path).expanduser().resolve()
            if args.wevisdoc_model_path else get_method_version(method).model_path)
        if args.wevisdoc_dpi < 1:
            raise ValueError("WeVisDoc DPI must be positive")
    elif method == "hunyuanocr_15_dflash":
        from methods.hunyuanocr.dflash import resolve_dflash_args
        resolve_dflash_args(args)
    elif method in ("hunyuanocr_10", "hunyuanocr_15"):
        from methods.hunyuanocr.commands import resolve_ar_gpu_memory_utilization
        args.hunyuan_vllm_gpu_memory_utilization = resolve_ar_gpu_memory_utilization(method, args.hunyuan_vllm_gpu_memory_utilization)
        if args.hunyuan_vllm_max_model_len is None:
            args.hunyuan_vllm_max_model_len = 65536 if method == "hunyuanocr_15" else 32768


def profile_for_args(method, args):
    return resolve_execution_profile(method, unlimited_request_mode=args.unlimited_ocr_request_mode).profile_id


def save_selected_profile(path, method, summary):
    if summary["status"] != "ok":
        raise ValueError("Only a successfully measured trial can be saved")
    profile = summary["execution_profile"]
    options = summary["runtime_options"]
    if set(options) != set(runtime_keys(method)):
        raise ValueError("Profile must contain exactly this method's runtime options")
    entry = {"pressure": summary["pressure"], "runtime_options": options}
    _check_profile_entry(method, profile, entry)
    data = json.loads(path.read_text()) if path.exists() else {"schema_version": 2, "methods": {}}
    if data["schema_version"] != 2:
        raise ValueError("Unsupported selected-profile schema; update explicitly before saving")
    data["methods"].setdefault(method, {})[profile] = entry
    path.parent.mkdir(parents=True, exist_ok=True)
    from common import write_json
    write_json(path, data)


def _check_profile_entry(method, profile, entry):
    if EXECUTION_PROFILE_BY_ID[profile].method_id != method or set(entry) != {"pressure", "runtime_options"}:
        raise ValueError("Selected-profile identity or fields mismatch")
    options = entry["runtime_options"]
    if set(options) != set(runtime_keys(method)):
        raise ValueError("Incomplete or unrelated runtime options in selected profile")
    # Restore through the same argument types/choices as the CLI. In particular,
    # the string "false" must never become a truthy disable-formula setting.
    parser = argparse.ArgumentParser(exit_on_error=False)
    add_runtime_arguments(parser)
    try:
        parsed = parser.parse_args(runtime_cli_args(argparse.Namespace(**options), method))
    except (argparse.ArgumentError, SystemExit) as exc:
        raise ValueError("Invalid runtime option value in selected profile") from exc
    resolve_runtime_defaults(parsed, method)
    effective = runtime_options(parsed, method)
    for key, value in options.items():
        if value != effective[key] or (isinstance(effective[key], bool) and type(value) is not bool):
            raise ValueError(f"Selected runtime option has an invalid type or unresolved value: {key}")
    from config.throughput_pressure import pressure_mapping
    pressure_mapping(method, entry["pressure"], unlimited_request_mode=options.get("unlimited_ocr_request_mode", "joint"))
    actual = resolve_execution_profile(method,
        unlimited_request_mode=options.get("unlimited_ocr_request_mode", "joint")).profile_id
    if actual != profile:
        raise ValueError("Selected profile cannot cross request routes")


def load_selected_profile(path, method, profile):
    data = json.loads(path.read_text())
    if data["schema_version"] != 2:
        raise ValueError("Unsupported selected-profile schema; no implicit migration")
    entry = data["methods"][method][profile]
    _check_profile_entry(method, profile, entry)
    return entry
