from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
from typing import Any


PressureKind = Literal[
    "hunyuan_page_pool",
    "qianfan_page_pool",
    "unlimited_page_pool",
    "service_concurrency",
    "deepseek_page_pool",
    "paddle_internal_workers",
    "glm_internal_workers",
    "youtu_query_batch",
    "mineru_api_tasks",
    "popo_document_clients",
    "native_workers",
    "pp_page_batch",
    "monkey_region_concurrency",
    "marker_shared_workers",
]


PRESSURE_GRID = (1, 2, 4, 8, 16, 32, 64, 128, 256)


@dataclass(frozen=True)
class PressureSpec:
    method_id: str
    kind: PressureKind
    notes: str
    primary_pressure: str = "outer_concurrency"
    secondary_pressure: tuple[str, ...] = ()
    default_pressures: tuple[int, ...] = PRESSURE_GRID
    max_pressure: int | None = None


PRESSURE_SPECS: dict[str, PressureSpec] = {
    "hunyuanocr_10": PressureSpec(
        "hunyuanocr_10",
        "hunyuan_page_pool",
        "One client and service; pressure bounds global page render/request tasks.",
        primary_pressure="inflight_pages",
        secondary_pressure=("vllm_capacity",),
    ),
    "hunyuanocr_15": PressureSpec(
        "hunyuanocr_15",
        "hunyuan_page_pool",
        "One client and service; pressure bounds global page render/request tasks.",
        primary_pressure="inflight_pages",
        secondary_pressure=("vllm_capacity",),
    ),
    "hunyuanocr_15_dflash": PressureSpec(
        "hunyuanocr_15_dflash",
        "hunyuan_page_pool",
        "One client and DFlash service; pressure bounds global page render/request tasks.",
        primary_pressure="inflight_pages",
        secondary_pressure=("vllm_capacity", "num_speculative_tokens"),
    ),
    "qianfan_ocr": PressureSpec(
        "qianfan_ocr",
        "service_concurrency",
        "Joint: concurrent document clients. Explicit page_pool: one client, pressure is global inflight pages.",
        primary_pressure="benchmark_client_workers",
        secondary_pressure=("vllm_capacity",),
    ),
    "qianfan_ocr_think": PressureSpec(
        "qianfan_ocr_think",
        "service_concurrency",
        "Joint: concurrent document clients. Explicit page_pool: one client, pressure is global inflight pages.",
        primary_pressure="benchmark_client_workers",
        secondary_pressure=("vllm_capacity",),
    ),
    "deepseek_ocr2": PressureSpec(
        "deepseek_ocr2",
        "deepseek_page_pool",
        "One resident client with a shared bounded render/request page pool and ordered document export; vLLM batches pages.",
        primary_pressure="inflight_pages",
        secondary_pressure=("preprocess_workers",),
    ),
    "paddleocr_vl_15": PressureSpec(
        "paddleocr_vl_15",
        "paddle_internal_workers",
        "Official PaddleOCR-VL queue path; keep one outer worker and tune vl_rec_max_concurrency.",
        primary_pressure="vl_rec_max_concurrency",
        secondary_pressure=("vllm_capacity", "outer_workers_fixed_1", "use_queues_fixed_true"),
    ),
    "paddleocr_vl_16": PressureSpec(
        "paddleocr_vl_16",
        "paddle_internal_workers",
        "Official PaddleOCR-VL-1.6 queue path; keep one outer worker and tune vl_rec_max_concurrency.",
        primary_pressure="vl_rec_max_concurrency",
        secondary_pressure=("vllm_capacity", "outer_workers_fixed_1", "use_queues_fixed_true"),
    ),
    "glmocr": PressureSpec(
        "glmocr",
        "glm_internal_workers",
        "One GLM-OCR SDK pipeline; tune only in-flight region recognition, with HTTP pool 128 and layout batch 1 fixed.",
        primary_pressure="pipeline_max_workers",
        max_pressure=128,
        secondary_pressure=("ocr_connection_pool_size_fixed_128", "layout_batch_size_fixed_1", "outer_workers_fixed_1"),
    ),
    "youtu": PressureSpec(
        "youtu",
        "youtu_query_batch",
        "Native Youtu shared-feature decoder; batch_size controls region queries inside one parser.",
        primary_pressure="query_batch_size",
        secondary_pressure=("official_decoder_num_gen_fixed_64", "outer_workers_fixed_1"),
    ),
    "mineru_303_2509": PressureSpec(
        "mineru_303_2509",
        "mineru_api_tasks",
        "MinerU 3.0.3 official mineru-api /tasks profile.",
        primary_pressure="api_clients_and_max_concurrent_requests",
        secondary_pressure=("processing_window_size", "hybrid_batch_ratio"),
    ),
    "mineru_344_2605": PressureSpec(
        "mineru_344_2605",
        "mineru_api_tasks",
        "MinerU 3.4.x official mineru-api /tasks profile.",
        primary_pressure="api_clients_and_max_concurrent_requests",
        secondary_pressure=("processing_window_size", "hybrid_batch_ratio"),
    ),
    "mineru_344_2605_popo": PressureSpec(
        "mineru_344_2605_popo",
        "popo_document_clients",
        "One MinerU API and one Popo vLLM service per GPU; P CPU document clients share both models.",
        primary_pressure="popo_concurrency",
        default_pressures=(1, 2, 4),  # Excluded from this finalization's search set.
        secondary_pressure=("mineru_vllm_gpu_memory_utilization", "popo_gpu_memory_utilization", "popo_max_model_len"),
    ),
    "monkeyocr_pro_3b": PressureSpec(
        "monkeyocr_pro_3b",
        "monkey_region_concurrency",
        "One vllm_async runtime; pressure bounds in-flight regions within each document.",
        primary_pressure="inflight_regions",
        secondary_pressure=(),
    ),
    "monkeyocr_pro_1_2b": PressureSpec(
        "monkeyocr_pro_1_2b",
        "monkey_region_concurrency",
        "One vllm_async runtime; pressure bounds in-flight regions within each document.",
        primary_pressure="inflight_regions",
        secondary_pressure=(),
    ),
    "pp_structurev3": PressureSpec(
        "pp_structurev3",
        "pp_page_batch",
        "One pipeline; P multiplies page/layout, general OCR recognition/orientation and formula leaf batches. Table and detectors unchanged; document mode counts workers.",
        primary_pressure="native_batch_scale",
        secondary_pressure=("submodule_batch_sizes",),
    ),
    "marker": PressureSpec(
        "marker",
        "marker_shared_workers",
        "One model-owning supervisor; pressure is spawned document workers sharing its models. Native component batches stay fixed.",
        primary_pressure="shared_model_document_workers",
    ),
    "unlimited_ocr_sglang": PressureSpec(
        "unlimited_ocr_sglang",
        "service_concurrency",
        "Official SGLang service with SM86 patches; pressure is persistent whole-PDF clients at benchmark 200 DPI.",
        primary_pressure="whole_pdf_service_concurrency",
        secondary_pressure=("context_length", "mem_fraction_static"),
    ),
}


DEFAULT_PRESSURE_METHODS: tuple[str, ...] = (
    "hunyuanocr_10",
    "hunyuanocr_15_dflash",
    "paddleocr_vl_16",
    "mineru_303_2509",
    "monkeyocr_pro_3b",
    "unlimited_ocr_sglang",
    "qianfan_ocr",
    "qianfan_ocr_think",
    "hunyuanocr_15",
    "deepseek_ocr2",
    "paddleocr_vl_15",
    "glmocr",
    "mineru_344_2605",
    "monkeyocr_pro_1_2b",
    "pp_structurev3",
    "marker",
    "youtu",
)


def pressure_mapping(method: str, pressure: int, qianfan_request_mode: str = "joint",
                     unlimited_request_mode: str = "joint") -> tuple[int, dict[str, Any], dict[str, Any], int | None]:
    if type(pressure) is not int or pressure <= 0:
        raise ValueError("Pressure must be a positive integer")
    if qianfan_request_mode not in {"joint", "page_pool"} or unlimited_request_mode not in {"joint", "page_pool"}:
        raise ValueError("Request mode must be joint or page_pool")
    spec = PRESSURE_SPECS[method]
    if spec.max_pressure is not None and pressure > spec.max_pressure:
        raise ValueError(f"{method} pressure exceeds its limit of {spec.max_pressure}")
    outer, docs = 1, None
    overrides = {}
    mapping = {"pressure": pressure, "kind": spec.kind, "notes": spec.notes,
               "primary_pressure": spec.primary_pressure, "secondary_pressure": list(spec.secondary_pressure)}
    if method == "unlimited_ocr_sglang" and unlimited_request_mode == "page_pool":
        overrides["unlimited_ocr_page_concurrency"] = pressure
        mapping.update(kind="unlimited_page_pool", primary_pressure="inflight_pages")
    elif method.startswith("qianfan") and qianfan_request_mode == "page_pool":
        overrides["qianfan_page_concurrency"] = pressure
        mapping.update(kind="qianfan_page_pool", primary_pressure="inflight_pages")
    elif spec.kind == "hunyuan_page_pool":
        overrides["hunyuan_page_concurrency"] = pressure
    elif spec.kind == "deepseek_page_pool":
        overrides["deepseek_page_concurrency"] = pressure
    elif spec.kind == "paddle_internal_workers":
        overrides.update(paddle_vl_rec_max_concurrency=pressure, paddle_use_queues=True)
    elif spec.kind == "glm_internal_workers":
        overrides.update(glm_max_workers=pressure, glm_ocr_connection_pool_size=128)
    elif spec.kind == "youtu_query_batch":
        overrides["youtu_ocr_batch_size"] = pressure
    elif spec.kind == "monkey_region_concurrency":
        overrides["monkey_region_concurrency"] = pressure
        mapping["primary_pressure"] = "inflight_regions"
    elif spec.kind == "pp_page_batch":
        overrides["pp_batch_scale"] = pressure
    elif spec.kind == "marker_shared_workers":
        overrides["marker_workers"] = pressure
        docs = 1
    elif spec.kind == "mineru_api_tasks":
        overrides["mineru_api_max_concurrent_requests"] = pressure
        outer, docs = pressure, 1
    elif spec.kind == "popo_document_clients":
        overrides.update(popo_concurrency=pressure, mineru_api_max_concurrent_requests=pressure)
        docs = 1
    elif spec.kind in {"service_concurrency", "native_workers"}:
        outer, docs = pressure, 1
    else:
        raise ValueError(f"Unsupported pressure kind: {spec.kind}")
    mapping.update(overrides, outer_concurrency=outer, docs_per_job=docs)
    return outer, overrides, mapping, docs
