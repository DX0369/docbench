"""Shared service/native task registrations; command builders live with each method."""
from __future__ import annotations

import argparse
from dataclasses import dataclass

from config.method_versions import get_method_version
from execution.commands import HealthcheckBuilder, RunnerBuilder, ServiceBuilder, WorkerBuilder
from paths import MODEL_ROOT
from methods.deepseek_ocr2.commands import (
    build_runner as build_deepseek_ocr2_runner,
    build_service as build_deepseek_ocr2_service,
    build_worker as build_deepseek_ocr2_worker,
)
from methods.glmocr.commands import (
    build_service as build_glm_service,
    build_worker as build_glm_worker,
)
from methods.hunyuanocr.commands import (
    build_10_runner as build_hunyuan_10_runner,
    build_10_service as build_hunyuan_10_service,
    build_15_dflash_runner as build_hunyuan_15_dflash_runner,
    build_15_dflash_service as build_hunyuan_15_dflash_service,
    build_15_runner as build_hunyuan_15_runner,
    build_15_service as build_hunyuan_15_service,
    build_10_worker as build_hunyuan_10_worker,
    build_15_worker as build_hunyuan_15_worker,
    build_15_dflash_worker as build_hunyuan_15_dflash_worker,
)
from methods.mineru.commands import (
    build_303_2509_service as build_mineru_303_2509_service,
    build_303_2509_worker as build_mineru_303_2509_worker,
    build_344_2605_service as build_mineru_344_2605_service,
    build_344_2605_worker as build_mineru_344_2605_worker,
    healthcheck_url as mineru_healthcheck_url,
)
from methods.mineru.versions.v303_2509 import PROFILE as MINERU_303_2509
from methods.mineru.versions.v344_2605 import PROFILE as MINERU_344_2605
from methods.monkeyocr.commands import (
    build_pro_1_2b_runner as build_monkey_pro_1_2b_runner,
    build_pro_1_2b_worker as build_monkey_pro_1_2b_worker,
    build_pro_3b_runner as build_monkey_pro_3b_runner,
    build_pro_3b_worker as build_monkey_pro_3b_worker,
)
from methods.paddleocr_vl.commands import (
    build_15_worker as build_paddle_15_worker,
    build_16_worker as build_paddle_16_worker,
    build_15_runner as build_paddle_15_runner,
    build_15_service as build_paddle_15_service,
    build_16_runner as build_paddle_16_runner,
    build_16_service as build_paddle_16_service,
)
from methods.qianfan_ocr.commands import (
    build_service as build_qianfan_service,
    build_worker as build_qianfan_worker,
    build_think_worker as build_qianfan_think_worker,
)
from methods.unlimited_ocr.commands import (
    build_service as build_unlimited_ocr_sglang_service,
    build_worker as build_unlimited_ocr_sglang_worker,
    healthcheck_url as unlimited_ocr_sglang_healthcheck_url,
)
from methods.youtu.commands import build_worker as build_youtu_worker
from methods.teleocr.commands import build_worker as build_teleocr_worker
from methods.jina_ocr.commands import build_worker as build_jina_worker
from methods.lightonocr2.commands import (
    build_service as build_lightonocr2_service,
    build_worker as build_lightonocr2_worker,
    healthcheck_url as lightonocr2_healthcheck_url,
)
from methods.xiaomi_ocr.commands import (
    build_service as build_xiaomi_service,
    build_worker as build_xiaomi_worker,
    healthcheck_url as xiaomi_healthcheck_url,
)
from methods.marker.commands import build_worker as build_marker_worker
from methods.pp_structurev3.commands import build_worker as build_ppstructure_worker
from methods.mineru_popo.commands import build_worker as build_mineru_popo_worker

PP_STRUCTUREV3 = get_method_version("pp_structurev3")
MARKER = get_method_version("marker")
MINERU_PROFILES = {
    MINERU_303_2509.method_id: MINERU_303_2509,
    MINERU_344_2605.method_id: MINERU_344_2605,
}


@dataclass(frozen=True)
class ServiceTask:
    method_id: str
    runner_env: str
    runner_builder: RunnerBuilder | None = None
    worker_builder: WorkerBuilder | None = None
    service_builder: ServiceBuilder | None = None
    healthcheck_builder: HealthcheckBuilder | None = None
    docs_per_job: int = 8
    env_overrides: dict[str, str] | None = None


def uses_persistent_worker(config: ServiceTask, args: argparse.Namespace) -> bool:
    if config.worker_builder is None:
        return False
    return True


def uses_local_managed_service(config: ServiceTask, args: argparse.Namespace) -> bool:
    if config.service_builder is None:
        return False
    return True


def managed_service_identity(method_id: str, command: list[str], port: int) -> dict:
    """Derive identity from the final command and the existing version profile."""
    base = f"http://127.0.0.1:{port}"
    if method_id in MINERU_PROFILES:
        profile = MINERU_PROFILES[method_id]
        return {"identity_url": base + "/ready", "expected_fields": {
            "status": "ready", "method_id": profile.method_id,
            "release_version": profile.version,
        }}
    return {"expected_model": command[command.index("--served-model-name") + 1]}


SERVICE_TASKS: dict[str, ServiceTask] = {
    "jina_ocr_v1": ServiceTask(
        method_id="jina_ocr_v1", runner_env="gdb_hunyuan_dflash",
        worker_builder=build_jina_worker, docs_per_job=8,
        env_overrides={"PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1"},
    ),
    "xiaomi_ocr_0": ServiceTask(
        method_id="xiaomi_ocr_0",
        runner_env="gdb_glmocr",
        worker_builder=build_xiaomi_worker,
        service_builder=build_xiaomi_service,
        healthcheck_builder=xiaomi_healthcheck_url,
        docs_per_job=1,
        env_overrides={"PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1",
                       "TOKENIZERS_PARALLELISM": "false"},
    ),
    "teleocr_vllm_async": ServiceTask(
        method_id="teleocr_vllm_async",
        runner_env="gdb_teleocr",
        worker_builder=build_teleocr_worker,
        docs_per_job=1,
        env_overrides={"PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1",
                       "TOKENIZERS_PARALLELISM": "false"},
    ),
    "lightonocr2_1b": ServiceTask(
        method_id="lightonocr2_1b",
        runner_env="gdb_glmocr",
        worker_builder=build_lightonocr2_worker,
        service_builder=build_lightonocr2_service,
        healthcheck_builder=lightonocr2_healthcheck_url,
        docs_per_job=1,
        env_overrides={
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "TOKENIZERS_PARALLELISM": "false",
        },
    ),
    "hunyuanocr_10": ServiceTask(
        method_id="hunyuanocr_10",
        runner_env="gdb_hunyuan",
        runner_builder=build_hunyuan_10_runner,
        worker_builder=build_hunyuan_10_worker,
        service_builder=build_hunyuan_10_service,
        healthcheck_builder=lambda port: f"http://127.0.0.1:{port}/v1/models",
        docs_per_job=1,
    ),
    "hunyuanocr_15": ServiceTask(
        method_id="hunyuanocr_15",
        runner_env="gdb_hunyuan",
        runner_builder=build_hunyuan_15_runner,
        worker_builder=build_hunyuan_15_worker,
        service_builder=build_hunyuan_15_service,
        healthcheck_builder=lambda port: f"http://127.0.0.1:{port}/v1/models",
        docs_per_job=1,
    ),
    "hunyuanocr_15_dflash": ServiceTask(
        method_id="hunyuanocr_15_dflash",
        runner_env="gdb_hunyuan",
        runner_builder=build_hunyuan_15_dflash_runner,
        worker_builder=build_hunyuan_15_dflash_worker,
        service_builder=build_hunyuan_15_dflash_service,
        healthcheck_builder=lambda port: f"http://127.0.0.1:{port}/v1/models",
        docs_per_job=1,
    ),
    "qianfan_ocr": ServiceTask(
        method_id="qianfan_ocr",
        runner_env="gdb_qianfan",
        worker_builder=build_qianfan_worker,
        service_builder=build_qianfan_service,
        healthcheck_builder=lambda port: f"http://127.0.0.1:{port}/v1/models",
        docs_per_job=1,
    ),
    "qianfan_ocr_think": ServiceTask(
        method_id="qianfan_ocr_think",
        runner_env="gdb_qianfan",
        worker_builder=build_qianfan_think_worker,
        service_builder=build_qianfan_service,
        healthcheck_builder=lambda port: f"http://127.0.0.1:{port}/v1/models",
        docs_per_job=1,
    ),
    "deepseek_ocr2": ServiceTask(
        method_id="deepseek_ocr2",
        runner_env="gdb_deepseek_ocr2",
        runner_builder=build_deepseek_ocr2_runner,
        worker_builder=build_deepseek_ocr2_worker,
        service_builder=build_deepseek_ocr2_service,
        healthcheck_builder=lambda port: f"http://127.0.0.1:{port}/health",
        docs_per_job=1,
    ),
    "unlimited_ocr_sglang": ServiceTask(
        method_id="unlimited_ocr_sglang",
        runner_env="gdb_unlimited_ocr_sglang",
        worker_builder=build_unlimited_ocr_sglang_worker,
        service_builder=build_unlimited_ocr_sglang_service,
        healthcheck_builder=unlimited_ocr_sglang_healthcheck_url,
        docs_per_job=1,
    ),
    "paddleocr_vl_15": ServiceTask(
        method_id="paddleocr_vl_15",
        runner_env="gdb_paddle",
        runner_builder=build_paddle_15_runner,
        worker_builder=build_paddle_15_worker,
        service_builder=build_paddle_15_service,
        healthcheck_builder=lambda port: f"http://127.0.0.1:{port}/v1/models",
        docs_per_job=1,
        env_overrides={
            "PADDLE_PDX_CACHE_HOME": str(MODEL_ROOT / "paddlex"),
            "PADDLE_PDX_MODEL_SOURCE": "modelscope",
            "PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK": "True",
        },
    ),
    "paddleocr_vl_16": ServiceTask(
        method_id="paddleocr_vl_16",
        runner_env="gdb_paddle",
        runner_builder=build_paddle_16_runner,
        worker_builder=build_paddle_16_worker,
        service_builder=build_paddle_16_service,
        healthcheck_builder=lambda port: f"http://127.0.0.1:{port}/v1/models",
        docs_per_job=1,
        env_overrides={
            "PADDLE_PDX_CACHE_HOME": str(MODEL_ROOT / "paddlex"),
            "PADDLE_PDX_MODEL_SOURCE": "modelscope",
            "PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK": "True",
        },
    ),
    "glmocr": ServiceTask(
        method_id="glmocr",
        runner_env="gdb_glmocr",
        worker_builder=build_glm_worker,
        service_builder=build_glm_service,
        healthcheck_builder=lambda port: f"http://127.0.0.1:{port}/v1/models",
        docs_per_job=1,
    ),
    "monkeyocr_pro_3b": ServiceTask(
        method_id="monkeyocr_pro_3b",
        runner_env="gdb_monkey",
        runner_builder=build_monkey_pro_3b_runner,
        worker_builder=build_monkey_pro_3b_worker,
        docs_per_job=1,
    ),
    "monkeyocr_pro_1_2b": ServiceTask(
        method_id="monkeyocr_pro_1_2b",
        runner_env="gdb_monkey",
        runner_builder=build_monkey_pro_1_2b_runner,
        worker_builder=build_monkey_pro_1_2b_worker,
        docs_per_job=1,
    ),
    "mineru_303_2509": ServiceTask(
        method_id="mineru_303_2509",
        runner_env=MINERU_303_2509.conda_env,
        worker_builder=build_mineru_303_2509_worker,
        service_builder=build_mineru_303_2509_service,
        healthcheck_builder=mineru_healthcheck_url,
        docs_per_job=1,
        env_overrides={
            **MINERU_303_2509.environment_overrides(),
        },
    ),
    "mineru_344_2605": ServiceTask(
        method_id="mineru_344_2605",
        runner_env=MINERU_344_2605.conda_env,
        worker_builder=build_mineru_344_2605_worker,
        service_builder=build_mineru_344_2605_service,
        healthcheck_builder=mineru_healthcheck_url,
        docs_per_job=1,
        env_overrides={
            **MINERU_344_2605.environment_overrides(),
        },
    ),
}



@dataclass(frozen=True)
class NativeTask:
    method_id: str
    output_subdir: str
    worker_builder: WorkerBuilder
    docs_per_job: int = 1
    env_overrides: dict[str, str] | None = None



NATIVE_TASKS: dict[str, NativeTask] = {
    "mineru_344_2605_popo": NativeTask(
        method_id="mineru_344_2605_popo", output_subdir="mineru_344_2605_popo",
        worker_builder=build_mineru_popo_worker,
    ),
    "youtu": NativeTask(
        method_id="youtu",
        output_subdir="youtu",
        worker_builder=build_youtu_worker,
        docs_per_job=1,
    ),
    "pp_structurev3": NativeTask(
        method_id="pp_structurev3",
        output_subdir="pp_structurev3",
        worker_builder=build_ppstructure_worker,
        docs_per_job=1,  # Native/single-GPU launchers assign a whole input stream in page_batch mode.
        env_overrides={
            "PADDLE_PDX_CACHE_HOME": str(PP_STRUCTUREV3.model_path.parent),
            "PADDLE_PDX_MODEL_SOURCE": "modelscope",
            "PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK": "True",
        },
    ),
    "marker": NativeTask(
        method_id="marker",
        output_subdir="marker",
        worker_builder=build_marker_worker,
        docs_per_job=1,
        env_overrides={
            "MODEL_CACHE_DIR": str(MARKER.model_path),
        },
    ),
}
