from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from paths import TMP_ROOT, MODEL_ROOT, PROJECT_ROOT


RUNTIME_CONFIG_ROOT = TMP_ROOT / "runtime" / "mineru"


@dataclass(frozen=True)
class MinerUVersionProfile:
    method_id: str
    display_name: str
    conda_env: str
    repository_relative_path: str
    source_revision: str
    version: str
    checkpoint: str
    model_relative_path: str
    pipeline_model_relative_path: str
    api_backend: str
    output_name: str

    @property
    def repository_path(self) -> Path:
        return PROJECT_ROOT / self.repository_relative_path

    @property
    def model_path(self) -> Path:
        return MODEL_ROOT / self.model_relative_path

    @property
    def pipeline_model_path(self) -> Path:
        return MODEL_ROOT / self.pipeline_model_relative_path

    @property
    def runtime_config_path(self) -> Path:
        return RUNTIME_CONFIG_ROOT / f"{self.method_id}.json"

    def environment_overrides(self) -> dict[str, str]:
        return {
            "GDB_MINERU_REPO_ROOT": str(self.repository_path),
            "GDB_METHOD_ID": self.method_id,
            "GDB_MINERU_VERSION": self.version,
            "GDB_MINERU_CHECKPOINT": self.checkpoint,
            "GDB_MINERU_SOURCE_REVISION": self.source_revision,
            "MINERU_TOOLS_CONFIG_JSON": str(self.runtime_config_path),
            "MINERU_MODEL_SOURCE": "local",
        }


def write_runtime_config(profile: MinerUVersionProfile) -> Path:
    profile.runtime_config_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "bucket_info": {},
        "models-dir": {
            "pipeline": str(profile.pipeline_model_path),
            "vlm": str(profile.model_path),
        },
        "model-source": "local",
        "config_version": "1.3.2",
    }
    profile.runtime_config_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return profile.runtime_config_path
