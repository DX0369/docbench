from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal, cast

from paths import MODEL_ROOT, REPO_ROOT


METHOD_CATALOG_PATH = REPO_ROOT / "config" / "methods.json"


def load_method_catalog(path: Path = METHOD_CATALOG_PATH) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload["schema_version"] != 4:
        raise ValueError(f"Unsupported method catalog schema: {payload['schema_version']}")
    repositories = {}
    for repository in payload["repositories"]:
        repo_id = repository["id"]
        if repo_id in repositories:
            raise ValueError(f"Duplicate repository id: {repo_id}")
        revision = repository["revision"]
        if not isinstance(revision, str) or len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
            raise ValueError(f"{repo_id}: revision must be a full lowercase commit hash")
        repositories[repo_id] = repository
    for variant in payload["variants"]:
        if "source_revision" in variant:
            raise ValueError("Store runtime revision only in repositories, not variants")
        repo_id = variant["source_repository_id"]
        if variant["source_binding"] == "model_only":
            if repo_id is not None:
                raise ValueError(f"{variant['method_id']}: model_only must not reference a repository")
        elif repo_id not in repositories:
            raise ValueError(f"{variant['method_id']}: unknown repository {repo_id!r}")
    return payload

SourceBinding = Literal[
    "exact_checkout",
    "compatibility_source",
    "reference_checkout",
    "model_only",
]
SOURCE_BINDINGS = frozenset(
    {"exact_checkout", "compatibility_source", "reference_checkout", "model_only"}
)


@dataclass(frozen=True)
class MethodVersion:
    method_id: str
    family_id: str
    display_name: str
    official_checkpoint: str
    model_relative_path: str
    additional_model_relative_paths: tuple[str, ...]
    model_required: bool
    source_repository_id: str | None
    source_revision: str | None
    source_binding: SourceBinding
    historical_source_revision: str | None
    adapter_module: str | None
    conda_environments: tuple[str, ...]
    benchmark_status: str
    role: str
    validation_status: str
    notes: str

    @property
    def model_path(self) -> Path:
        return MODEL_ROOT / self.model_relative_path

    @property
    def additional_model_paths(self) -> tuple[Path, ...]:
        return tuple(MODEL_ROOT / path for path in self.additional_model_relative_paths)

    @property
    def conda_environment(self) -> str:
        if not self.conda_environments:
            raise ValueError(f"No Conda environment recorded for {self.method_id}")
        return self.conda_environments[0]


@lru_cache(maxsize=1)
def load_method_versions() -> dict[str, MethodVersion]:
    payload = load_method_catalog()
    repositories = {item["id"]: item for item in payload["repositories"]}

    versions: dict[str, MethodVersion] = {}
    for item in payload["variants"]:
        source_binding = item.get("source_binding")
        if source_binding not in SOURCE_BINDINGS:
            raise ValueError(
                f"Unsupported source binding for {item.get('method_id')}: {source_binding}"
            )
        historical_source_revision = item.get("historical_source_revision")
        if source_binding == "compatibility_source" and not historical_source_revision:
            raise ValueError(
                f"compatibility_source requires historical_source_revision: {item.get('method_id')}"
            )
        if source_binding != "compatibility_source" and historical_source_revision:
            raise ValueError(
                f"historical_source_revision requires compatibility_source: {item.get('method_id')}"
            )
        version = MethodVersion(
            method_id=item["method_id"],
            family_id=item["family_id"],
            display_name=item["display_name"],
            official_checkpoint=item["official_checkpoint"],
            model_relative_path=item["model_path"],
            additional_model_relative_paths=tuple(item.get("additional_model_paths", [])),
            model_required=bool(item["model_required"]),
            source_repository_id=item["source_repository_id"],
            source_revision=(repositories[item["source_repository_id"]]["revision"]
                             if item["source_repository_id"] is not None else None),
            source_binding=cast(SourceBinding, source_binding),
            historical_source_revision=historical_source_revision,
            adapter_module=item.get("adapter_module"),
            conda_environments=tuple(item.get("conda_environments", [])),
            benchmark_status=item["benchmark_status"],
            role=item["role"],
            validation_status=item["validation_status"],
            notes=item.get("notes", ""),
        )
        if version.method_id in versions:
            raise ValueError(f"Duplicate method version id: {version.method_id}")
        versions[version.method_id] = version
    return versions


def get_method_version(method_id: str) -> MethodVersion:
    try:
        return load_method_versions()[method_id]
    except KeyError as exc:
        raise KeyError(f"Unknown method version: {method_id}") from exc
