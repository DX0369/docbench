"""Eager initialization for the project's pinned, local hybrid backends.

No PDF is processed here. Keep the official cache keys (including None paths)
so requests reuse these models rather than creating another engine.
"""
from __future__ import annotations

from importlib import import_module
import os
from pathlib import Path
import subprocess
from typing import Any


API_PRELOAD_BACKENDS = {
    "mineru_303_2509": ("hybrid-auto-engine", "hybrid-vllm-async-engine"),
    "mineru_344_2605": ("hybrid-engine",),
}
_VERSIONS = {"mineru_303_2509": "3.0.3", "mineru_344_2605": "3.4.4"}
# This released Git tag retains the preceding version.py string. Accept only
# the pinned checkout, never an arbitrary installed package with that version.
_SOURCE_VERSIONS = {"mineru_344_2605": "3.4.3"}


def supports_api_preload(method_id: str, backend: str) -> bool:
    return backend in API_PRELOAD_BACKENDS.get(method_id, ())


def prepare_preload_environment(*, table_enable: bool) -> None:
    # Match aio_do_parse's hybrid branch before an engine creates its processors.
    os.environ["MINERU_VLM_FORMULA_ENABLE"] = "true"
    os.environ["MINERU_VLM_TABLE_ENABLE"] = str(table_enable)


def preload_hybrid_models(
    *,
    method_id: str,
    backend: str,
    parse_method: str,
    language: str,
    formula_enable: bool,
    table_enable: bool,
    model_kwargs: dict[str, Any],
) -> dict[str, Any]:
    if not supports_api_preload(method_id, backend):
        raise ValueError(f"Unsupported MinerU API preload: {method_id} / {backend}")
    if parse_method not in ("auto", "txt", "ocr"):
        raise ValueError(f"Unsupported parse method: {parse_method}")
    version_module = import_module("mineru.version")
    actual_version = version_module.__version__
    if actual_version != _VERSIONS[method_id]:
        from config.method_versions import get_method_version
        from methods.mineru.versions import PROFILES
        profile = PROFILES[method_id]
        expected_file = profile.repository_path / "mineru/version.py"
        source_file = getattr(version_module, "__file__", None)
        pinned_source = (actual_version == _SOURCE_VERSIONS.get(method_id) and source_file is not None
                         and Path(source_file).resolve() == expected_file.resolve())
        if pinned_source:
            revision = subprocess.check_output(
                ["git", "-C", str(profile.repository_path), "rev-parse", "HEAD"], text=True).strip()
            pinned_source = revision == get_method_version(method_id).source_revision
        if not pinned_source:
            raise ValueError(f"MinerU preload requires {_VERSIONS[method_id]} or its exact pinned source, got {actual_version}")
    prepare_preload_environment(table_enable=table_enable)
    modern = method_id == "mineru_344_2605"
    engine = backend.removeprefix("hybrid-")
    if modern and engine == "engine":
        # This official helper preserves (resolved_engine, None, None).
        engine = import_module("mineru.cli.vlm_preload").preload_vlm_model(
            model_kwargs=dict(model_kwargs)
        )
    else:
        if engine in ("auto-engine", "engine"):
            engine = import_module("mineru.utils.engine_utils").get_vlm_engine(
                inference_engine="auto", is_async=True
            )
        import_module("mineru.backend.vlm.vlm_analyze").ModelSingleton().get_model(
            engine, None, None, **dict(model_kwargs)
        )

    pipeline = import_module("mineru.backend.pipeline.model_init")
    hybrid = pipeline.HybridModelSingleton()
    atoms = pipeline.AtomModelSingleton()
    if modern:
        # _predict_layout_for_window omits lang. Do not replace None with 'ch'.
        ocr_modes = (False, True) if parse_method == "auto" else (parse_method == "ocr",)
        formula_variants = list(dict.fromkeys(formula_enable and not ocr for ocr in ocr_modes))
        for enabled in formula_variants:
            hybrid.get_model(formula_enable=enabled)
        # The default medium-effort hybrid path initializes this on first table.
        # Its constructor also creates the dedicated orientation OCR instance.
        atoms.get_atom_model(atom_model_name=pipeline.AtomicModel.TableOrientationCls, lang=None)
        languages = ["ch"]
    else:
        # Benchmark inputs contain both languages. Explicit other languages are
        # additive; selecting the model for a PDF remains the adapter's job.
        languages = ["ch", "en"]
        if language not in ("auto", "ch", "en"):
            languages.append(language)
        formula_variants = [formula_enable]
        for lang in languages:
            hybrid.get_model(lang=lang, formula_enable=formula_enable)
        heading = import_module("mineru.backend.hybrid.hybrid_model_output_to_middle_json")
        if heading.heading_level_import_success:
            atoms.get_atom_model(
                atom_model_name="ocr", ocr_show_log=False,
                det_db_box_thresh=0.3, lang="ch_lite",
            )
    return {
        "method_id": method_id, "backend": engine,
        # Published only after the version / exact pinned-source check above.
        # The official 3.4.4 tag itself still reports runtime version 3.4.3.
        "release_version": _VERSIONS[method_id],
        "runtime_version": actual_version,
        "ocr_languages": languages, "hybrid_formula_variants": formula_variants,
    }
