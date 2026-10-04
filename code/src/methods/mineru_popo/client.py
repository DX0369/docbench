"""Persistent CPU document worker sharing two GPU services."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common import build_error_payload, get_doc_output_dir, runtime_unusable, write_json
from config.method_versions import get_method_version
from execution.workers import run_pdf_queue_worker
from execution.processes import http_service_ready
from methods.mineru.versions.v344_2605 import PROFILE
from methods.mineru_popo.api import GENERATION
from methods.mineru_popo.output import export_tree
from methods.mineru_popo import stages
from paths import TMP_ROOT

METHOD = "mineru_344_2605_popo"
EXECUTION_PROFILE = METHOD + "_vllm_services"


def process_pdf(pdf, args, runtime, mineru):
    output = get_doc_output_dir(args.input_root, args.output_root, pdf)
    if output.exists():
        if args.skip_existing and not (output / "error.json").exists() and (output / "metrics.json").is_file():
            metrics = json.loads((output / "metrics.json").read_text())
            if metrics.get("status") == "completed" and all((output / name).is_file() for name in
                    ("document.md", "document_tree.json", "popo_blocks.json", "image_boxes.json")):
                return False
        raise FileExistsError(f"Refusing to overwrite {output}; use a fresh output directory")
    output.mkdir(parents=True)
    work = Path(tempfile.mkdtemp(prefix="popo_", dir=TMP_ROOT))
    start = time.perf_counter()
    timings = {}
    stage = "mineru"
    try:
        for stage, operation in (("mineru", lambda: stages.parse_mineru(mineru, pdf, work)),
                                 ("popo", lambda: stages.process_document(runtime, pdf, work))):
            stage_start = time.perf_counter()
            operation()
            timings[stage] = time.perf_counter() - stage_start
        stage = "export"
        metadata = json.loads((work / "mineru_stage.json").read_text())
        blocks = json.loads((work / "inference" / f"{pdf.stem}.json").read_text())
        normalized = json.loads((work / "normalized.json").read_text())
        if Counter(b["source_id"] for page in normalized.values() for b in page) != Counter(b["source_id"] for b in blocks):
            raise ValueError("Popo inference lost or duplicated normalized blocks")
        tree = json.loads((work / "tree" / f"{pdf.stem}.json").read_text())
        stage_start = time.perf_counter()
        coverage = export_tree(tree, blocks, Path(metadata["middle"]).parent / "images", output)
        timings[stage] = time.perf_counter() - stage_start
        diagnostic = output / "diagnostics"
        diagnostic.mkdir()
        shutil.copyfile(work / "normalized.json", diagnostic / "normalized.json")
        shutil.copytree(work / "raw", diagnostic / "popo_raw")
        write_json(output / "metrics.json", {
            "method": METHOD, "status": "completed", "pdf_path": str(pdf),
            "output_scope": "document_only", "page_count": metadata["page_count"],
            "pages": [{"page_index": i, "latency_seconds": "N/A"} for i in range(metadata["page_count"])],
            "stage_seconds": timings, "doc_total_seconds": time.perf_counter() - start,
            "execution_profile": EXECUTION_PROFILE, "coverage": coverage,
            "timing_scope": "shared services; excludes startup; includes upload, polling, download, inference and export; no warmup",
            "base_method": PROFILE.method_id, "mineru_revision": PROFILE.source_revision,
            "mineru_backend": "hybrid-vllm-async-engine", "mineru_task_id": metadata["task_id"],
            "popo_revision": get_method_version(METHOD).source_revision,
            "popo_backend": "vllm", "popo_generation": GENERATION,
            "popo_patches": ["same_page_candidates", "community_PR12", "community_PR15", "failure_propagation"],
            "mineru_url": args.mineru_url, "popo_url": args.popo_url,
        })
        shutil.rmtree(work)
        return False
    except BaseException as exc:
        from methods.mineru.api_worker import MinerUServiceUnavailable

        if isinstance(exc, MinerUServiceUnavailable):
            exc.gdb_runtime_unusable = True
            exc.requeue_claimed_job = False
        if isinstance(exc, Exception) and stage in ("mineru", "popo") and not runtime_unusable(exc):
            health = (args.mineru_url.rstrip("/") + "/ready" if stage == "mineru"
                      else args.popo_url.removesuffix("/v1").rstrip("/") + "/health")
            if not http_service_ready(health):
                exc.gdb_runtime_unusable = True
        write_json(output / "error.json", build_error_payload(
            METHOD, pdf, exception=exc, stage=stage,
            extra={"diagnostics": str(work), "stage_seconds": timings}))
        if not isinstance(exc, Exception) or runtime_unusable(exc):
            raise
        print(traceback.format_exc(), flush=True)
        return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("input-root", "output-root", "queue-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("worker-id", "mineru-url", "popo-url"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        parser.error("Popo document clients must have CUDA_VISIBLE_DEVICES='' (services own the GPU)")
    runtime = stages.initialize(args)
    try:
        mineru = stages.mineru_options(args.mineru_url, args.input_root)
        failed = run_pdf_queue_worker(queue_dir=args.queue_dir, worker_id=args.worker_id,
            fail_fast=False, input_root=args.input_root, output_root=args.output_root,
            process_job=lambda _job, pdfs: any([process_pdf(pdf, args, runtime, mineru) for pdf in pdfs]))
    finally:
        runtime["client"].close()
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
