"""CPU regression tests; run with gdb_mineru_4, no GPU/model initialization."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx
import pypdfium2
from docvortex.schema import (DocumentMetadata, EquationBlock, ImageBlock, ImageBodyBlock, MiddleJson,
                             PageInfo, Producer, TableBlock, TableBodyBlock)
from mineru.parser import ParseResult
from pydantic import ValidationError

from common import concat_document_markdown, get_doc_output_dir
from config.execution_profiles import profile_for_args, runtime_keys
from config.throughput_pressure import latency_mapping, pressure_mapping
from execution.tasks import SERVICE_TASKS, managed_service_identity
from methods.mineru.v4_output import convert_result
from methods.mineru.v4_worker import process_documents
from methods.mineru.versions import v4011_standard as profile
from paths import TMP_ROOT
from validation.validate_outputs import ValidationResult, validate_doc_dir


def result_with_pages(pages):
    return ParseResult(MiddleJson(
        metadata=DocumentMetadata(file_suffix="pdf", producer=Producer(name="mineru", version="4.0.11")),
        pages=pages, is_full_document=True,
    ))


def table_page(index):
    box = (.1239, .2, .6, .7)
    return PageInfo(page_idx=index, blocks=[TableBlock(
        type="table", index=0, bbox=box, continues_prev=bool(index),
        content=[TableBodyBlock(type="table_body", index=0, bbox=box,
                               content=f"<table><tr><td>{index}</td></tr></table>")],
    )])


class OutputTests(unittest.TestCase):
    def test_continued_table_preserves_document_render_and_boxes_without_images(self):
        result = result_with_pages([table_page(0), table_page(1)])
        before = result.to_dict()
        document, pages, boxes = convert_result(result, input_page_count=2, pdf_path=Path("table.pdf"))
        self.assertEqual(document, result.markdown())
        self.assertNotEqual(document.strip(), concat_document_markdown(pages).strip())
        self.assertEqual(result.to_dict(), before)
        self.assertEqual(boxes["entry_count"], 2)
        self.assertEqual(boxes["entries"][0]["relative_box_0_1000"], [123, 200, 600, 700])
        self.assertIsNone(boxes["entries"][0]["page_marker_expression"])
        self.assertIsNone(boxes["entries"][0]["absolute_box"])

    def test_blank_page_is_retained_and_original_page_gap_fails(self):
        result = result_with_pages([PageInfo(page_idx=0), PageInfo(page_idx=1)])
        _, pages, _ = convert_result(result, input_page_count=2, pdf_path=Path("blank.pdf"))
        self.assertEqual(pages, ["", ""])
        with self.assertRaisesRegex(ValueError, "original PDF page"):
            convert_result(result_with_pages([PageInfo(page_idx=1)]), input_page_count=1, pdf_path=Path("gap.pdf"))

    def test_missing_page_index_and_missing_pdf_box_fail_at_schema_boundary(self):
        with self.assertRaises(ValidationError):
            PageInfo.model_validate({"blocks": []})
        with self.assertRaises(ValidationError):
            result_with_pages([PageInfo(page_idx=0, blocks=[ImageBlock(
                type="image", index=0, content=[ImageBodyBlock(type="image_body", index=0, content="")],
            )])])

    def test_actual_image_marker_is_matched_once(self):
        box = (0., 0., .5, .5)
        result = result_with_pages([PageInfo(page_idx=0, blocks=[ImageBlock(
            type="image", index=0, bbox=box,
            content=[ImageBodyBlock(type="image_body", index=0, bbox=box, content="", image_path="images/example.png")],
        )])])
        _, _, boxes = convert_result(result, input_page_count=1, pdf_path=Path("image.pdf"))
        self.assertEqual(boxes["entry_count"], 1)
        self.assertEqual(boxes["entries"][0]["page_marker_src"], "images/example.png")

    def test_equation_keeps_official_latex_and_box_without_invented_image(self):
        result = result_with_pages([PageInfo(page_idx=0, blocks=[EquationBlock(
            type="equation", index=0, bbox=(0., 0., .5, .5), content="x=1",
        )])])
        document, _, boxes = convert_result(result, input_page_count=1, pdf_path=Path("formula.pdf"))
        self.assertIn("x=1", document)
        self.assertEqual(boxes["entries"][0]["block_label"], "equation")
        self.assertIsNone(boxes["entries"][0]["page_marker_expression"])


class MappingTests(unittest.TestCase):
    def test_pressure_latency_and_service_use_new_defaults(self):
        for pressure in (1, 2, 4):
            outer, overrides, _, docs = pressure_mapping(profile.METHOD_ID, pressure)
            self.assertEqual((outer, docs), (pressure, 1))
            self.assertEqual(overrides, {"mineru4_concurrency": pressure})
            args = argparse.Namespace(input_root="input", output_root_base="output",
                                      service_port=9800, skip_existing=False, **overrides)
            cmd = profile.build_service(args, 0, 9800)
            self.assertEqual(cmd[cmd.index("--concurrency") + 1], str(pressure))
            self.assertIn("--preload-models", cmd)
            self.assertNotIn("--gpu-memory-utilization", cmd)
            self.assertEqual(managed_service_identity(profile.METHOD_ID, cmd, 9800)["expected_fields"], {"version": "4.0.11"})
        self.assertEqual(latency_mapping(profile.METHOD_ID)[1], {"mineru4_concurrency": 1})
        self.assertEqual(runtime_keys(profile.METHOD_ID), ())
        self.assertEqual(SERVICE_TASKS[profile.METHOD_ID].runner_env, "gdb_mineru_4")
        self.assertEqual(profile_for_args(profile.METHOD_ID, argparse.Namespace(unlimited_ocr_request_mode="joint")),
                         "mineru_4011_standard_v1_api")

    def test_managed_startup_does_not_disable_compilation(self):
        from benchmark import run_single_gpu as runner
        args = argparse.Namespace(startup_timeout=900, startup_poll_seconds=3, dry_run=False,
                                  output_root_base="unused", mineru4_concurrency=1)
        spec = runner.MethodSpec("vllm", SERVICE_TASKS[profile.METHOD_ID], profile.environment_overrides())
        with patch.object(runner, "assert_service_port_available"), patch.object(runner, "wait_for_http_or_process_exit"), \
                patch.object(runner, "run_command") as run:
            runner.start_vllm_service(method=profile.METHOD_ID, args=args, gpu_id=0, port=9800,
                                      config=spec.task_config, trial_log_dir=Path("unused"), method_spec=spec)
            self.assertNotIn("TORCHDYNAMO_DISABLE", run.call_args.kwargs["env_overrides"])
        self.assertTrue(runner.uses_model_ready_timing(profile.METHOD_ID, args))
        args.gpu_id = 0
        worker_env = runner._build_worker_env_overrides(
            method=profile.METHOD_ID, config=spec.task_config, base_args=args,
            method_spec=spec, worker_id="w00",
        )
        self.assertEqual(worker_env["CUDA_VISIBLE_DEVICES"], "")


class WorkerTests(unittest.TestCase):
    def setUp(self):
        TMP_ROOT.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="mineru4_cpu_", dir=TMP_ROOT)
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.args = argparse.Namespace(input_root=root / "inputs", output_root=root / "outputs", skip_existing=False)
        self.args.input_root.mkdir()
        self.pdfs = []
        for name in ("first", "second"):
            path = self.args.input_root / f"{name}.pdf"
            with pypdfium2.PdfDocument.new() as pdf:
                page = pdf.new_page(100, 100)
                pdf.save(path)
                page.close()
            self.pdfs.append(path)

    def test_document_error_skips_and_next_document_exports_valid_bundle(self):
        calls = []

        class Parser:
            def parse(_self, path, *, page_range, status_callback):
                calls.append(path)
                if path == self.pdfs[0]:
                    raise ValueError("official document failure")
                self.assertEqual(page_range, "all")
                status_callback("completed")
                return result_with_pages([PageInfo(page_idx=0)])

        failed, status = process_documents(self.args, self.pdfs, Parser())
        self.assertTrue(failed)
        self.assertEqual(calls, self.pdfs)
        self.assertEqual(len(status["failed_docs"]), 1)
        good = get_doc_output_dir(self.args.input_root, self.args.output_root, self.pdfs[1])
        validation = ValidationResult()
        self.assertTrue(validate_doc_dir(good, result=validation, allow_document_only=False))
        self.assertEqual(validation.issues, [])
        metrics = json.loads((good / "metrics.json").read_text())
        self.assertIsInstance(metrics["doc_total_seconds"], float)
        self.assertGreaterEqual(metrics["doc_total_seconds"], 0)
        self.assertNotIn("wall_seconds", metrics)
        bad = get_doc_output_dir(self.args.input_root, self.args.output_root, self.pdfs[0])
        self.assertEqual(json.loads((bad / "error.json").read_text())["stage"], "official_api_parse")

    def test_wrapped_connection_failure_stops_worker_without_document_retry(self):
        calls = []

        class Parser:
            def parse(_self, path, **kwargs):
                calls.append(path)
                try:
                    raise httpx.ConnectError("listener exited")
                except httpx.ConnectError as exc:
                    raise RuntimeError("SDK exhausted transport retries") from exc

        failed, status = process_documents(self.args, self.pdfs, Parser())
        self.assertTrue(failed)
        self.assertTrue(status["stop_worker"])
        self.assertEqual(calls, self.pdfs[:1])

    def test_partial_job_is_not_counted_as_success(self):
        class Parser:
            def parse(_self, path, *, status_callback, **kwargs):
                status_callback("partial")
                return result_with_pages([PageInfo(page_idx=0)])

        failed, status = process_documents(self.args, self.pdfs[:1], Parser())
        self.assertTrue(failed)
        self.assertEqual(status["processed_docs"], [])


if __name__ == "__main__":
    unittest.main()
