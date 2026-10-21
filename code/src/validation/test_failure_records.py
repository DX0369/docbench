"""CPU regression tests for failure/warning records; no GPU, service or model initialization."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark.collect_failed_pdfs import collect_paths
from benchmark.run_single_gpu import latency_unprocessed_docs
from methods.deepseek_ocr2.vllm import parse_ref_det_entries, write_pdf_outputs
from methods.qianfan_ocr import runtime as qianfan
from paths import PROJECT_ROOT, TMP_ROOT
from validation.validate_outputs import ValidationResult, validate_doc_dir


def fake_qianfan_client(content, finish_reason):
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content), finish_reason=finish_reason)],
        usage=None,
    )
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **_: response)))


class FailureRecordTests(unittest.TestCase):
    def setUp(self):
        TMP_ROOT.mkdir(parents=True, exist_ok=True)
        self.tmp = Path(tempfile.mkdtemp(dir=TMP_ROOT, prefix="test_failure_records_"))
        self.addCleanup(lambda: __import__("shutil").rmtree(self.tmp, ignore_errors=True))

    def test_error_json_document_fails_validation(self):
        doc = self.tmp / "doc"
        doc.mkdir()
        (doc / "error.json").write_text(json.dumps({"error": "boom"}), encoding="utf-8")
        result = ValidationResult()
        self.assertFalse(validate_doc_dir(doc, result=result, allow_document_only=False))
        self.assertEqual([issue.severity for issue in result.issues], ["error"])

    def test_latency_unrecorded_inputs_are_unprocessed(self):
        pdfs = [Path("/a.pdf"), Path("/b.pdf"), Path("/c.pdf")]
        statuses = [{"job_pdf_paths": ["/a.pdf"], "failed": True}]
        docs = latency_unprocessed_docs(pdfs, statuses)
        self.assertEqual([d["pdf_path"] for d in docs], ["/b.pdf", "/c.pdf"])
        self.assertTrue(all(d["reason"] == "latency_not_reached" for d in docs))

    def test_collect_failed_pdfs_reads_current_schema(self):
        summary = {"results": [{
            "failed_docs": [{"pdf_path": "/a.pdf", "reason": "x"}],
            "unprocessed_docs": [{"pdf_path": "/b.pdf"}, {"pdf_path": "/a.pdf"}],
        }]}
        self.assertEqual(collect_paths(summary, include_unprocessed=True), (["/a.pdf", "/b.pdf"], 1, 2))
        self.assertEqual(collect_paths(summary, include_unprocessed=False), (["/a.pdf"], 1, 0))

    def test_collect_failed_pdfs_rejects_missing_or_invalid_fields(self):
        with self.assertRaises(KeyError):
            collect_paths({"results": [{"unprocessed_docs": []}]}, include_unprocessed=True)
        with self.assertRaises(KeyError):
            collect_paths({}, include_unprocessed=True)
        with self.assertRaises(ValueError):
            collect_paths({"results": []}, include_unprocessed=True)
        with self.assertRaises(ValueError):
            collect_paths({"results": [{"failed_docs": [{"pdf_path": ""}], "unprocessed_docs": []}]},
                          include_unprocessed=True)

    def test_deepseek_counts_unparseable_ref_det(self):
        text = "<|ref|>image<|/ref|><|det|>[[1, 2, 3, 4]]<|/det|><|ref|>text<|/ref|><|det|>[[oops<|/det|>"
        entries, invalid = parse_ref_det_entries(text)
        self.assertEqual((len(entries), invalid), (1, 1))

    def test_deepseek_skip_repeat_empties_page_without_eos(self):
        args = argparse.Namespace(
            input_root=self.tmp, output_root=self.tmp / "out", server_url="http://x", model_name="m",
            dpi=200, max_tokens=8192, temperature=0.0, top_p=1.0, seed=0, page_concurrency=1, render_workers=1,
        )
        pdf = self.tmp / "doc.pdf"
        page = {"page_seconds": 1.0, "page_size": [100, 100], "render_seconds": 0.1}
        write_pdf_outputs(args=args, pdf_path=pdf, prompt="p", doc_started_at=0.0, page_results=[
            {**page, "page_index": 1, "raw_output": "kept", "finish_reason": "stop"},
            {**page, "page_index": 2, "raw_output": "looping", "finish_reason": "length"},
        ])
        out = args.output_root / "doc"
        metrics = json.loads((out / "metrics.json").read_text())
        self.assertEqual(metrics["warnings"], [
            {"page_number": 2, "reason": "skip_repeat_no_eos", "finish_reason": "length"}])
        self.assertNotIn("looping", (out / "document.md").read_text())
        self.assertNotIn("<|ref|>", (out / "document.md").read_text())
        self.assertIn("kept", (out / "document.md").read_text())
        self.assertEqual(metrics["page_count"], 2)

    def test_deepseek_image_refs_use_official_placeholders(self):
        from methods.deepseek_ocr2.vllm import clean_page_markdown
        text = ("<|ref|>image<|/ref|><|det|>[[1, 2, 3, 4]]<|/det|>\n"
                "<|ref|>text<|/ref|><|det|>[[5, 6, 7, 8]]<|/det|>Body"
                "<|ref|>image<|/ref|><|det|>[[9, 9, 9, 9]]<|/det|>")
        self.assertEqual(clean_page_markdown(text, 3),
                         "![](images/3_0.jpg)\n\nBody![](images/3_1.jpg)\n")

    def test_qianfan_think_split_and_warnings(self):
        args = argparse.Namespace(enable_think=True, model_name="m", max_tokens=16)
        image = self.tmp / "page.png"
        from PIL import Image
        Image.new("RGB", (4, 4)).save(image)
        text, record = qianfan.request(args, fake_qianfan_client("layout notes</think>\n# Title", "stop"), image)
        self.assertEqual(text, "# Title")
        self.assertNotIn("think_unterminated", record)
        text, record = qianfan.request(args, fake_qianfan_client("<think>truncated", "length"), image)
        self.assertEqual(text, "<think>truncated")
        self.assertTrue(record["think_unterminated"])

    def test_unlimited_page_pool_uses_official_single_image_recipe(self):
        from methods.unlimited_ocr import client as unlimited
        image = self.tmp / "page.png"
        from PIL import Image
        Image.new("RGB", (4, 4)).save(image)
        payload = unlimited.build_payload(model_name="m", image_paths=[image], logit_processor="lp",
                                          no_repeat_ngram_size=35, recipe=unlimited.RECIPES["page_pool"])
        self.assertEqual(payload["messages"][0]["content"][0]["text"], "document parsing.")
        self.assertEqual(payload["images_config"], {"image_mode": "gundam"})
        self.assertEqual(payload["custom_params"]["window_size"], 128)
        output = self.tmp / "out" / "doc"
        output.mkdir(parents=True)
        for name in ("document.md", "image_boxes.json"):
            (output / name).write_text("x" if name.endswith(".md") else "{}", encoding="utf-8")
        (output / "metrics.json").write_text(json.dumps({
            "execution_profile": "unlimited_ocr_sglang_page_pool", "prompt": "Multi page parsing.",
            "image_mode": "base", "ngram_window": 1024}), encoding="utf-8")
        args = argparse.Namespace(input_root=self.tmp, output_root=self.tmp / "out",
                                  skip_existing=True, request_mode="page_pool")
        with self.assertRaises(ValueError):
            unlimited.should_skip(args, self.tmp / "doc.pdf")

    def test_unlimited_refuses_to_overwrite_completed_output(self):
        from methods.unlimited_ocr import client as unlimited
        output = self.tmp / "out" / "doc"
        output.mkdir(parents=True)
        for name in ("document.md", "image_boxes.json", "metrics.json"):
            (output / name).write_text("x" if name.endswith(".md") else "{}", encoding="utf-8")
        args = argparse.Namespace(input_root=self.tmp, output_root=self.tmp / "out",
                                  skip_existing=False, request_mode="joint")
        with self.assertRaises(FileExistsError):
            unlimited.should_skip(args, self.tmp / "doc.pdf")

    def test_unlimited_page_pool_records_fixed_budget_and_page_warnings(self):
        from methods.unlimited_ocr import client as unlimited
        from PIL import Image
        image = self.tmp / "page.png"
        Image.new("RGB", (4, 4)).save(image)
        args = argparse.Namespace(
            input_root=self.tmp, output_root=self.tmp / "out", server_url="http://unused",
            model_name="m", dpi=200, request_timeout_seconds=10, no_repeat_ngram_size=35,
            request_mode="page_pool", page_concurrency=1, skip_existing=False, fail_fast=True,
        )
        with (
            patch.object(unlimited, "TMP_ROOT", self.tmp),
            patch.object(unlimited, "get_pdf_page_count", return_value=2),
            patch.object(unlimited, "render_pdf_page", side_effect=lambda *_: nullcontext(image)),
            patch.object(unlimited, "generate", return_value=(
                "<PAGE>generated text", {"finish_reason": "length", "usage": {"completion_tokens": 16384}},
            )) as generate,
        ):
            failed = unlimited.process_page_pool(args, [self.tmp / "doc.pdf"], logit_processor="lp")
        self.assertFalse(failed)
        self.assertEqual(generate.call_count, 2)
        self.assertTrue(all(call.kwargs["payload"]["max_tokens"] == 16384 for call in generate.call_args_list))
        output = args.output_root / "doc"
        metrics = json.loads((output / "metrics.json").read_text())
        self.assertEqual(metrics["max_tokens"], 16384)
        self.assertEqual(metrics["warnings"], [
            {"page_number": page, "reason": "generation_length_limit"} for page in (1, 2)])
        self.assertEqual(metrics["page_count"], 2)
        self.assertFalse((output / "error.json").exists())

    def _run_unlimited_trial(self, metrics, *, request_mode, keep_outputs=False):
        """Exercise real queue accounting/export cleanup without starting any process or monitor."""
        import time
        from benchmark import run_single_gpu as runner
        from execution.queue import ensure_queue_layout, claim_next_job, write_job_status
        root = Path(tempfile.mkdtemp(dir=self.tmp, prefix="unlimited_trial_"))
        pdf = root / "doc.pdf"
        argv = [
            "run_single_gpu.py", "--method", "unlimited_ocr_sglang", "--gpu-id", "0", "--pressure", "1",
            "--input-root", str(root), "--output-root-base", str(root / "out"),
            "--log-root", str(root / "logs"), "--unlimited-ocr-request-mode", request_mode,
        ]
        if not keep_outputs:
            argv.append("--discard-successful-outputs")
        with patch.object(sys, "argv", argv):
            args = runner.parse_args()

        def fake_worker(command, *_args, **_kwargs):
            output = Path(command[command.index("--output-root") + 1]) / "doc"
            output.mkdir(parents=True)
            (output / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
            (output / "document.md").write_text("generated text", encoding="utf-8")
            queue = Path(command[command.index("--queue-dir") + 1])
            pending, running, done = ensure_queue_layout(queue)
            claimed = claim_next_job(pending, running, "w00")
            write_job_status(done, claimed_path=claimed, worker_id="w00", failed=False,
                             job_pdf_paths=[str(pdf)],
                             extra={"successful_input_page_counts": {str(pdf): 2}})
            claimed.unlink()
            return Mock(wait=Mock(return_value=0), poll=Mock(return_value=0))

        with (
            patch.object(runner, "GpuMonitor") as gpu,
            patch.object(runner, "CpuMonitor") as cpu,
            patch.object(runner, "start_vllm_service", return_value=(None, 0.5)),
            patch.object(runner, "run_command", side_effect=fake_worker),
            patch.object(runner, "start_queue_processing", side_effect=lambda *_a, **_k: time.perf_counter()),
            patch.object(runner, "queue_processing_seconds", return_value=1.0),
            patch.object(runner, "terminate_process"),
        ):
            gpu.return_value.summary.return_value = {}
            cpu.return_value.summary.return_value = {}
            return runner.run_trial(
                method=args.method, method_spec=runner.build_method_spec(args.method, args),
                base_args=args, pdf_paths=[pdf], total_pages=2, concurrency=1, run_name="fixture",
            )

    def test_unlimited_trial_retains_quality_warnings_for_both_routes(self):
        cases = {
            "page_pool": {"input_page_count": 2, "warnings": [
                {"page_number": page, "reason": "generation_length_limit"} for page in (1, 2)]},
            "joint": {"input_page_count": 2, "output_segment_count": 1, "finish_reason": "length",
                      "warnings": [{"page_number": None, "reason": "generation_length_limit"},
                                   {"page_number": None, "reason": "page_count_mismatch"}]},
        }
        for route, metrics in cases.items():
            with self.subTest(route=route):
                summary = self._run_unlimited_trial(metrics, request_mode=route)
                self.assertEqual(summary["status"], "ok")
                self.assertEqual(summary["failed_doc_count"], 0)
                self.assertEqual(summary["successful_page_count"], 2)
                self.assertEqual(summary["quality_warning_doc_count"], 1)
                self.assertEqual(summary["truncated_doc_count"], 1)
                self.assertEqual(summary["page_mismatch_doc_count"], int(route == "joint"))
                self.assertEqual(summary["quality_warning_docs"][0]["warnings"], metrics["warnings"])
                output = Path(summary["trial_output_root"])
                self.assertTrue((output / "unlimited_ocr_sglang" / "doc" / "document.md").is_file())
                saved = json.loads((Path(summary["trial_log_dir"]) / "summary.json").read_text())
                self.assertEqual(saved["quality_warning_docs"], summary["quality_warning_docs"])

    def test_unlimited_trial_discards_only_warning_free_outputs(self):
        for route in ("joint", "page_pool"):
            for keep in (False, True):
                with self.subTest(route=route, keep_outputs=keep):
                    summary = self._run_unlimited_trial(
                        {"warnings": []}, request_mode=route, keep_outputs=keep)
                    self.assertEqual(summary["status"], "ok")
                    self.assertEqual(summary["quality_warning_doc_count"], 0)
                    self.assertEqual(summary["truncated_doc_count"], 0)
                    self.assertEqual(Path(summary["trial_output_root"]).exists(), keep)

    def test_unlimited_trial_requires_warning_records(self):
        with self.assertRaises(KeyError):
            self._run_unlimited_trial({}, request_mode="page_pool")

    def test_sdk_transport_retries_are_counted(self):
        import threading
        from http.server import BaseHTTPRequestHandler, HTTPServer
        from image_vllm_client import build_openai_client, count_http_attempts, sdk_retry_warning
        calls = []
        body = json.dumps({"id": "x", "object": "chat.completion", "created": 0, "model": "m", "choices": [
            {"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "ok"}}]}).encode()

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                self.rfile.read(int(self.headers["Content-Length"]))
                calls.append(1)
                status = 503 if len(calls) <= 2 else 200
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        server = HTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        client = build_openai_client(f"http://127.0.0.1:{server.server_port}/v1", 10)
        with count_http_attempts() as attempts:
            client.chat.completions.create(model="m", messages=[{"role": "user", "content": "x"}])
        self.assertEqual(attempts[0], 3)
        self.assertEqual(sdk_retry_warning(4, attempts[0]),
                         [{"page_number": 4, "reason": "sdk_transport_retry", "count": 2}])
        self.assertEqual(sdk_retry_warning(4, 1), [])

    def test_monkey_region_failures_become_attributable_placeholders(self):
        import asyncio
        from methods.monkeyocr.async_runtime import AsyncRecognitionLoop

        class Chat:
            async def async_batch_inference(self, images, questions):
                return ["ok", ValueError("bad region")]

        loop = AsyncRecognitionLoop(Chat())
        self.addCleanup(loop.loop.close)
        out = loop.batch_inference(["a", "b"], ["q", "q"])
        self.assertEqual(out[0], "ok")
        self.assertEqual(loop.region_failures, ["ValueError: bad region"])
        from methods.monkeyocr.pipeline import failed_region_ids
        self.assertEqual(failed_region_ids([{"text": "x"}, {"res": [{"text": out[1]}]}]), {0})

        class DeadChat:
            async def async_batch_inference(self, images, questions):
                exc = RuntimeError("engine gone")
                exc.gdb_runtime_unusable = True
                return [exc]

        dead = AsyncRecognitionLoop(DeadChat())
        self.addCleanup(dead.loop.close)
        with self.assertRaises(RuntimeError):
            dead.batch_inference(["a"], ["q"])

    def test_mineru_identity_requires_profile_environment(self):
        import types
        from unittest.mock import patch
        from methods.mineru.output import mineru_identity
        version = types.ModuleType("mineru.version")
        version.__version__ = "3.4.3"
        modules = patch.dict(sys.modules, {"mineru": types.ModuleType("mineru"), "mineru.version": version})
        modules.start()
        self.addCleanup(modules.stop)
        env = {"GDB_METHOD_ID": "mineru_344_2605", "GDB_MINERU_VERSION": "3.4.4",
               "GDB_MINERU_SOURCE_REVISION": "0" * 40, "GDB_MINERU_CHECKPOINT": "ckpt"}
        with patch.dict("os.environ", env):
            self.assertEqual(mineru_identity()["release_version"], "3.4.4")
        for missing in env:
            with patch.dict("os.environ", {k: v for k, v in env.items() if k != missing}, clear=True):
                with self.assertRaises(KeyError):
                    mineru_identity()

    def test_snapshot_patch_check_detects_missing_patch(self):
        import subprocess
        from method_paths import check_patches_applied, patch_apply_command
        repo = self.tmp / "snapshot"
        repo.mkdir()
        (repo / "a.py").write_text("x = 1\n", encoding="utf-8")
        patch = self.tmp / "change.patch"
        patch.write_text("--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-x = 1\n+x = 2\n", encoding="utf-8")
        # The snapshot lives inside the root work tree, where `git -C snapshot apply` skips files.
        self.assertIn("--directory=" + str(repo.resolve().relative_to(PROJECT_ROOT)), patch_apply_command(repo))
        with self.assertRaises(RuntimeError):
            check_patches_applied(repo, [patch])
        subprocess.run(patch_apply_command(repo, str(patch)), check=True)
        check_patches_applied(repo, [patch])

    def test_popo_export_keeps_image_reference_without_copying(self):
        from methods.mineru_popo.output import export_tree
        assets = self.tmp / "assets"
        (assets / "crops").mkdir(parents=True)
        (assets / "crops/fig.jpg").write_bytes(b"jpg")
        tree = {"type": "root", "content": "", "block_ids": [], "children": [
            {"type": "image", "content": "", "block_ids": [1], "img_path": "crops/fig.jpg", "title": "",
             "caption": "Figure 1", "metadata": "", "location": [[0, 0, 1, 1]]}]}
        output = self.tmp / "popo_out"
        output.mkdir()
        export_tree(tree, [{"id": 1, "type": "image", "level": 0}], assets, output)
        self.assertIn("![](<images/crops/fig.jpg>)", (output / "document.md").read_text())
        self.assertFalse((output / "images").exists())

    def test_snapshot_version_metadata(self):
        from method_repositories import Repository, snapshot_state
        root = self.tmp / "repo"
        (root / "pkg").mkdir(parents=True)
        (root / "pkg/version.py").write_text('__version__ = "3.4.3"\n', encoding="utf-8")
        repo = Repository(id="r", path=root, url="u", ref="r", update_ref="r", revision="0" * 40,
                          release="3.4.4", version_file="pkg/version.py")
        self.assertEqual(snapshot_state(repo), ("version-mismatch", "3.4.3"))
        self.assertEqual(snapshot_state(Repository(**{**repo.__dict__, "source_version": "3.4.3"})),
                         ("version-ok", "3.4.3"))
        self.assertEqual(snapshot_state(Repository(**{**repo.__dict__, "path": self.tmp / "none"}))[0], "missing")


if __name__ == "__main__":
    unittest.main()
