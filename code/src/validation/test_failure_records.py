"""CPU regression tests for failure/warning records; no GPU, service or model initialization."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark.collect_failed_pdfs import collect_paths
from benchmark.run_single_gpu import latency_unprocessed_docs
from methods.deepseek_ocr2.vllm import parse_ref_det_entries, write_pdf_outputs
from methods.qianfan_ocr import runtime as qianfan
from paths import TMP_ROOT
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
