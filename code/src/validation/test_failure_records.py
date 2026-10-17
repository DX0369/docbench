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
            dpi=200, max_tokens=8192, temperature=0.0, top_p=1.0, seed=0, page_concurrency=1,
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


if __name__ == "__main__":
    unittest.main()
