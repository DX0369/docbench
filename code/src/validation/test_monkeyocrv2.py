"""Small CPU checks for MonkeyOCRv2 AR registration and output boundaries."""
import argparse
import json
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.execution_profiles import resolve_execution_profile, runtime_keys
from config.throughput_pressure import latency_mapping, pressure_mapping
from methods.monkeyocrv2.commands import build_service, build_worker
from methods.monkeyocrv2.pipeline import export_document, process_document, process_pdf_paths
from method_paths import method_repo_path
from paths import TMP_ROOT


class MonkeyOCRv2Tests(unittest.TestCase):
    def test_ar_and_pressure_commands(self):
        service = build_service(None, 6, 8888)
        self.assertIn("VLLM_USE_V2_MODEL_RUNNER=1", service)
        for flag in ("--draft-model", "--speculative-config", "--enforce-eager", "--hf-overrides"):
            self.assertNotIn(flag, service)
        for p in (1, 2, 4):
            outer, overrides, _, _ = pressure_mapping("monkeyocrv2_b_parsing", p)
            args = argparse.Namespace(input_root="input", output_root_base="output", service_port=8888,
                                      skip_existing=False, **overrides)
            worker = build_worker(args, 6, Path("queue"), "worker")
            self.assertEqual(outer, 1)
            self.assertEqual(worker[worker.index("--page-max-inflight") + 1], str(p))
        self.assertEqual(runtime_keys("monkeyocrv2_b_parsing"), ())
        self.assertEqual(resolve_execution_profile("monkeyocrv2_b_parsing").profile_id,
                         "monkeyocrv2_b_parsing_vllm_ar")
        self.assertEqual(latency_mapping("monkeyocrv2_b_parsing")[1], {"monkeyocrv2_page_max_inflight": 64})

    def test_official_output_conversion(self):
        sys.path.insert(0, str(method_repo_path("monkeyocrv2") / "parsing"))
        import core_runner as official

        with tempfile.TemporaryDirectory(dir=TMP_ROOT) as temporary:
            root = Path(temporary)
            native, output = root / "native", root / "output"
            (native / "images").mkdir(parents=True)
            (native / "images/figure.jpg").write_bytes(b"picture")
            output.mkdir()
            record = {"image_size": [[100, 200], [100, 200]], "layouts": [
                {"page_num": 1, "label": "Page-header", "content": "header", "bbox": [0, 0, 10, 10]},
                {"page_num": 1, "label": "Picture", "content": "![image](../images/figure.jpg)", "bbox": [10, 20, 50, 100]},
                {"page_num": 2, "label": "Table", "content": "<table><tr><td>中文</td></tr></table>", "bbox": [0, 0, 100, 200]},
            ]}
            args = argparse.Namespace(page_max_inflight=2)
            export_document(args, root / "input.pdf", native, record, 2, output, time.perf_counter(), official)
            self.assertIn("../images/figure.jpg", (output / "pages/page_0001.md").read_text())
            document = (output / "document.md").read_text()
            self.assertIn("images/figure.jpg", document)
            self.assertIn("<table>", document)
            self.assertNotIn("header", document)
            self.assertTrue((output / "images/figure.jpg").is_file())
            self.assertEqual(json.loads((output / "metrics.json").read_text())["page_count"], 2)
            for broken in ({"image_size": record["image_size"]},
                           {"image_size": record["image_size"], "layouts": [{"page_num": 0}]}):
                with self.assertRaises((KeyError, ValueError)):
                    export_document(args, root / "input.pdf", native, broken, 2, output, time.perf_counter(), official)
            # The official resident pool uses the input basename in image_path.
            args.input_root, args.output_root, args.temp_root = root, root / "bundle", root
            def run(config):
                native_output = Path(config.output_path)
                shutil_source = native / "images/figure.jpg"
                (native_output / "images").mkdir()
                (native_output / "images/figure.jpg").write_bytes(shutil_source.read_bytes())
                result = native_output / "all_results.json"
                result.write_text(json.dumps([{**record, "image_path": "input.pdf"}]))
                return {"all_results_path": result}
            with patch("methods.monkeyocrv2.pipeline.pdf_page_count", return_value=2):
                process_document(args, root / "input.pdf", SimpleNamespace(run=run), None, official)
            self.assertEqual(len(list(args.output_root.rglob("metrics.json"))), 1)

    def test_document_failure_continues(self):
        with tempfile.TemporaryDirectory(dir=TMP_ROOT) as temporary:
            root = Path(temporary)
            pdfs = [root / "failed.pdf", root / "next.pdf"]
            args = argparse.Namespace(input_root=root, output_root=root / "output", skip_existing=False, fail_fast=False)
            def process(_args, pdf, *_rest):
                if pdf == pdfs[0]:
                    raise ValueError("bad layout")
            with patch("methods.monkeyocrv2.pipeline.process_document", side_effect=process) as mocked:
                self.assertTrue(process_pdf_paths(args, pdfs, None, None, None))
                self.assertEqual(mocked.call_count, 2)
            errors = list(args.output_root.rglob("error.json"))
            self.assertEqual(len(errors), 1)
            self.assertEqual(json.loads(errors[0].read_text())["error_type"], "ValueError")


if __name__ == "__main__":
    unittest.main()
