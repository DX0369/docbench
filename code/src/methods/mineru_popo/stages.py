"""CPU-only stages: MinerU /tasks artifacts -> official Popo inference/tree."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

from common import write_json
from method_paths import method_repo_path
from methods.mineru.versions.v344_2605 import PROFILE
from methods.mineru_popo.api import PopoClient


def initialize(args):
    source = method_repo_path("mineru_popo") / "post_processing"
    sys.path.insert(0, str(source))
    os.environ["POPO_INFERENCE_BACKEND"] = "vllm"
    import model_utils
    import inference
    from label_normalization import MineruReader, to_popo_pages
    from run_inference import validate_pages
    from get_json_tree import construct_json_tree

    client = PopoClient(args.popo_url)
    errors = []

    def observed_generate(*positional, **keywords):
        if errors:
            raise errors[0]
        try:
            return client.generate(*positional, **keywords)
        except Exception as exc:
            errors.append(exc)
            raise

    # inference imports model_utils with '*': replace both bindings, including
    # cached modules. Each CPU process owns its bindings and document state.
    model_utils.popo_generate = inference.popo_generate = observed_generate
    return dict(reader=MineruReader, normalize=to_popo_pages, validate=validate_pages,
                infer=inference.main, tree=construct_json_tree, errors=errors, client=client)


def mineru_options(server_url, input_root):
    from methods.mineru.api_worker import configure_http_timeouts

    options = argparse.Namespace(server_url=server_url, input_root=Path(input_root), backend=PROFILE.api_backend,
        method="auto", disable_formula=False, disable_table=False,
        http_timeout_seconds=None, task_timeout_seconds=3600.0, task_poll_seconds=1.0)
    configure_http_timeouts(options)
    return options


def parse_mineru(options, pdf, work):
    import fitz
    from methods.mineru.api_worker import (
        post_task, wait_for_task, download_and_extract_result, find_middle_json_path,
    )
    from methods.mineru.output import infer_mineru_lang

    with fitz.open(str(pdf)) as doc:
        page_count = len(doc)
    # "auto" is a project-side selector, not a valid official API language.
    language = infer_mineru_lang(options.input_root, pdf, "auto")
    task = post_task(options, pdf, language, return_images=True)
    result = wait_for_task(options, task)
    download_and_extract_result(options, result, work / "mineru")
    middle = find_middle_json_path(work / "mineru", pdf.stem)
    payload = json.loads(middle.read_text())
    if [page["page_idx"] for page in payload["pdf_info"]] != list(range(page_count)):
        raise ValueError("MinerU output does not cover all PDF pages")
    write_json(work / "mineru_stage.json", {"middle": str(middle), "page_count": page_count,
                                          "task_id": task["task_id"]})


def process_document(runtime, pdf, work):
    runtime["errors"].clear()
    middle = Path(json.loads((work / "mineru_stage.json").read_text())["middle"])
    reader = runtime["reader"](middle.parents[2])
    reader.inner_dir = middle.parent.name
    # MinerU may shorten a long upload name; use its actual archive identity.
    # Keep the original PDF name for Popo inference and our final output paths.
    result = reader.read_doc(middle.name.removesuffix("_middle.json"))
    if result.status != "ok":
        raise ValueError(result.message)
    pages = runtime["normalize"](result.blocks)
    runtime["validate"]("mineru", pdf.stem, pages, strict_bbox=True)
    write_json(work / "normalized.json", pages)
    for name in ("inference", "raw", "tree", "preview"):
        (work / name).mkdir()
    runtime["infer"](str(pdf), pages, str(work / "inference"), raw_output_dir=str(work / "raw"))
    if runtime["errors"]:
        raise runtime["errors"][0]
    runtime["tree"](str(work / "inference" / f"{pdf.stem}.json"),
                    str(work / "tree"), str(work / "preview"))
