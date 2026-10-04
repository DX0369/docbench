#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

THIS_DIR = Path(__file__).resolve().parent
CODES_ROOT = THIS_DIR.parents[1]

if str(CODES_ROOT) not in sys.path:
    sys.path.insert(0, str(CODES_ROOT))

from common import (
    add_shard_args,
    apply_shard_to_pdf_paths,
    build_error_payload,
    build_image_boxes_payload,
    build_page_records,
    cleanup_render_dir,
    collect_pdf_paths,
    concat_document_markdown,
    get_doc_rel_dir,
    get_doc_output_dir,
    get_image_size,
    load_pdf_paths_from_file,
    remove_file_if_exists,
    render_pdf_pages,
    save_markdown_bundle,
    write_json,
)
from image_vllm_client import build_openai_client, chat_with_image


DEFAULT_PROMPT = """You are an AI assistant specialized in converting PDF images to Markdown format. Please follow these instructions for the conversion:

1. Text Processing:
- Accurately recognize all text content in the PDF image without guessing or inferring.
- Convert the recognized text into Markdown format.
- Maintain the original document structure, including headings, paragraphs, lists, etc.

2. Mathematical Formula Processing:
- Convert all mathematical formulas to LaTeX format.
- Enclose inline formulas with \\( \\). For example: This is an inline formula \\( E = mc^2 \\)
- Enclose block formulas with \\[ \\]. For example: \\[ \\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a} \\]

3. Table Processing:
- Convert tables to HTML format.
- Wrap the entire table with <table> and </table>.

4. Figure Handling:
- Ignore figures content in the PDF image. Do not attempt to describe or convert images.

5. Header and Footer Handling:
- Ignore page headers and page footers.

6. Output Format:
- Ensure the output Markdown document has a clear structure with appropriate line breaks between elements.
- For complex layouts, try to maintain the original document's structure and format as closely as possible.

Please strictly follow these guidelines to ensure accuracy and consistency in the conversion. Your task is to accurately convert the content of the PDF image into Markdown format without adding any extra explanations or comments.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a generic multimodal model through a vLLM OpenAI-compatible server."
    )
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--pdf-list-file", type=Path)
    parser.add_argument("--server-url", type=str, required=True)
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--method-name", type=str, required=True)
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    parser.add_argument(
        "--thinking-mode",
        choices=["auto", "enabled", "disabled"],
        default="auto",
        help="Controls chat_template enable_thinking for models that support it.",
    )
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--timeout", type=float, default=3600.0)
    parser.add_argument("--limit", type=int)
    add_shard_args(parser)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--render-cache-root", type=Path)
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def build_extra_body(args: argparse.Namespace) -> Dict[str, Any] | None:
    if args.thinking_mode == "auto":
        return None
    return {
        "chat_template_kwargs": {
            "enable_thinking": args.thinking_mode == "enabled",
        }
    }


def run_one_pdf(args: argparse.Namespace, client: Any, pdf_path: Path) -> None:
    doc_output_dir = get_doc_output_dir(args.input_root, args.output_root, pdf_path)
    if args.skip_existing and (doc_output_dir / "document.md").exists():
        print(f"[skip] {pdf_path}")
        return

    notes = [
        "This method is evaluated in page-image mode because the benchmark requires stable page-level markdown outputs and per-page latency records.",
        "The source model accepts image inputs through a shared vLLM OpenAI-compatible service.",
        "image_boxes.json is left empty because this generic multimodal baseline does not expose stable structured image coordinates.",
    ]
    if args.thinking_mode != "auto":
        notes.append(f"thinking_mode={args.thinking_mode}")

    doc_total_start = time.perf_counter()
    render_seconds: float | None = None
    page_markdowns: List[str] = []
    page_latencies: List[float] = []
    page_sizes: List[Any] = []
    render_dir: Path | None = None
    created_temp_dir = False

    try:
        render_start = time.perf_counter()
        requested_render_dir = None
        if args.render_cache_root is not None:
            requested_render_dir = (
                args.render_cache_root / get_doc_rel_dir(args.input_root, pdf_path) / "rendered_pages"
            )
        page_images, render_dir, created_temp_dir = render_pdf_pages(
            pdf_path=pdf_path,
            dpi=args.dpi,
            render_dir=requested_render_dir,
        )
        page_sizes = [get_image_size(image_path) for image_path in page_images]
        render_seconds = round(time.perf_counter() - render_start, 6)

        extra_body = build_extra_body(args)
        for image_path in page_images:
            page_start = time.perf_counter()
            page_markdown = chat_with_image(
                client=client,
                model_name=args.model_name,
                image_path=image_path,
                prompt=args.prompt,
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens,
                seed=args.seed,
                extra_body=extra_body,
            )
            page_seconds = round(time.perf_counter() - page_start, 6)
            page_markdowns.append(page_markdown.rstrip() + "\n" if page_markdown.strip() else "")
            page_latencies.append(page_seconds)

        document_markdown = concat_document_markdown(page_markdowns)
        save_markdown_bundle(
            doc_output_dir=doc_output_dir,
            page_markdowns=page_markdowns,
            document_markdown=document_markdown,
        )

        doc_total_seconds = round(time.perf_counter() - doc_total_start, 6)
        metrics: Dict[str, Any] = {
            "method": args.method_name,
            "backend": "vllm",
            "input_mode": "pdf_to_images",
            "pdf_path": str(pdf_path),
            "output_dir": str(doc_output_dir),
            "server_url": args.server_url,
            "model_name": args.model_name,
            "prompt": args.prompt,
            "thinking_mode": args.thinking_mode,
            "dpi": args.dpi,
            "page_count": len(page_markdowns),
            "page_sizes": [[int(w), int(h)] for (w, h) in page_sizes],
            "render_seconds": render_seconds,
            "page_inference_seconds_sum": round(sum(page_latencies), 6),
            "doc_total_seconds": doc_total_seconds,
            "document_markdown_path": "document.md",
            "pages": build_page_records(page_markdowns, page_latencies),
            "notes": notes,
        }
        write_json(doc_output_dir / "metrics.json", metrics)
        write_json(
            doc_output_dir / "image_boxes.json",
            build_image_boxes_payload(
                method=args.method_name,
                pdf_path=pdf_path,
                entries=[],
                notes=[
                    "This generic multimodal baseline does not provide stable structured image coordinates.",
                ],
            ),
        )
        remove_file_if_exists(doc_output_dir / "error.json")
        print(f"[done] {pdf_path}")
    finally:
        if render_dir is not None and args.render_cache_root is None:
            cleanup_render_dir(render_dir, created_temp_dir)


def main() -> None:
    args = parse_args()
    args.input_root = args.input_root.resolve()
    args.output_root = args.output_root.resolve()

    client = build_openai_client(base_url=args.server_url, timeout=args.timeout)
    if args.pdf_list_file is not None:
        args.pdf_list_file = args.pdf_list_file.resolve()
        pdf_paths = load_pdf_paths_from_file(args.pdf_list_file, limit=args.limit)
    else:
        pdf_paths = collect_pdf_paths(args.input_root, limit=args.limit)
    pdf_paths = apply_shard_to_pdf_paths(
        pdf_paths,
        num_shards=args.num_shards,
        shard_index=args.shard_index,
    )

    for pdf_path in pdf_paths:
        try:
            run_one_pdf(args, client, pdf_path)
        except Exception:
            doc_output_dir = get_doc_output_dir(args.input_root, args.output_root, pdf_path)
            write_json(
                doc_output_dir / "error.json",
                build_error_payload(
                    method_name=args.method_name,
                    pdf_path=pdf_path,
                    extra={
                        "backend": "vllm",
                        "server_url": args.server_url,
                        "model_name": args.model_name,
                        "thinking_mode": args.thinking_mode,
                    },
                ),
            )
            print(f"[error] {pdf_path}", file=sys.stderr)
            if args.fail_fast:
                raise


if __name__ == "__main__":
    main()
