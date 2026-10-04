"""Qianfan bindings for the shared bounded page scheduler."""
from common import concat_document_markdown
from execution.page_pool import run_page_pool
from methods.qianfan_ocr.runtime import export, page_count, record_error, run_page, should_skip


def process_page_pool(args, client, pdf_paths):
    def export_document(pdf, ordered, started):
        pages = [v[0] for v in ordered]
        export(args, pdf, len(ordered), concat_document_markdown(pages), pages,
               [v[1] for v in ordered], [b for v in ordered for b in v[2]], started,
               warnings=[warning for v in ordered for warning in v[3]])

    return run_page_pool(pdf_paths, concurrency=args.page_concurrency,
        page_count=page_count, should_skip=lambda pdf: should_skip(args, pdf),
        run_page=lambda pdf, page: run_page(args, client, pdf, page),
        export_document=export_document, record_error=lambda pdf, exc: record_error(args, pdf, exc),
        fail_fast=args.fail_fast)
