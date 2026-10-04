#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys

PATCH_QWEN35_WARMUP_ENV = "GDB_PATCH_QWEN35_WARMUP"


def maybe_patch_qwen35_warmup(enable_patch: bool) -> None:
    if not enable_patch:
        return

    import vllm.model_executor.models.qwen3_next as qwen3_next

    def _skip_warmup_prefill_kernels(self, mixed_qkv):
        # vLLM 0.18.0 may spend an excessive amount of time autotuning
        # Qwen3.5's GDN prefill kernels during engine startup on RTX 3090.
        # Skipping this warmup keeps startup bounded for our single-GPU
        # benchmarking path; the model can still serve real requests after
        # initialization.
        self._prefill_kernels_warmed_up = True
        return None

    qwen3_next.Qwen3NextGatedDeltaNet._warmup_prefill_kernels = (
        _skip_warmup_prefill_kernels
    )


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Thin wrapper around `vllm serve` with optional model-specific patches."
    )
    parser.add_argument(
        "--patch-qwen35-warmup",
        action="store_true",
        help="Skip Qwen3.5 GDN prefill warmup during vLLM engine startup.",
    )
    return parser.parse_known_args()


def main() -> None:
    args, remaining = parse_args()
    if args.patch_qwen35_warmup:
        os.environ[PATCH_QWEN35_WARMUP_ENV] = "1"
    maybe_patch_qwen35_warmup(args.patch_qwen35_warmup)

    sys.argv = ["vllm", "serve"] + remaining
    from vllm.entrypoints.cli.main import main as vllm_main

    vllm_main()


if os.environ.get(PATCH_QWEN35_WARMUP_ENV) == "1":
    maybe_patch_qwen35_warmup(True)


if __name__ == "__main__":
    main()
