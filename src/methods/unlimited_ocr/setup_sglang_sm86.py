#!/usr/bin/env python3
"""Patch Unlimited-OCR's pinned SGLang wheel for RTX 3090 JIT compatibility."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


ROPE_PATCH_MARKER = "GDB RTX 3090 compatibility: use the packaged CUDA RoPE kernel"
ROPE_ORIGINAL = """        else:
            self.use_fallback_kernel = False
"""
ROPE_REPLACEMENT = f"""        else:
            # {ROPE_PATCH_MARKER}. The TVM-FFI JIT RoPE shipped by the
            # Unlimited-OCR SGLang wheel raises CUDA_ERROR_INVALID_HANDLE on
            # RTX 3090 (SM86), while sgl-kernel's packaged implementation is
            # numerically equivalent and supports this architecture.
            self.use_fallback_kernel = _is_cuda
            if _is_cuda:
                from sgl_kernel import rotary_embedding

                self.fallback_rotary_embedding = rotary_embedding
"""
KVCACHE_PATCH_MARKER = "GDB RTX 3090 compatibility: skip the TVM-FFI KV-cache kernel"
KVCACHE_ORIGINAL = """    if (_is_cuda or _is_hip) and same_kv_dim and can_use_store_cache(row_bytes):
        return store_cache(
"""
KVCACHE_REPLACEMENT = f"""    # {KVCACHE_PATCH_MARKER}, which loads on SM86 but
    # fails at launch with CUDA_ERROR_INVALID_HANDLE. Keep the existing indexed
    # assignment fallback on CUDA; the HIP path is unchanged.
    if _is_hip and same_kv_dim and can_use_store_cache(row_bytes):
        return store_cache(
"""
CLAMP_PATCH_MARKER = "GDB RTX 3090 compatibility: use native CUDA clamp_position"
CLAMP_ORIGINAL = """if is_cuda() or is_hip():
    from sglang.jit_kernel.clamp_position import clamp_position_cuda

    clamp_position = clamp_position_cuda
else:
    clamp_position = _clamp_position_native
"""
CLAMP_REPLACEMENT = f"""# {CLAMP_PATCH_MARKER}. The TVM-FFI implementation fails
# at launch on SM86; the existing native implementation has identical semantics.
if is_hip():
    from sglang.jit_kernel.clamp_position import clamp_position_cuda

    clamp_position = clamp_position_cuda
else:
    clamp_position = _clamp_position_native
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--python",
        type=Path,
        required=True,
        help="Python executable from the gdb_unlimited_ocr_sglang environment.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    python = args.python.resolve()
    purelib = subprocess.check_output(
        [
            str(python),
            "-c",
            "import sysconfig; print(sysconfig.get_paths()['purelib'])",
        ],
        text=True,
    ).strip()
    rope_target = (
        Path(purelib)
        / "sglang"
        / "srt"
        / "layers"
        / "rotary_embedding"
        / "base.py"
    )
    kvcache_target = Path(purelib) / "sglang" / "srt" / "mem_cache" / "memory_pool.py"
    clamp_target = (
        Path(purelib) / "sglang" / "srt" / "model_executor" / "forward_batch_info.py"
    )
    patch_file(
        rope_target,
        marker=ROPE_PATCH_MARKER,
        original=ROPE_ORIGINAL,
        replacement=ROPE_REPLACEMENT,
    )
    patch_file(
        kvcache_target,
        marker=KVCACHE_PATCH_MARKER,
        original=KVCACHE_ORIGINAL,
        replacement=KVCACHE_REPLACEMENT,
    )
    patch_file(
        clamp_target,
        marker=CLAMP_PATCH_MARKER,
        original=CLAMP_ORIGINAL,
        replacement=CLAMP_REPLACEMENT,
    )


def patch_file(target: Path, *, marker: str, original: str, replacement: str) -> None:
    text = target.read_text(encoding="utf-8")
    if marker in text:
        print(f"Already patched: {target}")
        return
    occurrences = text.count(original)
    if occurrences != 1:
        raise RuntimeError(
            f"Expected exactly one pinned SGLang block in {target}, found {occurrences}."
        )
    target.write_text(text.replace(original, replacement), encoding="utf-8")
    print(f"Patched: {target}")


if __name__ == "__main__":
    main()
