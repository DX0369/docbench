# Inference & Deployment Guide

HunyuanOCR-1.5 provides three self-contained inference setups under
[`../inference`](../inference). Each has its own README with the full,
validated environment recipe and usage — start there:

| Setup                                                 |       vLLM       | DFlash | transformers | CUDA        | Guide                                         |
| ----------------------------------------------------- | :--------------: | :----: | :----------: | ----------- | --------------------------------------------- |
| [`inference/vllm_0_18_1`](../inference/vllm_0_18_1)   | 0.18.1 (release) |   ❌   |      ❌      | 12.x        | [README](../inference/vllm_0_18_1/README.md)  |
| [`inference/nightly`](../inference/nightly)           |     nightly      |   ✅   |      ❌      | 13          | [README](../inference/nightly/README.md)      |
| [`inference/transformers`](../inference/transformers) |        —         |   —    |  ✅ 5.13.0   | host driver | [README](../inference/transformers/README.md) |

> ⚠️ These are **mutually exclusive environments**: vLLM (AR / DFlash) and native
> transformers inference require incompatible `transformers` versions and cannot
> coexist. See [`inference/README.md`](../inference/README.md) for the rationale.

This document only covers **performance tuning** and a couple of **advanced
notes** on top of those setups.

---

## 1. vLLM performance tuning

Key server arguments (set in the `serve*.sh` scripts; override via the documented
environment variables):

| Arg                        | Recommended             | Notes                                                              |
| -------------------------- | ----------------------- | ------------------------------------------------------------------ |
| `--gpu-memory-utilization` | `0.85`                  | Leave headroom for CUDA graph capture; DFlash's draft adds ~0.7 GB |
| `--max-model-len`          | `131072`                | Context window; lower it to save memory if your inputs are short   |
| `--max-num-batched-tokens` | `131072`                | Higher = better throughput, more memory                            |
| `--limit-mm-per-prompt`    | `{"image":4,"video":0}` | Image-only model; video disabled                                   |
| `--trust-remote-code`      | ✓                       | Required to load the HunyuanOCR model code                         |

**Multi-GPU throughput.** vLLM here uses `-tp 1` (one replica per GPU). For
higher throughput, launch one instance per GPU on separate ports
(`GPU=0 PORT=8000`, `GPU=1 PORT=8001`, …) and pass all ports to
`batch_infer.py --ports 8000,8001,...` — the batch client round-robins requests
across endpoints for near-linear scaling.

---

## 2. DFlash tuning (nightly setup only)

DFlash speculative decoding is lossless (it preserves the target model's output
distribution) and accelerates long structured outputs the most. The single knob
is `NUM_SPEC_TOKENS` (default 15, the official recommendation):

| `NUM_SPEC_TOKENS`   | Effect                                                         |
| ------------------- | -------------------------------------------------------------- |
| larger (e.g. 15)    | higher potential speedup, more per-step overhead               |
| smaller (e.g. 8–10) | less overhead; can win 5–10% when late positions rarely accept |

Inspect per-position acceptance in the server log to decide:

```bash
grep "Per-position acceptance rate" vllm_dflash_*.log | tail -5
# also look for lines like:
#   SpecDecoding metrics: Mean acceptance length: 7.36, ...
#   Avg Draft acceptance rate: 42.4%
```

If the position-15 acceptance rate is < 0.15, reducing `NUM_SPEC_TOKENS` to 8–10
usually helps; if it stays high (> 0.3), keep 15.

The DFlash server exposes the same OpenAI-compatible `/v1/chat/completions`
endpoint as the AR server, so **no client change is needed** — the clients in all
three setups are shared verbatim.

---

## 3. Advanced: multi-image requests

The shipped clients target **single-image** OCR (the validated path). For
requests with more than one image per prompt under vLLM, an extra vLLM shape-fix
patch may be required depending on your build; this is unrelated to single-image
OCR. Keep prompts to one image per request unless you specifically need
multi-image benches.

---

## 4. Benchmarking

See [`docs/benchmark.md`](benchmark.md) for the full speed comparison
(AR vs DFlash and cross-model) and a minimal reproduction script.
