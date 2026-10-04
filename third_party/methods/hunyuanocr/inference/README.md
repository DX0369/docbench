# HunyuanOCR-1.5 — Local Deployment & Inference

Three parallel deployment setups. Pick one based on your **hardware**, whether
you need **DFlash speculative-decoding acceleration**, and whether you need
**native transformers inference**.

All three share the same weights and the same sampling / post-processing
pipeline (task-type prompts + `repetition_penalty=1.08` + tail-repetition
early-stop + markdown normalization), so their outputs are directly comparable.

---

## Which one should I use?

| Your situation                                   | Use                               | vLLM AR | DFlash accel. | transformers |
| ------------------------------------------------ | --------------------------------- | :-----: | :-----------: | :----------: |
| CUDA 12, want the simplest setup                 | [`vllm_0_18_1/`](./vllm_0_18_1)   |   ✅    |      ❌       |      ❌      |
| Want DFlash speculative decoding (needs CUDA 13) | [`nightly/`](./nightly)           |   ✅    |      ✅       |      ❌      |
| Want native HuggingFace transformers inference   | [`transformers/`](./transformers) |    —    |       —       |      ✅      |

Each subfolder ships its own README and requirements — just follow it.

---

## Why three separate environments?

The key constraints are mutually exclusive, so a single environment cannot
satisfy all of them. This is a validated conclusion, not a preference:

|                   | `vllm_0_18_1`          | `nightly`                     | `transformers`                    |
| ----------------- | ---------------------- | ----------------------------- | --------------------------------- |
| vLLM              | **0.18.1** (release)   | **nightly** (0.23.1rc1)       | not used                          |
| transformers      | 4.57.6                 | 5.5.3 (must pin)              | **5.13.0**                        |
| CUDA              | 12.x (native, one pip) | 13 (torch cu130 + compat lib) | matches host driver (cu128/cu130) |
| Python            | 3.10                   | 3.12                          | 3.12                              |
| Runs AR           | ✅                     | ✅                            | ✅ (HF generate)                  |
| Runs DFlash       | ❌                     | ✅                            | ❌                                |
| Runs transformers | ❌                     | ❌                            | ✅                                |

1. **DFlash is registered only in vLLM nightly (cu130)** → DFlash requires
   nightly + CUDA 13; the 0.18.1 release has no `dflash` method.
2. **Native transformers inference needs the `hunyuan_vl` module → transformers
   ≥ 5.13.0**. vLLM 0.18.1 caps `transformers < 5`, and the nightly path must
   pin transformers back to 5.5.3 (see below) — neither ships `hunyuan_vl`.
3. **transformers 5.13.0 breaks vLLM's HunYuanVL loading** (`AutoImageProcessor.register`
   is called with a string; the 5.13 signature change raises `AttributeError`).
   Installing vLLM nightly pulls transformers 5.13.0 by default, so the nightly
   setup must explicitly downgrade to 5.5.3; conversely, a 5.13.0 environment
   can only do transformers, not vLLM.

**Bottom line:** vLLM inference (AR / DFlash) and native transformers inference
require two different transformers versions and cannot coexist in one
environment. For a three-way comparison, run both vLLM paths from `nightly/` and
transformers from `transformers/` — same kernel, same weights, same sampling, so
the comparison is methodologically sound.

---

## Common steps (identical across all three)

**Download the model weights** (each subfolder README repeats this):

```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli download tencent/HunyuanOCR --local-dir ./HunyuanOCR --exclude "v1.0/*"
```

The download contains both the **base model** and the **`dflash/` draft model**.

**Task types / prompts:** every setup selects the official recommended prompt via
`--task-type` (free-form prompt rewriting is intentionally not exposed, to avoid
degrading quality). There are 12 task types — see the "Task types" section in any
subfolder README.

**Layout**

```
inference/
├── README.md          # this file (overview + selection guide)
├── vllm_0_18_1/       # Setup A: vLLM 0.18.1 release, CUDA 12, AR only
├── nightly/           # Setup B: vLLM nightly, CUDA 13, AR + DFlash (draft config/code bundled; weight from HF)
└── transformers/      # Setup C: HF transformers 5.13.0, multi-GPU direct inference
```
