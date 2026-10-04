---
license: other
license_name: tencent-hunyuan-community
license_link: https://huggingface.co/tencent/HunyuanOCR/blob/main/LICENSE
tags:
  - ocr
  - speculative-decoding
  - draft-model
  - dflash
  - block-diffusion
  - vision-language-model
base_model: tencent/HunyuanOCR
library_name: transformers
---

# HunyuanOCR-1.5 · DFlash Draft &nbsp;·&nbsp; Preview

<div align="center">

**Speculative-decoding draft for [`tencent/HunyuanOCR`](https://huggingface.co/tencent/HunyuanOCR)**

</div>

> ⚠️ **This model is not usable standalone.** It is a _draft model_ used only
> for **speculative decoding** together with the target model
> [`tencent/HunyuanOCR`](https://huggingface.co/tencent/HunyuanOCR).

---

## 📖 What is DFlash?

End-to-end OCR is often accompanied by long autoregressive decoding — the major
bottleneck for dense documents, tables, formulas, and other long structured
outputs.

HunyuanOCR-1.5 adopts a speculative-decoding framework based on **DFlash**:

- A lightweight **block-diffusion** draft model (this repo) proposes multiple
  candidate tokens **in parallel**.
- The target model
  ([`tencent/HunyuanOCR`](https://huggingface.co/tencent/HunyuanOCR))
  verifies them in a **single forward pass**.
- Accepted tokens are committed as-is, so the **output distribution of the
  target model is preserved** — DFlash is a lossless acceleration.

The result is significantly reduced decoding latency for long structured OCR
outputs, without sacrificing accuracy.

Architecture: 5-layer Qwen3-style block-diffusion draft, predicting 16 masked tokens in a single block. The draft is bound to
target-layer indices `[1, 8, 15, 22]` of the 24-layer HunyuanOCR-1.5 base.

---

## ⚙️ Environment

- Python 3.10+
- PyTorch 2.1+ (CUDA 12.1+)
- **transformers**
- **vLLM nightly** — required for real speculative-decoding speedup at
  deployment time. DFlash support is included in the nightly wheel; no separate
  patch is needed.

```bash
uv pip install -U vllm \
    --torch-backend=cu130 \
    --extra-index-url https://wheels.vllm.ai/nightly
uv pip install runai-model-streamer
```

> 💡 On CUDA 12.x, replace `--torch-backend=cu130` with the matching tag
> (e.g. `cu121`, `cu124`).

---

## 🚀 How to use

The GitHub develop branch ships the draft **config and custom code** at
[`inference/nightly/hyocr_dflash/`](https://github.com/Tencent-Hunyuan/HunyuanOCR/tree/develop/inference/nightly)
(`config.json`, `dflash.py`, tokenizer files). The **weight file
(`model.safetensors`) is not committed to Git** — download it from the
`dflash/` subfolder of `tencent/HunyuanOCR` on HuggingFace into this folder:

```bash
git clone -b develop https://github.com/Tencent-Hunyuan/HunyuanOCR.git
cd HunyuanOCR/inference/nightly

# pull the draft weight from HuggingFace into hyocr_dflash/
python -c "from huggingface_hub import hf_hub_download; import shutil; \
shutil.copy(hf_hub_download('tencent/HunyuanOCR', 'dflash/model.safetensors'), \
'./hyocr_dflash/model.safetensors')"
```

`serve_dflash.sh` uses `hyocr_dflash/` as the default `DFLASH_PATH` once the
weight is in place.

> ⬇️ **Alternative — download the full draft into a flat directory.** vLLM's
> `--speculative-config` and `trust_remote_code` custom-code loading do not
> support HF subfolders, so pull the whole `dflash/` folder into a **flat local
> directory** and point `DFLASH_PATH` at it:
>
> ```bash
> python -c "from huggingface_hub import snapshot_download; import shutil, os; \
> d=snapshot_download('tencent/HunyuanOCR', allow_patterns=['dflash/*']); \
> shutil.copytree(os.path.join(d,'dflash'), './hyocr_dflash', dirs_exist_ok=True)"
> ```

### vLLM speculative decoding

From the `inference/nightly/` folder (with the draft weight downloaded into
`hyocr_dflash/`, which is the default `DFLASH_PATH`):

```bash
MODEL_PATH=./HunyuanOCR \
GPU=0 PORT=8000 NUM_SPEC_TOKENS=15 \
bash serve_dflash.sh
```

Under the hood the launch script passes the following to the vLLM entrypoint:

```
--speculative-config '{"method":"dflash","model":"<DFLASH_PATH>","num_speculative_tokens":15}'
```

Send an OpenAI-compatible request with the shipped single-image client:

```bash
python infer_vllm_client.py \
    --host 127.0.0.1 --port 8000 \
    --model tencent/HunyuanOCR \
    --image /path/to/document.png --task-type doc_parse
```

### llama.cpp (PC-side)

A DFlash-adapted `llama.cpp` fork is provided for CPU / consumer-GPU / laptop
speculative decoding. See `docs/llama_cpp.md` in the GitHub repo for the full
guide (GGUF conversion of both target + draft, `llama-server` launch, and a
smoke-test client).

---

## 📦 Files in this repo

| file                                                                                      | purpose                                                                 |
| ----------------------------------------------------------------------------------------- | ----------------------------------------------------------------------- |
| `model.safetensors`                                                                       | draft weights (float32)                                                 |
| `config.json`                                                                             | draft config; sets `auto_map` to `dflash.DFlashDraftModel`              |
| `dflash.py`                                                                               | `DFlashDraftModel` implementation (loaded via `trust_remote_code=True`) |
| `chat_template.jinja`, `tokenizer.json`, `tokenizer_config.json`, `processor_config.json` | tokenizer / processor, kept in sync with the target model               |

---

## 🔗 Related repositories

- **Target model** (required):
  [`tencent/HunyuanOCR`](https://huggingface.co/tencent/HunyuanOCR)
- **GitHub — training & inference toolkit** (branch `develop`):
  <https://github.com/Tencent-Hunyuan/HunyuanOCR>
- **HunyuanOCR-1.0** (previous generation, archived under `v1.0/`):
  [`tencent/HunyuanOCR/v1.0`](https://huggingface.co/tencent/HunyuanOCR/tree/main/v1.0)

---

## 📜 License

HunyuanOCR-1.5 (including the DFlash draft) is released under the same license as HunyuanOCR-1.0 — the **Tencent Hunyuan Community License Agreement**.
