# PC-side deployment with llama.cpp

HunyuanOCR-1.5 can be deployed on **CPU / consumer GPU / laptop** via
[`llama.cpp`](https://github.com/ggml-org/llama.cpp), converting the base model
(and optionally the DFlash draft) to the GGUF format and serving via
`llama-server` (OpenAI-compatible).

Two versions are supported:

| Version                   | Repo                                                                                                                                                    | Use when                                                                 |
| :------------------------ | :------------------------------------------------------------------------------------------------------------------------------------------------------ | :----------------------------------------------------------------------- |
| **Community (no DFlash)** | Upstream `ggml-org/llama.cpp` (main branch)                                                                                                             | You only need HunyuanOCR base — simplest, most stable.                   |
| **DFlash-adapted**        | Fork [`wendadawen/llama.cpp @ dflash-adapt-hunyuanocr-hunyuanstyle`](https://github.com/wendadawen/llama.cpp/tree/dflash-adapt-hunyuanocr-hunyuanstyle) | You want end-to-end acceleration with DFlash speculative decoding on PC. |

> ⚠️ Speculative decoding support in upstream `llama.cpp` is limited, DFlash has
> **not** been merged, and there are still known bugs. The fork above is our
> HunyuanOCR-specific DFlash port and is **not** the community version.

---

## 1. Community version (HunyuanOCR base, no DFlash)

### 1.1 Clone & build llama.cpp

```bash
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp

# Add -DGGML_CUDA=ON if you have an NVIDIA GPU and want CUDA acceleration.
cmake -B build -DLLAMA_BUILD_EXAMPLES=ON
cmake --build ./build --config Release -j
```

### 1.2 Set up a Python env for weight conversion

```bash
uv venv --python 3.12 venv-llamacpp
source venv-llamacpp/bin/activate
uv pip install huggingface_hub transformers torch openai
```

### 1.3 Download HunyuanOCR weights and convert to GGUF

```bash
hf download tencent/HunyuanOCR --local-dir ./HunyuanOCR

# Language / decoder weights → hyocr-f16.gguf
python3 convert_hf_to_gguf.py \
    --outfile ./HunyuanOCR/hyocr-f16.gguf \
    --outtype f16 \
    ./HunyuanOCR

# Vision (mmproj) weights → mmproj-hyocr-f16.gguf
python3 convert_hf_to_gguf.py \
    --outfile ./HunyuanOCR/mmproj-hyocr-f16.gguf \
    --outtype f16 \
    --mmproj \
    ./HunyuanOCR
```

### 1.4 Launch the OpenAI-compatible server

```bash
build/bin/llama-server \
    --model  "./HunyuanOCR/hyocr-f16.gguf" \
    --mmproj "./HunyuanOCR/mmproj-hyocr-f16.gguf" \
    --host 0.0.0.0 --port 8080 --alias HYVL \
    --ctx-size 10240 --n-predict 4096
```

The endpoint is `http://<host>:8080/v1/chat/completions`, alias `HYVL`.

---

## 2. DFlash-adapted version (HunyuanOCR + DFlash speculative decoding)

### 2.1 Clone & build the DFlash fork

```bash
git clone -b dflash-adapt-hunyuanocr-hunyuanstyle \
    https://github.com/wendadawen/llama.cpp.git
cd llama.cpp

cmake -B build -DLLAMA_BUILD_EXAMPLES=ON
cmake --build ./build --config Release -j
```

Weight download & base-model / mmproj GGUF conversion are identical to the
community version above — see steps 1.2 and 1.3.

### 2.2 Convert the DFlash draft weights to GGUF

`--target-model-dir` points to the base HunyuanOCR HF checkpoint (needed for
tokenizer / config), while the positional argument points to the DFlash
checkpoint directory.

```bash
python3 convert_hf_to_gguf.py \
    --outfile ./HunyuanOCR-Dflash/hyocr-dflash-bf16.gguf \
    --outtype bf16 \
    --target-model-dir ./HunyuanOCR \
    ./HunyuanOCR-Dflash
```

### 2.3 Launch the OpenAI-compatible server with DFlash

```bash
build/bin/llama-server \
    --model       "./HunyuanOCR/hyocr-f16.gguf" \
    --mmproj      "./HunyuanOCR/mmproj-hyocr-f16.gguf" \
    --model-draft "./HunyuanOCR-Dflash/hyocr-dflash-bf16.gguf" \
    --dflash --draft-max 16 \
    --host 0.0.0.0 --port 8080 --alias HYVL \
    --ctx-size 10240 --n-predict 4096 \
    --parallel 1 \
    --ubatch-size 8192 \
    --batch-size  8192
```

Key DFlash-specific flags:

| Flag                                | Meaning                                                     |
| :---------------------------------- | :---------------------------------------------------------- |
| `--model-draft <path>`              | GGUF path of the DFlash draft model                         |
| `--dflash`                          | Enable DFlash-style speculative decoding                    |
| `--draft-max 16`                    | Number of draft tokens per speculative step (K)             |
| `--parallel 1`                      | Single serial slot (recommended for DFlash on PC)           |
| `--ubatch-size / --batch-size 8192` | Large batch to keep the target model well fed during verify |

---

## 3. Quick verification

We ship a minimal OpenAI-compatible client and 26 test OCR images under
[`llama_cpp/`](../llama_cpp) so you can smoke-test the deployment end-to-end.

### 3.1 Install the client dep

```bash
pip install openai
```

### 3.2 Run

```bash
cd llama_cpp
python chat.py
```

By default `chat.py` targets `http://127.0.0.1:8080/v1` with alias `HYVL`
(matching the `llama-server` launch commands above), reads
`test_assets/data.jsonl`, sends the first `ocr` sample to the server, prints
the response and per-item elapsed time, and tees everything into
`logs/chat_<timestamp>.log`.

Tune the client behavior at the top of `chat.py`:

```python
BASE_URL     = "http://127.0.0.1:8080/v1"
MODEL        = "HYVL"
MAX_REQUESTS = 10                # cap total requests
TYPE_LIMITS  = {"ocr": 1}        # per-type cap; set None to disable
```

### 3.3 Example output

```
=== [ocr] ocr/0.png ===
Prompt: 请提取文档图片中正文的所有信息用 markdown 格式表示。
ring, and Jacobson semisimple, by Corollary 8.35(ii)]. The factor module
$ J^{q}/J^{q+1} $ is an $ (R/J) $-module; hence, by Corollary 8.43,
$ J^{q}/J^{q+1} $ is a semisimple module, and so it can be decomposed into a
direct sum of (possibly infinitely many) simple $ (R/J) $-modules. ...
...
[elapsed] 6.885s

[total] 1 items (ocr=1), elapsed: 6.885s
```

If you see the response streamed back and a `[total]` line at the end, the
llama.cpp deployment (± DFlash) is working correctly.
