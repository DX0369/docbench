# Speed Benchmark

## Setup

- **Hardware**: single NVIDIA H20 (80 GB) per server
- **Dataset**: OmnidocBench, document / PPT / book / textbook images
- **Config**: concurrency=1 (single-request latency), `max_tokens=8000`, `temperature=0.0`
- **Metric**: end-to-end latency from HTTP request to complete response

## HunyuanOCR AR vs DFlash

The core comparison — same base model, different inference paths:

| Config                                          |    wall time | avg latency |    page/s |   speedup |
| ----------------------------------------------- | -----------: | ----------: | --------: | --------: |
| **HunyuanOCR base (AR)**                        |     2821.9 s |     3.032 s |     0.330 |     1.00× |
| **HunyuanOCR + DFlash (14.7k packs, finetune)** | **1316.5 s** | **1.408 s** | **0.706** | **2.14×** |

**Key results:**

- DFlash achieves **2.14× end-to-end speedup** vs AR baseline
- Speculative decoding is essentially lossless: total-tokens difference < 0.15% vs AR

## Cross-Model Comparison

Comparison with other open-source OCR VLMs, all under identical eval conditions (c=1, `max_tokens=8000`):

| Rank | Model                       | Latency (s/img) |    Page/s | vs HunyuanOCR AR | Notes                          |
| :--: | --------------------------- | --------------: | --------: | ---------------: | ------------------------------ |
|  🥇  | **HunyuanOCR + DFlash**     |        **1.41** | **0.706** |        **2.14×** | Speculative decoding           |
|  🥈  | GLM-OCR (SDK page pipeline) |            1.65 |     0.604 |            1.83× | Layout + region OCR concurrent |
|  🥉  | PaddleOCR-VL 1.6 (0.9B)     |            1.74 |     0.562 |            1.71× | 0.9B small model + two-stage   |
|  4   | HunyuanOCR base (AR)        |            3.03 |     0.330 |            1.00× | Baseline                       |
|  5   | Unlimited-OCR (SGLang)      |            3.66 |     0.255 |            0.77× | image tiling (gundam mode)     |
|  6   | DeepSeek-OCR-2              |            5.46 |     0.179 |            0.54× | Large model + grounding        |
|  7   | dots.ocr                    |            7.15 |     0.136 |            0.41× | Slow but structured            |

## Reproducing These Numbers

1. Deploy your desired inference server (see `docs/inference.md`)
2. Run a compatible benchmark script — a minimal example:

```python
import time, base64, glob, json, os
from openai import OpenAI

# ← configure these
IMAGE_DIR = "/path/to/eval/images"
SAMPLE_LIST = "./speed_eval_set_930.txt"
SERVER_URL = "http://127.0.0.1:8001/v1"
MODEL_NAME = "/path/to/HunyuanOCR/base"
PROMPT = "提取文档图片中正文的所有信息用markdown格式表示，其中页眉、页脚部分忽略，表格用html格式表达，文档中公式用latex格式表示，按照阅读顺序组织进行解析。"

# collect images
want = set(l.strip() for l in open(SAMPLE_LIST))
imgs = sorted(p for p in glob.glob(f"{IMAGE_DIR}/*") if os.path.basename(p) in want)

# encode all to base64 upfront (avoids I/O in timing loop)
items = []
for p in imgs:
    with open(p, "rb") as f:
        items.append(base64.b64encode(f.read()).decode())

client = OpenAI(api_key="EMPTY", base_url=SERVER_URL)
os.environ.pop("http_proxy", None); os.environ.pop("https_proxy", None)

t0 = time.time()
latencies, tokens = [], []
for b64 in items:
    t = time.time()
    r = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                {"type": "text", "text": PROMPT},
            ],
        }],
        temperature=0.0, max_tokens=8000, extra_body={"top_k": 1},
    )
    latencies.append(time.time() - t)
    tokens.append(r.usage.completion_tokens)

wall = time.time() - t0
n = len(latencies)
print(json.dumps({
    "n": n,
    "wall_s": round(wall, 3),
    "avg_latency_s": round(sum(latencies)/n, 3),
    "page_per_s": round(n/wall, 4),
    "token_per_s": round(sum(tokens)/wall, 2),
}))
```

3. Compare `page_per_s` — this is the truly fair cross-model metric.

## Fairness Notes

- **Fair**: `wall time`, `avg_latency`, `page_per_s` — comparable across all models
- **NOT fair**: `token_per_s`, `total_tokens` — different tokenizers count differently
- Each model uses its native prompt (forcing a common prompt breaks output quality)
- All models tested on identical H20 (80GB) hardware
