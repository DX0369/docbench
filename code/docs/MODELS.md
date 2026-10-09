# GlobalDocBench 模型下载说明

本文档只记录模型下载、缓存和本地路径检查。模型统一保存到：

```text
/data/wangsq/code/global_doc_bench/models
```

代码默认直接从项目顶层 `models/` 读取权重，也可通过 `GDB_MODEL_ROOT` 覆盖。模型按明确的仓库名称和本地目录管理，目前不额外锁定模型仓库 commit。
迁移下载应保留完整snapshot（权重、config、processor/tokenizer及remote code），
不只拷贝权重文件。按同名仓库重下不承诺历史字节完全一致；本轮最终收束未改模型目录中的代码或权重。

GLM-OCR、MonkeyOCR 两版及两种 PP-DocLayoutV3 格式均直接存放在 `models/<模型名>/`，
不再使用 `models/stages/` 中间层。Monkey 模型内部组件目录保持原样；
Paddle inference 与 safetensors 格式分别保留，不能混用。`mineru/` 和框架缓存目录不变。

方法 ID、checkpoint、本地目录、源码 revision、环境和验证状态的统一映射见
[methods.json](../config/methods.json)。模型存在不代表已经完成推理验证；运行入口和验证边界见
[RUN_COMMANDS.md](RUN_COMMANDS.md) 与吞吐协议。
本文档不负责环境锁导出、吞吐测试或正式推理。

## 1. 准备下载终端

加载 Conda，并在下载终端清除代理（不修改全局配置）：

```bash
source /data/wangsq/software/miniconda/etc/profile.d/conda.sh
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export HF_ENDPOINT=https://hf-mirror.com
```

模型下载不要求独立 Conda 环境。先检查当前终端是否已有两个 CLI：

```bash
command -v modelscope
command -v hf
```

缺少 CLI 时才在选定的已有下载工具环境安装；不要为下载随意升级已验证的推理环境：

```bash
python -m pip install \
  --index-url https://repo.huaweicloud.com/repository/pypi/simple \
  modelscope "huggingface_hub[cli]"
```

设置模型和下载缓存目录：

```bash
export GDB_ROOT=/data/wangsq/code/global_doc_bench
export GDB_MODEL_ROOT="$GDB_ROOT/models"
export MODELSCOPE_CACHE=/data/wangsq/cache/modelscope
export HF_HOME=/data/wangsq/cache/huggingface
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"

mkdir -p "$GDB_MODEL_ROOT"
mkdir -p "$MODELSCOPE_CACHE"
mkdir -p "$HUGGINGFACE_HUB_CACHE"
```

Hugging Face 优先使用镜像；ModelScope 和明确的 GitHub/Surya 下载 URL 仍使用各自来源。
网络失败时保留错误并核对来源，不自动启用代理。中断后可重跑相同 CLI 命令复用已有文件。

## 2. 下载各方法独立权重

### HunyuanOCR 1.0 与 1.5

官方 `tencent/HunyuanOCR` 仓库当前根目录是 1.5，旧 1.0 位于 `v1.0/`
子目录。为了与 benchmark 的两个方法目录一致，推荐使用 Hugging Face CLI
的 include/exclude 过滤，不要让两个版本互相覆盖。

1.5 根权重（包含 `dflash/`，排除旧版）：

```bash
hf download tencent/HunyuanOCR \
  --repo-type model \
  --local-dir "$GDB_MODEL_ROOT/HunyuanOCR-1.5" \
  --exclude "v1.0/*"
```

1.0 权重需要把官方子目录内容提升为本地模型根目录：

```bash
export HUNYUAN_V1_TMP="$GDB_ROOT/tmp/HunyuanOCR-v1-repo"
mkdir -p "$HUNYUAN_V1_TMP" "$GDB_MODEL_ROOT/HunyuanOCR"

hf download tencent/HunyuanOCR \
  --repo-type model \
  --local-dir "$HUNYUAN_V1_TMP" \
  --include "v1.0/*"

rsync -a "$HUNYUAN_V1_TMP/v1.0/" "$GDB_MODEL_ROOT/HunyuanOCR/"
rm -rf "$HUNYUAN_V1_TMP"
```

最终必须同时存在：

```text
models/HunyuanOCR/config.json       # 1.0，32768 context
models/HunyuanOCR-1.5/config.json   # 1.5，131072 context
models/HunyuanOCR-1.5/dflash/
```

### DeepSeek-OCR-2

```bash
modelscope download \
  --model deepseek-ai/DeepSeek-OCR-2 \
  --local_dir "$GDB_MODEL_ROOT/DeepSeek-OCR-2"
```

### Qianfan-OCR

```bash
modelscope download \
  --model baidu-qianfan/Qianfan-OCR \
  --local_dir "$GDB_MODEL_ROOT/Qianfan-OCR"
```

### TeleOCR

GlobalDocBench 使用官方 `StarDoc-AI/TeleOCR` checkpoint，保存为与方法版本表一致的
`models/TeleOCR`；源码和 `TeleOCR-vllm` 插件由 `third_party/methods/TeleOCR` 提供。

```bash
hf download StarDoc-AI/TeleOCR \
  --repo-type model \
  --local-dir "$GDB_MODEL_ROOT/TeleOCR"
```

### Jina-OCR-v1

官方 checkpoint `jinaai/jina-ocr-v1`，当前已下载到 `models/jina-ocr-v1`。
FastMTP 复用同一 checkpoint；必须保留模型目录内官方 Python 文件，不能只复制权重。
清除代理后使用镜像与可续传下载：

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
    -u http_proxy -u https_proxy -u all_proxy \
    HF_ENDPOINT=https://hf-mirror.com \
    hf download jinaai/jina-ocr-v1 --local-dir "$GDB_ROOT/models/jina-ocr-v1"
```

当前 adapter 仅单页路线、严格 benchmark prompt；GPU 3 的同步/P1/P2/latency smoke 已通过。

### LightOnOCR-2-1B

GlobalDocBench 使用官方 `lightonai/LightOnOCR-2-1B` checkpoint：

```bash
hf download lightonai/LightOnOCR-2-1B \
  --repo-type model \
  --local-dir "$GDB_MODEL_ROOT/LightOnOCR-2-1B"
```

模型卡提供 Transformers 5.x 和 vLLM 两条官方调用方式，但没有独立的完整模型源码仓库。
当前正式 adapter 使用 vLLM 单页请求，服务参数和生成参数见
`environments/gdb_lightonocr2.md`。每个请求只包含一张页面图片，不登记整篇 PDF 的
joint 路线；P 表示跨 PDF 的页面在途数。

### Xiaomi-OCR-0

仅使用官方整页 checkpoint 和源码 helper，不使用布局权重、region 或整文档 joint。

```bash
hf download SeerRay-Lab/Xiaomi-OCR-0 \
  --repo-type model \
  --local-dir "$GDB_MODEL_ROOT/Xiaomi-OCR-0"
```

模型架构为 `Qwen3_5ForConditionalGeneration`。服务和 worker 复用 `gdb_glmocr`；
参数与安装边界见 `environments/gdb_xiaomi_ocr.md`；GPU 6 两文档 P1/P2 与 latency smoke 已通过，未进行正式调参。

### PaddleOCR-VL 1.5 与 1.6

```bash
modelscope download \
  --model PaddlePaddle/PaddleOCR-VL-1.5 \
  --local_dir "$GDB_MODEL_ROOT/PaddleOCR-VL-1.5"
```

1.6 使用独立目录和显式 `paddleocr_vl_16` profile，不能覆盖 1.5。
实际 adapter 与运行绑定见统一版本清单；下载命令：

```bash
hf download \
  --repo-type model \
  PaddlePaddle/PaddleOCR-VL-1.6 \
  --local-dir "$GDB_MODEL_ROOT/PaddleOCR-VL-1.6"
```

### GLM-OCR

GLM-OCR 主模型：

```bash
modelscope download \
  --model ZhipuAI/GLM-OCR \
  --local_dir "$GDB_MODEL_ROOT/GLM-OCR"
```

现有 `models/PP-DocLayoutV3` 是 Paddle inference 格式，可供 PaddleX 使用；GLM-OCR 的 Transformers layout detector 需要单独的 safetensors 仓库：

```bash
hf download \
  --repo-type model \
  PaddlePaddle/PP-DocLayoutV3_safetensors \
  --local-dir "$GDB_MODEL_ROOT/PP-DocLayoutV3_safetensors"
```

GLM smoke 时传入：

```text
--glm-layout-model-dir /data/wangsq/code/global_doc_bench/models/PP-DocLayoutV3_safetensors
```


Paddle inference 格式不是 GLM Transformers detector 的替代品。如其他 Paddle 路线需要
独立下载该格式，可使用 `PaddlePaddle/PP-DocLayoutV3`，或由下文 PaddleX 缓存流程获取。

### MonkeyOCR

原 benchmark 使用的 MonkeyOCR Pro 3B：

```bash
modelscope download \
  --model l1731396519/MonkeyOCR-pro-3B \
  --local_dir "$GDB_MODEL_ROOT/MonkeyOCR-pro-3B"
```

新版 MonkeyOCR Pro 1.2B。官方称其准确率、速度和效率优于此前的 3B 版本，建议作为新版方法单独测试：

```bash
modelscope download \
  --model l1731396519/MonkeyOCR-pro-1.2B \
  --local_dir "$GDB_MODEL_ROOT/MonkeyOCR-pro-1.2B"
```

### Unlimited-OCR

```bash
modelscope download \
  --model PaddlePaddle/Unlimited-OCR \
  --local_dir "$GDB_MODEL_ROOT/Unlimited-OCR"
```

### MinerU 3.0.3/2509 与 3.4.x/2605

当前锁定的 MinerU 3.4.4 tag（包报告 3.4.3）使用 `MinerU2.5-Pro-2605-1.2B`，不是旧示例中的 `2604`。

旧 MinerU 3.0.3 路线使用基础版 `MinerU2.5-2509-1.2B`：

```bash
modelscope download \
  --model OpenDataLab/MinerU2.5-2509-1.2B \
  --local_dir "$GDB_MODEL_ROOT/mineru/MinerU2.5-2509-1.2B"
```

该目录用于 `mineru_303_2509` 版本 profile，不要覆盖当前
`MinerU2.5-Pro-2605-1.2B`。

MinerU VLM：

```bash
modelscope download \
  --model OpenDataLab/MinerU2.5-Pro-2605-1.2B \
  --local_dir "$GDB_MODEL_ROOT/mineru/MinerU2.5-Pro-2605-1.2B"
```

MinerU pipeline 模型集合：

```bash
modelscope download \
  --model OpenDataLab/PDF-Extract-Kit-1.0 \
  --local_dir "$GDB_MODEL_ROOT/mineru/PDF-Extract-Kit-1.0"
```

调度器按方法 profile 自动配置本地模型路径，见下文“本地路径与检查”。
也可使用官方 `mineru-models-download --source modelscope --model_type all` 管理下载，
但不要同时维护重复权重；不同版本环境仍需选择正确的 checkpoint。

### MinerU 4.0.11 Standard

`mineru_4011_standard` 使用新版小模型包和原 2605 VLM；在独立
`gdb_mineru_4` 环境中用官方 downloader 准备，不能使用旧 3.x 下载命令：

```bash
conda activate gdb_mineru_4
type unsetproxy >/dev/null 2>&1 && unsetproxy
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export PYTHONNOUSERSITE=1
export MINERU_MODEL_BASE_DIR="$GDB_MODEL_ROOT/mineru"

mineru-kit models download --tier standard \
  --small-backend torch --vlm-engine vllm --source modelscope
mineru-kit models verify --tier standard --small-backend torch --vlm-engine vllm
```

实际目录为 `mineru/MinerU-4_models_torch` 和 `mineru/MinerU2.5-Pro-2605-1.2B`。
后者与 3.4.x 共用，不复制另一份；已有文件必须经官方 downloader 确认，
不要手工创建 `.mineru_complete`。两包已通过官方 verify，但这不证明 GPU 推理通过。
运行时自动设置 `source=local`，无需网络补模型，也不读取旧 `mineru.json`。

### Youtu-Parsing

目前未确认 Youtu-Parsing 存在可用的 ModelScope 模型仓库，使用官方 Hugging Face仓库：

```bash
hf download \
  --repo-type model \
  tencent/Youtu-Parsing \
  --local-dir "$GDB_MODEL_ROOT/Youtu-Parsing"
```

当前native主线关闭方向校正，不需要下载角度权重；旧可选下载步骤可从Git历史恢复。

在已安装的 `gdb_youtu` 中验证本地配置（下载本身不要求该环境）：

```bash
conda activate gdb_youtu
du -sh "$GDB_MODEL_ROOT/Youtu-Parsing"

HF_HUB_OFFLINE=1 python -c '
from transformers import AutoConfig, AutoProcessor
path = "/data/wangsq/code/global_doc_bench/models/Youtu-Parsing"
print(AutoConfig.from_pretrained(path, trust_remote_code=True).__class__.__name__)
print(AutoProcessor.from_pretrained(path, trust_remote_code=True).__class__.__name__)
'
```

## 3. 可选的框架组件缓存

以下步骤不是独立模型下载的必需步骤。只有需要把运行时组件预取到项目目录时才执行；
不执行也不会改变模型目录结构。

独立权重下载完成后，在对应已安装环境中预取运行时组件，先区分下载失败与推理失败。
以下变量只设置在对应方法进程/终端，不全局污染其他环境。

### Marker/Surya 权重

当前 Surya 0.14.7 的模型不是从 Hugging Face 下载，而是通过
`https://models.datalab.to` 的 manifest 下载。为避免权重散落在用户缓存目录，
本项目显式将其保存到：

```text
/data/wangsq/code/global_doc_bench/models/marker-surya
```

使用内部下载器只下载文件、不初始化 GPU 模型：

```bash
conda activate gdb_marker

export MODEL_CACHE_DIR=/data/wangsq/code/global_doc_bench/models/marker-surya
mkdir -p "$MODEL_CACHE_DIR"

python - <<'PY'
from pathlib import Path

from surya.common.s3 import download_directory
from surya.settings import settings

checkpoints = [
    settings.DETECTOR_MODEL_CHECKPOINT,
    settings.RECOGNITION_MODEL_CHECKPOINT,
    settings.LAYOUT_MODEL_CHECKPOINT,
    settings.TABLE_REC_MODEL_CHECKPOINT,
    settings.OCR_ERROR_MODEL_CHECKPOINT,
]

for checkpoint in checkpoints:
    relative = checkpoint.removeprefix("s3://")
    local_dir = Path(settings.MODEL_CACHE_DIR) / relative
    local_dir.mkdir(parents=True, exist_ok=True)
    print(f"[download] {checkpoint} -> {local_dir}")
    download_directory(relative, str(local_dir))

print("Marker/Surya model cache:", settings.MODEL_CACHE_DIR)
PY
```

Surya 的调试渲染功能还可能按需下载字体，但 Marker 的 Markdown 转换 smoke
不依赖该字体，因此这里不额外下载。也不要设置 Surya 0.14.7 不识别的
`FONT_PATH` 环境变量。

验证所有 manifest 文件是否齐全：

```bash
python - <<'PY'
from pathlib import Path
import json

root = Path("/data/wangsq/code/global_doc_bench/models/marker-surya")
for manifest in sorted(root.rglob("manifest.json")):
    payload = json.loads(manifest.read_text())
    missing = [name for name in payload["files"] if not (manifest.parent / name).exists()]
    print(manifest.parent, "OK" if not missing else f"MISSING {missing}")
PY
```

后续 Marker smoke 和正式运行也必须导出相同的 `MODEL_CACHE_DIR`，否则 Surya
会回到默认用户缓存目录并重新下载。

### PaddleX/PP-StructureV3/PaddleOCR-VL 组件

PaddleX 当前没有稳定的“只下载不初始化”CLI。使用 CPU 初始化官方 pipeline 来触发模型解析和缓存，避免占用 GPU：

```bash
conda activate gdb_paddle

export PADDLE_PDX_CACHE_HOME=/data/wangsq/code/global_doc_bench/models/paddlex
export PADDLE_PDX_MODEL_SOURCE=modelscope
export PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True

mkdir -p "$PADDLE_PDX_CACHE_HOME"
```

预取PP主线实际启用的组件，包括constructor延迟加载的表格方向和辅助OCR。
仅初始化、不解析PDF；不启用默认关闭的图表模型，不把单纯constructor成功当全部模型齐备：

```bash
PYTHONPATH="$GDB_ROOT/code/src" PYTHONNOUSERSITE=1 python - <<'PY'
from argparse import Namespace
from methods.pp_structurev3.pipeline import load_ppstructure_runtime

runtime = load_ppstructure_runtime(Namespace(device="cpu", lang=None, render_dpi=200, batch_scale=1))
print("PP mainline components initialized:", runtime["effective_batches"])
PY
```

预取 PaddleOCR-VL 1.5 客户端 pipeline 的非 VLM 组件。这里的服务 URL 只是占位，初始化阶段不应发送识别请求：

```bash
python - <<'PY'
from paddleocr import PaddleOCRVL

pipeline = PaddleOCRVL(
    pipeline_version="v1.5",
    device="cpu",
    vl_rec_backend="vllm-server",
    vl_rec_server_url="http://127.0.0.1:1/v1",
    vl_rec_api_model_name="PaddlePaddle/PaddleOCR-VL-1.5",
    vl_rec_max_concurrency=1,
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_queues=False,
)
print("PaddleOCR-VL pipeline initialized:", type(pipeline).__name__)
PY
```

验证缓存不为空：

```bash
du -sh "$PADDLE_PDX_CACHE_HOME"
find "$PADDLE_PDX_CACHE_HOME" -type f | wc -l
```

后续 smoke 和正式运行必须继续导出同一个 `PADDLE_PDX_CACHE_HOME`，否则 PaddleX 会换到默认目录并重新下载。

## 4. 本地路径与下载检查

benchmark 默认从项目顶层 `models/` 读取模型，不再需要创建 `code/external/models` 符号链接。验证默认解析结果：

```bash
cd /data/wangsq/code/global_doc_bench/code
PYTHONPATH=src python -c 'from paths import MODEL_ROOT; print(MODEL_ROOT)'
```

需要使用其他模型根目录时，显式设置：

```bash
export GDB_MODEL_ROOT=/path/to/models
```

### MinerU 调度器自动配置

`src/methods/mineru/version_profile.py` 按两个显式方法 ID 在
`tmp/runtime/mineru/<method_id>.json` 写入 pipeline/VLM 路径，并设置
`MINERU_TOOLS_CONFIG_JSON`、`MINERU_MODEL_SOURCE=local`。使用项目调度器无需
手写配置；不要以全局旧配置覆盖某个版本的模型选择。
旧手工配置已保留到 `tmp/runtime/mineru_manual/`，不作为调度器默认配置。
Marker、PP-StructureV3 和 Youtu 的可复用运行模型缓存默认位于 `models/runtime_cache/<method>/`，
不随 smoke 清理。实际模型路径和已有环境覆盖仍优先。




### 下载后检查

查看模型目录大小：

```bash
du -sh "$GDB_MODEL_ROOT"/*
du -sh "$GDB_MODEL_ROOT"/mineru/*
```

检查主要模型配置：

```bash
find "$GDB_MODEL_ROOT" \
  -maxdepth 3 \
  -name config.json \
  -print
```

ModelScope 或 Hugging Face下载被中断后，可以重新执行相同命令。下载工具通常会检查并复用已经完成的文件。


方法版本审计和 GPU 推理不在本文档执行；需要时参考
[methods.json](../config/methods.json) 和 [运行命令](RUN_COMMANDS.md)。
