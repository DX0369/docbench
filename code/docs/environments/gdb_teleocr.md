# `gdb_teleocr`

## 用途和版本依据

该环境只运行 TeleOCR 的正式异步路线：`vllm-async-engine`。
同步 `vllm-engine` 不登记为正式 benchmark profile。

- 源码目录：`third_party/methods/TeleOCR`；
- 源码 revision：以 `code/config/methods.json` 中的 `teleocr` 仓库记录为准；
- 模型目录：`models/TeleOCR`；
- Conda 环境：`gdb_teleocr`；
- Python：3.10；
- 正式并发参数：TeleOCR runtime 的 `max_concurrency`。
- 正式 benchmark method：`teleocr_vllm_async`；P 只覆盖当前 PDF 内共享的 layout/region 请求信号量，文档之间仍按官方顺序处理。

TeleOCR 仓库同时提供了 `pyproject.toml`、`requirements.txt` 和
`TeleOCR-vllm/pyproject.toml`。其核心版本以仓库文件及安装后的环境锁为准，当前目标组合为
Torch 2.8.0、Transformers 4.57.1、vLLM 0.11.0 和 xformers 0.0.32。

仓库的完整 `requirements.txt` 还包含可选 Gradio server 依赖。当前 benchmark 不使用
Gradio UI；同时 `gradio==5.49.1` 与 `pydantic==2.13.4` 存在元数据冲突，因此安装时只
排除 Gradio 及其 UI 相关包，不修改第三方仓库文件。

## 创建和安装

以下命令保留最小可复现安装流程。包源和代理由安装终端自行配置，不在方法配方中写死。

```bash
conda create -n gdb_teleocr python=3.10 -y
conda activate gdb_teleocr

export PYTHONNOUSERSITE=1
export TELEOCR_ROOT=/data/wangsq/code/global_doc_bench/third_party/methods/TeleOCR
export PY="$CONDA_PREFIX/bin/python"

grep -Ev '^(gradio|gradio-client|gradio-pdf|fastapi-cloud-cli)(==|$)' \
  "$TELEOCR_ROOT/requirements.txt" \
  | "$PY" -m pip install \
      --no-deps \
      -r /dev/stdin

"$PY" -m pip install \
  --no-deps \
  -e "$TELEOCR_ROOT"
```

`requirements.txt` 已经是冻结列表，因此使用 `--no-deps` 避免 pip 重新解析并改变版本；
editable 安装用于注册 `TeleOCR` 和 `TeleOCR_vllm` 包及其 vLLM plugin。

## 安装后核对

```bash
PYTHONNOUSERSITE=1 "$PY" -m pip check

PYTHONNOUSERSITE=1 "$PY" - <<'PY'
import torch
import transformers
import vllm
import TeleOCR
import TeleOCR_vllm
from importlib.metadata import version

print("torch:", version("torch"))
print("transformers:", version("transformers"))
print("vllm:", version("vllm"))
print("TeleOCR import: OK")
print("TeleOCR_vllm plugin import: OK")
print("CUDA available:", torch.cuda.is_available())
PY
```

完成安装后的实际版本以该核对命令和 `env/locks_rtx3090_*/gdb_teleocr.*` 为准。
环境锁应在 `pip check`、import probe 和最小 GPU smoke 均通过后再导出。

## 运行边界

GlobalDocBench 只接入 `teleocr_vllm_async`。TeleOCR 的官方 async 路线会在一个常驻
runtime 中并发处理当前 PDF 的页面和区域请求；它不是整篇 PDF 的联合上下文推理，
也不注册 `joint/page_pool` 两条路线。正式吞吐入口、压力映射和输出协议见
[`RUN_COMMANDS.md`](../RUN_COMMANDS.md) 与 [`THROUGHPUT_TUNING_PROTOCOL.md`](../THROUGHPUT_TUNING_PROTOCOL.md)。

adapter 使用 `aio_doc_analyze` 的官方页面流水线，并保留官方 PDF normalization；
传入 `image_writer=None`，因此只写统一 Markdown、metrics 和空的 `image_boxes.json`
元数据，不保存原始 middle JSON、裁剪图片或 layout PDF。模型、插件和 AsyncLLM
只在 worker 初始化时创建一次，ready 之后连续消费队列。
