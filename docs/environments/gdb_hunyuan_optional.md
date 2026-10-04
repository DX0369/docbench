# HunyuanOCR 可选比较环境

这些环境不是首轮 benchmark 必需项，只用于比较不同官方推理后端。它们与 `gdb_hunyuan` 的 release vLLM 依赖互斥。

## `gdb_hunyuan_dflash`

用途：HunyuanOCR-1.5 的 vLLM DFlash speculative decoding 后端。它与
`gdb_hunyuan` 中的 vLLM 0.18.1 AR 后端依赖互斥，必须使用独立环境。

### 已验证版本

2026-07-20 在 RTX 3090（24 GiB）、NVIDIA Driver 580.159.04 上完成了服务、
官方客户端和 GlobalDocBench 单页 adapter smoke。经过验证的精确组合是：

- Python 3.12。
- vLLM `0.23.1rc1.dev869+ge55cc59e5`，commit
  `e55cc59e52791fe7620f6808e409edcf68740b6b`。
- Torch `2.11.0+cu130`，TorchVision `0.26.0+cu130`。
- Transformers `5.5.3`。
- FlashInfer `0.6.13`。
- OpenAI `2.46.0`，Pillow `12.3.0`。
- 必须卸载 `torchcodec`。
- FlashInfer JIT 使用用户级 CUDA Toolkit 12.8 的 `nvcc` 已验证可行；Torch
  自身仍使用 wheel 内的 CUDA 13 runtime。

不要再次使用下面这种未锁版本的安装方式：

```bash
uv pip install -U vllm \
  --torch-backend=cu130 \
  --extra-index-url https://wheels.vllm.ai/nightly
```

nightly 简单索引只暴露当日最新 wheel。2026-07-20 该命令安装的是
`dev1279`，它已经要求 Transformers 提供 `HunYuanVLProcessor`，与官方固定的
Transformers 5.5.3 不兼容。临时换用 Transformers 5.13 虽能完成架构导入，
但图像请求会在当前 vLLM 的 `get_xdrope_input_positions` 中失败。因此正式环境
必须锁定上面的 `dev869`，不能写 `latest`。

### 删除并重建

以下命令会删除现有 `gdb_hunyuan_dflash`。执行前确认该环境没有正在运行的服务：

```bash
export GDB_ROOT=/data/wangsq/code/global_doc_bench
export CONDA_ROOT=/data/wangsq/software/miniconda
export CONDA="$CONDA_ROOT/bin/conda"
export DFLASH_ENV="$CONDA_ROOT/envs/gdb_hunyuan_dflash"
export DFLASH_PY="$DFLASH_ENV/bin/python"
export DFLASH_UV="$DFLASH_ENV/bin/uv"

"$CONDA" env remove -n gdb_hunyuan_dflash -y
"$CONDA" create -n gdb_hunyuan_dflash python=3.12 pip -y

"$DFLASH_PY" -m pip install "uv==0.11.29"
```

安装精确的官方 vLLM wheel。`--torch-backend=cu130` 用于让 uv 选择 cu130 的
Torch/TorchVision 构件；wheel URL 中已经同时锁定 vLLM commit 和 dev 版本：

```bash
"$DFLASH_UV" pip install \
  --python "$DFLASH_PY" \
  --torch-backend=cu130 \
  --extra-index-url https://wheels.vllm.ai/nightly \
  "https://wheels.vllm.ai/e55cc59e52791fe7620f6808e409edcf68740b6b/vllm-0.23.1rc1.dev869%2Bge55cc59e5-cp38-abi3-manylinux_2_28_x86_64.whl"
```

随后按官方要求及本项目 smoke 结果锁定滚动依赖。官方 nightly requirements
记录的 OpenAI 验证版本为 2.44.0，硬要求为 `>=1.30.0`；下面的 OpenAI 2.46.0
来自本项目已验证环境，不是官方指定的唯一版本。这里使用 `--no-deps`，避免再次改变
vLLM、Torch 或 CUDA wheel：

```bash
"$DFLASH_UV" pip install \
  --python "$DFLASH_PY" \
  --no-deps \
  "transformers==5.5.3" \
  "flashinfer-python==0.6.13" \
  "openai==2.46.0" \
  "pillow==12.3.0"

"$DFLASH_UV" pip uninstall \
  --python "$DFLASH_PY" \
  torchcodec
```

`vllm` 的包元数据仍声明依赖 `torchcodec`，因此 `pip check` 会报告这一项缺失。
这是 HunyuanOCR 官方为避开 TorchCodec/FFmpeg 导入错误而要求的主动卸载，不能
为了让 `pip check` 变绿而重新安装。

### 环境验证

```bash
"$DFLASH_PY" - <<'PY'
import inspect
import flashinfer
import torch
import transformers
import vllm
from vllm.config import SpeculativeConfig
from vllm.model_executor.models import ModelRegistry

print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("vllm:", vllm.__version__)
print("transformers:", transformers.__version__)
print("flashinfer:", flashinfer.__version__)
print("dflash registered:", "dflash" in inspect.getsource(SpeculativeConfig))
print(
    "hunyuan registered:",
    "HunYuanVLForConditionalGeneration" in ModelRegistry.get_supported_archs(),
)
PY
```

预期关键输出：

```text
torch: 2.11.0+cu130
torch cuda: 13.0
vllm: 0.23.1rc1.dev869+ge55cc59e5
transformers: 5.5.3
flashinfer: 0.6.13
dflash registered: True
hunyuan registered: True
```

检查模型文件：

```bash
test -s "$GDB_ROOT/models/HunyuanOCR-1.5/config.json"
test -s "$GDB_ROOT/models/HunyuanOCR-1.5/model.safetensors"
test -s "$GDB_ROOT/models/HunyuanOCR-1.5/dflash/config.json"
test -s "$GDB_ROOT/models/HunyuanOCR-1.5/dflash/model.safetensors"
```

### RTX 3090 profile

RTX 3090 的 DFlash 启动参数、显存限制和 CUDA/FlashInfer 路径见
[RUN_COMMANDS.md](../RUN_COMMANDS.md) 及吞吐 profile；本环境文档不重复服务命令。

验证和服务启动命令见 [RUN_COMMANDS.md](../RUN_COMMANDS.md)；本文件只保留安装和
硬件 profile 说明。

官方原始说明仍应作为语义和参数依据：

```text
third_party/methods/hunyuanocr/inference/nightly/README.md
third_party/methods/hunyuanocr/inference/nightly/requirements.txt
```

## `gdb_hunyuan_transformers`

官方关键约束：

- Python 3.12。
- Torch 2.11 + TorchVision 0.26，选择官方 cu128/cu130 wheel。
- Transformers 必须精确为 5.13.0。
- 不安装 vLLM。

候选步骤以官方文件为准：

```text
third_party/methods/hunyuanocr/inference/transformers/README.md
third_party/methods/hunyuanocr/inference/transformers/requirements.txt
```

native Transformers 环境仍只用于可选后端比较，不能与 DFlash 环境合并。
