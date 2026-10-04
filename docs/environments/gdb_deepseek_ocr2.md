# `gdb_deepseek_ocr2`

## 用途

运行官方 DeepSeek-OCR-2 模型注册与预处理代码，配合项目页面池／AsyncLLM HTTP 服务。

## 关键约束

- Python 3.12.9。
- Torch 2.6.0、TorchVision 0.21.0、TorchAudio 2.6.0。
- vLLM 0.8.5。
- Transformers 4.46.3、Tokenizers 0.20.3；模型 `config.json` 也明确记录 Transformers 4.46.3。
- 官方文档要求 FlashAttention 2.7.3；DeepSeek-OCR-2 的视觉编码器会直接导入它，因此是当前服务的运行时依赖。
- 官方 README 使用 cu118 vLLM wheel；项目已验证栈为 Torch `2.6.0+cu124`＋vLLM `0.8.5`＋FlashAttention `2.7.3`。CUDA发行版是项目环境选择，不称完全相同的官方二进制组合。

不要把这个环境与 Youtu 合并：两者虽然都基于 Torch 2.6，但 DeepSeek 依赖旧 vLLM、官方模型注册代码和 FlashAttention。

## 安装与迁移

```bash
export GDB_ROOT=/data/wangsq/code/global_doc_bench
export GDB_METHOD_ROOT="$GDB_ROOT/third_party/methods"
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1

conda create -n gdb_deepseek_ocr2 python=3.12.9 pip setuptools wheel -y

conda run -n gdb_deepseek_ocr2 python -m pip install \
  torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu124
conda run -n gdb_deepseek_ocr2 python -m pip install \
  'vllm==0.8.5'
conda run -n gdb_deepseek_ocr2 python -m pip install \
  -r "$GDB_METHOD_ROOT/deepseek_ocr2/requirements.txt"
```

DeepSeek 固定 Transformers 4.46.3/Tokenizers 0.20.3，而 vLLM 0.8.5 的包元数据要求 Transformers >=4.51.1/Tokenizers >=0.21.1，因此 `pip check` 预计会报告两项冲突。官方README明确说明Transformers冲突；Tokenizers是伴随固定版本出现的已记录冲突，不另称官方明确豁免。当前 `DeepseekOCR2ForCausalLM` 注册、页面池服务和GPU推理已验证；迁移后仍需单页验收，不能只凭import接受环境。

当前已验证的环境状态：

- Torch `2.6.0+cu124`；
- CUDA runtime `12.4`；
- FlashAttention `2.7.3`；
- `torch.cuda.is_available()` 为 `True`。

优先直接安装官方匹配 wheel，避免 pip 回退到源码编译：

```bash
conda activate gdb_deepseek_ocr2
mkdir -p "$GDB_ROOT/tmp/pip"
export TMPDIR="$GDB_ROOT/tmp/pip"

python -m pip install \
  "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3%2Bcu12torch2.6cxx11abiFALSE-cp312-cp312-linux_x86_64.whl" \
  --no-cache-dir
```

直接执行 `pip install flash-attn==2.7.3 --no-build-isolation` 可能进入源码编译；
如果临时目录和 pip 缓存位于不同文件系统，还可能出现 `Invalid cross-device link`。

## 首轮验收

```bash
conda run -n gdb_deepseek_ocr2 python -m pip check  # 保存预期的 Transformers/Tokenizers 冲突
conda run -n gdb_deepseek_ocr2 python -c \
  'import torch, transformers, vllm; print(torch.__version__, torch.version.cuda, transformers.__version__, vllm.__version__)'
```

这里的决定性验收不是 `pip check` 全绿，而是 vLLM 自定义服务能加载模型并完成单页请求。

## 迁移核对

迁移时使用根目录 `env/locks_rtx3090_20260720/gdb_deepseek_ocr2.*` 核对版本；
Torch/CUDA 与 FlashAttention wheel 来源必须同时核对。源码/模型身份以 `config/methods.json` 为准。
GPU覆盖和内容限制见运行命令与
[吞吐协议](../THROUGHPUT_TUNING_PROTOCOL.md#deepseek公共页面池与异步服务预处理)；
不在环境配方重复维护显存/worker调整时间线。
