# `gdb_paddle_vllm`

## 用途与已验证版本

仅运行PaddleOCR-VL 1.5／1.6的vLLM识别服务；本地pipeline使用[gdb_paddle](gdb_paddle.md)。
双环境是依赖隔离，不代表两张GPU。运行参数、显式eager及验证范围见
[吞吐协议](../THROUGHPUT_TUNING_PROTOCOL.md#paddleocr-vl单pipeline区域并发)。

- Python3.10；Torch2.8.0+cu128、TorchVision0.23.0、vLLM0.10.2、
  Transformers4.57.6、FlashAttention2.8.2、PaddleX3.7.2。
- PaddleOCR editable源码与pipeline相同，revision
  `211989f046cc1878460f9e65574690c00a127a1a`。
- 主版本来自锁定官方安装器及已验证解析结果；RTX3090满足CC≥8.0，
  官方NVIDIA路线要求CUDA≥12.6。FlashAttention wheel必须匹配Python/Torch/CUDA ABI。
- 本环境不装Paddle GPU。Torch要求cuDNN9.10.2.21，pipeline中的Paddle要求9.5.1.17，
  不能通过合并环境消除重复包。
- `psutil==7.2.2`已补入环境，避免aistudio-sdk/vLLM借用用户目录。

## 重建与验收

先按[通用迁移步骤](README.md)设置GDB路径变量，恢复对应PaddleOCR源码。
以下保留官方安装器路线，并显式固定本机已验证的核心解析结果；不调用浮动最新版。

```bash
git -C "$GDB_METHOD_ROOT/paddleocr" rev-parse HEAD

(
  unset PYTHONPATH PYTHONHOME PIP_EXTRA_INDEX_URL PIP_TARGET PIP_PREFIX PIP_USER
  export PYTHONNOUSERSITE=1
  conda create -n gdb_paddle_vllm python=3.10 pip setuptools wheel -y
  conda run -n gdb_paddle_vllm python -m pip --isolated install \
    -e "$GDB_METHOD_ROOT/paddleocr[doc-parser]" 'paddlex==3.7.2' 'psutil==7.2.2'
  conda run -n gdb_paddle_vllm python -m pip --isolated install \
    'vllm==0.10.2' 'transformers==4.57.6'
  conda run -n gdb_paddle_vllm python -m pip --isolated install \
    'https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.3.14/flash_attn-2.8.2+cu128torch2.8-cp310-cp310-linux_x86_64.whl'
  conda run -n gdb_paddle_vllm paddleocr install_genai_server_deps vllm
  conda run -n gdb_paddle_vllm python -m pip --isolated check
  conda run -n gdb_paddle_vllm python -c \
    'import torch, transformers, vllm, psutil; print(torch.__version__, torch.version.cuda, transformers.__version__, vllm.__version__, psutil.__file__)'
)
```

PyPI的FlashAttention源码包可能因构建隔离看不到Torch而失败，应优先使用上述官方文档
列出的匹配预编译wheel；GitHub wheel仍是直连，不因下载慢自动开代理。
不能只导入成功就宣称服务/GPU已验证；
新机器仍需两个模型版本的有界smoke。本轮没有执行安装或更新包。

## 运行注意

- 正常模式不要继承 `TORCHDYNAMO_DISABLE=1`；wrapper不替换torch.compile，
  显式eager是诊断选项，不做自动回退。
- vLLM IPC socket路径有长度限制，TMPDIR使用项目 `tmp/` 下独立短目录；
  日志/输出仍归各run，结束后清理本轮缓存。服务使用conda实时输出，避免退出前日志被缓冲。
- 当前wrapper兼容vLLM 0.10.2的 `vllm.utils` 和新版parser导入路径。
  PaddleX会参与vLLM插件发现，不能因wrapper直接启动vLLM就断言PaddleX无用。
- `torchaudio`、Ray、Numba、xformers等属于vLLM或官方extras依赖，不凭“只处理PDF”卸载。

迁移时按本配方重建，再用根目录 `env/locks_rtx3090_20260720/gdb_paddle_vllm.*` 核对版本；
同时恢复源码 revision、editable 安装和 FlashAttention wheel 来源。
