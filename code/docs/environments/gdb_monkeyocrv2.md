# `gdb_monkeyocrv2`

更新时间：2026-10-10 14:12 +08:00

## 用途与版本依据

用于 `monkeyocrv2_b_parsing` 的普通 AR 推理；当前不接入 DFlash。
源码和模型绑定以 `code/config/methods.json` 为准，安装依据为锁定源码的
`README.md` 中 Document Parsing / Install 和 `parsing/requirements.txt`。

现有环境为 Python 3.11.17、Torch `2.11.0+cu129`、vLLM `0.25.1+cu129`、
Transformers `5.19.0`、FlashInfer `0.6.13`。
官方提供 vLLM 0.25.1 / CUDA 12.9 路线，也允许不使用 DFlash 时采用 vLLM 0.11.2；
项目沿用已经验证的 0.25.1 环境，不因只接入 AR 而重新降级。
上述精确版本是项目现有组合，不全部属于官方 requirements 的固定要求。
`parsing/requirements.txt` 仅固定 Gradio 5.30.0 和 pypdfium2 5.10.1，不能单独恢复推理栈。

## 创建与安装

以下是按官方路线和现有核心版本整理的重建命令，**尚未在全新环境重新执行验证**，
不是已验证的完整环境锁。仅用于新建环境，不在现用环境上覆盖安装。
通用 Conda 初始化和路径设置见 [README.md](README.md)。

```bash
conda create -n gdb_monkeyocrv2 python=3.11.17 -y
conda activate gdb_monkeyocrv2
export GDB_ROOT=/data/wangsq/code/global_doc_bench
export MOCR2_SRC="$GDB_ROOT/third_party/methods/monkeyocrv2"
export PYTHONNOUSERSITE=1

(
  unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
  unset PIP_EXTRA_INDEX_URL UV_INDEX UV_EXTRA_INDEX_URL
  python -m pip install --index-url https://pypi.tuna.tsinghua.edu.cn/simple uv
  uv pip install --python "$CONDA_PREFIX/bin/python" \
    --default-index https://pypi.tuna.tsinghua.edu.cn/simple \
    --index https://wheels.vllm.ai/0.25.1/cu129 \
    --index https://download.pytorch.org/whl/cu129 \
    --index-strategy unsafe-best-match \
    'vllm==0.25.1+cu129' 'torch==2.11.0+cu129' \
    'transformers==5.19.0' 'flashinfer-python==0.6.13' \
    -r "$MOCR2_SRC/parsing/requirements.txt"
)
```

官方 CUDA wheel 仍从对应发布源直连获取，普通依赖显式使用镜像；不要把特殊
CUDA 构建替换为镜像中的另一版本。完整传递依赖尚未锁定，解析失败时应保留错误并核查，
不通过 `--no-deps` 隐藏缺包。

## 安装后检查与迁移边界

```bash
PYTHONNOUSERSITE=1 python -m pip check
PYTHONNOUSERSITE=1 python - <<'PY'
import torch, transformers, vllm, flashinfer
print("torch:", torch.__version__, "CUDA:", torch.version.cuda)
print("transformers:", transformers.__version__)
print("vllm:", vllm.__version__)
print("flashinfer:", flashinfer.__version__)
PY
```

当前尚未保存本环境的 Conda explicit、Conda env 和 pip freeze 迁移锁。
新服务器还需核查驱动、CUDA 编译工具及真实 GPU smoke，不能仅凭 import 成功认定复现完成。
现机使用 CUDA 12.8 toolkit 提供编译工具，与 Torch wheel 的 CUDA 12.9 运行时是不同概念，
不据此改写官方 CUDA 要求。

推理启动命令统一见 [RUN_COMMANDS.md](../RUN_COMMANDS.md)，运行机制和参数映射见
[THROUGHPUT_TUNING_PROTOCOL.md](../THROUGHPUT_TUNING_PROTOCOL.md)，验证结果仅记于
[PROJECT_STATE.md](../../../PROJECT_STATE.md)。此前 DFlash 旧 runner / eager / HF override
属于诊断兼容配置，不是本环境当前 AR 运行要求。
