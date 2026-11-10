# Jina-OCR-v1 FastMTP 环境

Jina 使用独立 `gdb_jinaocr`，不再复用 `gdb_hunyuan_dflash`。主要版本：

| 组件 | 固定版本 |
| --- | --- |
| Python | 3.12 |
| vLLM | `0.23.1rc1.dev869+ge55cc59e5` |
| Torch / TorchVision | `2.11.0+cu130` / `0.26.0+cu130` |
| Transformers | `5.14.1` |
| tokenizers | `0.22.2` |
| FlashInfer | `0.6.13` |

官方模型卡要求 vLLM ≥ 0.21；项目保留已有官方 wheel，不追随 nightly。
Transformers 5.14.1 是项目的兼容性约束，不是 Jina 模型卡指定的版本：
vLLM 将配置识别为 `deepseek_ocr`，5.5.3 未覆盖该类型的 tokenizer 兼容分支，
会按模型声明的 Llama 类重建 tokenizer；5.14.1 已覆盖该类型并保留
`tokenizer.json` 中的 ByteLevel 编解码。不能只用独立 HF tokenizer 的检查代替
vLLM 实际加载路径的检查。修复无需修改模型文件、官方插件或环境源码。

## 安装与迁移

新机器直接创建独立环境，不必先安装 DFlash。安装精确的官方 vLLM wheel
及兼容版本；以下用于新建环境，不要覆盖已有环境：

```bash
export GDB_ROOT="${GDB_ROOT:-$(git rev-parse --show-toplevel)}"
export GDB_CONDA_ROOT="${GDB_CONDA_ROOT:-$HOME/software/miniconda}"
export CONDA_ROOT="$GDB_CONDA_ROOT"
export JINA_PY="$CONDA_ROOT/envs/gdb_jinaocr/bin/python"

"$CONDA_ROOT/bin/conda" create -n gdb_jinaocr python=3.12 pip -y

(
    unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
    unset PIP_PROXY PIP_EXTRA_INDEX_URL UV_EXTRA_INDEX_URL
    export PIP_CONFIG_FILE=/dev/null PYTHONNOUSERSITE=1
    "$JINA_PY" -m pip install \
        --index-url https://repo.huaweicloud.com/repository/pypi/simple \
        'uv==0.11.29' 'setuptools==80.10.2'
    "$CONDA_ROOT/envs/gdb_jinaocr/bin/uv" pip install \
        --python "$JINA_PY" \
        --default-index https://repo.huaweicloud.com/repository/pypi/simple \
        --torch-backend=cu130 \
        'https://wheels.vllm.ai/e55cc59e52791fe7620f6808e409edcf68740b6b/vllm-0.23.1rc1.dev869%2Bge55cc59e5-cp38-abi3-manylinux_2_28_x86_64.whl' \
        'transformers==5.14.1' 'tokenizers==0.22.2' \
        'flashinfer-python==0.6.13' 'flashinfer-cubin==0.6.13' \
        'openai==2.46.0' 'pillow==12.3.0'
    "$CONDA_ROOT/envs/gdb_jinaocr/bin/uv" pip uninstall \
        --python "$JINA_PY" torchcodec
)
```

最后一条命令保持与基础环境相同的包集合。其他传递依赖由 wheel 元数据解析；
迁移后仍需与 [`gdb_jinaocr` 三份锁](../../../env/locks_rtx3090_20260720/)核对，
再完成 tokenizer 和有界 GPU 验证，不将新机器安装成功等同于推理验证。

## 运行要求

必须使用 conda run 带入 ninja 路径，并指定已有 CUDA Toolkit 12.8；系统旧 nvcc
已实测编译失败。adapter 自动配置工具链，迁移可传 `--jina-cuda-home`；
FlashInfer 缓存默认 `.cache/flashinfer/jina_ocr_v1`，smoke 可用
`--jina-flashinfer-workspace` 指向该轮 tmp。Torch CUDA runtime 仍来自 wheel。

模型目录必须包含官方 Python 文件和 tokenizer；adapter 从模型目录导入
`deepseek_ocr_mtp`，在创建 AsyncLLM 前调用官方 `register()`。
使用官方 `vllm_llm_kwargs`、`prepare_vllm_input` 和 `vllm_sampling_params`，
FastMTP 为 `eagle`、K=3、递归单 head；没有单独 draft checkpoint。
缺少 repetition detection 时显式失败，不静默改采样行为。

AsyncLLM 使用官方 worker extension 入口，在每个引擎进程导入项目的
`vllm_worker.py` 并调用官方 register；不改官方插件或环境文件。
运行命令见 [RUN_COMMANDS.md](../RUN_COMMANDS.md)，实际验证范围和结果只记录在
[`PROJECT_STATE.md`](../../../PROJECT_STATE.md)，不把历史环境的 P2/latency
验证自动算作新环境的验证。
