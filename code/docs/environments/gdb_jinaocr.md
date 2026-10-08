# Jina-OCR-v1 FastMTP 环境

状态：2026-10-08，官方同步、adapter P1/P2/latency smoke 已通过。
当前复用未改包的 `gdb_hunyuan_dflash`，不需要创建 `gdb_jinaocr`。
vLLM 为 `0.23.1rc1.dev869+ge55cc59e5`；迁移按
[DFlash 配方](gdb_hunyuan_optional.md) 和已有环境锁恢复。

必须使用 conda run 带入 ninja 路径，并指定已有 CUDA Toolkit 12.8；系统旧 nvcc
已实测编译失败。adapter 自动配置工具链，迁移可传 `--jina-cuda-home`；
FlashInfer 缓存默认 `.cache/flashinfer/jina_ocr_v1`，smoke 可用
`--jina-flashinfer-workspace` 指向该轮 tmp。Torch CUDA runtime 仍来自 wheel。

以下仅为未来需要独立环境时的候选配方，本轮未执行、未验证：

官方模型卡要求 vLLM ≥ 0.21，未提供精确 Python/Torch 锁。
项目候选为 Python 3.12、vLLM 0.21.0，由 vLLM wheel 决定 Torch 等依赖。
当前 DFlash 环境已完成官方同步和 adapter smoke，可作为 Jina 的实际运行环境。

```bash
conda create -n gdb_jinaocr python=3.12 pip -y
uv pip install \
    --python /data/wangsq/software/miniconda/envs/gdb_jinaocr/bin/python \
    --default-index https://repo.huaweicloud.com/repository/pypi/simple \
    'vllm==0.21.0' pillow pymupdf
```

以上是待验证安装配方，不是已验证环境锁。不得升级其他方法环境以适配 Jina。
模型目录必须包含官方 Python 文件和 tokenizer；adapter 从模型目录导入
`deepseek_ocr_mtp`，在创建 AsyncLLM 前调用官方 `register()`。
使用官方 `vllm_llm_kwargs`、`prepare_vllm_input` 和 `vllm_sampling_params`，
FastMTP 为 `eagle`、K=3、递归单 head；没有单独 draft checkpoint。
缺少 repetition detection 时显式失败，不静默改采样行为。

AsyncLLM 使用官方 worker extension 入口，在每个引擎进程导入项目的
`vllm_worker.py` 并调用官方 register；不改官方插件或环境文件。
正式调参仍未运行；当前 smoke 不能证明全量稳定性或最优吞吐。
