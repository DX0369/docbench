# `gdb_hunyuan`

## 用途

运行 HunyuanOCR 1.0 与 1.5 AR 的 release vLLM 服务，以及三种 Hunyuan 方法的页面客户端。
两版 AR 共用此环境；DFlash 服务使用独立的
[`gdb_hunyuan_dflash`](gdb_hunyuan_optional.md)，不能合并其核心依赖。

## 关键约束

- Python 3.10。
- vLLM 必须为 0.18.1。
- vLLM 0.18.1 会解析匹配的 Torch 2.10、Transformers 4.57 和 FlashInfer。
- 不安装 Hunyuan 根目录的 native Transformers/FlashAttention 依赖。
- 不把 DFlash nightly 或 Transformers 5.13 装入本环境。

官方提供了顶层要求和一份环境导出式锁文件：

```text
third_party/methods/hunyuanocr/inference/vllm_0_18_1/requirements.txt
third_party/methods/hunyuanocr/inference/vllm_0_18_1/requirements-lock.txt
```

`requirements-lock.txt` 第 114 行包含官方构建机上的 `packaging @ file:///home/conda/...` 本地路径，无法在其他机器复现。因此首轮必须使用短版 `requirements.txt`；不要直接安装原始 lock，也不要手工创建它引用的 `/home/conda/...` 路径。已有锁文件只用于迁移后的版本核对。

## 官方依赖安装入口

```bash
conda create -n gdb_hunyuan python=3.10 pip setuptools wheel -y

PYTHONNOUSERSITE=1 conda run -n gdb_hunyuan python -m pip install \
  -r "$GDB_METHOD_ROOT/hunyuanocr/inference/vllm_0_18_1/requirements.txt"
```

## 用户级包隔离修复（2026-10-02 14:17 +08:00）

设置 `PYTHONNOUSERSITE=1` 后，`pip check` 确认 vLLM 缺 `psutil`、FlashInfer 缺
`nvidia-ml-py`。原来的 `psutil==7.2.2` 和 `nvidia-ml-py==13.595.45` 实际位于
`~/.local/lib/python3.10/site-packages`，不是本 Conda 环境内的包。
历史 pip freeze 虽包含它们，不能证明环境隔离完整；不能靠重新启用 user-site 绕过。
经用户授权，已将这两个同版本包补入环境；安装前后隔离 freeze 比对仅新增两项，
其他包未升级或卸载，未修改模型或推理参数。迁移时，在官方依赖安装后执行：

```bash
PYTHONNOUSERSITE=1 \
    conda run -n gdb_hunyuan python -m pip --isolated install \
    --no-deps psutil==7.2.2 nvidia-ml-py==13.595.45

PYTHONNOUSERSITE=1 conda run -n gdb_hunyuan python -m pip check
PYTHONNOUSERSITE=1 conda run -n gdb_hunyuan python -c \
  'import psutil, pynvml, torch, transformers, vllm; print(psutil.__file__, pynvml.__file__)'
```

隔离 pip check/import 已通过，两包路径均位于 `gdb_hunyuan` 内。
环境锁已完成用户目录隔离核对；GPU 运行命令见 [RUN_COMMANDS.md](../RUN_COMMANDS.md)。

## 安装后核对（不含 smoke 或锁导出）

```bash
PYTHONNOUSERSITE=1 conda run -n gdb_hunyuan python -m pip check
PYTHONNOUSERSITE=1 conda run -n gdb_hunyuan python -c \
  'import torch, transformers, vllm; print(torch.__version__, torch.version.cuda, transformers.__version__, vllm.__version__)'
PYTHONNOUSERSITE=1 conda run -n gdb_hunyuan vllm --version
```

当前锁定 Python 3.10.20、vLLM 0.18.1、Torch 2.10.0、Transformers 4.57.6、
FlashInfer 0.6.6。官方短 requirements 是安装依据；现有锁文件用于迁移后的版本核对。
依赖闭包之外仅发现 pip/wheel 工具，未发现明显可删的大型额外依赖，不据此裁剪 vLLM 依赖。

迁移后仍需分别执行服务启动与短文档 smoke。1.0 的官方要求为 `vllm>=0.12.0`，
本项目验证版本为 0.18.1；环境检查成功不保证具体服务容量能启动。
上下文、显存默认、生成/早停及实测限制统一见
[吞吐协议](../THROUGHPUT_TUNING_PROTOCOL.md)，本文件不重复运行参数。
