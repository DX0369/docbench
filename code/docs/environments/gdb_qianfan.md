# `gdb_qianfan`

## 用途与版本依据

`qianfan_ocr` / `qianfan_ocr_think` 共用本环境与同一模型，当前唯一正式路线是单页请求的 page pool。
实际核心栈为Python 3.10.20、Torch 2.10.0（cu128）、vLLM 0.18.1、
Transformers 4.57.6。这是项目已验证组合，不是官方精确锁。

官方模型README给出 `vllm serve baidu/Qianfan-OCR --trust-remote-code`，
没有固定完整依赖；Qianfan-VL参考仓库的Docker示例不是OCR专属要求。
项目使用本地模型remote code及InternVL architecture override：
`--hf-overrides '{"architectures":["InternVLChatModel"],"model_type":"internvl_chat"}'`。
不需要editable安装Qianfan-VL参考仓库；源码/模型身份以 `config/methods.json` 为准。

基础安装路线是Python 3.10命名环境中安装 `vllm==0.18.1` 及Pillow/OpenAI/httpx/PyMuPDF/PyYAML。
仅此步骤不能精确复现当前全部依赖；同平台复现使用下列已核对快照，跨平台先按
[迁移流程](README.md#迁移时的安装流程)确认驱动、CUDA wheel与官方当前要求。

## 同平台恢复命令

适用Linux x86_64及兼容CUDA 12.8的驱动；先准备已有三份锁。
**只在目标前缀尚不存在时创建**，不要在已用环境上覆盖执行；按新服务器路径调整前缀。
Conda explicit恢复Conda部分（含pip/setuptools/wheel），pip快照恢复其余精确版本；
explicit中的Conda下载URL与下方PyPI镜像是不同来源。

```bash
export GDB_ROOT=/data/wangsq/code/global_doc_bench
export QIANFAN_PREFIX=/data/wangsq/software/miniconda/envs/gdb_qianfan
export QIANFAN_PY="$QIANFAN_PREFIX/bin/python"
export LOCK_DIR="$GDB_ROOT/env/locks_rtx3090_20260720"
mkdir -p "$GDB_ROOT/tmp"
unset PYTHONPATH PYTHONHOME PIP_EXTRA_INDEX_URL PIP_TARGET PIP_PREFIX PIP_USER
export PYTHONNOUSERSITE=1 TMPDIR="$GDB_ROOT/tmp"
conda create --prefix "$QIANFAN_PREFIX" \
  --file "$LOCK_DIR/gdb_qianfan.explicit.txt" -y
"$QIANFAN_PY" -m pip --isolated install \
  -r "$LOCK_DIR/gdb_qianfan.pip-freeze.txt"
env -u PYTHONPATH -u PYTHONHOME PYTHONNOUSERSITE=1 "$QIANFAN_PY" -m pip check
env -u PYTHONPATH -u PYTHONHOME PYTHONNOUSERSITE=1 "$QIANFAN_PY" -c \
  'import site, torch, transformers, vllm, psutil, pynvml; print(site.ENABLE_USER_SITE, torch.__version__, torch.version.cuda, transformers.__version__, vllm.__version__); print(psutil.__file__, pynvml.__file__)'
```

完整 YAML 和 pip 快照仅用于迁移核对；补入的 `psutil==7.2.2`、`nvidia-ml-py==13.595.45`
已包含在根目录锁文件中。真实服务和压力参数见 [RUN_COMMANDS.md](../RUN_COMMANDS.md) 与吞吐协议。

## eager / auto 执行模式

默认auto，不传 `--enforce-eager`；显式 `--qianfan-execution-mode eager` 用于诊断/历史复现。
auto启动应 `env -u TORCHDYNAMO_DISABLE`，避免继承禁用变量；
是否编译/捕获/使用CUDA graph看服务日志，不能仅凭开关认定更快。
执行模式与页面调度独立；参数来源、压力映射和测量口径统一见
[吞吐协议](../THROUGHPUT_TUNING_PROTOCOL.md)。模型卡基础 prompt 为
`Parse this document to Markdown.`；think 变体只追加字面量 `<think>`。
