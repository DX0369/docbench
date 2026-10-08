# `gdb_glmocr`

## 用途与版本

GLM-OCR 常驻 self-hosted SDK、PP-DocLayoutV3 布局模型和 vLLM 识别服务共用此环境。
LightOnOCR-2-1B 和 Xiaomi-OCR-0 的服务及 worker 也使用此环境。
运行主线及参数只在[吞吐协议](../THROUGHPUT_TUNING_PROTOCOL.md#glm-ocr-主线)维护；
 GPU 运行与内容限制见 [RUN_COMMANDS.md](../RUN_COMMANDS.md) 和吞吐协议。

- 官方要求：Python≥3.10、Torch≥2.10、TorchVision≥0.25、Transformers≥5.3；self-hosted 文档要求 vLLM≥0.19。
- 已验证栈：Python3.10.20、Torch2.11.0+cu130、TorchVision0.26.0、Transformers5.14.1、vLLM0.20.0、FlashInfer0.6.8.post1。这是项目锁定栈，不是官方唯一推荐组合。
- 已知例外：xgrammar0.2.4 的元数据要求 Transformers<5；隔离 `pip check` 仅报告这一条冲突（退出1）。已有真实 GPU 验证，不等于所有功能均兼容，也不能写成无冲突。
- 保持独立环境，不与使用旧 Transformers 的方法合并；不要为了消除告警卸载 xgrammar 或 CUDA 依赖。

## 被复用的方法

LightOnOCR-2-1B 的 vLLM page-pool adapter 也复用此环境，不额外安装或升级依赖。
LightOnOCR-2 的官方推理参数、PDF 预处理和 benchmark 规则见
[吞吐协议](../THROUGHPUT_TUNING_PROTOCOL.md)；复用关系和最小 import 检查见
[gdb_lightonocr2.md](gdb_lightonocr2.md)。

Xiaomi-OCR-0 的复用关系、依赖来源和迁移检查见
[gdb_xiaomi_ocr.md](gdb_xiaomi_ocr.md)；环境恢复统一使用下文配方。

## 恢复已验证环境

以下用于**新的 Linux x86_64 环境和干净源码目录**，不要在现有带补丁 checkout 上重复 sync。
变量沿用[通用安装说明](README.md)，迁移前须一并复制 `env/locks_rtx3090_20260720/`；
只克隆 `code/` 不包含这些环境快照。命令恢复包版本及源码，不保证跨平台或 wheel 字节完全相同；
新服务器仍需核对驱动、Torch CUDA 版本并独立 smoke。本轮没有实际创建新环境验证重建流程。

```bash
(
set -e
unset PYTHONPATH PYTHONHOME PIP_EXTRA_INDEX_URL PIP_PROXY
export PYTHONNOUSERSITE=1
export TMPDIR="$GDB_ROOT/tmp"
mkdir -p "$TMPDIR"
LOCK_DIR="$GDB_ROOT/env/locks_rtx3090_20260720"

conda create -n gdb_glmocr --file "$LOCK_DIR/gdb_glmocr.explicit.txt" -y
conda env config vars set -n gdb_glmocr PYTHONNOUSERSITE=1
GLM_PY="$(dirname "$(command -v conda)")/../envs/gdb_glmocr/bin/python"

# 精确恢复已审计的依赖集合，不重新解析宽下界。
"$GLM_PY" -m pip install \
  --no-deps -r "$LOCK_DIR/gdb_glmocr.pip-freeze.txt"

# 恢复 methods.json 锁定的源码，再应用项目异常传播补丁。
python3 "$GDB_ROOT/code/src/method_repositories.py" --method glmocr sync
GLM_PATCH="$GDB_ROOT/code/src/methods/glmocr/patches/failure_propagation.patch"
git -C "$GDB_METHOD_ROOT/glmocr" apply --check "$GLM_PATCH"
git -C "$GDB_METHOD_ROOT/glmocr" apply "$GLM_PATCH"
"$GLM_PY" -m pip install --no-deps --no-build-isolation \
  -e "$GDB_METHOD_ROOT/glmocr[selfhosted]"
)
```

`--no-deps` 仅用于这里已审计的完整版本集合，也避免重新解析已记录的 xgrammar 冲突；
不是允许缺包。`--no-build-isolation` 使用快照中的构建工具。若改用官方宽下界安装，
即先安装 vLLM 再安装 `.[selfhosted]`，解析出的版本可能不同，必须作为新环境重新验收。

## 隔离、补丁与验收

`psutil` 和 `nvidia-ml-py` 已补入环境。保持 `PYTHONNOUSERSITE=1`，避免依赖用户目录。

补丁处理布局、裁剪、区域失败和 watchdog 的异常传播，不改变模型参数或 SDK 重试。
它不在官方 HEAD 中；只恢复环境锁或 Git revision 不够。现有 editable 安装直接使用
已打补丁源码，运行时仅检查，不自动修改。方法应标注“官方源码＋项目异常传播补丁”。

```bash
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1
git -C "$GDB_METHOD_ROOT/glmocr" apply --reverse --check \
  "$GDB_ROOT/code/src/methods/glmocr/patches/failure_propagation.patch"
conda run -n gdb_glmocr python -m pip check
# 仅接受上文的一条已知冲突；有其他缺包/冲突时先停止处理。
conda run -n gdb_glmocr python -c \
  'import site, torch, transformers, vllm, glmocr, psutil, pynvml; assert not site.ENABLE_USER_SITE; print(torch.__version__, torch.version.cuda, transformers.__version__, vllm.__version__); print(psutil.__file__, pynvml.__file__)'
```

布局模型需要环境内 `site-packages/nvidia/cu13/lib` 的 CUDA 库；项目 launcher 已设置
`LD_LIBRARY_PATH`，不可按 Python imports 删减这些动态库。随后按运行命令验证完整服务＋SDK，
不能只用 `/v1/models` 或 import 代替端到端验收。

迁移时按本配方重建，再用根目录 `env/locks_rtx3090_20260720/gdb_glmocr.*` 核对版本。
