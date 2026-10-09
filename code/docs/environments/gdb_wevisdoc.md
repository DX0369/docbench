# WeVisDoc：复用 `gdb_glmocr`

## 官方依赖与当前环境

WeVisDoc-2B 的 vLLM 服务和页面 worker 复用 `gdb_glmocr`，不创建独立环境，
不额外安装或升级包。仅接入 vLLM 路线，不安装 `requirements-local.txt`。

官方安装说明见 [README.md](../../../third_party/methods/WeVisDoc/README.md#serve-with-vllm)。
[requirements-vllm.txt](../../../third_party/methods/WeVisDoc/requirements-vllm.txt)
要求 `vllm>=0.11.1,<1`，并引用客户端
[requirements.txt](../../../third_party/methods/WeVisDoc/requirements.txt)，要求
`openai>=1.40.0,<3`。官方 README 要求 Python 3.10+；vLLM 路线没有单独固定
Torch 或 Transformers 版本。

当前复用版本为 Python 3.10、vLLM 0.20.0、OpenAI SDK 2.46.0、Torch 2.11.0、
Transformers 5.14.1。这些是项目共享环境的实际版本，不是 WeVisDoc 官方精确锁定版本。
PDF 页面渲染使用项目公共 Poppler 工具 `pdfinfo`/`pdftoppm`；worker 从第三方
源码目录直接导入官方 prompt 和消息构造函数，不需要额外 editable 安装。

## 安装与迁移

按 [gdb_glmocr.md 的恢复流程](gdb_glmocr.md#恢复已验证环境) 建立共享环境，
镜像与代理设置沿用 [通用环境说明](README.md)。不要在现有共享环境中重新执行
官方宽范围依赖安装，以免改变其他方法使用的依赖栈。

恢复后可执行以下最小检查，不启动模型或 GPU 服务：

```bash
export GDB_ROOT=/data/wangsq/code/global_doc_bench
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1

conda run -n gdb_glmocr python -c '
import os, site, sys
from importlib.metadata import version
import openai, torch, transformers, vllm
assert not site.ENABLE_USER_SITE
sys.path.insert(0, os.path.join(os.environ["GDB_ROOT"], "third_party/methods/WeVisDoc"))
from wevisdoc.common import DEFAULT_PROMPT, DEFAULT_SYSTEM_PROMPT, build_messages
print("Python", sys.version)
for package in ("openai", "vllm", "torch", "transformers"):
    print(package, version(package))
print("WeVisDoc official helpers imported")
'

command -v pdfinfo
command -v pdftoppm
conda run -n gdb_glmocr python -m pip check
```

共享环境已有 xgrammar 0.2.4 要求 `transformers<5`、实际安装 Transformers 5.14.1
的元数据冲突，`pip check` 不能描述为无冲突；详情见 `gdb_glmocr.md`。
若出现其他缺包或冲突，应先核查，不通过启用用户级包目录掩盖问题。

## 项目 eager 兼容设置

当前 vLLM 0.20.0 / Torch 2.11.0 栈在 RTX 3090 上启动 WeVisDoc 时出现
`aot_compile is not supported by the current configuration`。项目
[服务命令](../../src/methods/wevisdoc/commands.py) 因此显式添加 `--enforce-eager`，
禁用相关编译和 CUDA Graph 优化路径。

该参数不是 WeVisDoc 官方默认设置，未修改第三方源码或模型；可能影响性能，
当前执行路线应视为官方 vLLM 路线加项目 eager 兼容设置。根因尚未完全定位，
不能据此认定所有 RTX 3090 或其他版本栈都必须启用。以后移除此设置时，
应重新验证服务启动与真实推理，并同步更新本节。
