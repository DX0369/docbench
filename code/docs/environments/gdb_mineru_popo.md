# gdb_mineru_popo

更新时间：2026-10-03 14:28 +08:00

## 用途与验证范围

当前用于 MinerU-Popo 的纯CPU文档客户端：官方归一化/分块/建树及共享API调用。
保留原Transformers依赖，但新入口不加载本地模型；不安装MinerU、vLLM、SGLang、flash-attn。
MinerU 服务使用 `gdb_mineru`；Popo vLLM 服务是独立进程。GPU 运行和吞吐参数见
[RUN_COMMANDS.md](../RUN_COMMANDS.md) 与吞吐协议，本文不重复记录实验结果。

历史Transformers环境已验证：Python 3.10.20、Torch 2.11.0+cu130、TorchVision 0.26.0+cu130、
Transformers 5.14.1、Accelerate 1.14.0、PyMuPDF 1.28.0、BeautifulSoup 4.13.5。
`pip check` 无冲突；CPU 3+5 组测试通过（后者含社区 11 项测试）；GPU 1 RTX 3090 的
三页论文通过官方推理与建树，23/23 块及 2/2 图片路径、图注保留。

本环境由用户安装，随后独立验证；验证时清除 PYTHONPATH/PYTHONHOME 并禁用 user site，
BeautifulSoup 来自本环境，已不依赖 `tmp/popo_smoke_deps`。没有升级或修改其他方法环境。

## 依据与源码

- 官方 `README_zh.md` 推荐 Python 3.10；[维护者说明](https://github.com/opendatalab/MinerU-Popo/issues/1#issuecomment-4577361548)
  允许参考 Qwen3-VL 环境并安装精简依赖。没有照搬包含金融/Spark 等无关包的完整 requirements。
- 下面固定版本来自本项目成功环境及独立环境 smoke，不是官方唯一指定版本。
- 源码：`third_party/methods/MinerU-Popo`，revision `97d560172361e772b3d079ce9eac35796b998782`。
- 当前必须应用同页候选、社区 PR #12/#15 与 `failure_propagation.patch`；后者让表格分支
  异常原样抛出，避免失败输出冒充成功。详见 [补丁说明](../../src/methods/mineru_popo/patches/README.md)。
- `resident_init.patch` 属于历史Transformers常驻支持；当前API客户端不调用其初始化函数。
- 模型：`models/mineru/MinerU-Popo`，来源 `DreamEternal/MinerU-Popo`。
- 本阶段不用 qwen-vl-utils、zss 的相关功能；未因官方完整清单列出它们而额外安装。

## 从零安装

以下路径适用于当前服务器，迁移时替换项目路径。环境已存在时不要重复创建或删除。
```bash
export GDB_ROOT=/data/wangsq/code/global_doc_bench
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1
mkdir -p "$GDB_ROOT/tmp"
export TMPDIR="$GDB_ROOT/tmp"

conda create -n gdb_mineru_popo python=3.10.20 pip -y
conda activate gdb_mineru_popo
export POPO_PY="$CONDA_PREFIX/bin/python"

"$POPO_PY" -m pip install \
  uv
export UV_LINK_MODE=copy
"$POPO_PY" -m uv pip install --python "$POPO_PY" \
  "torch==2.11.0" "torchvision==0.26.0" \
  "transformers==5.14.1" "tokenizers==0.22.2" \
  "accelerate==1.14.0" "huggingface-hub==1.24.0" \
  "safetensors==0.8.0" "numpy==2.2.6" \
  "pillow==12.3.0" "pymupdf==1.28.0" \
  "beautifulsoup4==4.13.5" "soupsieve==2.9.2" \
  "requests==2.34.2" "tqdm==4.69.0" "openai==2.46.0" "loguru==0.7.3"
```

本机从标准 wheel 安装后实际 CUDA runtime 为 13.0；其他平台必须重新核对 wheel 与驱动。
该流程不编译 CUDA 扩展，不要求额外安装 CUDA 13 Toolkit。

## 验收

```bash
"$POPO_PY" -m pip check
"$POPO_PY" - <<'PY'
import sys, torch, torchvision, transformers, accelerate, fitz, bs4
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
print(sys.executable)
print(torch.__version__, torch.version.cuda, torchvision.__version__)
print(transformers.__version__, accelerate.__version__, fitz.VersionBind)
print(bs4.__version__, bs4.__file__)
print('CUDA available:', torch.cuda.is_available())
assert torch.__version__ == '2.11.0+cu130'
assert transformers.__version__ == '5.14.1'
PY

cd "$GDB_ROOT"
REPO="$GDB_ROOT/third_party/methods/MinerU-Popo"
PATCHES="$GDB_ROOT/code/src/methods/mineru_popo/patches"
# 仅对尚未应用修复的锁定源码执行；已有环境用 --reverse --check 检查即可。
git -C "$REPO" apply --check "$PATCHES/same_page.patch" "$PATCHES/pr12.patch" "$PATCHES/pr15.patch" "$PATCHES/failure_propagation.patch"
git -C "$REPO" apply "$PATCHES/same_page.patch" "$PATCHES/pr12.patch" "$PATCHES/pr15.patch" "$PATCHES/failure_propagation.patch"
测试源已在定版前验证后删除。迁移时使用 [运行命令](../RUN_COMMANDS.md) 中的 Popo
组合入口，并用 `validate_outputs.py` 检查生成的输出目录。
```

定版前的核心 adapter 检查使用自建小样例，不再依赖历史 smoke 输出；测试源已删除，
不替代真实模型链路验证。

历史诊断脚本和一次性输出已清理，不再引用实验目录内的临时脚本进行复现。
请使用 [RUN_COMMANDS.md](../RUN_COMMANDS.md) 中受版本管理的正式 adapter 命令，
将新输入、输出和日志放在 `tmp/smoke/<run_id>/`。历史环境验证使用官方 BF16，
曾复用已有 MinerU middle JSON，因此不包含 MinerU 重新计时。
图注保留为元数据，非独立图注关联候选；跨页表格实际合并及吞吐未验证。

## 共享API客户端补包（2026-10-02 23:59 +08:00）

用户授权后，仅新增 `loguru==0.7.3`，与已有MinerU环境相同。它是复用锁定MinerU
官方API超时工具所需的依赖，不是模型/采样修改。PDF计数复用已有PyMuPDF，未装PDFium。

```bash
env -u PYTHONPATH -u PYTHONHOME PYTHONNOUSERSITE=1 \
  /data/wangsq/software/miniconda/envs/gdb_mineru_popo/bin/python -m pip install \
  --no-deps loguru==0.7.3
```

实际CPU导入归一化/推理/建树与MinerU超时工具通过，确认未初始化Transformers模型；
隔离pip check通过。该验证不调用模型服务，不代表vLLM环境或双服务显存就绪。
官方README仅提供API配置及Docker latest，没有可直接锁定的pip版vLLM版本；
服务须支持本checkpoint的Qwen3VLForConditionalGeneration，并单独做GPU验证。

## Popo vLLM服务：复用gdb_mineru

2026-10-03 GPU3单服务验证通过：Python3.10.20、vLLM0.11.2、
Torch2.9.0+cu128、Transformers4.57.6，解释器为
`/data/wangsq/software/miniconda/envs/gdb_mineru/bin/python`。
隔离user site后的pip check、模型注册、本地AutoConfig/AutoTokenizer/AutoProcessor通过；
真实纯文本表格行请求及带图标题请求正常结束，官方解析器可解析，标题ID覆盖通过。
这不是完整组合、自然触发的续表合并或吞吐验证，详见吞吐协议中的当前限制。

早期同卡组合P=1实测：MinerU预加载成功后，Popo启动时仅余8.84/23.56 GiB，低于
0.40预算所需9.42 GiB，在加载Popo权重前拒绝启动；不是CUDA OOM或依赖错误。
两份5页PDF当时均未处理。随后经用户确认降低仅组合路线的MinerU预算，0.20/0.68
已通过同卡加载及两文档smoke；独立MinerU0.55不变。单服务权重加载实测8.591 GiB，
仍需KV/cache/视觉激活；该容量不是长文档/高并发已验证的最优设置。

迁移时恢复[gdb_mineru配方](gdb_mineru.md)及其现有环境锁，不向本CPU客户端环境
安装vLLM，也不额外创建同栈环境。vLLM0.11.2要求Transformers<5，与本客户端的5.14.1
不同；通过HTTP隔离依赖。官方支持vLLM但未指定此精确版本，本栈属于项目实测版本。
本环境只负责客户端和依赖安装。Popo 服务启动、健康检查及请求参数见
[RUN_COMMANDS.md](../RUN_COMMANDS.md)。

迁移时按本配方重建，再用根目录 `env/locks_rtx3090_20260720/gdb_mineru_popo.*`
核对版本；Conda explicit 不含 pip 安装的模型依赖。

当前 Popo 服务默认上下文为 `32768`，可通过 `--popo-max-model-len` 显式覆盖；
生成请求仍固定为 `max_tokens=8192`。该默认值用于统一命令和迁移复现，不能视为
双模型组合在 RTX 3090 上已经完成高并发容量验证。
