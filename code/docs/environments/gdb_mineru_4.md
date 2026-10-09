# `gdb_mineru_4`

本页只记录环境安装和迁移配方；实验进展、结果及验证范围统一维护于
[PROJECT_STATE.md](../../../PROJECT_STATE.md)。

## 路线与版本依据

`mineru_4011_standard` 使用官方 **MinerU 4.0.11 PyPI wheel**、Standard 档位、Torch 小模型和
本地 vLLM；通过官方 V1 文档解析 API 与 `MinerUApiParser` 执行完整 PDF 流程。
不覆盖 `gdb_mineru` / `gdb_mineru_303`，不使用 3.x 的 JSON 配置或 `/tasks`。

当前环境由用户克隆 `gdb_glmocr` 后安装。以下是项目选定的可复现组合，
不是声称官方逐包精确推荐：Python 3.10.20、MinerU 4.0.11、Torch/torchaudio 2.11.0、
torchvision 0.26.0、Transformers 5.14.1、vLLM 0.20.0、tokenizers 0.22.2、
mineru-vl-utils 2.0.5、docvortex 0.5.13。官方安装入口为 `mineru[full]`；
各可选引擎仍须满足对应 wheel、驱动和平台要求。

参考源码在 `third_party/methods/mineru4/`，revision 由 `config/methods.json` 维护。
官方 `mineru-4.0.11-released` tag 的 `version.py` 写作 4.0.10；与当前安装包的
`mineru/` 目录比对仅此文件不同。因此登记为 `reference_checkout`，实际运行固定 wheel，
不把参考源码加入 `PYTHONPATH`，不冒充源码安装的 4.0.11。

## 重建命令

先按 [通用安装设置](README.md#迁移时的安装流程) 初始化 Conda 和路径。
环境不存在时，选择一种创建方式，不要覆盖已有环境：

```bash
# 与当前服务器创建方式一致；不改动 GLM 原环境
conda create -n gdb_mineru_4 --clone gdb_glmocr -y
# 新服务器没有 GLM 环境时，可改为：
# conda create -n gdb_mineru_4 python=3.10 pip -y

conda activate gdb_mineru_4
type unsetproxy >/dev/null 2>&1 && unsetproxy
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
unset PYTHONPATH PYTHONHOME PIP_EXTRA_INDEX_URL PIP_PROXY
export PYTHONNOUSERSITE=1
export GDB_ROOT=/data/wangsq/code/global_doc_bench
export TMPDIR="$GDB_ROOT/tmp"
mkdir -p "$TMPDIR/mineru4_install"

# 当时华为镜像尚未同步 4.0.11：只下载 MinerU wheel 时直连官方 PyPI
python -m pip download --no-deps \
  --index-url https://pypi.org/simple \
  --dest "$TMPDIR/mineru4_install" 'mineru==4.0.11'

# 其余依赖显式走镜像，保留当前 Torch/vLLM 栈
python -m pip install \
  --index-url https://repo.huaweicloud.com/repository/pypi/simple \
  "$TMPDIR/mineru4_install/mineru-4.0.11-py3-none-any.whl[full]" \
  'torch==2.11.0' 'torchaudio==2.11.0' 'torchvision==0.26.0' \
  'transformers==5.14.1' 'vllm==0.20.0' 'tokenizers==0.22.2' \
  'mineru-vl-utils==2.0.5' 'docvortex==0.5.13'

python -m pip check
mineru version --json
```

镜像缺少其他包时，先核对来源，不开启代理或混入多个索引。
克隆环境中会保留其他方法的依赖；本配方不包含环境裁剪。
当前三份锁保存实际状态，不保证 fresh-create 与 clone 的全部包完全相同。
pip freeze 中 MinerU 指向本机临时 wheel，迁移时按上述命令重新下载相同版本，
不能依赖旧服务器的 `file://` 路径。

## 模型路径与必要变量

按 [模型说明](../MODELS.md#mineru-4011-standard) 准备并使用官方 `models verify` 检查：
`MinerU-4_models_torch` 和 `MinerU2.5-Pro-2605-1.2B`。旧版 2605 VLM 可共享，
但新小模型包不能用 `PDF-Extract-Kit-1.0` 替代。完成标记必须由官方 downloader 生成。

调度器自动设置 `MINERU_MODEL_BASE_DIR`、`MINERU_MODEL_SOURCE=local`、
`MINERU_MODEL_SMALL_BACKEND=torch`、`MINERU_MODEL_VLM_ENGINE=vllm`，
并为本地服务设置 `NO_PROXY/no_proxy`。不依赖用户级 Python 包或旧 `mineru.json`。
并发、预加载、计时和失败规则见 [吞吐协议](../THROUGHPUT_TUNING_PROTOCOL.md#mineru-4-standard-的并发边界)。

## 环境锁

三份实际环境锁保存于 `env/locks_rtx3090_20260720/gdb_mineru_4.*`。
仅在依赖实际变化后更新锁；修改 adapter 或执行测试不需要重新导出。
通用运行命令见 [RUN_COMMANDS.md](../RUN_COMMANDS.md)。
