# 方法环境安装配方

本目录是环境安装和迁移的唯一入口：本文件维护通用设置，各环境配方维护自己的版本、安装命令和已知问题。运行命令见 [RUN_COMMANDS.md](../RUN_COMMANDS.md)，吞吐规则见 [THROUGHPUT_TUNING_PROTOCOL.md](../THROUGHPUT_TUNING_PROTOCOL.md)。

这些配方不是 lock 文件本身。锁文件统一位于根目录 `env/locks_rtx3090_20260720/`，用途和文件类型见
[`env/README.md`](../../../env/README.md)；迁移时按配方重建，再用对应锁文件核对。

## 服务器迁移时的依据优先级

迁移或重装环境时按以下优先级决策：

1. 当前锁定第三方源码 revision 内的官方安装文档；
2. 当前源码的 `requirements*.txt`、`pyproject.toml`、`setup.py` 和容器说明；
3. 本目录记录的已验证版本、冲突和安装顺序；
4. 已保存的环境快照，仅用于安装后的版本核对；
5. 旧服务器/A800 环境导出，仅作为历史候选版本参考。

不要直接使用旧服务器的完整 YAML 或 pip freeze 创建环境。旧导出可能包含构建机
本地路径、旧 CUDA wheel、已废弃环境、旧第三方源码和不再成立的依赖组合。

## 查看官方环境要求

先确认当前实际源码 revision，再在对应方法仓库内搜索安装入口：

```bash
cd "$GDB_METHOD_ROOT/<method>"
git rev-parse HEAD

find . -maxdepth 3 -type f \
  \( -iname 'README*' -o -iname 'requirements*.txt' \
     -o -name 'pyproject.toml' -o -name 'setup.py' \
     -o -iname 'Dockerfile*' \) \
  -print

rg -n -i \
  'install|python|torch|cuda|vllm|transformers|flash.?attn|paddle|optional-dependencies' \
  README* docs requirements*.txt pyproject.toml setup.py 2>/dev/null
```

阅读时区分四类信息：

- 官方精确固定版本，例如 `vllm==0.18.1`；
- 官方允许范围，例如 `torch>=2.6,<3`；
- 不同 backend 的互斥依赖，例如 vLLM、Transformers、LMDeploy；
- 可选组件与实际硬依赖，最终以真实 import/model smoke 为准。

如果官方 README、包元数据和实际 import 相互矛盾，保留错误日志，在独立环境中做
最小修正，并把最终验证结果写回对应环境配方。不要为了消除一条元数据警告而覆盖
整套 Torch/vLLM/Transformers 栈。

## 迁移时的安装流程

1. 记录新服务器 GPU、驱动、系统和磁盘信息。
2. 按 `config/methods.json` checkout 第三方源码。
3. 按本目录逐环境创建 Miniconda 命名环境，不在 `base` 安装方法依赖。
4. 按 [MODELS.md](../MODELS.md) 下载模型并准备 Marker、PaddleX、MinerU 项目缓存。
5. 按需要运行 readiness、smoke 或吞吐测试；这些步骤分别记录在对应文档中。

锁文件只用于迁移后的版本核对，不是每次安装结束都必须重新导出的产物。

各配方应分别说明：**官方要求**（对应 backend 和源码 revision）、**项目选定版本**
（范围内锁定或有说明的兼容修正）、**实际验证范围**（import、smoke 或有界吞吐）。
三者不能互相替代，项目补丁也不能描述成未经修改的官方实现。

## 目录与命名

统一使用 Miniconda 的命名环境：

```text
/data/wangsq/software/miniconda/envs/<environment-name>
```

这样与当前代码中的 `CONDA_ENV_ROOT`、`conda activate <name>` 和批跑参数保持一致。不要同时使用 `/data/wangsq/software/conda-envs/` 前缀环境，除非先修改代码使环境根目录可配置。

环境配方不重复锁定第三方源码 commit；源码 revision 统一来自 `config/methods.json`。安装前应确认工作树实际 commit 与该锁文件一致。

通用路径：

如终端尚未初始化 Conda，先执行（路径按新服务器调整）：

```bash
source /data/wangsq/software/miniconda/etc/profile.d/conda.sh
```

```bash
export GDB_ROOT=/data/wangsq/code/global_doc_bench
export GDB_METHOD_ROOT="$GDB_ROOT/third_party/methods"
export GDB_MODEL_ROOT="$GDB_ROOT/models"
export PIP_CACHE_DIR=/data/wangsq/.cache/pip

command -v conda
conda --version
```

安装前在专用安装终端执行以下设置；仅影响当前 shell 和后续子进程，不修改全局配置：

```bash
type unsetproxy >/dev/null 2>&1 && unsetproxy
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
unset PYTHONPATH PYTHONHOME PIP_EXTRA_INDEX_URL PIP_PROXY
export PYTHONNOUSERSITE=1
export PIP_INDEX_URL=https://repo.huaweicloud.com/repository/pypi/simple
export UV_DEFAULT_INDEX=https://repo.huaweicloud.com/repository/pypi/simple
export TMPDIR="$GDB_ROOT/tmp"
mkdir -p "$TMPDIR"
```

uv 不读取 `pip.conf`；单条命令还可显式使用 pip 的 `--index-url` 或 uv 的
`--default-index`。专用 Torch/Paddle 索引、官方 wheel URL 保留各配方要求，
国内镜像只提供可用的 PyPI 包，不保证包含专用构件。当前优先华为云，清华可作替代，
不保证某个源永远最快。镜像缺包时先确认来源，再单次直连官方源，不自动开启代理、
不永久混入多个通用源。清代理变量不会停止代理软件，也不会清除 Git/工具的独立代理配置。
日志应核对实际下载地址；镜像设置不会改写显式 wheel URL。超时设置不等于下载加速。

Marker/Surya、PaddleX 和 MinerU 的模型缓存与本地配置见 [MODELS.md](../MODELS.md)。
必须在对应方法启动前设置其变量；通用名称 `MODEL_CACHE_DIR` 不应全局写入 `.bashrc`，
避免污染其他方法环境。

只有编译 CUDA 扩展时才设置：

```bash
export CUDA_HOME=/data/wangsq/software/cuda-envs/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

预编译 wheel 通常不需要激活 CUDA toolkit 环境。编译前核对 nvcc、目标 Torch CUDA
runtime 及 ABI 的兼容性，不能因为本机有 CUDA 12.8 就直接编译其他 runtime 的扩展。

## 环境矩阵

| 环境 | 方法 | Python | 已验证关键版本 | 版本依据 |
|---|---|---:|---|---|
| `gdb_hunyuan` | HunyuanOCR 1.0/1.5 release vLLM | 3.10 | vLLM 0.18.1 | 官方顶层精确约束 |
| `gdb_hunyuan_dflash` | HunyuanOCR-1.5 DFlash | 3.12 | vLLM dev869；Torch 2.11 cu130；Transformers 5.5.3；FlashInfer 0.6.13 | 官方验证区间 + 精确历史 wheel + 3090 smoke |
| `gdb_youtu` | Youtu-Parsing native HF | 3.10 | Torch 2.6.0；Transformers 4.51.3 | 官方精确锁定 |
| `gdb_marker` | Marker 1.7.5 | 3.10.20 | Torch 2.7.0+cu126；Transformers 4.53.3；Surya 0.14.7 | 锁定 fork 依赖范围＋项目已验证版本 |
| `gdb_mineru` | MinerU 3.4.x/2605 API/direct；Popo vLLM服务 | 3.10 | MinerU 3.4.3；Torch 2.9.0；Transformers 4.57.6；vLLM 0.11.2 | 官方范围 + 项目已验证版本；Popo单服务及双服务小smoke通过，20文档仍有容量/上下文/建树失败 |
| `gdb_mineru_303` | MinerU 3.0.3/2509 API/direct | 3.10 | Torch 2.9.0；Transformers 4.57.6；vLLM 0.11.2 | 原始路线显式版本；API/direct 单页 smoke |
| `gdb_mineru_popo` | MinerU-Popo CPU API客户端/后处理，保留历史TF依赖 | 3.10 | 原Torch/Transformers不变；新增loguru 0.7.3 | CPU导入及组合小smoke通过；20文档P=1/2/4未全通过，尚无稳定配置 |
| `gdb_glmocr` | GLM-OCR service/pipeline；LightOnOCR-2-1B、Xiaomi-OCR-0 service/worker | 3.10 | Torch 2.11.0 cu130；Transformers 5.14.1；vLLM 0.20.0 | 官方下界 + 项目已验证版本；已知 xgrammar 冲突 |
| `gdb_paddle` | PP-StructureV3、PaddleOCR-VL pipeline/client | 3.10 | PaddlePaddle GPU 3.2.1 cu126；PaddleX 3.7.x | 官方精确/范围约束 |
| `gdb_paddle_vllm` | PaddleOCR-VL vLLM service | 3.10 | Torch 2.8.0 cu128；Transformers 4.57.6；vLLM 0.10.2 | 官方动态配方 + 项目已验证解析结果 |
| `gdb_monkey` | MonkeyOCR Pro-3B / Pro-1.2B vllm_async + PP-DocLayout | 3.10 | Torch 2.7.0 cu126；vLLM 0.9.1；Transformers 4.51.0；NumPy <2 | 官方 vLLM 分支 + 项目兼容约束/裁剪 |
| `gdb_deepseek_ocr2` | DeepSeek-OCR-2 | 3.12.9 | Torch 2.6.0+cu124；vLLM 0.8.5；Transformers 4.46.3 | 官方核心版本＋项目cu124组合；保留两项已记录冲突 |
| `gdb_qianfan` | Qianfan OCR / think | 3.10 | Torch 2.10.0 cu128；Transformers 4.57.6；vLLM 0.18.1 | 官方未锁版本；项目已验证版本 |
| `gdb_teleocr` | TeleOCR vllm-async-engine | 3.10 | Torch 2.8.0；Transformers 4.57.1；vLLM 0.11.0；xformers 0.0.32 | 官方 requirements/插件；排除可选 Gradio UI 冲突依赖 |
| `gdb_glmocr`（LightOnOCR-2 复用） | GLM-OCR；LightOnOCR-2-1B vLLM page pool | 3.10 | Torch 2.11.0；Transformers 5.14.1；vLLM 0.20.0；Pillow；pypdfium2 5.12.1 | LightOnOCR-2 不重复安装依赖；当前不创建独立环境 |
| `gdb_unlimited_ocr_sglang` | Unlimited OCR 官方定制SGLang service | 3.12.13 | SGLang dev11416；Torch 2.9.1 cu128；Transformers 5.3.0；FlashInfer 0.6.7.post3 | 官方定制wheel＋SM86补丁；安装见配方，验证范围见smoke，测量见吞吐协议 |

可选比较环境见 [gdb_hunyuan_optional.md](gdb_hunyuan_optional.md)。Hunyuan DFlash 和 native Transformers 与 release vLLM 明确不兼容，不应装进 `gdb_hunyuan`。

## 配方索引

- [gdb_hunyuan.md](gdb_hunyuan.md)
- [gdb_hunyuan_optional.md](gdb_hunyuan_optional.md)：独立 DFlash 环境及可选 backend。
- [gdb_youtu.md](gdb_youtu.md)
- [gdb_marker.md](gdb_marker.md)
- [gdb_mineru.md](gdb_mineru.md)
- [gdb_mineru_303.md](gdb_mineru_303.md)
- [gdb_mineru_popo.md](gdb_mineru_popo.md)：当前CPU API客户端与历史TF环境；旧smoke不能替代新共享vLLM验证。
- [gdb_glmocr.md](gdb_glmocr.md)
- [gdb_paddle.md](gdb_paddle.md)
- [gdb_paddle_vllm.md](gdb_paddle_vllm.md)
- [gdb_monkey.md](gdb_monkey.md)
- [gdb_deepseek_ocr2.md](gdb_deepseek_ocr2.md)
- [gdb_qianfan.md](gdb_qianfan.md)
- [gdb_teleocr.md](gdb_teleocr.md)
- [gdb_lightonocr2.md](gdb_lightonocr2.md)：说明 LightOnOCR-2 如何复用 `gdb_glmocr`，不重复安装依赖。
- [gdb_xiaomi_ocr.md](gdb_xiaomi_ocr.md)：Xiaomi-OCR-0 服务和 worker 复用 `gdb_glmocr`；依赖来源、恢复链接与迁移检查。
- [gdb_unlimited_ocr_sglang.md](gdb_unlimited_ocr_sglang.md)

## 安装后的最小核对

1. 根据配方创建环境并安装官方依赖、锁定源码或专用 wheel。
2. 在禁用 user site 的状态下执行 `pip check` 和最小 import probe。
3. 需要真实推理时，按 [RUN_COMMANDS.md](../RUN_COMMANDS.md) 选择正式入口；不要把运行结果或锁导出混入安装脚本。

不并行安装多个大型环境。顺序安装可以复用 Conda 与 pip 缓存，也更容易确认是哪个顶层依赖改变了 Torch 栈。

## 最小核对命令

每个环境至少执行：

```bash
ENV_NAME=gdb_hunyuan

unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1
conda run -n "$ENV_NAME" python -m pip check
conda run -n "$ENV_NAME" python -c \
  'import sys; print("python", sys.version); import torch; print("torch", torch.__version__); print("torch_cuda", torch.version.cuda); print("cuda_available", torch.cuda.is_available())'
```

没有 Torch 的纯客户端环境应把第二条换成对应框架的 import probe。不要在 `conda run` 后使用 heredoc；它不会可靠转发标准输入，可能出现命令返回成功但 Python 内容没有执行的假象。

若隔离后出现缺包，应在目标环境补齐并重新检查，不能恢复 user site 掩盖缺失。
`pip check` 不验证完整的 extras 使用需求；已知冲突也不等于无冲突。
`gdb_glmocr` 已持久配置 `PYTHONNOUSERSITE=1`；其他环境不能据此宣称已完成相同审计。
直接调用 Python 绝对路径不会加载 Conda 激活变量；迁移核对时使用 `conda run` 或先激活环境。
GLM 迁移时还需按其配方执行 `conda env config vars set`。历史快照未逐一重审，
不能仅凭 freeze 列出了某包就认定它安装在目标环境内。
GLM 的特殊 pip 核对方式按其[环境配方](gdb_glmocr.md)执行；不要用普通 freeze 覆盖已审计清单。

正式锁文件另行记录方法源码 commit、模型目录、GPU、驱动和实际命令。旧 A800 导出仅用于对比，不直接恢复。

## 推荐安装批次

1. 基础验证：`gdb_hunyuan`、`gdb_youtu`。
2. 原生与集成 parser：`gdb_marker`、`gdb_mineru`、`gdb_glmocr`。
3. 复合 pipeline：`gdb_paddle`、`gdb_paddle_vllm`、`gdb_monkey`、`gdb_teleocr`。
4. 高风险兼容栈：`gdb_deepseek_ocr2`、`gdb_qianfan`。
5. Unlimited-OCR 当前只保留 `gdb_unlimited_ocr_sglang` 正式路线；使用项目200 DPI、
   常驻整文档客户端。旧 `gdb_unlimited_ocr` 环境/锁保留，但不再进入当前readiness和调度器；
   HF adapter及旧环境配方可从主代码 Git `5c2b4a5` 恢复，新路线按运行入口和吞吐协议核对。

## 安装后检查代码默认值

Qianfan、DeepSeek-OCR-2、Youtu、Paddle pipeline 与 Paddle service 必须分别使用本文定义的独立环境。环境安装完成后、开始批跑前，执行调度 dry run，禁止重新创建一个混合环境来兼容历史默认值。
