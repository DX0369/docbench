# `gdb_marker`

## 用途与版本依据

运行 Marker/Surya 原生 parser；唯一项目路线为 `marker_shared_workers`，不需要 vLLM。
并发、200/200 DPI benchmark 设置及选定 P=4 见[吞吐协议](../THROUGHPUT_TUNING_PROTOCOL.md#marker-共享模型文档-worker)；
已完成有界内容和吞吐核对，具体运行命令及边界见 [RUN_COMMANDS.md](../RUN_COMMANDS.md) 与吞吐协议。

- 实装：Python **3.10.20**、Marker **1.7.5**、Surya **0.14.7**、
  Torch **2.7.0+cu126**、torchvision **0.22.0+cu126**、Transformers **4.53.3**。
- 锁定源码来自 `jerzydziewierz/marker_ocr` fork，revision 以 `config/methods.json` 为准。
  上游范围为 Torch `^2.7.0`、Transformers `^4.45.2`、Surya `^0.14.6`；
  Surya 0.14.7 进一步要求 Transformers `>=4.51.2,<4.54`。上述精确版本是项目选择，
  不是上游唯一推荐组合。

## 重建步骤

先完成[通用安装设置](README.md#目录与命名)，
按方法登记表恢复精确 fork revision，再安装；不要用 PyPI 的同版本 Marker 替代源码。
以下核心 pins 不能固定全部传递依赖，安装后仍须与环境锁核对。

```bash
conda create -n gdb_marker python=3.10 pip setuptools wheel -y

conda run -n gdb_marker python -m pip install \
  'torch==2.7.0' 'torchvision==0.22.0' \
  'transformers==4.53.3' 'surya-ocr==0.14.7'
conda run -n gdb_marker python -m pip install \
  -e "$GDB_METHOD_ROOT/marker"
```

当前实测是 CUDA **12.6** 构建；freeze 元数据只显示 `torch==2.7.0`，应同时核对
下方 import 输出的 `+cu126` 和 `torch.version.cuda`，不能把其他 CUDA wheel 当成同一环境。
完整 YAML 只记录 `marker-pdf==1.7.5`，丢失 editable/fork 身份；源码恢复步骤不能省略。

## 安装后核对（锁文件仅用于迁移比对）

```bash
PYTHONNOUSERSITE=1 conda run -n gdb_marker python -m pip check
PYTHONNOUSERSITE=1 conda run -n gdb_marker python -c \
  'import torch, torchvision, transformers; print(torch.__version__, torch.version.cuda, torchvision.__version__, transformers.__version__)'
```

2026-10-03 隔离检查通过；没有为主线整理安装、卸载或升级包。保留上游声明依赖，
不因 `use_llm=False` 删除 Google/OpenAI/Anthropic SDK；部分服务在导入阶段即引用 SDK。
torchvision 尚无可安全移除的验证，不为精简改变已测依赖组合。

现有环境锁位于 [环境锁目录](../../../env/locks_rtx3090_20260720/)；新服务器需要时再按
按 [RUN_COMMANDS.md](../RUN_COMMANDS.md) 执行有界运行核对。
