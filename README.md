# GlobalDocBench

多页文档解析测评框架，集成不同方法的官方推理路线，提供统一输出、单卡吞吐搜索、固定 P 运行和逐文档 latency 入口。

## 仓库内容

- [`code/`](code/)：方法 adapter、配置、执行入口和使用文档。
- [`third_party/methods/`](third_party/methods/)：受版本清单管理的第三方源码快照及必要构件；保留各项目的许可证和来源说明。
- [`env/`](env/)：环境版本锁及安装后核对材料，不是实际 Conda 环境。
- [`PROJECT.md`](PROJECT.md)：研究范围和项目约定；[`PROJECT_STATE.md`](PROJECT_STATE.md)：验证进展、已知限制和待办。

仓库不包含输入、预测、实验产物、模型权重、实际环境、缓存、论文 PDF 或历史归档。模型按官方仓库名称重新下载，不额外锁定模型 snapshot；下载时须保留 config、processor、tokenizer 和 remote code，不能只下载权重。

## 新机器恢复顺序

```bash
git clone https://github.com/DX0369/docbench.git
cd docbench
export GDB_ROOT="$(pwd)"
export GDB_CONDA_ROOT="$HOME/software/miniconda"  # 改成新机器实际的 Miniconda 路径
export GDB_DATA_ROOT="$GDB_ROOT/data"
export GDB_MODEL_ROOT="$GDB_ROOT/models"
```

1. 按[环境配方](code/docs/environments/README.md)创建所需方法环境，再以 [`env/`](env/) 中的锁核对版本。第三方源码已包含在本仓库，不需要重新 clone 或重复应用现有补丁。
2. 按[模型下载说明](code/docs/MODELS.md)准备模型和必要辅助组件。
3. 按[运行命令](code/docs/RUN_COMMANDS.md)选择输入、GPU 和独立输出目录；迁移后先做有界 smoke，再决定是否正式运行。

路径设置只影响当前终端。CUDA 路径和历史 profile 的迁移说明见[运行命令](code/docs/RUN_COMMANDS.md#迁移时的路径与历史-profile)。现有 P4 profile 是历史配置，不是已确认最优配置；新机器安装和推理尚需独立验证。

## 其他说明

- [代码导航与 Git 工作流](code/README.md)
- [吞吐与 latency 协议](code/docs/THROUGHPUT_TUNING_PROTOCOL.md)
- [方法、模型、环境与源码绑定](code/config/methods.json)
