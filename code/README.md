# GlobalDocBench Integration Code

`code/` 保存 GlobalDocBench 的 benchmark adapter、批跑工具和输出校验代码。当前进展只记录在
[`PROJECT_STATE.md`](../PROJECT_STATE.md)。

## 项目布局

Git 仓库是项目根目录 `global_doc_bench/`，`code/` 是其中的一个目录：

```text
global_doc_bench/          # 唯一 Git 仓库
├── code/                  # adapter、脚本和文档
│   ├── config/            # 第三方仓库版本锁等配置
│   ├── docs/              # 当前维护的工程、环境和模型文档
│   └── src/               # methods、benchmark、execution、config、validation 五类源码
├── third_party/methods/   # 按 revision 保存的第三方源码（普通文件纳入根仓库）
├── data/                  # 当前 benchmark 输入
├── models/                # 当前模型权重
├── env/                   # 当前 RTX 3090 已验证环境锁
├── experiments/           # tuning/<run_id> 调参；benchmark/<run_id> 正式结果
├── tmp/                   # smoke、调参中间文件和可再生运行配置
└── archive/               # A800 旧实验、源码快照、技术报告和历史文档（不入 Git）
```

数据、模型、运行环境、实验输出、缓存和 `archive/` 由根目录 `.gitignore` 排除。第三方源码的上游
URL 和 revision 记录在 `config/methods.json`；除 `monkeyocrv2` 外，快照不保留上游 `.git`。

固定输入清单：`config/throughput_manifests/rtx3090_20.json`；选定运行参数：
`config/throughput_profiles/rtx3090.json`。后者保留Marker和Unlimited joint的既有P4配置；
新搜索使用 `run_pressure_sweep.py --save-profile <JSON>`显式保存，
`run_single_gpu.py --pressure P`或`--profile-file <JSON>`固定运行一次。
另有 `rtx3090_4.json`（4文档/10页）供首次有界检查；命令与产物规则见吞吐文档。

## 从哪里开始

普通批量与吞吐入口统一在 `src/benchmark/`；迁移、资产和输出检查统一在
`src/validation/`，第三方仓库管理入口为 `src/method_repositories.py`。
源码结构与 CPU 验证命令见 [源码导航](src/README.md)。旧分析/队列工具的源码快照保存在
`../archive/code/src_pre_consolidation_6f608b9/`；对应旧提交不在当前仓库历史中。

- [项目目标与约定](../PROJECT.md)、[当前状态与下一步](../PROJECT_STATE.md)
- [环境安装与迁移](docs/environments/README.md)
- [源码与方法版本管理](docs/METHOD_REPOSITORY_UPGRADES.md)
- [吞吐规则、方法机制与实验记录](docs/THROUGHPUT_TUNING_PROTOCOL.md)
- [迁移后的运行命令](docs/RUN_COMMANDS.md)
- [Runner 说明](src/README.md)
- [模型下载说明](docs/MODELS.md)
- [旧实验归档说明](../archive/README.md)

## 当前边界

- 版本锁和源码同步成功只说明源码可追溯，不代表新版方法与现有 adapter 兼容。
- 旧 A800 结果不能与 RTX 3090 上的新质量或吞吐结果直接拼接。
- 每个方法需要依次完成环境安装、模型检查、import/CLI smoke、输出协议校验和完整实验。
- 正式结果必须记录源码 revision、模型名称、环境、GPU、参数和输入清单。

多数逐页 VLM 即使输出一个 `document.md`，也不代表模型一次读取了整份 PDF；
必须区分逐页拼接与整文档原生推理。不可靠的原生页级时间应记为 `N/A`，不得伪造。
输出协议及 document-only 方法的例外见 [吞吐协议](docs/THROUGHPUT_TUNING_PROTOCOL.md)。

## Git 管理

根仓库跟踪 `code/`、`env/`、`third_party/` 和根级项目文档；不提交数据、模型、运行环境、缓存、
实验产物或 `archive/`。

按明确路径暂存，用一次提交表达一个完整改动；提交前检查差异和大文件，不直接 `git add .`：

```bash
git status --short
git diff
git add -- path/to/changed_file
git diff --cached --stat
git diff --cached
git commit -m "Describe the change"
```

第三方版本升级应同时提交版本映射、相关 adapter/config 和必要验证说明；具体流程见
[第三方源码升级](docs/METHOD_REPOSITORY_UPGRADES.md)。
