# GlobalDocBench Integration Code

本仓库保存 GlobalDocBench 的 benchmark adapter、批跑工具、输出校验和分析代码。当前第三方源码、模型、方法环境和 active adapter 已完成版本绑定及 RTX 3090 单页 smoke；下一阶段是多页验证、吞吐调参与正式实验。

## 项目布局

`code/` 只是整个项目的一部分。相关目录位于它的上一级：

```text
global_doc_bench/
├── code/                  # 当前 Git 仓库：adapter、脚本和文档
│   ├── config/            # 第三方仓库版本锁等配置
│   ├── docs/              # 当前维护的工程、环境和模型文档
│   └── src/               # methods、benchmark、execution、config、validation 五类源码
├── third_party/methods/   # 按版本锁检出的第三方 Git 仓库
├── data/                  # 当前 benchmark 输入
├── models/                # 当前模型权重
├── env/                   # 当前 RTX 3090 已验证环境锁
├── experiments/           # tuning/<run_id> 调参；benchmark/<run_id> 正式结果
├── tmp/                   # smoke、调参中间文件和可再生运行配置
├── archive/               # A800 旧实验、源码快照、环境和历史文档
└── tech_report/           # 方法技术报告
```

大型数据、模型、当前实验输出和第三方仓库不纳入 `code/` 的 Git 历史。第三方仓库各自保留 `.git`，其正式实验版本由 `config/methods.json` 记录。

固定输入清单：`config/throughput_manifests/rtx3090_20.json`；选定运行参数：
`config/throughput_profiles/rtx3090.json`。后者保留Marker和Unlimited joint的既有P4配置；
新搜索使用 `run_pressure_sweep.py --save-profile <JSON>`显式保存，
`run_single_gpu.py --pressure P`或`--profile-file <JSON>`固定运行一次。
另有 `rtx3090_4.json`（4文档/10页）供首次有界检查；命令与产物规则见吞吐文档。

## 从哪里开始

普通批量与吞吐入口统一在 `src/benchmark/`；迁移、资产和输出检查统一在
`src/validation/`，第三方仓库管理入口为 `src/method_repositories.py`。
源码结构与 CPU 验证命令见 [源码导航](src/README.md)。旧分析/队列工具已归档到
`../archive/code/src_pre_consolidation_6f608b9/`；完整旧代码可从提交 `6f608b9` 恢复。

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

主 Git 仓库是 `code/`，跟踪核心代码、脚本、稳定配置/manifest 和必要文档；
不提交数据、模型、第三方仓库本体、环境目录、缓存或实验产物。第三方仓库保留
自己的 Git，选定 revision 由 `config/methods.json` 管理。

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
[第三方源码升级](docs/METHOD_REPOSITORY_UPGRADES.md)。项目根目录的 `PROJECT.md` 和
`PROJECT_STATE.md` 是工作区文档，不随单独克隆 `code/` 自动恢复；安装、模型下载、
补丁和运行协议的唯一说明保留在本仓库，不依赖根目录状态文档才能复现。
