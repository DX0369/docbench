# 项目说明

更新时间：2026-10-04 14:06 +08:00

## 研究目标与范围

GlobalDocBench 用统一的多页 PDF 输入，比较文档解析方法的准确率和单卡吞吐量。

- 准确率路线尽量保持官方模型、prompt、图像处理、生成参数和后处理。
- 吞吐路线只在不改变输出语义的前提下，使用官方支持且适合方法特性的 backend、batch、并发和队列机制。
- 吞吐调参是有界工程搜索，不把一次小规模实验描述为理论最优。
- 当前方法族包括 HunyuanOCR、Qianfan-OCR、DeepSeek-OCR-2、PaddleOCR-VL、GLM-OCR、MonkeyOCR、MinerU、Youtu、Marker、PP-StructureV3、Unlimited-OCR，以及 MinerU-Popo 组合。
- 正式比较按单张 RTX 3090 执行，不将多卡总吞吐与单卡结果混排。

## 顶层目录

- `code/`：主 Git 仓库，包含实现、配置、稳定入口和项目文档。
- `code/src/`：方法 adapter、benchmark 入口、执行工具和必要核对工具。
- `code/config/`：方法绑定、源码锁、模型映射、输入清单和吞吐 profile。
- `data/`：输入 PDF 和数据清单。
- `models/`：按版本组织的模型权重和项目级缓存。
- `third_party/methods/`：按精确 revision 保存的第三方源码，各目录独立维护 Git。
- `env/`：环境锁文件和迁移用快照；具体安装命令位于 `code/docs/environments/`。
- `experiments/tuning/<run_id>/`：吞吐调参结果及该轮必要诊断。
- `experiments/benchmark/<run_id>/`：正式运行结果及该轮必要诊断。
- `tmp/`：一次性 smoke、调试和可再生中间产物，完成核对后清理。
- `archive/`：历史代码、实验和迁移材料，不作为当前默认入口或配置来源。

## 稳定约定

- 每个方法必须绑定显式 method ID、模型、源码 revision、环境、adapter 和 execution profile。
- 精确历史源码、兼容源码和仅作参考的源码必须明确区分，不能用含糊的版本名代替 revision。
- 输入 PDF 通过固定 manifest 引用，不复制到实验目录；运行使用独立 `run_id`，不得覆盖历史结果。
- 吞吐调参与正式运行分开；调参使用 `run_pressure_sweep.py`，固定配置运行使用 `run_single_gpu.py`。
- 正式吞吐默认使用一张 GPU。方法可根据自身机制使用页面任务、文档客户端、区域请求或原生 batch 作为压力参数，但不强行把 P 解释成同一种内部量。
- 计时、ready、成功/失败和输出保存遵循统一协议；失败必须显式记录，不能通过静默 fallback 或弱化校验隐藏。
- 模型权重、环境、输入数据和第三方源码不纳入主代码仓库；缓存、临时文件和一次性诊断不进入正式结果。
- 不在当前项目状态文档中重复保存方法参数、实验曲线或环境安装细节；这些信息只维护在对应配置和专题文档中。

## 正式配置位置

- `code/config/methods.json`：正式 method ID、模型、环境、adapter 和执行路线。
- `code/config/method_repositories.lock.json`：第三方仓库 URL、本地目录和精确 revision。
- `code/config/throughput_manifests/`：固定输入 PDF 清单及其选择说明。
- `code/config/throughput_profiles/`：已经选定的吞吐运行参数。
- `env/`：环境锁和迁移快照。
- `experiments/tuning/`、`experiments/benchmark/`：按 run_id 隔离的运行产物。

## 文档职责

- [`PROJECT_STATE.md`](PROJECT_STATE.md)：当前阶段、实验状态、有效限制和下一步。
- `code/README.md`：代码导航和 Git 工作流。
- `code/docs/environments/`：各方法环境的安装和迁移命令。
- `code/docs/MODELS.md`：模型来源、版本和下载方式。
- `code/docs/RUN_COMMANDS.md`：可复制的运行入口；验证边界和吞吐规则见吞吐协议。
- `code/docs/THROUGHPUT_TUNING_PROTOCOL.md`：吞吐测试规则、压力映射和实验记录。
- `code/docs/METHOD_REPOSITORY_UPGRADES.md`：源码同步、revision 审计和版本区别。

根目录的 `PROJECT.md` 与 `PROJECT_STATE.md` 是工作区级说明；主实现和可迁移操作说明以 `code/` Git 仓库中的文件为准。
