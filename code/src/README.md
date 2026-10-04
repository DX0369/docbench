# 源码导航

`src/` 按职责保留五个目录；方法版本、模型和环境绑定以
[`config/methods.json`](../config/methods.json) 为准，不在这里重复维护。

```text
src/
├── methods/               # adapter、版本 profile、方法命令及必要补丁
├── benchmark/             # 普通批量解析、吞吐调参和正式运行入口
├── execution/             # 共享任务表、进程、队列、worker 和资源采样
├── config/                # 方法清单读取、execution profile、pressure、输入清单
├── validation/            # 迁移前资产、就绪和输出检查 CLI
├── method_repositories.py # 第三方仓库同步与 revision 核对
├── method_paths.py        # 第三方源码目录解析
├── paths.py               # 公共资产路径
├── common.py              # PDF 列表、输出路径和通用输出工具
└── image_vllm_client.py    # 共享图像请求客户端
```

## 调用关系

吞吐主链为 `benchmark/run_pressure_sweep.py →
run_single_gpu.py → execution/tasks.py → methods/<family>/`。
固定吞吐运行从 `run_single_gpu.py` 进入同一任务表；压力搜索只由
`run_pressure_sweep.py` 调用它。`run_service_batch.py` 和 `run_native_batch.py`
仅保留为普通批量/Popo 兼容入口，不参与压力搜索。

- `methods/` 保留方法差异：模型/prompt、backend、布局组件、专用 worker、
  版本分支、服务 wrapper 和兼容补丁。共享 adapter 的版本仍用显式 method ID 区分。
- `execution/tasks.py` 只登记两类已有任务及其命令构建器，不统一改写 native/service 流程。
- DFlash 配置在 `methods/hunyuanocr/dflash.py`；
  Paddle 服务 wrapper 在 `methods/paddleocr_vl/serve.py`；
  通用 VLM 的可选服务补丁在 `methods/generic_vlm/serve.py`。
- Qianfan 的 `vllm.py` 同时是直接/常驻队列入口；`runtime.py` 管 joint 请求与输出，
  `page_pool.py` 绑定共享页面调度，`prompt.py` 固定官方长 prompt。
  joint（默认）与 page_pool 均为正式支持路线，普通/think 共用；旧短 prompt 从 Git
  `f578eeda` 恢复。模式和参数见[吞吐协议](../docs/THROUGHPUT_TUNING_PROTOCOL.md#qianfanjoint-与-page_pool-双路线)。
- `execution/page_pool.py` 是 Hunyuan、Qianfan/Unlimited page_pool 与 DeepSeek 共用的有界调度；
  `common.py` 提供同一单页 Poppler 渲染，推理/后处理/文档导出仍在方法目录。
  Hunyuan 默认一个常驻客户端、全输入一个 job，pressure 只映射全局页面任务数；
  `methods/hunyuanocr/vllm.py` 统一直接/常驻队列入口；已删除的旧独立页面池和逐文档实现
  可从 Git `52ccf8b` 恢复，不在当前源码中保留并行旧路径。

## 运行入口

以下文件均位于 `benchmark/`：

| 入口 | 职责 |
| --- | --- |
| `run_service_batch.py` | 兼容用普通 service 批量入口；不用于吞吐调参 |
| `run_native_batch.py` | native/pipeline 普通批量及 MinerU-Popo 入口 |
| `run_pressure_sweep.py` | 唯一搜索：P=1～256倍增、失败或提升不足5%停止，显式保存配置 |
| `run_single_gpu.py` | `--pressure P`或`--profile-file`固定运行一次，默认保留输出，不搜索 |
| `build_pdf_subset.py` | 按需构建 PDF 路径子集；不替代固定调参 manifest |
| `collect_failed_pdfs.py` | 从单方法 aggregate summary 或旧 method-queue summary 提取失败输入名单 |

## 公共执行工具

`execution/` 供运行入口和方法专用 worker 复用，不导入 `benchmark/` 入口：

- `tasks.py`：`SERVICE_TASKS`、`NATIVE_TASKS` 及各自任务类型、本地服务管理判断。
- `commands.py`：Conda/vLLM 命令基础函数与 builder 类型。
- `processes.py`：子进程日志、进程组退出、HTTP 健康检查。
- `queue.py`：文件队列、文档分批与完成/失败记录。
- `workers.py`：常驻 worker 的队列消费循环。
- `trial.py`：页数统计和 CPU/GPU 采样；不含并发搜索。

队列流转为 `pending/job_*.txt → running/job_*__worker.txt → done/job_*__worker.json`。
方法专用命令在 `methods/<family>/commands.py`；模型推理和专用环境修复不放在公共执行层。

## 最小使用与验证

以下命令从 `code/` 执行。GPU 编号仅为示例，实际运行须由用户明确指定。

```bash
# 普通批量入口：仅查看生成的命令。
python src/benchmark/run_service_batch.py --methods hunyuanocr_15 --gpus 0 --dry-run
python src/benchmark/run_native_batch.py --methods marker --gpus 0 --dry-run

# 吞吐调参入口：dry-run 不会更新选定配置。
python src/benchmark/run_pressure_sweep.py \
  --methods hunyuanocr_15 --gpus 0 --execution-policy serial --dry-run \
  --run-name layout_check

# 固定P：这里仍只是dry-run；真实运行前明确输入与GPU。
python src/benchmark/run_single_gpu.py \
  --method hunyuanocr_15 --gpu-id 0 --pressure 4 --dry-run \
  --run-name fixed_p_check
```

固定调参输入见 [`rtx3090_20.json`](../config/throughput_manifests/rtx3090_20.json)，
只引用原始 PDF；选定配置见 [`rtx3090.json`](../config/throughput_profiles/rtx3090.json)。
实际调参显式使用 `--save-profile <JSON>` 才保存参数，后续单次入口用
`--profile-file <JSON> --execution-profile <路线ID>` 加载完整配置。
schema v2按方法/路线保存P和相关参数；旧schema、缺字段或跨路线配置明确失败。
两入口默认20文档清单，均可用`--pdf-list-file <JSON或TXT>`或`--input-root <目录>`切换。
旧full/coarse-refine实现从Git `f2c0196`恢复，不保留兼容入口。

真实 smoke 应显式指定项目 `tmp/smoke/<run_id>/` 下的输出和日志，
不要混入正式结果。dry-run 只能核对命令连线，不能代替真实 smoke 或性能结论。

迁移前只保留以下无模型核对命令；历史 `test_*.py` 回归已在定版前执行，
测试源不属于迁移运行时：

```bash
python src/validation/check_method_versions.py \
  --json-report ../tmp/migration/method_versions.json
python src/validation/check_smoke_readiness.py \
  --json-report ../tmp/migration/smoke_readiness.json
python -m compileall -q src
```

资产检查、单页准备、输出检查仍是 `validation/` 下的独立 CLI。
固定 P、压力搜索以及各方法的 method ID 清单见
[运行命令](../docs/RUN_COMMANDS.md)。

## 详细说明

这里仅维护代码结构和入口。计时、调参、调度、失败规则及完整产物规范统一见吞吐协议；
当前验证进展见项目状态，不在源码说明中重复维护。

- [运行命令与输出协议](../docs/RUN_COMMANDS.md)
- [吞吐规则、方法运行机制及实验记录](../docs/THROUGHPUT_TUNING_PROTOCOL.md)
- [源码版本管理](../docs/METHOD_REPOSITORY_UPGRADES.md)
- [环境配方](../docs/environments/README.md)
- [项目状态（工作区文档）](../../PROJECT_STATE.md)

旧论文分析、旧多方法队列协议及一次性输出清理工具已移到项目
`archive/code/src_pre_consolidation_6f608b9/`，完整历史代码可从 Git 提交
`6f608b9` 恢复。旧入口不保留兼容 wrapper；使用这里的新路径。
