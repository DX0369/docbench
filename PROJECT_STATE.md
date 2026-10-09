# 项目状态

更新时间：2026-10-09 16:42 +08:00

## 当前阶段

项目已完成主要方法、模型、第三方源码、隔离环境和 adapter 的整理，代码入口和输出协议已完成一次定版精简。当前处于正式吞吐实验前的有限验证阶段：先核对统一入口、计时和失败记录，再决定各方法的固定吞吐配置。当前不计算 GT 或 scorer，也不把已有小规模结果宣称为最终最优。

## 已完成

- Jina-OCR-v1 `jina_ocr_v1` 已在 GPU 3 完成官方同步单页和两文档 P1/P2/latency smoke：中文试卷、英文杂志各 2 页，各轮输出校验 2/2 通过，无执行失败。固定严格 benchmark prompt、FastMTP eagle K=3，复用未改包的 DFlash 环境。修复仅涉及启动工具路径（conda run/ninja、CUDA 12.8）和 spawned worker 调用官方 register；不修改模型或环境源码。P1/P2 ready 后处理 29.16/21.95 秒，吞吐 0.137/0.182 页/秒；latency 中文→英文 24.20/7.82 秒、页面并发1。GPU 3 恢复18 MiB；未正式调参，不宣称全量稳定性或最优。一轮一次性产物核对后清理，仅保留结论。

- 正式方法和执行路线已登记在 [`code/config/methods.json`](code/config/methods.json)，源码与模型绑定由方法版本清单和源码锁维护。
- MinerU 3.0.3/2509、MinerU 3.4.x/2605、Hunyuan AR/DFlash、Unlimited SGLang 等版本已拆分为显式 method ID；兼容源码和精确历史源码已区分记录。
- 环境安装配方和锁文件已整理；锁文件仍位于 `env/`，具体命令见 [`code/docs/environments/`](code/docs/environments/) 和 [`code/docs/MODELS.md`](code/docs/MODELS.md)。
- MinerU 4.0.11 独立 `mineru_4011_standard` / `mineru_4011_standard_v1_api` 已接入官方 V1 SDK、Standard/Torch/local vLLM 和统一计时，保留 3.x 与 Popo 绑定。用户已建立独立环境，两模型官方 verify 与禁用 user site 的 pip check 通过；10 项 CPU 回归、P1/P2/P4/latency 及默认压力网格 dry-run 通过，旧两版 API dry-run 通过。安装配方与三份实际锁已保存；release tag 的版本文件仍写 4.0.10，仅作参考，实际运行固定官方 4.0.11 wheel。
- 2026-10-09 16:42 +08:00 核对：用户在 GPU 6 完成 MinerU 4.0.11 P1/P2 smoke，均为中文试卷、英文杂志各两页，共两文档四页；执行及输出结构校验均通过，无失败或遗漏。P1 的 `doc_total_seconds` 缺失 warning 已在 adapter 修正，10 项 CPU 回归通过；P2 真实输出包含该字段，validator 为 2/2、零 issue。P1/P2 初始化 103.804/76.847 秒，就绪后处理 14.071/12.218 秒，处理吞吐 0.284/0.327 页/秒，总墙钟 117.875/89.065 秒。仅为单次短文档链路验证，首次编译和缓存未控制，不据此选最优 P；正文、复杂跨页表格、长文档及真实 P4/latency 尚未核查。本轮不重写旧结果、不重跑 GPU、不改依赖或环境锁。
- 统一入口已收束为：固定 P 的 `run_single_gpu.py` 和逐档搜索的 `run_pressure_sweep.py`。旧的 full/coarse-refine 搜索路径不再作为正式入口。
- 默认吞吐清单为 [`code/config/throughput_manifests/rtx3090_20.json`](code/config/throughput_manifests/rtx3090_20.json)，包含 20 份 PDF、146 页；输入 PDF 不复制到实验目录。
- 统一计时记录启动、ready 后处理和处理结束三个时间点；正式处理时间从 ready 开始，到输出及失败记录写完结束，不包含退出清理。
- 统一保留最小输出：结构化结果、运行状态、失败记录和必要诊断；不默认保存原始响应或图片资产。
- 已完成方法版本审计、源码 HEAD 审计、smoke/dry-run、P 映射和失败传播的定向核对。详细运行规则见 [`code/docs/RUN_COMMANDS.md`](code/docs/RUN_COMMANDS.md) 和 [`code/docs/THROUGHPUT_TUNING_PROTOCOL.md`](code/docs/THROUGHPUT_TUNING_PROTOCOL.md)。
- MinerU-Popo 已接入组合流程，但由于双模型容量和稳定吞吐配置尚未确认，暂不纳入本轮正式吞吐调参。
- GPU 7 两文档 latency smoke 已完成并保存在 `tmp/latency_smoke/gpu7_2pdf_20261004_231416_retry1/`：除 MinerU-Popo 外的 17 个方法 ID（共 20 条执行路线）均完成顺序处理；Qianfan 历史 joint 路线因 14 页英文文档输入超过 32768 上下文而明确失败，page_pool 路线通过。该结果仅验证入口顺序、输出和失败记录，不代表正式吞吐结论。
- TeleOCR 已创建 `gdb_teleocr` 环境；`pip check`、TeleOCR/`TeleOCR_vllm` import probe 和 CUDA 检查通过。GPU 6 上已完成统一 adapter 的单文档 throughput smoke 和双文档 latency smoke，均通过输出 validator：latency 使用官方 `max_concurrency=100`，处理顺序为文档 1→2，render DPI 为 200；throughput P=1 路线也成功。产物保留于 `tmp/teleocr_adapter_smoke/teleocr_adapter_smoke_20261008_110329/`，不代表正式吞吐结果。
- LightOnOCR-2-1B 已登记为 `lightonocr2_1b`。模型卡确认其正式路线是 vLLM 单页请求（`limit-mm-per-prompt=image:1`），不注册整文档 joint；新增 page-pool adapter，P 映射为页面在途数，官方生成参数和 200 DPI/1540px 预处理均写入 metrics。GPU 2 上使用两份 2 页 PDF 完成 P1/P2 有界 smoke，输出校验通过；P1 处理吞吐约 0.218 页/秒，P2 约 0.334 页/秒。当前先复用 `gdb_glmocr`（Transformers 5.14.1、vLLM 0.20.0）；服务日志仅有 NIXL/CUDA graph 等 warning，无 LightOn 推理错误。产物保留于 `tmp/lightonocr2_smoke/lightonocr2_gpu2_20261008_132000/`，不代表正式吞吐结论。
- Qianfan 普通/think 已按官方单页基础 prompt 和 page pool 在 GPU 5 完成同一份 3 页英文 PDF 的 P1 smoke；两条路线均输出 3 个页面、每页一张图请求，validator 通过，Think metrics 确认 prompt suffix。产物保留于 `tmp/smoke/qianfan_gpu5_20261008/`，仅验证新入口，不代表正式吞吐结论。
- Latency 入口已统一为 `run_single_gpu.py --latency-mode` 和 native 入口的同名选项：四个页面请求路线固定页面级并发 1，其他方法串行提交文档但保留官方/default 内部并发。TeleOCR 已接入常驻 async worker、统一输出和 `P` 映射；所有主要路线的 latency mapping、dry-run 和 Python 语法检查通过。一次性 dry-run 产物已清理，尚未启动真实 latency smoke。

## 当前实验状态

| 方法族/路线 | 当前状态 |
| --- | --- |
| Jina-OCR-v1 | 官方严格 prompt、单常驻 FastMTP page pool 和 latency 已通过 GPU 3 两文档 smoke；复用 DFlash 环境，未正式调参。 |
| Marker | 既有 20 文档 P1/P2/P4 共享 worker 验证通过；P4 仅为该子集候选，未统一选定最终 profile。 |
| PP-StructureV3 | 既有 20 文档 P1/P2/P4 的 batch 倍率搜索有提速但不单调；P4 仅为已测最快，未统一选定最终 profile。 |
| PaddleOCR-VL 1.5/1.6 | 既有短多页及 1.6 P1/P2/P4/P8 pipeline/区域服务验证通过；P 不等于所有内部 batch，未统一选定最终 profile。 |
| GLM-OCR | 既有 20 文档 P1/P2/P4 初筛显示区域并发有提速；输出存在原生内容差异，未统一选定最终 profile。 |
| DeepSeek-OCR-2 | 既有 20 文档 P1/P2 验证中 P2 完成；部分 P4 受端口/基础设施影响，不能当作 OOM 上限，未统一选定最终 profile。 |
| Youtu native | 既有短多页和 P1/P2/P4 验证中 query 分组有提速，不能与普通文档请求并发混称；未统一选定最终 profile。 |
| MonkeyOCR 两个版本 | 既有短多页及 20 文档 P1/P2/P4 async 验证通过；P4 为该子集已测最快，存在内容差异和表格重复等 warning，不能据此宣称全局最优。 |
| Qianfan 普通/think | 已收束为官方单页基础 prompt + page_pool；普通/think 共用服务，P 映射页面在途数。旧 joint 结果仅作历史证据，不再属于正式调度。 |
| MinerU 两个版本 | 既有 20 文档 P1/P2/P4 API 路线完成有界压力验证，记录表格、公式及长上下文失败和内容差异；未选定最终 profile。 |
| MinerU 4.0.11 Standard | GPU 6 P1/P2 两文档四页执行及结构校验通过，P2 验证计时字段修正；正文人工核查、复杂跨页/长文档及真实 P4/latency 尚未覆盖，未调参。 |
| Hunyuan 1.0、1.5 AR、1.5 DFlash | 三条路线已登记并通过有界验证。既有 1.5 AR 20 文档 P1/P2/P4 测试中页面并发有提速，P4 存在内容差异、未保存无损最优 profile；DFlash 页面 smoke 通过，独立环境与 3090 profile 不与 AR 混用。尚未统一吞吐定版。 |
| Unlimited-OCR SGLang | joint/page_pool 路线通过有界 smoke，分页与截断 warning 仍需报告；使用 RTX 3090 SM86 兼容补丁，不能描述为原封不动的官方实现。 |
| TeleOCR | `teleocr_vllm_async` 常驻 async adapter、统一输出和 P 映射已接入；GPU 6 上单文档吞吐与双文档 latency smoke 已通过，尚未进行正式吞吐搜索。 |
| LightOnOCR-2-1B | `lightonocr2_1b` 官方 vLLM 单页 page-pool adapter 已接入；GPU 2 上 P1/P2 有界 smoke 通过，尚未完成正式吞吐调参。 |
| Xiaomi-OCR-0 | `xiaomi_ocr_0` 仅整页路线，200 DPI、官方 RGB/2048px 编码和 Markdown 后处理；复用 `gdb_glmocr`，无依赖安装或源码/模型补丁。2026-10-08 21:21 +08:00，GPU 6 中文试卷 `exam_zh_2014______________0_2`、英文杂志 `magazine_en_The_New_Yorker-2025-12-15_68_70` 各 2 页，P1/P2/latency 各一次全部完成，输出校验均 2/2 通过、4 页覆盖完整。P1/P2 处理耗时 19.73/12.93 秒、吞吐 0.203/0.309 页/秒；延迟按中文→英文顺序，9.48/6.14 秒，单页并发 1。启动耗时分别 216.57/164.85/88.54 秒，首次编译/缓存影响未控制，不作为正式性能结论。只有非致命 NIXL/弃用和退出 semaphore warning；无 OOM、请求失败或残留 GPU 进程，GPU 恢复约 18 MiB。本轮一次性产物清理，仅保留结论；未正式调参，不登记 region/joint。 |
| MinerU-Popo | 组合 adapter 和常驻路径已接入，CPU 导入及单/双服务小 smoke 通过；既有 20 文档 P1/P2/P4 未全部成功，有容量、上下文及建树失败。双模型稳定容量、跨页表格和图文关联仍待验证，暂不调参。 |

## 已知限制

- 本项目当前结果主要证明执行链路、输出结构和错误记录可用，不等于内容准确率或最终吞吐最优。
- 部分整文档路线可能出现截断、分页标记不可靠或内容差异；这些作为质量 warning 保存，不冒充执行失败，也不自动补页。
- Paddle/GLM 等 pipeline 方法仍使用其官方组件和默认内部 batch；本轮只搜索各自主压力参数，不同时展开页面 batch、布局 batch、线程等多维调参。
- Unlimited 使用定制 SGLang wheel 及 SM86 kernel 兼容补丁；需要在报告中单列该实现边界。
- TeleOCR 上游 async smoke 曾出现 vLLM engine teardown warning；本次 adapter smoke 无残留进程，warning 未影响结果，但仍不将 smoke 结果当作稳定正式 profile。
- Qianfan、Monkey、MinerU 和 Popo 的历史实验产物仅作证据，不自动合并为新的 profile 或正式结论。
- 20 文档清单用于当前有限验证；163 文档全量正式运行尚未启动。

## 文档与产物边界

- 项目目标和目录约定：[`PROJECT.md`](PROJECT.md)。
- 当前方法、环境和模型细节：[`code/docs/`](code/docs/)。
- 本文件是唯一的人工实验进展与简短结论记录；环境配方、通用命令、吞吐协议和源码管理文档不复制测试过程或结果。
- 吞吐调参规则：[`code/docs/THROUGHPUT_TUNING_PROTOCOL.md`](code/docs/THROUGHPUT_TUNING_PROTOCOL.md)。
- 运行入口和验证边界：[`code/docs/RUN_COMMANDS.md`](code/docs/RUN_COMMANDS.md) 与 [`code/docs/THROUGHPUT_TUNING_PROTOCOL.md`](code/docs/THROUGHPUT_TUNING_PROTOCOL.md)。
- 调参运行：`experiments/tuning/<run_id>/`；正式运行：`experiments/benchmark/<run_id>/`；临时 smoke 和诊断产物：`tmp/`。
- 实验目录只保存该运行的状态、结果和必要诊断，不复制输入 PDF，不把缓存、模型或环境写入 Git。

## 下一步

先核查 MinerU 4.0.11 P1/P2 smoke 正文及复杂跨页、长文档覆盖需求，再决定是否扩展有界验证或进行真实吞吐搜索。

1. 使用固定的 20 文档清单，对各方法执行有限 P 搜索或指定 P 运行；一次只使用一张 GPU，不自动并行分发正式任务。
2. 为每个方法记录可复现的选定 profile，并区分官方路线、项目兼容补丁和实验性路线。
3. 在 profile 稳定后，再决定是否启动 163 文档正式吞吐实验；GT、scorer 和 MinerU-Popo 不在当前阶段范围内。
