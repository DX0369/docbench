# 吞吐测试协议

本文档是项目唯一的吞吐测试协议，同时保留必要的方法运行机制和实验摘要。环境和模型说明分别见
[环境配方](environments/README.md) 与 [MODELS.md](MODELS.md)；可复制的运行命令见
[RUN_COMMANDS.md](RUN_COMMANDS.md)。
本文不进行准确率评分，也不把短 smoke 当作正式排名。

## 一、通用测试规则

### 入口和输入

- `run_pressure_sweep.py` 是唯一调参入口，按 `1/2/4/8/16/32/64/128/256` 逐档搜索；
- `run_single_gpu.py` 只运行一个明确的 P 或 profile，不自动搜索；
- 默认输入为 `code/config/throughput_manifests/rtx3090_20.json`，共 20 份 PDF、146 页；
- 输入只引用 `data/pdf_tree/`，不复制原始 PDF；短检查可显式传入更小清单；
- 一次 trial 只使用一张 GPU、一个方法和一个独立 `run_id`，正式 workflow 不默认跨卡并行。

### 调参停止和 profile

每档只运行一次。OOM、服务死亡、最终请求失败、程序异常或结果写盘失败时，记录失败并停止继续升压；
非零退出、缺失或损坏结果不能被 JSON 的 `ok` 覆盖。相邻成功档的处理吞吐提升小于 5% 时停止，恰好 5% 继续。
首档失败不生成 profile，历史成功档可以保留但必须保留失败事实。

只有显式传入 `--save-profile` 才保存选定配置，不自动启动正式实验。profile 按方法 ID 和 execution profile
保存，只记录本方法实际需要的参数；不同模型版本和 execution profile 不能互相套用。

### 计时和成功判定

三个时间点为：`t0` 启动服务/runtime，`t1` 全部必要组件 ready 并放行输入，`t2` 输出、错误记录和写盘完成。
记录 `startup_seconds=t1-t0`、`processing_seconds=t2-t1`、`total_seconds=t2-t0`。

主吞吐为 `successful_input_pages / processing_seconds`，总墙钟吞吐为
`successful_input_pages / total_seconds`。处理时间包含读取、渲染、预处理、推理、后处理、导出和失败尝试，
不包含退出清理，也不额外加入预热 PDF；首次 kernel/JIT/缓存开销仍计入。

- 成功页数来自成功输入 PDF；失败或未处理文档的页数不进入分母；
- 失败耗时保留，默认失败 trial 不参与选优；
- 正常 `stop`、`length`、空文本、分页警告和图片框质量问题不是执行失败；
- OOM、引擎死亡、请求中断、协议损坏、程序异常和写盘失败是真实失败；
- 文档失败后继续处理其他文档；服务整体死亡时停止 trial 并保留未处理清单；
- 不自动重试、不切换 backend、不用内容质量判断吞吐成功，质量问题作为 warning 记录。

### 目录和记录

```text
code/config/throughput_manifests/rtx3090_20.json   # 固定输入引用
code/config/throughput_profiles/rtx3090.json       # 已选配置
experiments/tuning/<run_id>/                       # 调参结果
experiments/benchmark/<run_id>/                   # 正式运行结果
tmp/tuning/<run_id>/                               # 临时清单和 dry-run
```

每个 run 记录完整命令、主代码 commit、源码 revision、模型、环境、GPU、输入清单、P、计时边界、成功/失败数量、
峰值显存和 OOM/timeout 信息。dry-run 只写 `tmp/`；一次性输出和临时缓存核对后清理。

## 二、方法运行机制与 P 映射

P 不要求在所有方法中同名，只要求映射明确、可复现，并且不改变官方模型、prompt 和生成参数。除特别说明，
各方法使用锁定引擎默认 `max_num_seqs`，不额外覆盖。

| 方法/路线 | 运行机制 | P 控制对象 | 跨文档情况 |
|---|---|---|---|
| HunyuanOCR AR / DFlash | vLLM 页面池 | 页面在途数 | 页面级请求 |
| DeepSeek-OCR-2 | AsyncLLM 页面流水线 | 页面在途数 | 页面级请求 |
| Qianfan 普通/think | 官方单页请求 + 共享 page pool | 页面在途数 | 可跨文档供给，按 PDF 页序导出 |
| Unlimited `page_pool` | SGLang 页面请求 | 页面在途数 | 可跨文档供给 |
| Unlimited `joint` | SGLang 整篇请求 | 文档客户端数 | 文档级联合上下文 |
| MonkeyOCR 3B/1.2B | 常驻 vLLM async runtime | 文档内区域在途数 | 官方分组供给 |
| PaddleOCR-VL 1.5/1.6 | pipeline + 区域 VLM 服务 | 区域请求并发 | 页面可跨文档供给 |
| GLM-OCR | 布局 pipeline + 区域服务 | 区域识别在途数 | 页面可跨文档供给 |
| PP-StructureV3 | pipeline + 页面预取 | 原生组件 batch 倍率 | 页面跨文档批处理 |
| Marker | 共享模型 worker | 文档 worker 数 | worker 跨文档复用 |
| MinerU 3.0.3/3.4.x | hybrid API + 文档客户端 | 客户端数及 API 任务上限 | 整 PDF 请求 |
| Youtu native | shared-feature/query | query 分组数 | query 可跨文档供给 |
| TeleOCR vLLM async | 常驻 AsyncLLM 两阶段页面流水线 | 当前 PDF 内共享的 layout/region 请求信号量 | 文档之间按官方顺序处理 |
| LightOnOCR-2-1B | 官方 vLLM 单页 VLM + 全局 page pool | 页面在途数 | 页面可跨文档供给；不支持已登记的整文档 joint |
| Xiaomi-OCR-0 | 官方 batch 整页 VLM + 常驻 vLLM 全局 page pool | 页面渲染／请求在途数 | 单 worker、单服务，页面可跨文档供给；不登记 region 或 joint |

### 生成长度和上下文参数来源

长度参数不在 P 搜索中调整。`max_tokens`/`max_new_tokens` 是请求的生成上限，
不是要求每次请求都生成到该长度；`max_model_len`/`context_length` 是服务侧输入、
图片 token 和输出共用的上下文上限。下表记录官方推荐或默认值，以及本项目当前实际值。
“项目值”若与官方不同，必须按项目侧覆盖理解，不能写成官方默认。

| 方法 | 官方推荐/默认 | 本项目实际值 |
|---|---|---|
| HunyuanOCR 1.0 | 官方客户端 `max_tokens=16384` | 生成 16384；上下文 32768 |
| HunyuanOCR 1.5 AR | 官方 vLLM `max-model-len=131072`、`max-num-batched-tokens=131072`、客户端 `max_tokens=32768` | 生成 32768；上下文/batched tokens 65536（RTX 3090 安全值） |
| HunyuanOCR 1.5 DFlash | 官方上下文/batched tokens 131072、speculative tokens 15、生成 32768 | 生成 32768；上下文/batched tokens 65536；speculative tokens 15 |
| Qianfan-OCR 普通/Think | 模型卡 `max_new_tokens=16384`；基础 prompt 为 `Parse this document to Markdown.`，think 追加 `<think>` | 生成 16384；服务上下文 32768 |
| DeepSeek-OCR-2 | 当前官方 adapter/API 默认 8192 | 生成 8192；上下文 8192 |
| PaddleOCR-VL 1.5/1.6 | 官方 `max_new_tokens` 默认不传；vLLM 上下文按服务配置 | 不传 `max_new_tokens`；服务上下文 16384 |
| GLM-OCR | 官方 YAML `max_tokens=8192`；上下文按硬件设置 | 生成 8192；服务上下文 32768 |
| MonkeyOCR Pro 两版 | 官方 async vLLM `max_tokens=4096`；捕获长度 10240 | 生成 4096；沿用 async runtime 的模型配置，未额外设置 `max_model_len` |
| Unlimited-OCR SGLang | 官方示例上下文 32768；请求不固定 `max_tokens` | 生成上限 16384（项目值）；上下文 32768 |
| Youtu-Parsing | 官方 HF parser 默认 `max_new_tokens=8192` | 生成 8192；上下文由模型配置决定 |
| MinerU 3.0.3/3.4.x | 官方直连 vLLM 示例有 16384，但 hybrid API 不暴露统一长度参数 | 由 MinerU hybrid backend 内部默认决定 |
| PP-StructureV3 | 各识别子模型使用官方 native 配置，无全局长度值 | 不设置统一生成/上下文上限 |
| Marker | 当前关闭 LLM | 不适用 |
| MinerU-Popo | Popo 官方脚本生成上限 8192；未提供固定服务上下文 | Popo 生成 8192；默认 `popo_max_model_len=32768`，仍可显式覆盖 |
| TeleOCR vLLM async | 官方 `vllm-async-engine`、`MAX_MODEL_LEN=16384`、`GPU_MEMORY_UTILIZATION=0.95`、PDF worker 4；上游 `max_concurrency=100` | backend/layout/PDF worker 保持上述值；P 显式覆盖请求 semaphore；不设置 `max_num_seqs` |
| LightOnOCR-2-1B | 官方 vLLM 示例：每请求一张图片、`max_tokens=4096`、`temperature=0.2`、`top_p=0.9`；服务参数为 image limit 1、processor cache 0、关闭 prefix caching | 生成 4096；200 DPI、最长边 1540；不显式设置 `max_num_seqs` 或 `max_model_len` |
| Xiaomi-OCR-0 | 官方 batch e2e：生成 16384、temperature 0、top_p 0.8、presence penalty 0、repetition penalty 1、thinking 关闭、超时 600 秒、RGB 最长边 2048；Demo/MCP 为 4096 输出和 150 DPI，服务示例上下文 16384 | 采用 batch 请求及官方后处理；项目指定 200 DPI、服务上下文 32768；单图请求，`max_num_seqs`、显存比例、引擎 batch 和编译模式沿用默认 |

Xiaomi 直接复用官方 `PROMPTS["document"]`、`encode_image_base64()` 和
`finalize_document()`，不调用会将空响应判失败的官方 `infer_one()`。
空响应和 `finish_reason=length` 按统一执行成功规则处理；真实 SDK、结构或写盘
错误显式记录。不开启官方可选的重复内容重试；OpenAI SDK 默认传输重试保持不变，
其耗时计入处理时间。临时页面自动清理，不保存原始响应或图片资产。
CPU/dry-run 与 GPU 6 两文档/四页 P1、P2、逐文档 latency smoke 已通过；
真实服务可在当前共享环境启动，但不据此宣称全量稳定或吞吐最优。

本节中的硬件安全值不等于官方最大值。特别是 Hunyuan 1.5 的 65536、GLM 的 32768、
Paddle 的 16384、Unlimited 的 16384 以及 Popo 的 32768，都必须随运行 metadata 记录。
Paddle 不传 `max_new_tokens` 后，metrics 不再伪造该字段；实际上限由 PaddleOCR-VL
官方 pipeline 和识别服务处理。Popo 的 32768 只是新的默认上下文值，不代表双模型组合
已经完成 3090 高并发容量验证。

Paddle/GLM 的页面、布局 batch 和内部线程先保持官方值；PP-StructureV3 只调整已确认的原生 batch；
Monkey 两个 checkpoint 使用独立 ID 但共用 async runtime；MinerU 不调整内部 VLM batch、页窗口和组件 batch；
Qianfan 当前只登记单页 page_pool；旧 joint 结果属于历史诊断，不能与当前 profile 混排。MinerU-Popo 需要同时驻留两个模型，3090 尚未形成稳定 profile，
暂不纳入正式调参。

### Latency 参数口径

Latency 不复用吞吐压力映射。HunyuanOCR、DeepSeek-OCR-2、Qianfan、LightOnOCR-2、Xiaomi-OCR-0 和
Unlimited `page_pool` 使用页面级 `P=1`；其他方法只
固定一个外层文档客户端。Monkey 使用官方区域并发 64，TeleOCR 使用 async 默认 100，
Youtu 使用 query batch 5；Paddle 不显式传 `vl_rec_max_concurrency`，GLM 不显式传
`max_workers`（锁定 YAML 的 32），PP 使用基础 `batch_scale=1`，Marker 使用一个
文档 worker，MinerU 不覆盖 API 的默认并发和 processing window。Latency summary 会
记录 `latency_profile=official_defaults`；命令行中的 `--pressure 1` 只表示文档级
latency 标识，不代表所有内部并发均为 1。

## 三、实验记录摘要

本节只保留当前有界实验结论；完整数值、命令和异常证据仍在 `experiments/tuning/<run_id>/` 中。

| 方法 | 已测范围 | 当前结论 |
|---|---|---|
| HunyuanOCR 1.5 AR | 20 文档 P1/P2/P4 | 页面并发有提速；P4 存在内容差异，未保存无损最优 profile |
| Xiaomi-OCR-0 | GPU 6 中文试卷/英文杂志各 2 页，P1/P2 与 latency 各一次 | 三项全部通过输出/覆盖检查；ready 后处理吞吐 P1=0.203、P2=0.309 页/秒。逐文档 latency=9.48/6.14 秒；缓存和单次小样本影响未控制，不作为最终配置或正式速度结论 |
| HunyuanOCR DFlash | 有界页面 smoke | 环境和 DFlash 路线通过；3090 profile 与 AR 分开 |
| DeepSeek-OCR-2 | 20 文档 P1/P2 | P2 完成；部分 P4 受端口/基础设施影响，不能当作 OOM 上限 |
| Qianfan 普通/think | 单页 smoke 与有界 page pool | 当前路线按输入 PDF 页边界导出；joint 已退出正式调度 |
| PaddleOCR-VL 1.5/1.6 | 短多页及 1.6 P1/P2/P4/P8 | pipeline 和区域服务通过；P 不等于内部所有 batch |
| GLM-OCR | 20 文档 P1/P2/P4 初筛 | 区域并发有提速；输出存在原生内容差异 |
| MonkeyOCR 3B/1.2B | 短多页及 20 文档 P1/P2/P4 | P4 为当前子集已测最快；输出差异单独记录 |
| Marker | 20 文档 P1/P2/P4 | 共享 worker 通过；P4 仅为当前子集候选 |
| PP-StructureV3 | 20 文档 P1/P2/P4 | batch 倍率有提速但不单调；P4 仅为已测最快 |
| MinerU 3.0.3/3.4.x | 20 文档 P1/P2/P4 | API 路线完成；存在表格、公式和长上下文失败 |
| Youtu native | 短多页及 P1/P2/P4 | query 分组有提速，不能与普通文档并发混称 |
| Unlimited SGLang | joint/page_pool 有界测试 | 两条路线可运行；分页和截断 warning 仍需报告 |
| MinerU-Popo | 容量诊断 | 双模型组合尚未稳定，排除正式调参 |

单次 P1/P2/P4 或 P1/P2/P4/P8 只表示该输入子集上的有界结果，不是理论最优；不同 GPU、输入、模型、prompt、
生成参数或计时边界的结果不能直接排名。内容差异和分页 warning 必须记录，但不自动判定吞吐失败。
正式论文结果应使用选定 profile 重新运行，并同时报告处理吞吐、总墙钟吞吐和失败情况。旧 A800/rebuttal 数字只作历史参考。

## 四、固定 P 执行模板

准备输入清单和 GPU 后，在 `code/` 目录执行：

```bash
python src/benchmark/run_single_gpu.py \
  --method <method_id> \
  --gpu-id <GPU_ID> \
  --pressure <P> \
  --input-root "$GDB_ROOT/data/pdf_tree" \
  --pdf-list-file "$GDB_ROOT/code/config/throughput_manifests/rtx3090_20.json" \
  --run-name <run_name> \
  --output-root-base "$GDB_ROOT/experiments/benchmark/<run_id>/outputs" \
  --log-root "$GDB_ROOT/experiments/benchmark/<run_id>/logs"
```

调参使用 `run_pressure_sweep.py`，执行前先 dry-run，确认模型、源码 revision、环境、GPU、P 和方法专用参数。
执行后核对 summary、失败清单、显存采样和端口释放；任何一档失败都不能静默跳到更高 P。
