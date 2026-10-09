# 吞吐测试协议

## Jina-OCR-v1

- 路线：单常驻 AsyncLLM、全局单页 page pool；不支持整文档 joint。
- Prompt：固定官方模型卡的严格 benchmark prompt，包含 HTML 表格、LaTeX、
  忽略图形、删除页眉页脚并保留脚注；完整原文维护于
  `src/methods/jina_ocr/runtime.py::STRICT_PROMPT`，写入每份文档 metrics。
  不使用推荐简短 prompt，不在压力搜索中更换 prompt。
- 官方生成示例：`max_tokens=4096`、temperature 0、repetition penalty 1.05；
  重复检测 35/35/10。项目沿用这些设置，缺少重复检测支持时显式失败。
- 官方 FastMTP：同 checkpoint、`eagle`、K=3、递归单 head；项目不另下载 draft，
  不自动关闭 MTP、不回退 Transformers。
- 项目设置：200 DPI、context 32768；不设置 max_num_seqs 或 GPU memory fraction。
- P：跨 PDF 的全局页面渲染/请求在途上限，单 worker、单模型 runtime。
  Latency：逐 PDF 顺序运行，页面在途上限 1，其他参数相同。
- 空文本、`length` 正常完成仍算执行成功；真正请求/引擎/写盘失败记录并传播。
  不保存原始响应、图片资产；不根据模型页标记猜测页归属。

本文档是项目唯一的吞吐测试协议，维护测试规则、方法运行机制和参数依据，不记录实验过程或结果。
实验进展与简短结论统一见 [PROJECT_STATE.md](../../PROJECT_STATE.md)。环境和模型说明分别见
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

### 结果可比性

单次压力搜索只表示该输入子集上的有界结果，不是理论最优；不同 GPU、输入、模型、prompt、
生成参数或计时边界的结果不能直接排名。内容差异和分页 warning 必须记录，但不自动判定吞吐失败。
正式论文结果应使用选定 profile 重新运行，并同时报告处理吞吐、总墙钟吞吐和失败情况。
旧 A800/rebuttal 数字只作历史参考，不与当前协议的结果混排。

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
| MonkeyOCR 3B/1.2B | 常驻 vLLM async runtime | 分组内区域识别并发上限 | 同组可跨 PDF，分组顺序处理 |
| PaddleOCR-VL 1.5/1.6 | 队列 pipeline + 异步 HTTP 区域 VLM 服务 | 区域识别并发上限 | 多 PDF 页面进入官方 pipeline，可跨文档供给 |
| GLM-OCR | 布局—识别队列 + HTTP 线程池 | 区域识别并发上限 | 多 PDF 共用页面、区域队列，可跨文档供给 |
| PP-StructureV3 | pipeline + 页面预取 | 指定原生组件 batch 倍率 | 顶层页面可跨 PDF 组批；部分子组件仍逐页调用 |
| Marker | 多文档进程调用共享模型 | 文档 worker 数 | 文档可并行；没有统一跨文档区域合批器 |
| MinerU 3.0.3/3.4.x | hybrid API + 文档客户端 | 文档客户端数及 API 任务上限 | 整 PDF 任务可交错向共享 VLM 供给，非整文档联合上下文 |
| MinerU 4.0.11 Standard | 官方 V1 API + 共享常驻 Torch/AsyncLLM | 文档客户端数及服务 concurrency | 整 PDF 提交，内部按页面/区域识别并后处理 |
| Youtu native | 页面视觉特征复用 + 顺序 query | 同页单次文字查询包含的 bbox 数 | 不跨页、不跨文档分组 |
| TeleOCR vLLM async | 常驻 AsyncLLM 两阶段页面流水线 | 当前 PDF 内布局及区域生成请求的共享在途上限 | 文档内页面异步交错，文档之间顺序处理 |
| LightOnOCR-2-1B | 官方 vLLM 单页 VLM + 全局 page pool | 页面在途数 | 页面可跨文档供给；不支持已登记的整文档 joint |
| Xiaomi-OCR-0 | 官方 batch 整页 VLM + 常驻 vLLM 全局 page pool | 页面渲染／请求在途数 | 单 worker、单服务，页面可跨文档供给；不登记 region 或 joint |

### 区域识别与供给边界

以下说明针对当前 `run_single_gpu.py` 吞吐入口：Monkey、Paddle、GLM 和 PP 接收本次选定的
PDF 列表；普通批量入口 `run_service_batch.py` 的 job 粒度可能不同，不能直接套用相同的跨文档范围。

Monkey、Paddle 和 GLM 的“区域识别并发上限”统一表示最多同时在途的区域识别任务数，
不表示 GPU 的固定 batch size，也不调整布局检测等其他组件的 batch。
Monkey 将 P 映射为 `region_concurrency`，在每次 async 识别调用中用信号量限制提交给
进程内 AsyncLLM 的任务；Paddle 将 P 映射为 `vl_rec_max_concurrency`，用客户端信号量
限制区域识别 HTTP 请求；GLM 将 P 映射为 `max_workers`，用线程池限制区域识别任务，
当前上限和 HTTP 连接池均为 128。实际 GPU batch 由推理引擎调度。

Monkey 的吞吐供给按官方分组入口构造 `MultiFileDataset`，项目固定分组页数上限为 20；
同组可以包含多份 PDF。单份超过 20 页的 PDF 保持完整、独占一组，不拆分。
每组先完成布局，再收集区域进行识别；分组顺序处理，不是跨组连续供给流水线。
识别结果仍按原 PDF 和页序导出；latency 模式逐 PDF 提交，不跨文档分组。

Paddle 的输入、CV/布局和 VLM 阶段通过队列交叠执行，识别阶段分批汇集区域并发送异步 HTTP 请求；
GLM 的加载、布局和识别通过页面/区域队列交叠执行，由线程池发送区域 HTTP 请求。
两者的 P 都不扩大布局 batch、预处理线程数或模型副本数。
TeleOCR 则由同一个 VLM 处理布局和裁剪区域；当前 PDF 的各页异步推进，布局与区域生成共享 P，
完成一份 PDF 后才处理下一份。

PP 的 P 缩放页面、布局、文本识别、文本行方向和公式识别五项 batch，不增加模型副本。
顶层页面 batch 可跨 PDF，但文本识别仍有按页调用，表格也逐页处理；不能将其解释为所有组件统一跨文档合批。
Marker 的 P 增加文档进程，各进程逐 PDF 调用共享模型；文档内部可汇集跨页 OCR、公式等区域，
但进程之间没有统一区域调度器。Youtu 的 P 只给同页文字 bbox 分组后组成一个 prompt，
并非 P 个并发请求，也不统一扩大表格、公式等类别的并发；其 parallel decoder 并行的是生成内部的 token 候选。

跨文档请求供给不等于跨文档静态 batch，也不等于多文档联合上下文。P=1 不代表内部完全串行，
仍可保留官方组件 batch、页面窗口及内部并发；latency 逐文档提交也不将所有内部参数置为 1。
上述源码机制只能确定并发边界，不能证明实际供给充足、batch 已填满或吞吐最优。

来源：项目各方法 adapter；锁定 Monkey 的 `magic_pdf/model/batch_analyze_llm.py`、`async_vllm.py`；
实际 PaddleX 环境的 `inference/pipelines/paddleocr_vl/pipeline.py`、`inference/models/common/genai.py`、
`inference/pipelines/ocr/pipeline.py`；GLM 的 `glmocr/pipeline/_workers.py`；
TeleOCR 的 `TeleOCR/vlm_utils/TeleOCR_client.py`；Youtu 的 `youtu_hf_parser/youtu_ocr_parser_hf.py`；
Marker 的 `marker/builders/ocr.py` 及实际 Surya recognition 实现。

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
| MinerU 4.0.11 Standard | 由官方本地 VLM runtime 决定 | 不额外覆盖生成、上下文或显存配置；不继承 3.x 的 0.55 |
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

本节中的硬件安全值不等于官方最大值。特别是 Hunyuan 1.5 的 65536、GLM 的 32768、
Paddle 的 16384、Unlimited 的 16384 以及 Popo 的 32768，都必须随运行 metadata 记录。
Paddle 不传 `max_new_tokens` 后，metrics 不再伪造该字段；实际上限由 PaddleOCR-VL
官方 pipeline 和识别服务处理。Popo 的 32768 只是新的默认上下文值，不代表双模型组合
已经完成 3090 高并发容量验证。

Paddle/GLM 的页面、布局 batch 和内部线程先保持官方值；PP-StructureV3 只调整已确认的原生 batch；
Monkey 两个 checkpoint 使用独立 ID 但共用 async runtime；MinerU 不调整内部 VLM batch、页窗口和组件 batch；
Qianfan 当前只登记单页 page_pool；旧 joint 结果属于历史诊断，不能与当前 profile 混排。
MinerU-Popo 的组合路径需要同时驻留两个模型，执行前应核对双模型容量；当前参与方法集合见 `PROJECT_STATE.md`。

### MinerU 3.x 的版本与并发边界

3.0.3 当前 hybrid 路线使用 VLM 两步布局/内容提取，按文档类型与设置结合传统 OCR、公式等组件。
3.4.x 当前默认 medium：先小模型布局，再由区域 VLM 识别并结合传统组件；high effort 才使用 VLM 两步提取，
不能将两版都描述为默认 VLM 两步布局。
两版均提交整 PDF、内部按页面窗口处理，P 只扩大文档客户端数及 API 在途解析任务上限，
不覆盖内部窗口或组件 batch。不同文档可交错向共享 VLM 引擎供给请求，但小组件 batch 来自各文档窗口，
不是自动跨文档静态合批，也不是整篇 PDF 一次进入 VLM 上下文。
3.x 区域请求信号量通常按每次提取调用建立，不是整个服务共用一个全局额度；不能与 4.x 的共享额度混称。
来源：锁定两版的 `mineru/backend/hybrid/hybrid_analyze.py`、3.4.x 的 `mineru/cli/backend_options.py`、
实际环境的 `mineru_vl_utils/mineru_client.py`。

### MinerU 4 Standard 的并发边界

`mineru_4011_standard` 使用官方 V1 服务和 SDK，提交整份 PDF、明确解析全部页；
不是把整个文档一次性放进 VLM 上下文。Standard 使用小模型布局、区域 VLM 识别及
官方后处理。P=1/2/4 同时设置客户端数量和服务 `--concurrency`（官方默认 1），
所有客户端共享一个服务和模型，不复制 GPU pipeline。
窗口 64、布局 batch 2、公式 batch 32、OCR 检测 batch 上限 16 和 VLM 请求额度 100 保持默认。
小模型 prepare/finish 阶段共享设备锁，等待 VLM 时可释放锁形成文档重叠；
这不等于跨文档自动合并小模型 batch，也不保证增大 P 必然提速。
来源：锁定参考源码 `mineru/model/vlm/runtime.py`、`mineru/model/vlm/engine_utils.py`、
`mineru/parser/api_server.py`、`mineru/backend/analysis/pdf/window.py`、
`mineru/model/runtime/execution.py`；安装包为官方 4.0.11 wheel。

服务官方预加载完成、CPU 客户端完成本地输入能力发现后进入统一 ready/start/done；
SDK 默认传输重试计入处理时间，不追加文档重跑。
官方完整文档 renderer 保留跨页合并；page views 单独投影，不能拼接后替代整文档输出。
生成参数、渲染及上下文沿用官方管线，不额外覆盖；VLM 显存预算由官方按设备决定，
24 GiB 卡约为 0.25，不继承 3.x 的 0.55。仅完整成功且覆盖全部输入页的文档计为成功。
文档 metrics 使用 `doc_total_seconds`；文档并发时该耗时有重叠，不能相加代替 trial 的 `processing_seconds`。
环境与模型说明见
[gdb_mineru_4.md](environments/gdb_mineru_4.md)。

### Latency 参数口径

Latency 不复用吞吐压力映射。HunyuanOCR、DeepSeek-OCR-2、Qianfan、LightOnOCR-2、Xiaomi-OCR-0 和
Unlimited `page_pool` 使用页面级 `P=1`；其他方法只
固定一个外层文档客户端。Monkey 使用官方区域并发 64，TeleOCR 使用 async 默认 100，
Youtu 使用 query batch 5；Paddle 不显式传 `vl_rec_max_concurrency`，GLM 不显式传
`max_workers`（锁定 YAML 的 32），PP 使用基础 `batch_scale=1`，Marker 使用一个
文档 worker，MinerU 3.x 不覆盖 API 的默认并发和 processing window；4.0.11 的默认
服务 concurrency 为 1，latency 使用一个客户端且不改内部 window/batch。Latency summary 会
记录 `latency_profile=official_defaults`；命令行中的 `--pressure 1` 只表示文档级
latency 标识，不代表所有内部并发均为 1。

通用固定 P、调参和 latency 命令统一见 [RUN_COMMANDS.md](RUN_COMMANDS.md)。
