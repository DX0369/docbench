# 项目状态

更新时间：2026-11-10 21:45 +08:00

## 当前阶段

项目已完成主要方法、模型、第三方源码、隔离环境和 adapter 的整理，代码入口和输出协议已完成一次定版精简。当前转入服务器退出前的迁移封版，只整理受管源码、配置、环境配方和运行说明，不在本机继续启动模型测试或吞吐调参。既有验证范围和未解决问题沿用下文；不将小规模结果宣称为最终最优，也不将异机论文重跑视为本机已核验的正式结果。

## 已完成

- 历史 Jina-OCR-v1（旧 `gdb_hunyuan_dflash` 环境 tokenizer 加载错误，输入 prompt 编码与输出解码均受影响，该环境下的结果不可用、不复用） `jina_ocr_v1` 在 GPU 3 完成官方同步单页和两文档 P1/P2/latency smoke：中文试卷、英文杂志各 2 页，各轮输出校验 2/2 通过，无执行失败。固定严格 benchmark prompt、FastMTP eagle K=3，当时复用未改包的 DFlash 环境。修复仅涉及启动工具路径（conda run/ninja、CUDA 12.8）和 spawned worker 调用官方 register；不修改模型或环境源码。P1/P2 ready 后处理 29.16/21.95 秒，吞吐 0.137/0.182 页/秒；latency 中文→英文 24.20/7.82 秒、页面并发1。GPU 3 恢复18 MiB；未正式调参，不宣称全量稳定性或最优。一轮一次性产物核对后清理，仅保留结论。2026-10-20 已确认该旧环境的 tokenizer 兼容性问题，因此上述记录只能证明当时执行链和输出结构通过，不能证明正文正确；当前独立环境及验证范围见下文。

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
- GPU 7 两文档 latency smoke 已完成（产物已不存在，仅保留本条结论，无法复核）：除 MinerU-Popo 外的 17 个方法 ID（共 20 条执行路线）均完成顺序处理；Qianfan 历史 joint 路线因 14 页英文文档输入超过 32768 上下文而明确失败，page_pool 路线通过。该结果仅验证入口顺序、输出和失败记录，不代表正式吞吐结论。
- TeleOCR 已创建 `gdb_teleocr` 环境；`pip check`、TeleOCR/`TeleOCR_vllm` import probe 和 CUDA 检查通过。GPU 6 上已完成统一 adapter 的单文档 throughput smoke 和双文档 latency smoke，均通过输出 validator：latency 使用官方 `max_concurrency=100`，处理顺序为文档 1→2，render DPI 为 200；throughput P=1 路线也成功。产物已不存在，仅保留结论，不代表正式吞吐结果。
- LightOnOCR-2-1B 已登记为 `lightonocr2_1b`。模型卡确认其正式路线是 vLLM 单页请求（`limit-mm-per-prompt=image:1`），不注册整文档 joint；新增 page-pool adapter，P 映射为页面在途数，官方生成参数和 200 DPI/1540px 预处理均写入 metrics。GPU 2 上使用两份 2 页 PDF 完成 P1/P2 有界 smoke，输出校验通过；P1 处理吞吐约 0.218 页/秒，P2 约 0.334 页/秒。当前先复用 `gdb_glmocr`（Transformers 5.14.1、vLLM 0.20.0）；服务日志仅有 NIXL/CUDA graph 等 warning，无 LightOn 推理错误。产物已不存在，仅保留结论，不代表正式吞吐结论。
- Qianfan 普通/think 已按官方单页基础 prompt 和 page pool 在 GPU 5 完成同一份 3 页英文 PDF 的 P1 smoke；两条路线均输出 3 个页面、每页一张图请求，validator 通过，Think metrics 确认 prompt suffix。产物已不存在，仅验证新入口，不代表正式吞吐结论。
- Latency 入口统一为 `run_single_gpu.py --latency-mode`（含 Marker、Youtu、PP；native 入口的 latency 模式已于 2026-10-17 删除）：整页 VLM 页面池路线固定页面级并发 1，其他方法串行提交文档但保留官方/default 内部并发。TeleOCR 已接入常驻 async worker、统一输出和 `P` 映射；所有主要路线的 latency mapping、dry-run 和 Python 语法检查通过。一次性 dry-run 产物已清理，尚未启动真实 latency smoke。

## 当前实验状态

- 2026-11-10 21:45 +08:00 迁移收束（未运行模型或安装环境）：
  - 新增简短根 README；公共安装、模型下载和运行模板使用项目、Conda、数据和模型路径变量，CUDA 路径通过现有方法参数指定。历史 P4 profile 保持原记录，明确不是已确认最优配置，也不能假定绝对路径自动迁移。
  - 根仓库按精确文件路径补入 MonkeyOCRv2 `recognition/strhub/data/` 的六个原始 Python 文件；没有修改上游源码内容或放开真实数据目录。
  - Popo 补丁说明改为根仓库 `--directory --reverse --check`，不重复应用已有补丁；修正 Bash 代码块中的中文正文。Paddle 安装命令为无上游 `.git` 的快照显式标注现有包版本，并显式指定 PyPI 镜像；没有更改包版本或实际环境，无 `.git` 重建流程尚未实际安装验证。
  - 按用户确认，不上传本项目论文、历史代码、输入、预测、权重或缓存，不增加模型版本锁或模型小文件备份，也不扩展系统工具与启动器依赖安装流程。模型继续按官方仓库名下载完整文件，补充 Popo 下载命令。
  - 定向静态检查通过：103 个 Shell 代码块（另有 7 个占位模板未直接解析）、Markdown 代码块和文件链接、六个源码的语法及与上游的一致性、Popo 补丁反向检查；待上传路径检查未发现被排除的资产或高置信度凭据模式。不据此宣称新机器安装或推理已验证。

- 2026-10-21 12:01 +08:00 LightOnOCR-2 重复生成追加定位完成（GPU 0，CPU 输入核对及五次单图请求）：
  - 沿用上一轮 PDFium 异常整页，直接 HTTP 绕过项目 adapter 后，与此前输出去除首尾空白后完全一致；再用官方 HF 模型类/processor 独立生成，两行短语各重复283次，同样用满4096 tokens。HF 生成参数有意对齐 vLLM 的 temperature=0.2/top_p=0.9/上限4096、seed=0，不声称逐字照搬 HF 示例的1024上限默认解码；权重加载无缺失、意外或形状不匹配。
  - 同环境 HF 控制：官方收据359 tokens、简单中文封面154 tokens、同报纸单个文章裁图1206 tokens，均正常 EOS、不再出现数百次循环。人工抽查裁图仍有明显错误，如“3岁以下”变“3岁以上”，“托育费用仍偏高”变“仍偏低”；正常结束不代表正文准确。
  - CPU 实际 processor 核对：两后端 tokenizer 组件、特殊 token 和非图像模板 token 一致；regex 修复 flag 仅内存试验未改变该 image-only 模板 IDs，未应用于模型或正式运行。图像路径有真实差异：HF 张量 `[3,1540,1078]`/输入2102 tokens，vLLM `[3,1540,1092]`/2157 tokens，来自当前14/28对齐差异；两边均重复，不能把它判为唯一根因。
  - 结论限于本案例：重复已在服务响应中，独立官方 HF 也复现，更支持官方推理链在密集中文整页上的退化，而非项目拼接/调用错误。裁图同时改变有效分辨率、布局和内容量，不能单独定因；共用 Transformers/底层算子与模型资产尚未完全隔离，不宣称模型的绝对能力上限或推广到所有异常。
  - 诊断产物已于 2026-10-21 清理，仅保留本条结论。未改正式 adapter、环境、权重或历史输出；已清理可再生缓存及一次推理前的诊断脚本序列化失败记录。GPU 0 恢复32 MiB，自建服务端口已释放。本轮到此停止，不自动加去重、修改 tokenizer、切换裁图路线或扩展实验。

- 2026-10-21 10:47 +08:00 LightOnOCR-2 异常页预处理对照完成（GPU 0，两次单页请求）：
  - 对中文报纸 `supplement_newspaper_zh_______2025-07-17__1_4` 第 1 页，复用同一个 `gdb_glmocr` / vLLM 服务和现有 image-only 请求函数；两条均固定诊断 seed=0、temperature=0.2、top_p=0.9、max_tokens=4096。只比较现有 Poppler 200 DPI + PIL/LANCZOS 1540 预缩放，与官方模型卡 PDFium scale=2.77 + PNG、不做客户端缩放；不修改主线 adapter、环境或模型。
  - 现有路线请求图 1065×1540，输出 4,245 字符，末段出现连续 1,891 次“是，”；官方路线请求图 3117×4505，输出 4,468 字符，“2025年7月 版摘”出现 372 次。两条均为 2,157 输入 tokens、4,096 生成 tokens、finish_reason=length；均未复现历史特定短语“深圳市教育局”，但都发生明显重复退化。
  - 结论：切换官方模型卡预处理没有消除该页异常，不能将问题单独归因于 Poppler 或客户端预缩放；单页单 seed 不能确定重复发生率，也未分离模型能力与当前后端/环境的影响。服务另有 tokenizer regex 兼容 warning，本轮不修改，也不认定其为根因。没有改生成参数、加去重或展开其它测试。
  - 诊断产物已于 2026-10-21 清理，仅保留本条结论。首次启动因诊断临时路径过长触发 Unix socket 限制，缩短路径后完成；已清理该次启动失败产物及可再生缓存。自建服务已停止，GPU 0 恢复约 32 MiB。

- 2026-10-21 00:38 +08:00 Unlimited 多页异常追加定位，GPU 0 五次有界请求完成：
  - 不改正式 adapter、环境、权重或配置。SGLang 复用上一轮报纸原图，分别只送前 1/2 页；再用现有 `gdb_unlimited_ocr`（Torch 2.10.0+cu128、Transformers 4.57.1）直接调用官方 `infer_multi`，对同图前 1/2 页各运行一次；最后使用官方 README 的 PyMuPDF 300 DPI 渲染前两页，再运行一次官方 Transformers。均为 joint/base、1024 图像尺寸、greedy、ngram 35/window 1024、P1；固定输出上限 16384，HF 的 `max_length` 按实际输入长度加 16384 设置，未误作输出上限。
  - SGLang 前 1 页：278 输入、2283 生成 tokens、9773 字符、1 个 PAGE；前 2 页：551 输入、5044 生成 tokens、20922 字符、仍仅 1 个 PAGE。HF 同图前 1 页：2360 生成 tokens、10123 字符、1 个 PAGE；前 2 页：5796 生成 tokens、25851 字符、仍仅 1 个 PAGE，并有局部问答重复。HF + PyMuPDF 300 DPI 前 2 页：4301 生成 tokens、17670 字符、仍仅 1 个 PAGE。五次均正常 EOS 1 结束，未触及生成上限；这些时间不用于后端速度比较。
  - 原图人工核对：单页两后端均识别出首页大标题及对应正文，但不能据此宣称内容全部正确；两页三个配置均未输出第二页大标题，第二页部分正文混入同一段，缺 PAGE 不只是导出器的计数问题。HF 加载 warning 中的 `vision_model.embeddings.position_ids` 已在实际模型实例核对为完整 `arange(257)`，不是随机初始化的视觉权重。
  - 当前结论：该两页异常在独立官方 Transformers 路线及官方渲染方式下同样出现，不依赖本项目 SSE/Markdown 导出、SGLang SM86 补丁或 Poppler 200 DPI；更指向官方多页推理链在此文档上的问题。尚不能把模型权重能力与上游实现缺陷完全分开，也没有在独立 backend 重跑原 26 页以复现其短句 111 次重复，不将两页结论扩大为所有异常的根因。未发现需要立即修改正式调用或重装环境的证据；仍保留质量 warning，不升级为执行失败、不自动补分页。
  - 诊断产物已于 2026-10-21 清理，仅保留本条结论。本轮可再生缓存和空输出目录已清理。所有诊断进程已退出，GPU 0 恢复约 32 MiB，端口 29887 无监听。

- 2026-10-20 22:54 +08:00 Unlimited 异常定位，有界对照完成：
  - CPU 实际加载官方 SGLang tokenizer/processor：输入 processor 的原始分词组件完全一致；输出 tokenizer 的 ByteLevel 预分词器/解码器一致，空 normalizer 表示及 `byte_fallback`/`fuse_unk` 标志不同。原始 ByteLevel 字母表 256/256 均在词表内，词表、merges 和特殊 token 配置一致；中英文、空白、两个官方 prompt、标签及完整 32,360 字符异常文本的编码、解码、批量解码均与原始 tokenizer 一致。未发现该案例的 Jina 式编解码错乱，不宣称所有 tokenizer 状态逐字相同。
  - 按用户指定 GPU 0，仅一个服务、P1、两文档各两条请求路径，共四次请求。官方 README 请求函数与项目 adapter 共用同一组 Poppler 200 DPI 图片，固定 joint/base、temperature 0、ngram 35/window 1024、max_tokens 16384、context 32768、显存比例 0.9；实际生成 payload 一致，仅 adapter 额外请求流式 usage。官方请求也显式加上项目的 16384 上限，不称为原封不动的完整官方示例。
  - RFC6263 控制文档 3 页：两条路径原始文本逐字一致，8,819 字符、3 段、stop，adapter 分页导出检查通过且无 warning。China Daily 报纸 26 页：两条路径及导入的历史异常文本逐字一致，32,360 字符、1 个 PAGE、目标短句重复 111 次；adapter usage 为 prompt 7,103、completion 10,051，以 stop/EOS 1 结束，未触及上限；保存内容与原始响应一致，只记录 page_count_mismatch。
  - 第一轮结论限于所测样本：重复/缺分页已在服务返回内容中存在，不是 adapter 的流收集或 Markdown 导出造成。该轮尚未分离共同的图像预处理、模型推理及 SM86 兼容路径，不能直接定性为模型固有能力限制；后续页数、渲染和独立 backend 对照见上条。未做吞吐搜索，也未修改环境、模型和历史预测。
  - 诊断产物已于 2026-10-21 清理，仅保留本条结论。可再生编译缓存与运行临时目录已清理。服务已退出，端口 29887 释放，GPU 0 恢复约 32 MiB。

- 2026-10-20 22:32 +08:00 Unlimited 记录修复（未运行 GPU）：
  - 保留固定 `max_tokens=16384` 和现有推理配方；`page_pool` metrics 补记实际生成上限。
  - `page_pool` 的逐页长度 warning 纳入 trial 汇总，按文档计数，并复用现有调参产物保留规则；有 warning 的输出不再被成功产物清理删除。正常结束的长度 warning 仍不算执行失败，joint 的记录不变。
  - 六项定向 CPU 测试通过，覆盖实际导出、请求上限、两路线 warning 汇总/保留、无 warning 时的清理与保留、缺失 warning 字段显式报错及既有输出保护；服务、模型响应和监控均模拟，一次性产物已清理。未修改环境、模型或历史预测。
  - 报纸案例的重复与缺分页根因仍未证实；后续 tokenizer 和同图同参数对照已完成，结果见上条。

- 2026-10-20 21:55 +08:00 Jina tokenizer 环境修复与 P1 验证：
  - 旧环境 CPU 对照实际复现：vLLM 将配置识别为 `deepseek_ocr`，Transformers 5.5.3 按 Llama 路径加载出 `Metaspace`，不符合模型 `tokenizer.json` 的 ByteLevel 配置。独立 `gdb_jinaocr` 从旧环境克隆，恢复克隆产生的 setuptools 重复登记后，包版本逐项对比确认仅 Transformers 改为 5.14.1；原 DFlash 环境未修改，未新增依赖冲突。
  - 新环境实际 vLLM tokenizer 与 prompt tokenizer 均保留原始预分词器和 ByteLevel decoder；中英文、空格、换行、特殊 token 及完整严格 prompt 的编码与解码均与原 tokenizer.json 一致。官方 helper、AsyncLLM 与重复检测接口检查通过，不引入自定义 tokenizer、模型源码补丁或输出字符替换。
  - GPU 0 只运行 P1：NIST `industry_standard_en_NIST_SP_800-87r2_74_76` 2 页、中文试卷 `exam_zh_2019_______________________0_3` 3 页，共两文档五页。执行成功 2/2、页覆盖 5/5，validator 为 2/2、零 issue，无失败或遗漏；全部页面以 `stop` 结束、质量 warnings 为空。
  - 五页均非空且无字面 `Ġ/Ċ`；NIST 原空白第一页生成 1,716 字符，第二页 1,811 字符，抽查机构代码与 PDF 正文相符；中文三页分别为 991、1,379、1,115 字符。这里只确认所选案例的异常消失，不代表全部历史空页的根因均已验证，也不构成完整准确率评测。
  - 保留严格 prompt、200 DPI、4096 生成上限、32768 上下文及官方 FastMTP eagle K=3；日志确认 drafter 加载、MTP 共享权重和推测解码路径。启动 250.890 秒，ready 后处理 27.233 秒，总计 278.123 秒；首次 JIT/缓存影响未控制，不据此宣称吞吐最优。引擎正常退出，GPU 0 恢复约 32 MiB。
  - 默认环境绑定、迁移配方、运行命令与独立环境三份锁已同步。按用户要求未运行 P2、latency、压力搜索或全量重推理；历史预测未改写。本轮一次性 smoke、诊断及缓存核对后清理，只保留本条结论。

- 2026-10-17 15:59 +08:00 审计与修复（未启动 GPU/正式运行）：
  - 证据：`experiments/` 为空；tmp、archive 中找不到任何 3090 20 文档（146 页）P1/P2/P4 结果，下表“既有 20 文档”类结论及 `methods.json` 中相应 `validation_status` 仅为历史记录，不可复核；`throughput_profiles/rtx3090.json` 的 Unlimited joint P4、Marker P4 无调参产物，不视为定版。
  - 框架修复：`validate_outputs` 将含 `error.json` 的文档判为失败；latency 中途停止时未到达的 PDF 记入 `unprocessed_docs`；`collect_failed_pdfs.py` 按当前 aggregate schema 重写并显式失败；新增可选 `--processing-timeout-seconds`；删除 native 入口的第二套 latency 实现与 legacy 计时分支。
  - Adapter 修复（需真实 smoke 复核）：Jina 官方重复检测终止改记 warning；Hunyuan 1.0 补官方 `clean_repeated_substrings`，1.0/1.5 记录 finish_reason 与长度 warning；LightOn、Qianfan、DeepSeek 补长度 warning；Qianfan 按官方模板取最后一个 `</think>` 之后的文本，未闭合时记 warning；DeepSeek 按官方 `SKIP_REPEAT` 将无 EOS 页置空并记 warning，统计 ref/det 解析失败；TeleOCR 对上游静默跳过的布局行、区域、后处理异常和截断区域计数；WeVisDoc 缺 finish_reason 时显式失败。
  - CPU 验证：新增 `validation/test_failure_records.py` 7 项，`test_mineru4` 10 项、`test_monkeyocrv2` 3 项通过；修改路线的 throughput/latency dry-run 通过。
  - 2026-10-17 16:25 +08:00 按用户决定继续修改（CPU 测试 9 项与 dry-run 通过，未跑 GPU）：Unlimited `page_pool` 改用官方单图配方（"document parsing."、gundam、ngram window 128），`joint` 保持多页配方，`--skip-existing` 不复用不同配方的输出；Hunyuan 1.5 AR 去掉仅属 1.0 README 的 `--no-enable-prefix-caching --mm-processor-cache-gb 0`，1.5 上下文在所有入口统一为 65536；MonkeyOCRv2 取消 8 文档供给上限，提交全部文档、仅由官方页面窗口 P 约束；DeepSeek 图片 ref 按官方写 `images/{jdx}_{idx}.jpg` 悬空占位（不保存裁图）；放弃 MonkeyOCRv2 DFlash 并删除其诊断产物。其他与官方不同的显存/上下文/max_num_seqs 维持现状。
  - 2026-10-17 17:07 +08:00 第三轮修改（CPU 测试 14 项、`test_mineru4`、`test_monkeyocrv2` 与 dry-run 通过，未跑 GPU）：Unlimited 与 `run_service_batch`/`run_native_batch` 拒绝覆盖已有输出；OpenAI SDK 传输重试按页计为 `sdk_transport_retry`；GPU 监控记录采样失败；metrics 质量 warning 统一为 `warnings: [{page_number, reason, ...}]`（Unlimited 改名 `generation_length_limit`，旧产物不改写、新 validator 不再兼容旧字段）；MonkeyOCRv2 不再复制图片，TeleOCR 恢复官方图片引用但不落盘（裁图开销与官方一致，计入吞吐）；DeepSeek 渲染进程数 = min(P, CPU 核数)（旧结果为串行渲染，吞吐不可直接比较）；MinerU 3.x 身份字段缺失即失败；旧 MonkeyOCR 区域失败按所属 PDF 隔离（修改 failure_propagation 补丁与快照）；版本核对工具改为快照与版本元数据检查，`sync` 删除，MinerU 三个仓库登记 `version_file`/`source_version`。MinerU 4 缺 body 由上游 schema 保证、无需修改；MonkeyOCRv2 兼容入口已按全量输入成 job，无需修改。
  - 快照无 `.git` 引发的问题：`mineru/preload.py` 的 `git rev-parse` 取到根仓库 HEAD，MinerU 3.4.x preload 会因 revision 不符报错；`require_failure_patch` 与环境配方中的 `git -C <快照> apply --check` 在子目录下跳过全部文件、校验形同虚设（现有补丁实际均已应用）。2026-10-17 18:52 +08:00 已修复：preload 与 Popo preflight 改为核对 `methods.json` 快照与版本元数据（3.4.3 由 `source_version` 登记，去掉硬编码）；补丁检查与配方改为在根目录用 `--directory` 执行，CPU 测试确认能检出未应用补丁。MinerU-Popo 不再复制图片资产，保留悬空引用。均未跑 GPU。
  - 补 `gdb_teleocr`、`gdb_monkeyocrv2` 环境锁；删除旧仓库副本 `/data/wangsq/code/docbench_repo`（其 HEAD 已包含在当前历史中）。

- 2026-10-10 14:12 +08:00 同步：MonkeyOCRv2 B-Parsing 普通 AR 已接入 `monkeyocrv2_b_parsing` / `monkeyocrv2_b_parsing_vllm_ar`，复用官方常驻 pipeline 和 V2 runner / CUDA graphs，不启用 DFlash。GPU 6 上中文试卷 2 页、英文杂志 2 页、英文试卷 6 页，共 3 文档 10 页，P1/P2 执行与输出结构校验均通过（3/3、零 issue、无失败）。启动耗时 77.32/67.66 秒，ready 后处理 26.78/19.15 秒，处理吞吐 0.373/0.522 页/秒；仅为单次有界 smoke，缓存未统一控制，不据此选择最优 P 或宣称内容准确率。初次 P1 因 adapter 将官方 JSON 文件名与绝对路径比较而导出失败，已修正并通过回归，不改上游源码、模型或依赖。抽查中文正文、表格和图片引用；英文试卷末页仅含页眉页脚，官方过滤后 Markdown 为空但页覆盖完整。服务已退出，GPU 6 恢复约 18 MiB。Latency 已实现逐文档提交、内部页面窗口 64，并通过代码测试和 dry-run，真实 GPU latency smoke 尚未完成。smoke 产物已于 2026-10-21 清理。环境迁移锁已于 2026-10-17 导出。
- MonkeyOCRv2 DFlash：曾通过旧 runner + eager + HF override 的六页 smoke；默认 V2 runner warmup 失败定位至 Triton top-k/top-p kernel 非法写入，根因未修复。2026-10-17 用户决定放弃该路线，诊断产物已删除；不接入、不推广其兼容配置。

| 方法族/路线 | 当前状态 |
| --- | --- |
| Jina-OCR-v1 | 独立 `gdb_jinaocr` / Transformers 5.14.1 修复实际 vLLM ByteLevel 加载；GPU 0 两文档五页 P1 通过，原 NIST 空页与所选页 `Ġ/Ċ` 异常消失。新环境未跑 P2/latency，未正式调参或全量重推理。 |
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
| WeVisDoc-2B | `methods.json` 记为 GPU 7 P1 短多页 smoke 通过（无保留产物）；`--enforce-eager` 为本机兼容设置，服务 `max-num-seqs 8` 为官方默认；未调参。 |
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

当前先完成受管迁移材料的提交与远端同步核对；退出本服务器后，在新机器按根 README 恢复所需环境和模型，并另行决定有界验证。现机环境与模型不会随 Git clone 恢复。

以下为恢复后的研究待办，不在本轮启动：

先核查 MinerU 4.0.11 P1/P2 smoke 正文及复杂跨页、长文档覆盖需求，再决定是否扩展有界验证或进行真实吞吐搜索。

0. 正式调参前，对本轮改动的 Hunyuan、LightOn、Qianfan、DeepSeek、TeleOCR、WeVisDoc、Unlimited page_pool、MonkeyOCRv2 路线做真实 GPU 短 smoke（用户暂缓）；Paddle 服务配置暂不改。Jina 新环境已完成上述 P1 验证，后续是否重推理历史异常页由用户决定，不自动扩大运行范围。

1. 使用固定的 20 文档清单，对各方法执行有限 P 搜索或指定 P 运行；一次只使用一张 GPU，不自动并行分发正式任务。
2. 为每个方法记录可复现的选定 profile，并区分官方路线、项目兼容补丁和实验性路线。
3. 在 profile 稳定后，再决定是否启动 163 文档正式吞吐实验；GT、scorer 和 MinerU-Popo 不在当前阶段范围内。
