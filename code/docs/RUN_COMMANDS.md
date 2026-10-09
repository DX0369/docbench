# GlobalDocBench 运行命令

本文档只维护迁移后可直接复制的运行命令。环境安装见
[`environments/`](environments/README.md)，模型下载见 [`MODELS.md`](MODELS.md)，
吞吐口径和失败规则见 [`THROUGHPUT_TUNING_PROTOCOL.md`](THROUGHPUT_TUNING_PROTOCOL.md)。
本页只保留可复用命令模板；实际运行记录和结果见
[`PROJECT_STATE.md`](../../PROJECT_STATE.md)。

## 公共变量

从 `code/` 目录执行。启动 Python 只负责读取配置和启动子进程；各方法 worker
会按 `config/methods.json` 和 `execution/tasks.py` 选择自己的 Conda 环境。

```bash
export GDB_ROOT=/data/wangsq/code/global_doc_bench
export CODE="$GDB_ROOT/code"
export PDF_ROOT="$GDB_ROOT/data/pdf_tree"
export INPUTS="$CODE/config/throughput_manifests/rtx3090_20.json"
export GPU=6                         # 运行前由用户确认空闲
export RUN_ID="benchmark_$(date +%Y%m%d_%H%M%S)"
export RUN_ROOT="$GDB_ROOT/experiments/benchmark/$RUN_ID"
export PY=/data/wangsq/software/miniconda/bin/python

mkdir -p "$RUN_ROOT/outputs" "$RUN_ROOT/logs"
cd "$CODE"
```

不要复制 PDF；清单只引用 `data/pdf_tree` 中的原始文件。每次运行必须使用新的
`RUN_ID`，不要覆盖既有输出。

`--pdf-list-file` 和 `--input-root` 二选一：传入清单时只运行清单中的 PDF；只传
`--input-root` 时递归发现该目录下的全部 PDF。当前固定 20 文档协议使用
`rtx3090_20.json`；完整数据运行应使用 `$PDF_ROOT`。

## 固定 P：所有正式吞吐方法

```bash
run_fixed() {
    local method="$1"
    local pressure="$2"
    shift 2
    "$PY" src/benchmark/run_single_gpu.py \
        --method "$method" \
        --gpu-id "$GPU" \
        --pressure "$pressure" \
        --pdf-list-file "$INPUTS" \
        --run-name "${RUN_ID}_${method}_p${pressure}" \
        --output-root-base "$RUN_ROOT/outputs" \
        --log-root "$RUN_ROOT/logs/${method}_p${pressure}" \
        "$@"
}
```

示例：

```bash
run_fixed deepseek_ocr2 1
run_fixed pp_structurev3 1
run_fixed monkeyocr_pro_3b 4
run_fixed teleocr_vllm_async 1
run_fixed lightonocr2_1b 1
run_fixed xiaomi_ocr_0 1
run_fixed jina_ocr_v1 1             # 复用 DFlash 环境；自动指定 CUDA 12.8 工具链
```

当前正式 method ID 和 P 的实际含义如下：

| method ID | execution route | P 控制的参数 |
| --- | --- | --- |
| `jina_ocr_v1` | 常驻 AsyncLLM FastMTP page pool | 全局页面在途数 |
| `hunyuanocr_10` | vLLM AR page pool | 页面在途数 |
| `hunyuanocr_15` | vLLM AR page pool | 页面在途数 |
| `hunyuanocr_15_dflash` | vLLM DFlash page pool | 页面在途数 |
| `deepseek_ocr2` | resident page pool | 页面在途数 |
| `qianfan_ocr` | vLLM 单页 page pool | 页面在途数 |
| `qianfan_ocr_think` | vLLM 单页 page pool | 页面在途数 |
| `unlimited_ocr_sglang` | SGLang joint 或 page pool | 文档客户端数或页面在途数 |
| `monkeyocr_pro_3b` | vLLM async multifile | 分组内区域识别并发上限 |
| `monkeyocr_pro_1_2b` | vLLM async multifile | 分组内区域识别并发上限 |
| `paddleocr_vl_15` | 官方多 PDF queue pipeline | 区域识别并发上限（异步 HTTP） |
| `paddleocr_vl_16` | 官方多 PDF queue pipeline | 区域识别并发上限（异步 HTTP） |
| `glmocr` | 单 pipeline 区域 worker | 区域识别并发上限（HTTP 线程池） |
| `mineru_303_2509` | MinerU 3.0.3 API | API 客户端/任务上限 |
| `mineru_344_2605` | MinerU 3.4.x API | API 客户端/任务上限 |
| `mineru_4011_standard` | MinerU 4.0.11 Standard V1 API | 文档客户端数及官方服务 `--concurrency` |
| `pp_structurev3` | 单 pipeline page prefetch | 原生 batch 缩放 |
| `marker` | shared model workers | 文档 worker 数 |
| `youtu` | native resident parser | 同页单次文字查询包含的 bbox 数 |
| `teleocr_vllm_async` | resident TeleOCR vLLM async pipeline | 当前 PDF 内共享的 layout/region 请求并发 |
| `lightonocr2_1b` | official vLLM single-page page pool | 跨 PDF 的页面在途数；每个请求固定一张图片 |
| `xiaomi_ocr_0` | official batch whole-page + vLLM page pool | 跨 PDF 的页面渲染／请求在途数；不登记 region 或 joint |

Qianfan 固定使用单页 page pool；Unlimited 的 page pool 路线仍需显式选择：

```bash
run_fixed unlimited_ocr_sglang 4 --unlimited-ocr-request-mode page_pool
```

如果已经保存选定 profile，固定运行改用：

```bash
"$PY" src/benchmark/run_single_gpu.py \
    --method <METHOD_ID> \
    --gpu-id "$GPU" \
    --profile-file "$CODE/config/throughput_profiles/rtx3090.json" \
    --execution-profile <EXACT_PROFILE_ID> \
    --pdf-list-file "$INPUTS" \
    --run-name "${RUN_ID}_<METHOD_ID>" \
    --output-root-base "$RUN_ROOT/outputs" \
    --log-root "$RUN_ROOT/logs/<METHOD_ID>"
```

### 完整 PDF 目录的正式固定 P 运行

下面的入口不传 `--pdf-list-file`，会递归处理 `$PDF_ROOT` 下的全部 PDF。它适用于
已经选定 P 后的正式运行；当前数据目录预计包含 163 份 PDF。

```bash
"$PY" src/benchmark/run_single_gpu.py \
    --method <METHOD_ID> \
    --gpu-id "$GPU" \
    --pressure <P> \
    --input-root "$PDF_ROOT" \
    --run-name "${RUN_ID}_<METHOD_ID>_full" \
    --output-root-base "$RUN_ROOT/outputs" \
    --log-root "$RUN_ROOT/logs/<METHOD_ID>_full"
```

正式吞吐运行不添加 `--latency-mode` 或 `--dry-run`。完整目录运行与 20 文档清单运行
必须使用不同的 `RUN_ID` 和输出目录。

## 压力搜索

调参只使用固定 20 文档清单；不在搜索入口复制 PDF：

```bash
"$PY" src/benchmark/run_pressure_sweep.py \
    --methods <METHOD_ID> \
    --pressures 1,2,4,8,16,32,64,128,256 \
    --gpus "$GPU" \
    --execution-policy serial \
    --pdf-list-file "$INPUTS" \
    --run-name "${RUN_ID}_sweep" \
    --save-profile "$CODE/config/throughput_profiles/rtx3090.json"
```

TeleOCR 只使用单个常驻 async runtime；P 映射为当前 PDF 内 layout/region 请求的
`max_concurrency`，文档之间保持官方顺序：

```bash
"$PY" src/benchmark/run_pressure_sweep.py \
    --methods teleocr_vllm_async \
    --pressures 1,2,4,8,16,32,64,128,256 \
    --gpus "$GPU" \
    --execution-policy serial \
    --pdf-list-file "$INPUTS" \
    --run-name "${RUN_ID}_teleocr_sweep" \
    --save-profile "$CODE/config/throughput_profiles/rtx3090.json"
```

LightOnOCR-2 使用官方 vLLM 单页路线，不支持整篇 PDF joint。服务端固定每个请求一张
图片，客户端 page pool 用 P 提供跨文档页面供给：

```bash
"$PY" src/benchmark/run_pressure_sweep.py \
    --methods lightonocr2_1b \
    --pressures 1,2,4,8,16,32,64,128,256 \
    --gpus "$GPU" \
    --execution-policy serial \
    --pdf-list-file "$INPUTS" \
    --run-name "${RUN_ID}_lightonocr2_sweep" \
    --save-profile "$CODE/config/throughput_profiles/rtx3090.json"
```

固定 P 运行：

```bash
run_fixed lightonocr2_1b 4
```

该 adapter 使用模型卡的 `max_tokens=4096`、`temperature=0.2`、`top_p=0.9`、
200 DPI 和最长边 1540 像素；不显式设置 `max_num_seqs` 或额外上下文上限。

如果要对完整目录搜索，去掉 `--pdf-list-file`，改为显式指定：

```bash
"$PY" src/benchmark/run_pressure_sweep.py \
    --methods <METHOD_ID> \
    --pressures 1,2,4,8,16,32,64,128,256 \
    --input-root "$PDF_ROOT" \
    --gpus "$GPU" \
    --execution-policy serial \
    --run-name "${RUN_ID}_<METHOD_ID>_full_sweep" \
    --save-profile "$CODE/config/throughput_profiles/rtx3090.json"
```

压力搜索会在该 run 内保存实际输入清单，便于复核；这不是需要用户预先创建的
job 文件。

Qianfan 直接搜索单页 page_pool；Unlimited 的 `page_pool` 仍不能把
`joint` 的搜索结果当作其配置：

```bash
# Qianfan page_pool
"$PY" src/benchmark/run_pressure_sweep.py \
    --methods qianfan_ocr \
    --pressures 1,2,4,8,16,32,64,128,256 \
    --gpus "$GPU" \
    --execution-policy serial \
    --pdf-list-file "$INPUTS" \
    --run-name "${RUN_ID}_qianfan_page_pool_sweep" \
    --save-profile "$CODE/config/throughput_profiles/rtx3090.json"

# Unlimited-OCR page_pool
"$PY" src/benchmark/run_pressure_sweep.py \
    --methods unlimited_ocr_sglang \
    --pressures 1,2,4,8,16,32,64,128,256 \
    --unlimited-ocr-request-mode page_pool \
    --gpus "$GPU" \
    --execution-policy serial \
    --pdf-list-file "$INPUTS" \
    --run-name "${RUN_ID}_unlimited_page_pool_sweep" \
    --save-profile "$CODE/config/throughput_profiles/rtx3090.json"
```

先只检查命令映射时加 `--dry-run`；只有真实运行并确认成功的配置才保存 profile。
失败档必须保留在该 run 的日志中，不能由前一个成功档替代。

## Xiaomi-OCR-0：整页路线

仅接入官方整页解析，不启用 PP-DocLayoutV3、region 或 joint。服务和 worker
默认复用 `gdb_glmocr`。
复用官方 batch prompt、RGB/最长边 2048px 图像编码和 Markdown 后处理；
PDF 渲染使用项目指定的 200 DPI。生成上限 16384、temperature 0、top_p 0.8、
thinking 关闭、请求超时 600 秒；服务上下文 32768 是项目设置。
不设置 `max_num_seqs`，显存比例、引擎 batch 和编译模式沿用 vLLM 默认。

固定 P 使用上面的公共变量和 `run_fixed`：

```bash
run_fixed xiaomi_ocr_0 4
```

20 文档压力搜索：

```bash
"$PY" src/benchmark/run_pressure_sweep.py \
    --methods xiaomi_ocr_0 \
    --pressures 1,2,4,8,16,32,64,128,256 \
    --gpus "$GPU" \
    --execution-policy serial \
    --pdf-list-file "$INPUTS" \
    --run-name "${RUN_ID}_xiaomi_sweep" \
    --save-profile "$CODE/config/throughput_profiles/rtx3090.json"
```

逐文档延迟测试（PDF 和页请求都顺序执行，服务只加载一次）：

```bash
"$PY" src/benchmark/run_single_gpu.py \
    --method xiaomi_ocr_0 \
    --gpu-id "$GPU" \
    --pressure 1 --latency-mode \
    --pdf-list-file "$INPUTS" \
    --run-name "${RUN_ID}_xiaomi_latency" \
    --output-root-base "$RUN_ROOT/outputs" \
    --log-root "$RUN_ROOT/logs/xiaomi_latency"
```

全量输入可将 `--pdf-list-file "$INPUTS"` 换为 `--input-root "$PDF_ROOT"`。
Smoke 需把输出和日志显式改到独立 `tmp/<run_id>/`，不能写入正式结果目录。
必要覆盖参数只有 `--xiaomi-env`、`--xiaomi-model-path`、`--xiaomi-dpi`、
`--xiaomi-max-tokens`、`--xiaomi-max-model-len`；P 仍由统一 `--pressure` 控制。

## 正式 Latency 测试

Latency 与吞吐是两类独立的正式指标：吞吐测试不同 P 下的 pages/s、docs/s；
Latency 使用单 GPU、单外层 worker，逐个 PDF 记录处理耗时。四个页面请求路线
（HunyuanOCR、DeepSeek-OCR-2、LightOnOCR-2、Qianfan `page_pool`、Unlimited `page_pool`）保持
页面级 `P=1`；其他方法只把文档级并发固定为 1，内部区域并发、query batch、
pipeline worker 和原生 batch 使用官方/default 参数。命令中的 `--pressure 1`
只是兼容性的 latency 标识，不再把所有方法的内部参数改成 1。
Latency 结果不能与吞吐结果共用 run ID 或输出目录。

正式 Latency 使用 `experiments/latency/<run_id>/`；两文档流程验证使用下面的
`tmp/latency_smoke/` 小节，不把 smoke 结果当作正式结论。

### 固定清单的正式 Latency

```bash
export LATENCY_RUN_ID="latency_$(date +%Y%m%d_%H%M%S)"
export LATENCY_ROOT="$GDB_ROOT/experiments/latency/$LATENCY_RUN_ID"

mkdir -p "$LATENCY_ROOT/outputs" "$LATENCY_ROOT/logs"

"$PY" src/benchmark/run_single_gpu.py \
    --method <METHOD_ID> \
    --gpu-id "$GPU" \
    --pressure 1 \
    --latency-mode \
    --pdf-list-file "$INPUTS" \
    --run-name "${LATENCY_RUN_ID}_<METHOD_ID>" \
    --output-root-base "$LATENCY_ROOT/outputs" \
    --log-root "$LATENCY_ROOT/logs/<METHOD_ID>"
```

### 完整 PDF 目录的正式 Latency

不传 `--pdf-list-file` 时，入口递归处理 `$PDF_ROOT` 下的全部 PDF：

```bash
"$PY" src/benchmark/run_single_gpu.py \
    --method <METHOD_ID> \
    --gpu-id "$GPU" \
    --pressure 1 \
    --latency-mode \
    --input-root "$PDF_ROOT" \
    --run-name "${LATENCY_RUN_ID}_<METHOD_ID>_full" \
    --output-root-base "$LATENCY_ROOT/outputs" \
    --log-root "$LATENCY_ROOT/logs/<METHOD_ID>_full"
```

Latency 模式只改变文档之间的调度：runtime/service 只启动一次，前一个 PDF
完成并写出结果后才提交下一个 PDF；页面路线按上面的页面规则运行，其他方法的
区域、query 和组件内部官方/default 并发保持不变。summary 会记录
`latency_profile=official_defaults` 及实际映射。
每个方法应生成 `latency_summary.json`，记录单文档耗时、处理总时间、启动时间和
失败文档。Latency 不计算 GT 或 scorer；MinerU-Popo 当前不纳入该协议。

## Jina-OCR-v1

环境见 [gdb_jinaocr](environments/gdb_jinaocr.md)。仅 page pool，不注册 joint。
固定官方严格 benchmark prompt，FastMTP `eagle` K=3。
默认复用 DFlash 环境，工具链见环境说明；smoke 的 FlashInfer workspace 也应指向 tmp。

```bash
run_fixed jina_ocr_v1 4
run_fixed jina_ocr_v1 1 --latency-mode

"$PY" src/benchmark/run_pressure_sweep.py \
    --methods jina_ocr_v1 --pressures 1,2,4,8,16,32,64,128,256 \
    --gpus "$GPU" --execution-policy serial --pdf-list-file "$INPUTS" \
    --run-name "${RUN_ID}_jina_sweep" \
    --save-profile "$CODE/config/throughput_profiles/rtx3090.json"
```

## 两文档 latency smoke（不属于正式吞吐结果）

该模式只验证“单个 PDF 完成后再处理下一个 PDF”。输入清单或 native 临时输入根目录
应放在 `tmp/latency_smoke/<run_id>/`，输出和日志也必须留在 `tmp`，不能写入
`experiments/benchmark`。

服务型方法命令：

```bash
export LATENCY_ROOT="$GDB_ROOT/tmp/latency_smoke/<run_id>"
export LATENCY_INPUTS="$LATENCY_ROOT/inputs/pdf_list.txt"

"$PY" src/benchmark/run_single_gpu.py \
    --method <METHOD_ID> \
    --gpu-id "$GPU" \
    --pressure 1 \
    --latency-mode \
    --pdf-list-file "$LATENCY_INPUTS" \
    --output-root-base "$LATENCY_ROOT/outputs" \
    --log-root "$LATENCY_ROOT/logs/<METHOD_ID>" \
    --run-name "latency_<METHOD_ID>"
```

Marker、Youtu 和 PP-StructureV3 使用 native 入口。`inputs/native/` 中应只放两个
输入 PDF 的软链接，不复制原始 PDF：

```bash
"$PY" src/benchmark/run_native_batch.py \
    --methods marker,youtu,pp_structurev3 \
    --gpus "$GPU" \
    --latency-mode \
    --input-root "$LATENCY_ROOT/inputs/native" \
    --output-root-base "$LATENCY_ROOT/outputs/native" \
    --log-root "$LATENCY_ROOT/logs/native" \
    --run-id "latency_native"
```

latency 模式固定使用单 GPU、单 worker，不生成 `job_*.txt`；每个方法的
`latency_summary.json` 应记录两个文档且顺序为 1、2。

## MinerU-Popo 组合入口

Popo 仍保留独立 native 组合入口，不能使用上述 `run_fixed`：

```bash
"$PY" src/benchmark/run_native_batch.py \
    --methods mineru_344_2605_popo \
    --gpus "$GPU" \
    --popo-vllm-python /data/wangsq/software/miniconda/envs/gdb_mineru/bin/python \
    --popo-gpu-memory-utilization 0.68 \
    --mineru-vllm-gpu-memory-utilization 0.20 \
    --popo-max-model-len 32768 \
    --run-id "${RUN_ID}_popo" \
    --input-root "$GDB_ROOT/data/pdf_tree" \
    --output-root-base "$GDB_ROOT/tmp/smoke/${RUN_ID}_popo/outputs" \
    --log-root "$GDB_ROOT/tmp/smoke/${RUN_ID}_popo/logs"
```

## 迁移后最小核对

```bash
"$PY" src/validation/check_method_versions.py \
    --json-report "$GDB_ROOT/tmp/migration/method_versions.json"
"$PY" src/validation/check_smoke_readiness.py \
    --json-report "$GDB_ROOT/tmp/migration/smoke_readiness.json"
PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 "$PY" -m compileall -q src
```

之后选择一个明确的 method ID，以 `run_fixed <METHOD_ID> 1` 做单卡 smoke，
再用 `validate_outputs.py` 检查该次输出。历史 `test_*.py` 不再作为迁移步骤。
