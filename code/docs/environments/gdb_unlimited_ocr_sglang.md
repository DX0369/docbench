# `gdb_unlimited_ocr_sglang`

## 用途和版本依据

该环境运行当前唯一正式 Unlimited-OCR 路线：官方定制 SGLang + 项目 SM86 补丁。
方法 ID 保持 `unlimited_ocr_sglang`，没有将旧 `unlimited_ocr` ID 重定向到新路线。
旧 Transformers adapter 和环境配方可从主代码 Git `5c2b4a5` 恢复；其 Conda 环境、
模型和 `env/locks_rtx3090_20260720/gdb_unlimited_ocr.*` 历史锁未删除。
移除 HF 运行入口不代表卸载 Transformers：SGLang 仍依赖下列官方 wheel 指定版本。

当前配方绑定：

- 官方源码 revision：`528fca4e2161e23231d05666a6d35155dcb1957e`；
- 官方定制 wheel：
  `third_party/methods/unlimited_ocr/wheel/sglang-0.0.0.dev11416+g92e8bb79e-py3-none-any.whl`；
- Python 3.12.13、uv 0.11.29；
- SGLang `0.0.0.dev11416+g92e8bb79e`；
- Torch 2.9.1+cu128、Transformers 5.3.0；
- FlashInfer 0.6.7.post3、`sglang-kernel` 0.4.1、FlashAttention-4 4.0.0b31；
- `kernels==0.11.7`、`PyMuPDF==1.27.2.2`。

Torch 2.9.1、Transformers 5.3.0、FlashInfer 0.6.7.post3 是上述官方 wheel
元数据的精确依赖，不是本项目任意选定的版本；CUDA 构件及其他包以实际环境快照核对。

官方 README 的说明文字仍写着 `kernels==0.9.0`，但同一段实际安装命令是
`kernels==0.11.7`；当前环境和 RTX 3090 smoke 验证的是后者。

## 删除并重建

以下命令使用 Miniconda 命名环境，并只为当前安装命令临时清除六个代理变量。
镜像由环境总览中的公共设置统一配置。

```bash
export GDB_ROOT=/data/wangsq/code/global_doc_bench
export SGLANG_ENV=/data/wangsq/software/miniconda/envs/gdb_unlimited_ocr_sglang
export SGLANG_PY="$SGLANG_ENV/bin/python"
export SGLANG_UV="$SGLANG_ENV/bin/uv"

conda create -n gdb_unlimited_ocr_sglang \
  python=3.12 pip setuptools wheel -y

"$SGLANG_PY" -m pip install "uv==0.11.29"

"$SGLANG_UV" pip install \
  --python "$SGLANG_PY" \
  "$GDB_ROOT/third_party/methods/unlimited_ocr/wheel/sglang-0.0.0.dev11416+g92e8bb79e-py3-none-any.whl" \
  "kernels==0.11.7" \
  "pymupdf==1.27.2.2"
```

若要重建已有环境，先确认没有 SGLang 服务正在运行，再执行：

```bash
conda env remove -n gdb_unlimited_ocr_sglang -y
```

然后从创建命令重新开始。不要在旧环境中叠加另一版官方 wheel。

## RTX 3090 SM86 补丁

官方定制 wheel 的三处 TVM-FFI JIT CUDA kernel 在 RTX 3090（SM86）上可以完成
加载，但实际 launch 会报 `CUDA error: invalid resource handle`：

- RoPE；
- KV-cache store；
- `clamp_position`。

安装完成后必须执行项目内可重复补丁：

```bash
"$GDB_ROOT/code/src/methods/unlimited_ocr/setup_sglang_sm86.py" \
  --python "$SGLANG_PY"
```

补丁将 CUDA RoPE 切换到 wheel 已提供并完成数值对照的
`sgl_kernel.rotary_embedding`，KV-cache store 使用现有 indexed-assignment
fallback，`clamp_position` 使用现有 PyTorch native 实现；HIP 路径保持不变。
脚本会校验精确原文且可重复执行，因此 wheel 变化时会显式失败，而不会静默修改未知代码。

### 吞吐结果的实现标识

当前环境实际启用了上述三处替换，应报告为 **官方定制 SGLang + 项目 RTX 3090
（SM86）兼容补丁**，不能称为未经修改的官方实现。补丁未修改模型权重、prompt
或生成策略，但改变了 kernel 执行路径，可能影响性能；现有数值对照和有界 smoke
不证明吞吐不变，也不证明整套推理在所有输入上完全等价。
目前没有量化补丁的性能影响，不能声称该结果代表官方实现的最快速度。
后续吞吐报告应在已有配置/结果说明中记录官方 wheel 版本、项目补丁脚本 revision
和上述三项替换；不把带补丁与不带补丁的结果混为同一执行配置。

## 环境验收

FlashInfer import 会创建 JIT 日志，因此显式放到项目 workspace：

```bash
export FLASHINFER_WORKSPACE_BASE="$GDB_ROOT/.cache/unlimited_ocr_sglang"
export TVM_FFI_CACHE_DIR="$FLASHINFER_WORKSPACE_BASE/tvm-ffi"
mkdir -p "$TVM_FFI_CACHE_DIR"

PYTHONNOUSERSITE=1 "$SGLANG_PY" -m pip check
PYTHONNOUSERSITE=1 "$SGLANG_PY" - <<'PY'
import fitz
import flashinfer
import sglang
import torch
import transformers

print("torch:", torch.__version__, "cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("sglang:", sglang.__version__)
print("transformers:", transformers.__version__)
print("flashinfer:", flashinfer.__version__)
print("pymupdf:", fitz.VersionBind)
PY
```

当前环境 `pip check` 为 `No broken requirements found`。

## 运行与验证入口

唯一正式入口为 `unlimited_ocr_sglang`，普通批量和吞吐均使用常驻客户端共享单服务。
请求路线现可显式选 `--unlimited-ocr-request-mode joint|page_pool`，共用该环境、
模型、官方prompt及SM86补丁，不需另装环境。fixed-P入口均用 `--pressure P`；
joint调整文档客户端数，page_pool调整跨文档页面任务窗口，保存配置时分开execution profile。
项目200 DPI/显存0.9区别于官方示例300 DPI/0.8；其余请求和服务参数、计时与质量warning
规则统一见[吞吐协议](../THROUGHPUT_TUNING_PROTOCOL.md#单模型服务hunyuanqianfandeepseekunlimited-ocr-sglang)。
不要将官方wheel加SM86补丁描述为未经修改的官方实现。

从 `code/` 目录执行；先按 [RUN_COMMANDS.md](../RUN_COMMANDS.md) 准备输入，并设置当轮获授权 GPU：

```bash
python src/validation/check_smoke_readiness.py --methods unlimited_ocr_sglang
python src/benchmark/run_service_batch.py \
  --methods unlimited_ocr_sglang --gpus "$GPU" \
  --input-root "$SMOKE_RUN/inputs" \
  --output-root-base "$SMOKE_RUN/outputs" \
  --log-root "$SMOKE_RUN/logs" --dry-run
```

实际GPU smoke去掉 `--dry-run`；运行前确认指定GPU和端口空闲。
已验证范围及内容限制见吞吐协议中的 Unlimited 小节，
最新P1/P2/P4结果仅在吞吐协议维护，不在安装配方重复实验历程。

迁移时按本配方重建，再用根目录 `env/locks_rtx3090_20260720/gdb_unlimited_ocr_sglang.*`
核对版本；环境锁不携带 SM86 site-packages 补丁，迁移时仍须重新应用。
迁移时按目标目录调整源码、模型、CUDA 和 FlashInfer 路径，并用现有锁文件核对依赖。
