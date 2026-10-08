# Xiaomi-OCR-0：复用 `gdb_glmocr`

## 环境与依赖

服务与 CPU worker 共用 `gdb_glmocr`，不创建独立 `gdb_xiaomi_ocr` 环境。
当前共享栈为 Python 3.10、Torch 2.11.0+cu130、Transformers 5.14.1、vLLM 0.20.0；
这些是项目选定版本，不是 Xiaomi 官方精确锁定版本。

依赖来源分为两部分：

- 官方 [requirements.txt](../../../third_party/methods/Xiaomi-OCR-0/requirements.txt)
  列出 OpenAI SDK、Pillow、pypdfium2、NumPy、tqdm 和 wordfreq；安装说明见
  [INSTALL.md](../../../third_party/methods/Xiaomi-OCR-0/INSTALL.md)。
- 当前 adapter 的服务使用 vLLM；worker 直接引用官方 prompt、图像编码及后处理，
  使用 OpenAI SDK、Pillow、NumPy。PDF 渲染复用项目公共 Poppler 工具
  `pdfinfo`/`pdftoppm`，不调用官方 PDFium 渲染入口。

## 环境恢复

按 [gdb_glmocr.md 的恢复流程](gdb_glmocr.md#恢复已验证环境) 建立共享环境，
安装的镜像与代理设置沿用 [通用环境说明](README.md)。源码目录随项目迁移，
模型准备见 [MODELS.md](../MODELS.md#xiaomi-ocr-0)。

复用以下现有快照，不新增 Xiaomi 专用锁：

- [gdb_glmocr.explicit.txt](../../../env/locks_rtx3090_20260720/gdb_glmocr.explicit.txt)：Conda 包与构建来源。
- [gdb_glmocr.pip-freeze.txt](../../../env/locks_rtx3090_20260720/gdb_glmocr.pip-freeze.txt)：Python 包版本。
- [gdb_glmocr.full.no-builds.yml](../../../env/locks_rtx3090_20260720/gdb_glmocr.full.no-builds.yml)：完整状态参考。

## 迁移后的检查命令

```bash
export GDB_ROOT=/data/wangsq/code/global_doc_bench
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1

# 隔离依赖与官方 helper import，不加载模型。
conda run -n gdb_glmocr python -c '
import os, site, sys
import openai, PIL, numpy
assert not site.ENABLE_USER_SITE
sys.path.insert(0, os.path.join(os.environ["GDB_ROOT"], "third_party/methods/Xiaomi-OCR-0"))
from pipeline.e2e_img2md import PROMPTS, encode_image_base64
from postprocess.document import finalize_document
print("Xiaomi dependencies and official helpers imported")
'

conda run -n gdb_glmocr python \
  "$GDB_ROOT/code/src/methods/xiaomi_ocr/vllm.py" --help
command -v pdfinfo
command -v pdftoppm

conda run -n gdb_glmocr python \
  "$GDB_ROOT/code/src/validation/check_smoke_readiness.py" \
  --methods xiaomi_ocr_0

# 只生成命令和状态，不启动 GPU 服务；临时产物留在 tmp。
CHECK_ROOT="$GDB_ROOT/tmp/xiaomi_env_check/$(date +%Y%m%d_%H%M%S)"
conda run -n gdb_glmocr python \
  "$GDB_ROOT/code/src/benchmark/run_single_gpu.py" \
  --method xiaomi_ocr_0 --gpu-id 0 --pressure 1 --dry-run \
  --pdf-list-file "$GDB_ROOT/code/config/throughput_manifests/rtx3090_20.json" \
  --run-name xiaomi_env_check \
  --output-root-base "$CHECK_ROOT/outputs" \
  --log-root "$CHECK_ROOT/logs"
```

核对完成后清理该次 `CHECK_ROOT`；若检查失败，先保留对应诊断。

参数来源和计时规则见 [吞吐协议](../THROUGHPUT_TUNING_PROTOCOL.md)，
固定 P、搜索和逐文档延迟命令见 [运行命令](../RUN_COMMANDS.md)。
