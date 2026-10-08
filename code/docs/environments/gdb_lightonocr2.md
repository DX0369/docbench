# LightOnOCR-2-1B：复用 `gdb_glmocr`

## 环境结论

LightOnOCR-2-1B 当前不创建独立 Conda 环境。正式 adapter 的服务和 worker
都复用已经验证的 `gdb_glmocr`：

- Python 3.10；
- Torch 2.11.0+cu130；
- Transformers 5.14.1；
- vLLM 0.20.0；
- Pillow；
- pypdfium2 5.12.1。

这些是项目当前复用环境的实际版本，不是 LightOnOCR 官方锁定的完整依赖版本。
官方推理参数、PDF 预处理和 benchmark 约定统一记录在
[`THROUGHPUT_TUNING_PROTOCOL.md`](../THROUGHPUT_TUNING_PROTOCOL.md)，
可复制命令统一记录在 [`RUN_COMMANDS.md`](../RUN_COMMANDS.md)。

## 安装和检查

先按照 [`gdb_glmocr.md`](gdb_glmocr.md) 创建并检查 `gdb_glmocr`。
LightOnOCR-2 不在该环境中额外安装或升级包。

迁移或排障时可执行最小 import 检查：

```bash
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY
unset http_proxy https_proxy all_proxy
export PYTHONNOUSERSITE=1

conda run -n gdb_glmocr python -c '
from importlib.metadata import version
from transformers import LightOnOcrForConditionalGeneration, LightOnOcrProcessor
import torch, vllm
print(torch.__version__)
print(vllm.__version__)
print(version("pypdfium2"))
print(LightOnOcrForConditionalGeneration.__name__)
print(LightOnOcrProcessor.__name__)
'
```

当前共享环境的 `pip check` 会报告已知冲突：

```text
xgrammar 0.2.4 requires transformers<5,
but transformers 5.14.1 is installed.
```

这是 `gdb_glmocr` 原有的环境冲突，不是 LightOnOCR-2 新增的依赖。
不要为了 LightOnOCR-2 在共享环境中升级或降级 Transformers、xgrammar 或 vLLM。
如果未来需要完全无冲突的环境，再单独创建 `gdb_lightonocr2`。

## 环境锁和验收边界

在独立环境创建前，LightOnOCR-2 沿用 `gdb_glmocr` 的环境锁，不新增重复锁文件。
import 成功只说明 Python 依赖可见；真实模型服务是否支持该架构，必须通过 GPU smoke
确认。smoke 和正式实验状态统一记录在根目录 `PROJECT_STATE.md`，吞吐调参按吞吐协议单独进行。
