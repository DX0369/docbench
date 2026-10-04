# `gdb_mineru_303`

## 用途

复现原始 A800 benchmark 使用的 MinerU 3.0.3、MinerU2.5-2509-1.2B
和官方hybrid API tasks路线。当前只保留受管API；原始direct实现从Git `69d689a`恢复。
环境必须继续保留官方hybrid的VLM、pipeline及vLLM依赖。

## 固定版本

- 源码：`third_party/methods/mineru_303`
- revision：`ff2caf3a842a54b3d31979a2945003cf67858199`
- Python 3.10
- Torch 2.9.0 + CUDA 12.8 runtime
- Transformers 4.57.6
- vLLM 0.11.2
- mineru-vl-utils 0.2.8
- 当前锁定 `mineru-vl-utils==0.2.8`；旧 A800 组合不属于本环境。
- six 1.17.0
- albumentations 2.0.8
- albucore 0.0.24
- pypdfium2 5.10.1，配合下述图片坐标接口补丁；不是未经修改的上游源码。

上游 `mineru-3.0.3-released` tag 所在提交仍将包版本写为 3.0.2；紧随其后的
`ff2caf3a` 只修正版本号为 3.0.3，因此项目锁定后一个提交。

## 安装命令

```bash
export GDB_ROOT=/data/wangsq/code/global_doc_bench
export MINERU_303_SRC="$GDB_ROOT/third_party/methods/mineru_303"
export MINERU_303_PY=/data/wangsq/software/miniconda/envs/gdb_mineru_303/bin/python
export MINERU_303_UV=/data/wangsq/software/miniconda/envs/gdb_mineru_303/bin/uv

conda create -n gdb_mineru_303 python=3.10 pip setuptools wheel uv -y

"$MINERU_303_UV" pip install \
  --python "$MINERU_303_PY" \
  'torch==2.9.0' 'transformers==4.57.6' 'vllm==0.11.2'

"$MINERU_303_UV" pip install \
  --python "$MINERU_303_PY" \
  -e "$MINERU_303_SRC[vlm,pipeline,vllm]"

"$MINERU_303_UV" pip install \
  --python "$MINERU_303_PY" \
  'mineru-vl-utils==0.2.8' 'six==1.17.0' \
  'albumentations==2.0.8' 'albucore==0.0.24' 'pypdfium2==5.10.1'
```

`albumentations` 是 MinerU 公式识别代码的实际 import，但 3.0.3 的
`pipeline` extra 没有声明它；这里采用原 A800 环境导出的精确版本。

当前 3090 smoke 验证环境使用 `mineru-vl-utils==0.2.8`，因此安装配方显式
固定该版本。旧 A800 环境导出记录的是 0.1.22；如果需要逐包复原 A800 环境，
应建立另一个环境锁定 0.1.22 并重新 smoke，不能把它与当前已验证环境混称。

## 项目 API 预加载

受管 API 通过项目 wrapper 提前加载 `ch/en` 辅助 OCR 和适用的标题 OCR，复用同一 VLM。
这是启动生命周期调整，不改变按文档选择语言的规则；不新增依赖、不改官方源码或环境锁。
迁移需恢复本项目代码，单独运行官方 `mineru-api` 不会启用这套预加载。
范围见[吞吐协议](../THROUGHPUT_TUNING_PROTOCOL.md#mineru-api-预加载)。

## PDFium 图片坐标接口补丁

旧源码调用 `get_pos()`，当前锁定的
`pypdfium2==5.10.1`仅提供新接口 `get_bounds()`；官方依赖声明 `pypdfium2>=4.30.0`
未限制上界，pip check不能发现此功能性不兼容。官方catch会直接返回ocr，改变解析分支。

经用户授权，仅将 `mineru/utils/pdf_classify.py` 的这一处调用改成 `get_bounds()`。
旧4.30.0的get_pos与当前get_bounds均使用 `FPDFPageObj_GetBounds`，返回相同的
left/bottom/right/top；未改变面积算法、分类阈值、异常策略、模型或生成参数。
补丁只用于 `mineru_303` 锁定HEAD，**不应用到新版MinerU**，也不改用legacy或强制txt。

迁移恢复源码及环境后，显式应用并检查（若已应用，只运行最后的反向检查）：

```bash
git -C "$MINERU_303_SRC" apply --check \
  "$GDB_ROOT/code/src/methods/mineru/patches/pdfium_image_bounds.patch"
git -C "$MINERU_303_SRC" apply \
  "$GDB_ROOT/code/src/methods/mineru/patches/pdfium_image_bounds.patch"
git -C "$MINERU_303_SRC" apply --reverse --check \
  "$GDB_ROOT/code/src/methods/mineru/patches/pdfium_image_bounds.patch"
```

如需恢复未补丁源码，使用同一patch的 `git apply --reverse`，不要删除记录或覆盖其他修改。
实际环境包和三份锁未变，安装命令中的pypdfium2精确版本来自现有锁；HEAD不编码本地补丁。
只恢复环境锁和HEAD、不应用补丁，仍会复现原错误。

补丁可从锁定 HEAD 精确重放；CPU 回归命令如下：

```bash
测试源已在定版前验证后删除；迁移时按 [运行命令](../RUN_COMMANDS.md) 执行源码、
模型和环境审计，再运行 MinerU 的固定 P smoke。
```

GPU 运行命令见 [RUN_COMMANDS.md](../RUN_COMMANDS.md)。

迁移时按本页配方重建，再用根目录 `env/locks_rtx3090_20260720/gdb_mineru_303.*`
核对版本；源码仍须 checkout 到 `ff2caf3a842a54b3d31979a2945003cf67858199`。

## 验收

```bash
"$MINERU_303_PY" -m pip check

"$MINERU_303_PY" - <<'PY'
import torch
import transformers
import vllm
from mineru.version import __version__ as mineru_version

print("mineru:", mineru_version)
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("transformers:", transformers.__version__)
print("vllm:", vllm.__version__)
PY
```

预期版本为 MinerU 3.0.3、Torch 2.9.0、Transformers 4.57.6 和 vLLM 0.11.2，
`mineru-vl-utils` 为 0.2.8，且 `pip check` 无 broken requirements。

现有锁文件已完成用户目录隔离核对，位于项目根 `env/`。
