# `gdb_youtu`

更新时间：2026-10-03 13:01 +08:00

## 唯一运行路线与版本来源

官方 HF/native＋BF16＋FlashAttention2＋专用并行解码器；一个常驻模型，P仅调整文字
query分组，普通默认5。方向校正关闭；模型就绪后计时、不预热。项目不再提供
API/service、vLLM或SGLang入口；旧项目API实现可从Git `d1e6c92`恢复。

官方 `youtu_hf_parser/requirements.txt` 精确约束 Python环境中的 Torch2.6.0、
TorchVision0.21.0、Transformers4.51.3、NumPy2.0.2、PyMuPDF1.26.5、
Accelerate1.12.0、OpenCV4.12.0.88、Numba0.60.0；Python使用3.10。
官方CUDA分支选择FA2，项目已验证的ABI wheel为FlashAttention2.7.4.post1。
psutil7.2.2、Pillow12.3.0属于当前已测锁定版本，不称官方精确要求：
utils的requirements写Pillow12.1.0，但实际包元数据为Pillow>=8.0.0。

## 安装

先按 `code/config/methods.json` 恢复锁定源码。官方requirements含远程Git依赖，
pyproject又引用GitHub utils；因此过滤Git行并使用本地editable安装，parser不重新解析依赖。
下列命令沿用[公共配方变量](README.md)，清除代理、PyPI包直连华为镜像：

```bash
export PYTHONNOUSERSITE=1

conda create -n gdb_youtu python=3.10 pip setuptools wheel -y
mkdir -p "$GDB_ROOT/tmp"
grep -v '^git+' "$GDB_METHOD_ROOT/youtu/youtu_hf_parser/requirements.txt" \
  > "$GDB_ROOT/tmp/gdb_youtu_requirements.txt"
conda run -n gdb_youtu python -m pip install \
  -r "$GDB_ROOT/tmp/gdb_youtu_requirements.txt" \
  'psutil==7.2.2' 'Pillow==12.3.0'
conda run -n gdb_youtu python -m pip install \
  -e "$GDB_METHOD_ROOT/youtu/youtu_parsing_utils"
conda run -n gdb_youtu python -m pip install --no-deps \
  -e "$GDB_METHOD_ROOT/youtu/youtu_hf_parser"
conda run -n gdb_youtu python -m pip install \
  --no-cache-dir \
  "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1%2Bcu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
rm "$GDB_ROOT/tmp/gdb_youtu_requirements.txt"
```

临时清单只在全部安装成功后删除。FlashAttention使用已验证wheel，不现场编译；
该GitHub直链不经PyPI镜像。完整传递依赖以现有环境锁核对，不保证未来重新解析范围依赖
仍得到相同版本；复现差异时先检查锁，不擅自升级核心栈。

## 必需补丁与隔离检查

在新的干净checkout应用既有补丁；已应用时只做reverse-check，不重复修改：

```bash
git -C "$GDB_METHOD_ROOT/youtu" apply --check \
  "$GDB_ROOT/code/src/methods/youtu/patches/failure_propagation.patch"
git -C "$GDB_METHOD_ROOT/youtu" apply \
  "$GDB_ROOT/code/src/methods/youtu/patches/failure_propagation.patch"
git -C "$GDB_METHOD_ROOT/youtu" apply --reverse --check \
  "$GDB_ROOT/code/src/methods/youtu/patches/failure_propagation.patch"

PYTHONNOUSERSITE=1 conda run -n gdb_youtu python -m pip check
PYTHONNOUSERSITE=1 conda run -n gdb_youtu python -c \
  'import youtu_hf_parser, youtu_parsing_utils, flash_attn, psutil; print("native imports ok")'
```

只保留当前HF/native异常传播补丁；退役client补丁已撤销，对应官方client源码恢复HEAD。
不删除官方API文件，不改变当前识别算法；旧补丁集由主代码Git `f2c0196`恢复。
editable安装复用该源码，迁移必须恢复**精确checkout＋补丁＋环境锁**。
已完成用户目录隔离核对。

迁移时按本配方重建，再用根目录 `env/locks_rtx3090_20260720/gdb_youtu.*` 核对版本。
