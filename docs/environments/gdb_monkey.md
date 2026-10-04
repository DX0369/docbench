# `gdb_monkey`

更新时间：2026-10-03 12:53 +08:00

## 当前路线

两个显式 ID `monkeyocr_pro_1_2b`、`monkeyocr_pro_3b` 共用此环境和锁定源码。
项目只保留单常驻 `vllm_async` runtime：外层 worker=1，pressure 控制官方多文件分组内共享的区域在途数，
普通入口默认64；吞吐入口由显式P决定。官方MultiFileDataset按20页示例值固定分组，超长PDF独组；不复制模型或额外叠加区域池。
3B 使用登记的 compatibility_source，不冒称历史 A800 的精确旧 runner。

生成与处理配置沿用锁定官方 async；参数、运行机制与 P=1/2/4 结果统一见
[吞吐协议](../THROUGHPUT_TUNING_PROTOCOL.md#monkeyocr单runtime区域并发)和
[RUN_COMMANDS.md](../RUN_COMMANDS.md)。
README 的 async 建议针对服务并发，一般安装说明仍推荐 LMDeploy；不声称当前 backend 全局最快。
已移除项目 API/其他 backend 入口；整理前代码可从主代码 Git `d152a4a29da395de415c8fc644a1761c373ce4cb` 恢复。
第三方仓库和三份现用补丁保留，不通过卸载依赖来裁剪 backend。

## 版本与已知例外

当前隔离环境：Python3.10.20、Torch2.7.0+cu126、vLLM0.9.1、Transformers4.51.0、
PaddlePaddle GPU3.0.0、PaddleX3.3.0、Accelerate1.12.0、LangChain0.3.26。
NumPy<2，PyMuPDF1.24.14；具体全量版本以 `env/locks_rtx3090_20260720/gdb_monkey.*` 为准。

官方 `docs/install_cuda_pp.md` 的 Torch2.6 属于 LMDeploy/Transformers 分支；
vLLM0.9.1 元数据要求 Torch2.7.0，不混用不同 backend 配方。
当前隔离 `pip check` 仍有两条已记录的冲突：

- Paddle3.0 要求 NCCL2.25.1，Torch2.7 cu126 要求当前的2.26.2。
- Monkey 固定 Transformers4.51.0，vLLM0.9.1 要求≥4.51.1。

不为消除警告覆盖这些核心版本。GPU 验证边界见 smoke，不称环境无冲突。
FastAPI0.115.12／Starlette0.46.2／instrumentator7.1.0 是已验证兼容组合；
Accelerate 供官方 pipeline 的 `init_empty_weights` 使用。

当前主动省略 CuPy 与 Decord，属于项目裁剪而非官方完整 extras：
vLLM→Ray[cgraph]→CuPy，PaddleX[base]→Decord；不能凭 `pip check` 没报 extras 就称无依赖。
CuPy14 与本项目 NumPy<2 冲突，文档图像路线不使用 Decord 视频功能。
Transformers 还参与阅读顺序；OpenAI、qwen-vl-utils、boto3 有顶层导入。
Gradio、FastAPI 及 PaddleX extras 不仅按“当前没有调用 UI/API”就可安全卸载；
本轮不再精简真实环境包。

现有锁文件仅用于迁移后的版本核对；环境内 `psutil==7.2.2` 保留。
Conda 元数据的 packaging26.0 已被 pip25.0 覆盖，恢复时必须先 Conda 后 pip。

## 同平台恢复

先按 `code/config/methods.json` 恢复源码 revision 和两个版本的模型。
`magic_pdf==1.1.0` 是 MonkeyOCR 自身的 editable 安装，不是本项目应另装的旧 MinerU。
YAML 中的 `magic-pdf==1.1.0` 不足以恢复正确源码；freeze 的 `-e git+...` 只含 HEAD，
不含三份本地补丁。不要直接用同名 PyPI 包或未打补丁的新克隆替代。

以下为 Linux x86_64、同 CUDA wheel 路线的锁定恢复命令，仅用于新环境，不在当前环境覆盖执行。
先设置通用环境文档中的 `GDB_ROOT/GDB_METHOD_ROOT`，并确认 `conda` 可用；需要时再使用 `env/` 快照核对。
这次核对没有在全新服务器重建；镜像可用性、驱动及新机器 GPU smoke 仍须现场验证。

```bash
conda create -n gdb_monkey --file \
  "$GDB_ROOT/env/locks_rtx3090_20260720/gdb_monkey.explicit.txt" -y
conda activate gdb_monkey
MONKEY_PY="$CONDA_PREFIX/bin/python"
MONKEY_LOCK="$GDB_ROOT/env/locks_rtx3090_20260720/gdb_monkey.pip-freeze.txt"

(
  export PYTHONNOUSERSITE=1
  unset PIP_EXTRA_INDEX_URL

  # 特定 GPU 构建从原发布源获取，其余已锁定依赖走镜像。
  "$MONKEY_PY" -m pip install --no-deps \
    --index-url https://download.pytorch.org/whl/cu126 \
    'torch==2.7.0+cu126' 'torchaudio==2.7.0+cu126' 'torchvision==0.22.0+cu126'
  "$MONKEY_PY" -m pip install --no-deps \
    --index-url https://www.paddlepaddle.org.cn/packages/stable/cu126/ \
    'paddlepaddle-gpu==3.0.0'
  "$MONKEY_PY" -m pip install --no-deps -r \
    <(sed -E '/^(-e |torch==|torchaudio==|torchvision==|paddlepaddle-gpu==)/d' "$MONKEY_LOCK")
)
```

这里的 `--no-deps` 用于逐项恢复已核对快照，避免解析器重新选择不兼容组合，
不是省略依赖；不要将这套命令描述为一次无冲突的官方 resolver 安装。
之后应用下节补丁，再从该本地目录安装 editable：

```bash
PYTHONNOUSERSITE=1 "$MONKEY_PY" -m pip install --no-deps --no-build-isolation \
  -e "$GDB_METHOD_ROOT/monkeyocr"
```

## 源码补丁

恢复锁定 HEAD `70f6e2c5583a563796d7e0ec8a319d5bcfb86219` 后，顺序应用三份现用补丁：

1. `failure_propagation.patch`：只保留当前async路径的真实错误传播。
2. `region_concurrency.patch`：将官方固定信号量64暴露为正整数配置，不改变生成参数。
3. `tiny_image.patch`：原max_size=1600加载后，仅短边<28像素的区域调用官方HF路线
   使用的`qwen_vl_utils.fetch_image`；两版锁定processor均为patch_size14×merge_size2。
   原区域框、区域数量、正常尺寸图片及生成参数不变，小区域像素会变化，须视为项目预处理兼容补丁。
   async显式启用`load_image(raise_on_error=True)`，图片加载/缩放错误继续抛出，不替换dummy图。
   正规化前后尺寸写入worker日志，metrics登记`region_image_normalization`策略；
   `--skip-existing`拒绝缺失或不同策略的旧结果，不迁移或改写历史产物。
   不安装新包、不改变环境锁，当前已安装的qwen-vl-utils被复用。

已撤销API消息格式及未使用的Transformers/queue异常补丁，对应源码恢复原HEAD；
历史API配方从主代码Git `f2c0196`恢复。当前分组、文件级后处理与统一导出位于项目adapter，
不再修改官方分组代码；split_files启用的整页识别行为和旧单PDF路线并非完全等价。

```bash
# 仅用于干净的锁定 checkout；已应用时不要重复执行。
for patch in failure_propagation region_concurrency tiny_image; do
  git -C "$GDB_METHOD_ROOT/monkeyocr" apply --check \
    "$GDB_ROOT/code/src/methods/monkeyocr/patches/$patch.patch" || break
  git -C "$GDB_METHOD_ROOT/monkeyocr" apply \
    "$GDB_ROOT/code/src/methods/monkeyocr/patches/$patch.patch" || break
done
# 当前 checkout 的只读已应用检查：
for patch in failure_propagation region_concurrency tiny_image; do
  git -C "$GDB_METHOD_ROOT/monkeyocr" apply --reverse --check \
    "$GDB_ROOT/code/src/methods/monkeyocr/patches/$patch.patch" || break
done
```

项目 `async_runtime.py` 另负责同一事件循环跨文档复用与整轮引擎释放；
修复了每批 `asyncio.run()` 取消后台任务及退出 join 卡死，不新建模型或线程。
迁移必须同时恢复项目 adapter 与三份现用补丁，仅导入包成功不足以验收。

## 安装后最小核对与短 IPC 路径

```bash
PYTHONNOUSERSITE=1 "$MONKEY_PY" -m pip check
PYTHONNOUSERSITE=1 "$MONKEY_PY" -c \
  'import torch, transformers, vllm, paddle, paddlex, fitz, numpy; print(torch.__version__, torch.version.cuda, transformers.__version__, vllm.__version__, paddle.__version__, paddlex.__version__, fitz.VersionBind, numpy.__version__)'
```

`pip check` 预期只有上述两条已知冲突；之后执行独立临时目录的单页/连续文档 smoke，
检查输出、失败记录和自然退出，不把旧 GPU 结果转记为新环境验收。

每轮为 IPC 使用项目内独立短目录，同时设置 `TMPDIR` 和 `VLLM_RPC_BASE_PATH`。
例如选择一个尚不存在的短 run 名：

```bash
mkdir -p "$GDB_ROOT/tmp/runtime/mk01"
export TMPDIR="$GDB_ROOT/tmp/runtime/mk01"
export VLLM_RPC_BASE_PATH="$TMPDIR"
```

迁移后按实际根目录核对：目录路径＋`/`＋36字符 UUID 不得超过107字节。
进程退出后清理本轮目录，不删除其他运行的 socket/cache。
