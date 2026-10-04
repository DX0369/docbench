# `gdb_mineru`

## 用途

运行锁定的MinerU 3.4.4/2605受管hybrid API；当前不提供direct/router/其他backend入口。
官方hybrid内部仍需要VLM、pipeline及vLLM依赖，不能因撤下direct而卸载这些包。

2026-10-03 14:05 +08:00：现有环境另通过Popo Qwen3-VL的独立vLLM服务及文本/带图请求smoke，
无需补包；MinerU0.20＋Popo0.68双服务同卡也通过GPU4两文档/5页smoke。
固定20文档P=1/2/4仍有公式组件OOM、上下文超限及建树遗漏，不能视为稳定吞吐配置；
结果见[吞吐协议](../THROUGHPUT_TUNING_PROTOCOL.md#mineru-popo-共享服务计时)，环境锁不变。
复用解释器的命令与边界见[Popo环境配方](gdb_mineru_popo.md#popo-vllm服务复用gdb_mineru)。

2026-09-30 13:54 +08:00 核对：锁定的 `mineru-3.4.4-released` /
`0dfc9460cd9ab693b9af60ae3fbffd7bc111b062` 内部 `mineru/version.py` 仍为 3.4.3。
这是官方 tag 元数据差异；项目预载校验同时核对实际源码路径与精确 HEAD，不能据此换装任意
3.4.3 包。历史Popo常驻同步路径在GPU9验证过（旧入口现从Git c533601恢复）；受管API在2026-10-02 GPU2上
完成4文档／15页smoke（P=2、显存0.55、官方编译开启），两条路线仍分开记录。

2026-10-03 13:02 +08:00：修正项目受管服务身份检查，`/ready`分别报告经上述核验的
`release_version=3.4.4`与实际`runtime_version=3.4.3`；启动器核对前者，不改写后者，
也不放行错误源码路径/HEAD。无需重装、改上游version.py或更新环境锁。

## 关键约束

- Python `>=3.10,<3.14`；本项目统一选 3.10。
- Torch `>=2.6,<3`。
- Transformers `>=4.57.3,<5`。
- vLLM `>=0.10.1.1,<0.22`。
- 项目已验证版本：Torch 2.9.0、Transformers 4.57.6、vLLM 0.11.2；最初参考旧环境选定，满足当前官方范围。验证范围见环境总表及本配方记录，不表示所有入口都已通过。
- 不额外安装 LMDeploy、Gradio 或 S3 支持；benchmark 只安装 `vlm,pipeline,vllm` extras。

## 安装命令

```bash
conda create -n gdb_mineru python=3.10 pip setuptools wheel -y

conda run -n gdb_mineru python -m pip install \
  'vllm==0.11.2' 'transformers==4.57.6'
conda run -n gdb_mineru python -m pip install \
  -e "$GDB_METHOD_ROOT/mineru[vlm,pipeline,vllm]"
conda run -n gdb_mineru python -m pip install 'six==1.17.0'
```

这里故意先固定 vLLM/Transformers，再安装 MinerU 源码。如果 resolver 想替换锚点，先查看冲突原因，不使用 `--no-deps` 强行压过官方范围。

## 首轮验收

```bash
conda run -n gdb_mineru python -m pip check
conda run -n gdb_mineru python -c \
  'import torch, transformers, vllm, mineru; print(torch.__version__, torch.version.cuda, transformers.__version__, vllm.__version__)'
conda run -n gdb_mineru mineru --help
conda run -n gdb_mineru mineru-api --help
```

如果新 MinerU 与 vLLM 0.11.2 的 API smoke 失败，再在官方允许范围内升级 vLLM，而不是直接跟随最新 `<0.22`。

当前 3.4.x API 后端名为 `hybrid-engine`，旧名称
`hybrid-vllm-async-engine` 会被 API 拒绝。pipeline 的 PaddleOCR2Pytorch
兼容代码还会导入 `six`，但当前 extra 未声明它，因此配方显式补装 1.17.0。

## 项目 API 预加载

受管 `mineru_344_2605` 的 hybrid-engine API 通过项目 wrapper 预载 VLM 及共享辅助模型，
不额外安装包或应用第三方源码补丁。迁移时恢复本项目代码和既定源码/权重即可；
直接使用官方 `mineru-api` 不包含项目补充的 hybrid 预加载。
范围与计时见[吞吐协议](../THROUGHPUT_TUNING_PROTOCOL.md#mineru-api-预加载)，
运行命令见 [RUN_COMMANDS.md](../RUN_COMMANDS.md)。
本轮未安装/卸载包；隔离pip check及三份锁核对通过，环境锁不变。

## 隔离环境审计（2026-09-19 00:32 +08:00）

MinerU + Popo 新组合 adapter 禁用 user site 后暴露了现有环境的隐式依赖：
`psutil==7.2.2` 与 `nvidia-ml-py==13.595.45` 实际来自
`~/.local/lib/python3.10/site-packages`，不在 `gdb_mineru` 内。普通 import 可以成功，
但 `PYTHONNOUSERSITE=1 python -m pip check` 报 vLLM/Accelerate 缺 psutil，FlashInfer
缺 nvidia-ml-py。因此不能仅凭此前普通 pip check 断言环境已自包含。

`psutil` 和 `nvidia-ml-py` 已补入环境；保持 `PYTHONNOUSERSITE=1`，避免依赖用户目录。
现有锁文件仅用于迁移后的版本核对。

迁移时在原安装流程后执行：

```bash
env -u PYTHONPATH -u PYTHONHOME PYTHONNOUSERSITE=1 \
  "$(dirname "$(command -v conda)")/../envs/gdb_mineru/bin/python" -m pip install \
  --no-deps 'psutil==7.2.2' 'nvidia-ml-py==13.595.45'

PYTHONNOUSERSITE=1 "$(dirname "$(command -v conda)")/../envs/gdb_mineru/bin/python" -m pip check
```

环境路径由当前 `conda` 安装位置决定；当前为
`/data/wangsq/software/miniconda`。这里 `--no-deps` 仅用于补入已确认版本的两个小包，
不适用于忽略其他环境冲突。
