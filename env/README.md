# 环境锁文件

`locks_rtx3090_20260720/` 保存当前 RTX 3090 环境的迁移核对材料。它不属于源码、模型或
安装脚本，不能单独保证跨服务器复现。

每个环境通常包含三类文件：

- `*.explicit.txt`：Conda 包及构建来源；
- `*.full.no-builds.yml`：Conda 与 pip 的完整状态参考；
- `*.pip-freeze.txt`：Python 可见的包版本和直接安装来源。

迁移时先按 [`code/docs/environments/README.md`](../code/docs/environments/README.md) 及对应
环境配方重建，再用锁文件核对版本、wheel 来源和已知例外。不要直接用完整 YAML 或 pip freeze
覆盖创建环境；其中可能包含本机路径、旧 CUDA 构件或 editable 来源。

源码 revision、模型目录和必要补丁分别由 `code/config/methods.json`、
`code/docs/MODELS.md` 及环境配方管理。锁文件不记录源码补丁、激活变量或 GPU smoke 结果。

锁目录名称是历史导出批次，不代表所有文件同日导出。依赖发生实际变化后才更新对应快照；
仅修改代码或文档时不重新导出。
