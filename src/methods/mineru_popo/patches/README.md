# MinerU-Popo 最终修复

基准：`opendatalab/MinerU-Popo@97d560172361e772b3d079ce9eac35796b998782`。
这是锁定官方源码加本地/社区修复，不是未经修改的官方版本。升级源码前应检查上游是否已包含修复。

| 补丁 | 来源与作用 |
|---|---|
| `same_page.patch` | 项目修复：非空同页候选生成一个完整分块，避免零模型调用；空候选与原多页分块逻辑不变 |
| `pr12.patch` | [社区 PR #12](https://github.com/opendatalab/MinerU-Popo/pull/12)，abinch，head `1d38614e39d2e7f9f68cb023bd0f89542aabf555`；传递图片路径与图注 |
| `pr15.patch` | [社区 PR #15](https://github.com/opendatalab/MinerU-Popo/pull/15)，0mao0，head `9a8b30cc9e2de96a56a47e19f4b74deb33f3a1b7`；读取嵌套表格 span 中的已有 HTML |
| `resident_init.patch` | 项目常驻入口：从原 Transformers 惰性加载函数提取可独立调用的初始化，复用相同模型和 processor；不改变 BF16、device_map、prompt 或生成参数 |
| `failure_propagation.patch` | 当前共享vLLM入口要求：表格分支捕获异常后重新抛出，避免请求/解析/写盘失败变成成功文档 |

社区补丁保持原始 diff、作者来源及 MIT 许可。图注是附加元数据，不会自动成为独立关联候选；
表格补丁不重新识别或合并表格。图片路径仍相对原始 MinerU 输出，单独搬走树 JSON 不保证图片可访问。
三个补丁均不改变模型、prompt 或生成参数。
当前入口要求前三项及 `failure_propagation.patch`。`resident_init.patch`保留用于解释
本机已应用的历史TF初始化改动；新API客户端不调用它，不要求新机器安装TF常驻路径。

## 应用与检查

从项目根目录执行，`GDB_ROOT` 指向该目录：

```bash
export GDB_ROOT="$(pwd)"
REPO="$GDB_ROOT/third_party/methods/MinerU-Popo"
PATCHES="$GDB_ROOT/code/src/methods/mineru_popo/patches"
git -C "$REPO" apply --check "$PATCHES/same_page.patch" "$PATCHES/pr12.patch" "$PATCHES/pr15.patch" "$PATCHES/failure_propagation.patch"
git -C "$REPO" apply "$PATCHES/same_page.patch" "$PATCHES/pr12.patch" "$PATCHES/pr15.patch" "$PATCHES/failure_propagation.patch"
```

已安装环境不要重复应用。可用以下只读命令确认当前四项修复均存在；adapter也做相同检查：

```bash
git -C "$REPO" apply --reverse --check "$PATCHES/same_page.patch" "$PATCHES/pr12.patch" "$PATCHES/pr15.patch" "$PATCHES/failure_propagation.patch"
```

只有明确需要撤销时，才执行：

```bash
git -C "$REPO" apply --reverse --check "$PATCHES/same_page.patch" "$PATCHES/pr12.patch" "$PATCHES/pr15.patch" "$PATCHES/failure_propagation.patch"
git -C "$REPO" apply --reverse "$PATCHES/same_page.patch" "$PATCHES/pr12.patch" "$PATCHES/pr15.patch" "$PATCHES/failure_propagation.patch"
```

检查失败应停止，区分源码版本变化、部分应用和其他本地修改，不强制套用。

仅恢复Git `c533601`的历史Transformers常驻路径时，另需以下初始化补丁：

```bash
git -C "$REPO" apply --check "$PATCHES/resident_init.patch"
git -C "$REPO" apply "$PATCHES/resident_init.patch"
# 已应用时只检查，不重复应用：
git -C "$REPO" apply --reverse --check "$PATCHES/resident_init.patch"
# 只有明确需要撤销时：
# git -C "$REPO" apply --reverse "$PATCHES/resident_init.patch"
```

当前 `stages.py` 将官方 `model_utils` 与 `inference` 两处生成绑定接到同一常驻API客户端，
不加载HF模型。记录原异常/traceback，文档结束前检查，失败不建树；不新增自动重试。
新补丁还覆盖官方表格catch内的解析/写盘异常。正常四任务顺序、prompt、JPEG与建树不改。

## 验证范围

2026-09-19 的历史 CPU/GPU 验证结论见 [环境配方](../../../../docs/environments/gdb_mineru_popo.md)
和 [运行命令](../../../../docs/RUN_COMMANDS.md)。历史一次性脚本和输出已删除。
核心 adapter 测试已在定版前运行并删除。迁移时使用
[运行命令](../../../../docs/RUN_COMMANDS.md) 中的 Popo 组合入口，并用
`validate_outputs.py` 检查导出与失败边界；这不代表模型预测、跨页表格合并或吞吐性能。
