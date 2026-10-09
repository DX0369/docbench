# 源码与方法版本管理

本文档只说明第三方源码的身份、同步和升级规则。唯一的机器可读版本清单是
[`config/methods.json`](../config/methods.json)；环境安装、模型下载、smoke 和吞吐规则分别见：

- [环境配方](environments/README.md)
- [模型下载](MODELS.md)
- [吞吐协议](THROUGHPUT_TUNING_PROTOCOL.md)

第三方源码位于 `third_party/methods/`，每个目录保留自己的 Git 仓库。旧服务器或 A800
源码只作为历史证据，不替换当前锁定 checkout。

## 源码身份规则

- `repositories[].revision` 是实际运行源码的完整 commit；不要只记录分支或短 hash。
- `variants[].source_repository_id` 指向源码仓库，运行 revision 从仓库清单解析，不在方法条目中重复维护。
- `exact_checkout` 表示模型和源码来自同一历史版本。
- `compatibility_source` 表示当前源码兼容运行历史模型；必须同时记录
  `historical_source_revision`，不能冒充历史源码完全复原。
- `reference_checkout` 只表示参考源码，`model_only` 只表示模型来源，二者都不能作为运行源码。
- 不同源码 revision 保留独立仓库条目；不同模型版本保留独立模型目录。
- 本地兼容补丁单独记录并按环境配方应用。HEAD 锁定不代表工作树没有补丁或 dirty 文件。
- `active` 只表示入口已登记；`validation_status` 必须反映实际验证层级，模型存在不等于 GPU smoke 通过。
- 含糊的历史 runnable ID 只能读取历史结果，不能用于新实验调度。

## 同步与审计

以下命令从 `code/` 目录执行，只需要 Python 3.10+ 和 Git：

```bash
python src/method_repositories.py validate
python src/method_repositories.py status
python src/method_repositories.py sync
```

含义如下：

- `validate`：离线检查清单、仓库引用和字段一致性；
- `status`：核对本地仓库 HEAD、origin 和工作树状态；
- `sync`：clone/fetch 后 checkout 清单中的精确 commit，遇到 dirty 或非预期仓库会拒绝操作。

需要查看上游是否出现新 revision 时，再执行：

```bash
python src/method_repositories.py check-upstream
```

该命令只读取远程 refs，不修改清单或 checkout。指定仓库时，`--method` 使用
`repositories` 中的仓库 ID，而不是 benchmark runnable ID：

```bash
python src/method_repositories.py --method mineru sync
```

完整绑定审计：

```bash
python src/validation/check_method_versions.py \
  --json-report ../tmp/runtime/method_version_audit.json
```

它核对方法 ID、模型、源码 HEAD、兼容源码历史 revision、adapter、环境和 active 注册。
审计通过不代表模型推理成功；实际推理应使用 `RUN_COMMANDS.md` 中的正式入口，
遵循吞吐协议，并将进展与简要结论记录在 `PROJECT_STATE.md`。

## 升级流程

任何第三方源码升级都按以下顺序进行：

1. 阅读上游 release、README 和依赖变更，确认是否改变 import、配置、backend、prompt 或输出 schema；
2. 更新 `methods.json` 中的完整 revision、release/ref 和版本关系；
3. 检查模型、tokenizer/processor、环境和 adapter 是否仍兼容；
4. 需要不同依赖时建立独立环境，不覆盖已验证环境；
5. 执行 import/CLI、短文档和复杂文档 smoke，确认输出语义；
6. 更新验证状态和必要的 adapter/config；
7. 提交源码映射、代码修改和验证说明，不把一个版本的质量与另一个版本的吞吐混在一起。

源码同步本身不证明兼容，也不自动应用本项目补丁。环境配方维护必要补丁和安装要求，
实验进展、smoke 结果与已知限制统一维护于 [PROJECT_STATE.md](../../PROJECT_STATE.md)。

## 必须保留的版本例外

### MinerU

- `mineru_303_2509` 使用 MinerU 3.0.3、`MinerU2.5-2509-1.2B` 和独立环境；
- 当前 3.0.3 锁定 `ff2caf3a842a54b3d31979a2945003cf67858199`。官方
  `mineru-3.0.3-released` tag 的 `version.py` 仍写作 3.0.2，不能只依据 tag 名称；
- 旧版需要 `pdfium_image_bounds.patch`，新版 3.4.x 不应用该补丁；
- 旧的含糊 `mineru` runnable ID 不进入正式调度。
- `mineru_4011_standard` 独立使用官方 PyPI 4.0.11 wheel、V1 API、新小模型包和原 2605 VLM。
  `third_party/methods/mineru4` 是 release tag 的参考快照，不用于 runtime import；
  该 tag 的版本文件仍写 4.0.10，与安装包比对仅版本文件不同，明确标为 `reference_checkout`。
  不替换旧两版 API 或 Popo 绑定。

### HunyuanOCR

1.0 与 1.5 是官方明确版本。1.0 当前使用新版仓库中的兼容路径和模型 remote code，标记为
`compatibility_source`；历史 `ce4a967` 只记录来源，不代表当前 checkout。

### MonkeyOCR

3B 和 1.2B 是两个 checkpoint，不改写成 v1.0/v1.5。官方源码同时支持两者，因此可以共用
源码 checkout 和环境；3B 的旧源码 commit 只保留为 provenance，当前运行以清单中的兼容 revision 为准。

### PaddleOCR-VL

1.5 与 1.6 使用独立 checkpoint 和显式 profile。即使上游声明架构兼容，也不能覆盖模型目录或
把两个 runnable ID 合并。

### Qianfan-OCR

当前 `qianfan_vl` 是参考源码仓库；Qianfan-OCR 官方链接和独立公开源码状态需单独记录。
现有推理依赖本地模型 remote code，不把参考仓库冒充 OCR 官方源码版本。

### Unlimited-OCR

SM86 kernel 兼容补丁作用于安装环境，不在 Git checkout 中。迁移时必须同时恢复官方定制 wheel、
补丁和环境配方，不能只 checkout 源码就宣称复现完成。

## Dirty 工作树

运行 Python 或 editable 安装可能产生 `__pycache__`、`*.egg-info` 等副产物。它们不等于源码
版本升级，但正式冻结或提交前应确认没有人工修改，再清理或单独记录；不要把运行副产物提交到主代码仓库。
