# `gdb_paddle`

## 用途与已验证版本

PP-StructureV3及PaddleOCR-VL 1.5／1.6的布局、预处理、后处理和客户端共用此环境。
VLM服务单独使用[gdb_paddle_vllm](gdb_paddle_vllm.md)；不合并两个CUDA依赖栈。
模型下载见[MODELS](../MODELS.md)，运行参数、计时和输出边界只在[吞吐协议](../THROUGHPUT_TUNING_PROTOCOL.md)维护。

- Python 3.10；Paddle GPU 3.2.1（CUDA 12.6 wheel）；PaddleX 3.7.2。
- PaddleOCR editable版本为 `3.8.0.dev9+g211989f04`，源码为
  `third_party/methods/paddleocr`，revision `211989f046cc1878460f9e65574690c00a127a1a`。
  两版VL和PP共用该源码，不额外克隆或安装一份历史SDK。
- 官方源码的 `doc-parser` extra要求 `paddlex[ocr,genai-client]>=3.7,<3.8`。
  3.7.2是本机已验证解析结果，不是把官方范围改成唯一支持版本。
- 当前无Torch/Transformers/vLLM；不要在此运行 `install_genai_server_deps vllm`。
- `psutil==7.2.2`必须装在环境内：早期aistudio-sdk曾借用用户目录的该包，现已修复。

## 重建与验收

先按[通用迁移步骤](README.md)设置GDB路径变量并恢复上述源码revision；
不要仅按YAML中的PaddleOCR版本号从PyPI安装。

```bash
git -C "$GDB_METHOD_ROOT/paddleocr" rev-parse HEAD

(
  unset PYTHONPATH PYTHONHOME PIP_EXTRA_INDEX_URL PIP_TARGET PIP_PREFIX PIP_USER
  export PYTHONNOUSERSITE=1
  conda create -n gdb_paddle python=3.10 pip setuptools wheel -y
  conda run -n gdb_paddle python -m pip --isolated install \
    --index-url https://www.paddlepaddle.org.cn/packages/stable/cu126/ \
    'paddlepaddle-gpu==3.2.1'
  conda run -n gdb_paddle python -m pip --isolated install \
    -e "$GDB_METHOD_ROOT/paddleocr[doc-parser]" 'paddlex==3.7.2' 'psutil==7.2.2'
  conda run -n gdb_paddle python -m pip --isolated check
  conda run -n gdb_paddle python -c \
    'import paddle, paddleocr, paddlex, psutil; print(paddle.__version__, paddleocr.__version__, paddlex.__version__, psutil.__file__)'
)
```

Paddle专用index只用于Paddle GPU wheel，其余依赖使用显式镜像直连。
重建后须分别验证PP与外部vLLM支持的VL pipeline；本轮仅整理配方，未重新安装。

迁移时按本配方重建，再用根目录 `env/locks_rtx3090_20260720/gdb_paddle.*` 核对版本。

## PP-StructureV3 接入边界

唯一profile为`pp_structurev3_page_prefetch`，无新增依赖或第三方源码补丁。
页面预取使用锁定PaddleX私有sampler，升级SDK需重新验证；主线ready前还会初始化
constructor延迟加载的表格方向和辅助OCR。模型预取见[MODELS](../MODELS.md#paddlexpp-structurev3paddleocr-vl-组件)，
参数、计时及测量见[吞吐协议](../THROUGHPUT_TUNING_PROTOCOL.md#native-multi-modelpp-structurev3marker)，
验收／原生内容限制见吞吐协议，不在环境配方重复维护。
