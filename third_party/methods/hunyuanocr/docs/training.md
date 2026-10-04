# Training Guide

This document describes the three training modes supported by this repo:

1. **SFT the base HunyuanOCR model** (full end-to-end fine-tuning)
2. **DFlash draft — train from scratch** (large data, e.g. ~1M packs)
3. **DFlash draft — continue-finetune** (small domain data, e.g. ~10k packs)

---

## 1. SFT the Base HunyuanOCR Model

Full end-to-end supervised fine-tuning of HunyuanOCR: vision encoder + MLP + LLM.

### Entry

`train/train_hunyuan.py` — invoked by `scripts/sft_base.sh`.

### Quick run

```bash
# Single-node 8-GPU
MODEL_PATH=/path/to/HunyuanOCR \
TRAIN_DATA=./data/parsing_packed_20480.jsonl \
    bash scripts/sft_base.sh
```

### Key hyperparameters

| Env var          | Default | Meaning                     |
| ---------------- | ------: | --------------------------- |
| `LR`             |  `2e-5` | Learning rate               |
| `EPOCHS`         |     `5` | Number of training epochs   |
| `BATCH_SIZE`     |     `1` | Per-device train batch size |
| `GRAD_ACCUM`     |     `1` | Gradient accumulation steps |
| `SAVE_STEPS`     |   `200` | Checkpoint every N steps    |
| `NPROC_PER_NODE` |     `8` | GPUs per node               |

### Multi-node

Set standard PyTorch distributed env vars:

```bash
# Node 0 (master)
NNODES=4 NODE_RANK=0 NPROC_PER_NODE=8 \
MASTER_ADDR=10.0.0.1 MASTER_PORT=29500 \
    bash scripts/sft_base.sh

# Node 1
NNODES=4 NODE_RANK=1 NPROC_PER_NODE=8 \
MASTER_ADDR=10.0.0.1 MASTER_PORT=29500 \
    bash scripts/sft_base.sh

# ... etc for node 2, 3
```

Adjust `NCCL_SOCKET_IFNAME` / `NCCL_IB_HCA` in `scripts/env_common.sh` to match your cluster's NIC layout.

---

## 2. Train a DFlash Draft From Scratch

Train a small MTP-style draft model from randomly initialized weights.
**Recommended when you have plenty of data** (~1M packs).

### Entry

`train/train_draft.py` — invoked by `scripts/sft_dflash.sh`.

### Quick run

```bash
MODEL_PATH=/path/to/HunyuanOCR \
TRAIN_DATA=./data/parsing_packed_20480.jsonl \
    bash scripts/sft_dflash.sh
```

### Key hyperparameters (from-scratch profile)

| Env var            | Default | Meaning                           |
| ------------------ | ------: | --------------------------------- |
| `LR`               |  `1e-4` | Larger LR for from-scratch        |
| `EPOCHS`           |     `2` | Fewer epochs (large data)         |
| `NUM_MASK_TOKENS`  |    `16` | K speculative tokens per position |
| `SAMPLE_BLOCK_NUM` |     `8` | Number of blocks sampled per pack |
| `LOOP_NUM`         |     `1` | MTP-style iteration count         |
| `SAVE_STEPS`       |  `2000` | Checkpoint every N steps          |

### Output

Training produces:

- `output/{run_name}/model.safetensors` — DFlash draft weights (~350 MB)
- `output/{run_name}/config.json` — DFlash draft config
- `output/{run_name}/checkpoint-XXXX/` — intermediate checkpoints

Only `model.safetensors` + `config.json` + your `dflash.py` are needed for inference; see [`inference.md`](inference.md).

---

## 3. Continue-Finetune from an Existing DFlash Draft

Start from a pre-trained DFlash checkpoint (e.g. our released v1) and adapt to a smaller/domain-specific dataset.

**Recommended when data is small** (~10k packs) and a decent DFlash draft already exists.

### Entry

`train/train_draft_from_dflash.py` — invoked by `scripts/sft_dflash_finetune.sh`.

### Quick run

```bash
MODEL_PATH=/path/to/HunyuanOCR \
DFLASH_INIT=/path/to/existing/dflash/checkpoint \
TRAIN_DATA=./data/domain_packed_20480.jsonl \
    bash scripts/sft_dflash_finetune.sh
```

### Key hyperparameters (v3 finetune profile)

| Env var       |      Default | Meaning                               |
| ------------- | -----------: | ------------------------------------- |
| `LR`          |       `2e-5` | **5× smaller** than from-scratch      |
| `EPOCHS`      |         `10` | **More passes** for small data        |
| `WARMUP`      |       `0.05` | Slightly longer warmup                |
| `SAVE_STEPS`  |        `500` | ~10 ckpts over full run               |
| `DFLASH_INIT` | _(required)_ | Path to existing draft checkpoint dir |

### Empirical results (v3 vs v1 on 930 OCR eval set)

|                               | v1 (1M packs, from-scratch) |     v3 (14.7k packs, finetune) |
| ----------------------------- | --------------------------: | -----------------------------: |
| Data volume                   |                1.02 M packs | **14.7 k packs** (70× smaller) |
| End-to-end speedup vs base AR |                       1.92× |             **2.14×** (+11.6%) |
| Mean acceptance length (/15)  |                        6.06 |              **7.36** (+21.4%) |
| Avg draft accept rate         |                       33.8% |            **42.4%** (+8.7 pp) |

**Takeaway**: for domain adaptation, finetuning from a pretrained draft with a smaller LR + more epochs consistently beats training from scratch with the same/less data.

---

## Training Data Format

All three modes consume the same packed JSONL format produced by `tools/pipeline_count_and_pack.py`. See [`data_format.md`](data_format.md) for details.

## Distributed / DeepSpeed

To use DeepSpeed ZeRO-2 (recommended for models > 3B params on ≤ 24GB GPUs), add:

```bash
--deepspeed scripts/zero2.json \
```

to the `args` block inside your training script. Not enabled by default — H20 80GB with `gradient_checkpointing=True` fits full training without ZeRO for most cases.

## Debugging

- Enable NCCL debug logging: `export NCCL_DEBUG=INFO` (already in `env_common.sh`)
- Enable CUDA sync for stack traces: `export CUDA_LAUNCH_BLOCKING=1`
- Enable full torch dist debug: `export TORCH_DISTRIBUTED_DEBUG=DETAIL`

Common issues:

| Symptom                  | Fix                                                      |
| ------------------------ | -------------------------------------------------------- |
| NCCL timeout after N min | Check IB fabric; increase `NCCL_TIMEOUT`                 |
| OOM on 80GB GPU          | Reduce `packed_max_length` from 20480 to 16384           |
| Loss doesn't decrease    | For DFlash: check draft init dir has correct config.json |
