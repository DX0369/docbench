#!/bin/bash
# ============================================================================
# DFlash draft model training FROM SCRATCH.
#
# Trains a small MTP-style draft model that predicts K speculative tokens
# for HunyuanOCR to enable speculative decoding at inference time.
#
# Default profile (from-scratch on large data, e.g. ~1M packs):
#   - lr        = 1e-4 (from-scratch LR)
#   - epochs    = 2
#   - num_mask_tokens  = 16     (K spec tokens)
#   - sample_block_num = 8
#   - packed_max_length = 20480
#
# For continue-finetuning from an existing DFlash checkpoint on smaller data,
# use scripts/sft_dflash_finetune.sh instead (lr=2e-5, ep=10).
# ============================================================================

set -e

# Activate your Python environment before running:
#   source /path/to/miniconda3/bin/activate && conda activate hyocr

# ────────────── Common env ──────────────
source scripts/env_common.sh

# ────────────── Distributed configuration ──────────────
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
NNODES=${NNODES:-1}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
NODE_RANK=${NODE_RANK:-0}

echo "总节点数 (Total nodes) : $NNODES"
echo "主节点地址 (Master)     : $MASTER_ADDR"
echo "主节点端口 (Port)       : $MASTER_PORT"
echo "NODE_RANK              : $NODE_RANK"

# ────────────── Model & data paths ──────────────
model_name_or_path=${MODEL_PATH:-/path/to/HunyuanOCR/base/model}
train_data_path=${TRAIN_DATA:-./data/parsing_packed_20480.jsonl}
image_path="not_needed"

# ────────────── Hyperparameters (from-scratch profile) ──────────────
lr=${LR:-1e-4}
batch_size=${BATCH_SIZE:-1}
grad_accum_steps=${GRAD_ACCUM:-1}
num_epochs=${EPOCHS:-2}
save_steps=${SAVE_STEPS:-2000}

# DFlash-specific
num_mask_tokens=${NUM_MASK_TOKENS:-16}
loop_num=${LOOP_NUM:-1}
sample_block_num=${SAMPLE_BLOCK_NUM:-8}

entry_file=train/train_draft.py

# ────────────── Output ──────────────
run_name="hyocr_dflash_scratch_lr${lr}_ep${num_epochs}_$(date +%m%d_%H%M)"
output_dir="./output/${run_name}"
TENSORBOARD_DIR="${output_dir}/tensorboard/$(date "+%Y.%m.%d-%H.%M.%S")"
mkdir -p "${TENSORBOARD_DIR}"

echo "========================================"
echo "Run name        : ${run_name}"
echo "Output dir      : ${output_dir}"
echo "Base model      : ${model_name_or_path}"
echo "Train data      : ${train_data_path}"
echo "  lr            : ${lr}"
echo "  epochs        : ${num_epochs}"
echo "  batch_size    : ${batch_size}"
echo "  grad_accum    : ${grad_accum_steps}"
echo "  num_mask_toks : ${num_mask_tokens}"
echo "  sample_blocks : ${sample_block_num}"
echo "========================================"

# ────────────── Training args ──────────────
args="
    --model_name_or_path ${model_name_or_path} \
    --train_data_path ${train_data_path} \
    --image_folder ${image_path} \
    --data_flatten True \
    --data_packing True \
    --num_mask_tokens ${num_mask_tokens} \
    --use_kv_cache False \
    --loop_num ${loop_num} \
    --sample_block_num ${sample_block_num} \
    --tune_mm_vision True \
    --tune_mm_mlp True \
    --tune_mm_llm True \
    --bf16 \
    --output_dir ${output_dir} \
    --num_train_epochs ${num_epochs} \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size $((batch_size*2)) \
    --gradient_accumulation_steps ${grad_accum_steps} \
    --eval_strategy no \
    --save_strategy steps \
    --save_steps ${save_steps} \
    --save_total_limit 3 \
    --learning_rate ${lr} \
    --weight_decay 0.01 \
    --warmup_ratio 0.03 \
    --max_grad_norm 1 \
    --lr_scheduler_type cosine_with_min_lr \
    --logging_steps 1 \
    --packed_max_length 20480 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --run_name ${run_name} \
    --logging_dir ${TENSORBOARD_DIR} \
    --report_to tensorboard"

# ────────────── Launch ──────────────
echo "Launch training on NODE_RANK=${NODE_RANK}"
torchrun --nproc_per_node=${NPROC_PER_NODE} \
         --master_addr=${MASTER_ADDR} \
         --master_port=${MASTER_PORT} \
         --node_rank=${NODE_RANK} \
         --nnodes=${NNODES} \
         ${entry_file} ${args} 2>&1 | tee "${output_dir}/train_${NODE_RANK}.log"
