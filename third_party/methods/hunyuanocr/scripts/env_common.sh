#!/bin/bash
# ============================================================================
# Common environment variables shared across all training scripts.
# Source this from within a training script:  source scripts/env_common.sh
#
# Covers:
#   - NCCL / InfiniBand tuning for multi-node training
#   - Timeout / error-handling settings
#
# Adjust NCCL_SOCKET_IFNAME / NCCL_IB_HCA to match your cluster's NIC layout.
# ============================================================================

# ────────────── NCCL / InfiniBand ──────────────
export NCCL_IB_GID_INDEX=3
export NCCL_IB_SL=3
export NCCL_CHECK_DISABLE=1
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0
export NCCL_LL_THRESHOLD=16384
export NCCL_IB_CUDA_SUPPORT=1
export NCCL_TOPO_AFFINITY=6
export NCCL_COLLNET_ENABLE=0
export SHARP_COLL_ENABLE_SAT=0
export NCCL_NET_GDR_LEVEL=2
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_IB_TC=160
export NCCL_PXN_DISABLE=0

# Cluster-specific — edit these to match your network setup
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-bond1}
export UCX_NET_DEVICES=${UCX_NET_DEVICES:-bond1}
export NCCL_IB_HCA=${NCCL_IB_HCA:-mlx5_bond_1,mlx5_bond_5,mlx5_bond_3,mlx5_bond_7,mlx5_bond_4,mlx5_bond_8,mlx5_bond_2,mlx5_bond_6}

# ────────────── Timeout / Error Handling ──────────────
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=1800            # 30 min
export NCCL_SOCKET_TIMEOUT=1800     # 30 min
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}

# For debugging hangs, uncomment:
# export CUDA_LAUNCH_BLOCKING=1
# export TORCH_DISTRIBUTED_DEBUG=DETAIL
