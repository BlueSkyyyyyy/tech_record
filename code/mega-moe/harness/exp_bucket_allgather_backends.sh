#!/usr/bin/env bash
# P102-1 实验 driver：DeepEP bucket all_gather 的两个 SM kernel 后端（pure-RDMA / hybrid）。
#
# 为什么 H100(sm90) 可跑（真机真实代码/协议路径）：
#   * `layout::AllGatherSignals` / `CopyEngineSignals` / `BucketSignals` 是真实头文件里的 POD 布局，
#     直接 include、host sizeof/offsetof 即真值。
#   * `rdma_all_gather` 的 dst-rank(lane) 分解、`hybrid_all_gather` 的每 chunk QP 均分（:`hybrid.cuh:61-64`）
#     是纯整数寻址，与 NCCL GIN 无关，逐行搬运后在真机枚举验证。
#   * hybrid 的完成协议（peer 对每个 (chunk,QP) 在远端 `tails[qp][chunk]` +1；host 用
#     `cuStreamBatchMemOp(WAIT_VALUE_GEQ)` 等 `>= num_rdma_ranks-1`）用真实 device atomic +
#     真实驱动 mem-op 端到端验证。
#   * 真正的 `gin.put`/`ncclGin_WeakVASignalInc` 数据面用 `ncclDevComm`/`ncclGin`（NCCL GIN 设备 API），
#     本机 NCCL 2.29.7 不具备 → 仅源码分析（见 docs）。
#   * host 算术层（hybrid chunk 尺寸自选择代价模型）由 Python 抽真实源码复刻 + 独立对拍。
#
# 用法：bash harness/exp_bucket_allgather_backends.sh > harness/exp_bucket_allgather_backends.out.txt
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$ROOT/src/bucket_allgather_backends.cu"
PY="$ROOT/harness/exp_bucket_allgather_backends.py"
DEP=${HOME}/github/DeepEP

"$ROOT/scripts/lab.sh" exec "nvcc -arch=sm_90a -O3 -std=c++17 --expt-relaxed-constexpr -w \
  -I$DEP/deep_ep/include -I/usr/local/cuda/include/cccl \
  '$SRC' -o /tmp/bucket_ag_backends -lcuda"

echo "DeepEP bucket all_gather 的 SM kernel 后端（pure-RDMA / hybrid）实验 —— H100 (sm90a)，真实代码/协议路径"
"$ROOT/scripts/lab.sh" exec "nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader | head -1"
echo ""

"$ROOT/scripts/lab.sh" exec "timeout 120 /tmp/bucket_ag_backends; echo 'EXIT='\$?"

echo ""
echo "=== Python: host 算术层（hybrid chunk 尺寸自选择代价模型 + 独立复刻）==="
"$ROOT/scripts/lab.sh" exec "python '$PY'"
