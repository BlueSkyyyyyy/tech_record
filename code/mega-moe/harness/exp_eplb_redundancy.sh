#!/usr/bin/env bash
# P43-1 实验 driver：DeepEP EPLB 冗余专家的「权重预取 / 梯度归约」切块（Mega MoE 反向侧的通信原语）。
#
# 为什么 H100(sm90) 可跑：
#   * EPLBSharedMemoryLayout::build_tasks（cub::BlockScan + smem）与 EPWeightIterator 的整数切块逻辑
#     都不依赖 NCCL GIN（EPWeightIterator 只在一处调 gin.get_sym_ptr，用等价对称指针桩替换）；
#     两段均**逐行原样抽取**真实设备代码，见 src/eplb_redundancy.cu。
#   * host 侧公式（lb_get_theoretical_num_sms / align）纯 Python，跑真实 nvidia-smi nvlink。
# 真正的 lb_prefetch_weights / lb_reduce_grads kernel 用 ncclDevComm/ncclGin，本机 NCCL 2.29.7 头
# 不提供 GIN 设备 API → 仅源码分析。
#   bash harness/exp_eplb_redundancy.sh > harness/exp_eplb_redundancy.out.txt
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$ROOT/src/eplb_redundancy.cu"
PY="$ROOT/harness/exp_eplb_redundancy.py"
DEP=~/github/DeepEP
"$ROOT/scripts/lab.sh" exec "nvcc -arch=sm_90a -O3 -std=c++17 --expt-relaxed-constexpr -w \
  -I$DEP/deep_ep/include -I/usr/local/cuda/include/cccl \
  '$SRC' -o /tmp/eplb_redundancy \
  && nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader | head -1 \
  && /tmp/eplb_redundancy \
  && echo '' && echo '=== Python: host 公式 + 归约语义 ===' \
  && python '$PY'"
