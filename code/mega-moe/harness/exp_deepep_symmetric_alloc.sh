#!/usr/bin/env bash
# P56-1 实验 driver：DeepEP 的对称内存分配层（csrc/kernels/comm/symmetric.hpp）。
#
# 为什么 H100(sm90) 可跑：这一层是纯 host-side CUDA VMM + NCCL 主机 API（cuMemCreate/Map/
#   SetAccess/Export/Import + pidfd_getfd + ncclMemAlloc），与 UMMA/tcgen05/GIN 设备 API 无关。
#   故**直接 include 真实头文件**调用真实类：DeviceContext/cumem_create_with_fallback/
#   ElasticSymmetricMemory/HybridElasticSymmetricMemory/GPUSymmetricMemory/alloc。
#   依赖 NCCL 设备 API 的 window/GIN 部分不在本层，不涉及。
#   bash harness/exp_deepep_symmetric_alloc.sh > harness/exp_deepep_symmetric_alloc.out.txt
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$ROOT/src/deepep_symmetric_alloc.cu"
PY="$ROOT/harness/exp_deepep_symmetric_alloc.py"
DEP=${HOME}/github/DeepEP
DUMP=/tmp/deepep_sym_alloc_dev.txt
SHA="$(sha256sum "$DEP/csrc/kernels/comm/symmetric.hpp" | cut -c1-16)"
"$ROOT/scripts/lab.sh" exec "nvcc -arch=sm_90a -O3 -std=c++17 --expt-relaxed-constexpr -w \
  -I$DEP -I$DEP/deep_ep/include '$SRC' -o /tmp/deepep_sym_alloc -lcuda -lnccl \
  && echo 'symmetric.hpp sha256(前16): $SHA' \
  && nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader | head -1 \
  && /tmp/deepep_sym_alloc | tee $DUMP \
  && echo '' && echo '=== Python: 真实 BufferAllocator 源码计划/落地 + 与设备常量交叉核对 ===' \
  && python '$PY' $DUMP"
