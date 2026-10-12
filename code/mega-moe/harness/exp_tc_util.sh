#!/usr/bin/env bash
# P26-1 实验 driver：在 kernel_lab(H100) 上验证 DeepGEMM 的 `set_tc_util` 语义与影响面。
#   bash harness/exp_tc_util.sh > harness/exp_tc_util.out.txt
# 源码 exp_tc_util.py；参考 ~/github/DeepGEMM 的
#   csrc/apis/config.hpp, csrc/runtime/runtime.hpp,
#   csrc/jit_kernels/impls/sm100_bf16_gemm.hpp,
#   deep_gemm/include/deep_gemm/impls/sm100_bf16_gemm.cuh。
# 用独立 JIT 缓存目录（冷启动），这样 §4 的「切 tc_util 不重编译」结论才干净。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
"$ROOT/scripts/lab.sh" exec "cd '$ROOT/harness' && export DG_JIT_CACHE_DIR=\$(mktemp -d /tmp/dg_tcutil_XXXX) && nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader | head -1 && python exp_tc_util.py"
