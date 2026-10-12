#!/usr/bin/env bash
# P19-1 实验 driver：在 kernel_lab(H100) 上跑 DeepGEMM 的 SM90 BF16 dense GEMM 主 kernel 实测。
#   bash harness/exp_bf16_sm90_gemm.sh > harness/exp_bf16_sm90_gemm.out.txt
# 对应源码 exp_bf16_sm90_gemm.py；参考 ~/github/DeepGEMM 的
# csrc/jit_kernels/impls/sm90_bf16_gemm.hpp 与
# deep_gemm/include/deep_gemm/impls/sm90_bf16_gemm.cuh。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
"$ROOT/scripts/lab.sh" exec "cd '$ROOT/harness' && nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader | head -1 && python exp_bf16_sm90_gemm.py"
