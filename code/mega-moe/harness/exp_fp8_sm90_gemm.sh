#!/usr/bin/env bash
# P22-1 实验 driver：在 kernel_lab(H100) 上跑 DeepGEMM 的 SM90 FP8 dense GEMM（1D1D vs 1D2D）实测。
#   bash harness/exp_fp8_sm90_gemm.sh > harness/exp_fp8_sm90_gemm.out.txt
# 对应源码 exp_fp8_sm90_gemm.py；参考 ~/github/DeepGEMM 的
# csrc/jit_kernels/impls/sm90_fp8_gemm_1d1d.hpp / sm90_fp8_gemm_1d2d.hpp 与
# deep_gemm/include/deep_gemm/impls/sm90_fp8_gemm_1d1d.cuh / sm90_fp8_gemm_1d2d.cuh。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
"$ROOT/scripts/lab.sh" exec "cd '$ROOT/harness' && nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader | head -1 && python exp_fp8_sm90_gemm.py"
