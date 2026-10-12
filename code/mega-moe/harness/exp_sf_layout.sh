#!/usr/bin/env bash
# P15-1 实验 driver：在 kernel_lab(H100) 上跑 DeepGEMM SF 布局转换 / UE8M0 打包实测。
#   bash harness/exp_sf_layout.sh > harness/exp_sf_layout.out.txt
# 对应源码 exp_sf_layout.py；参考 ~/github/DeepGEMM 的
# csrc/apis/layout.hpp、csrc/jit_kernels/impls/smxx_layout.hpp、
# deep_gemm/include/deep_gemm/impls/smxx_layout.cuh。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
"$ROOT/scripts/lab.sh" exec "cd '$ROOT/harness' && nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader | head -1 && python exp_sf_layout.py"
