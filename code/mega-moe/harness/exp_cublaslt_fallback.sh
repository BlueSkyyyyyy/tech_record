#!/usr/bin/env bash
# P17-1 实验 driver：在 kernel_lab(H100) 上跑 DeepGEMM cuBLASLt 回退路径与选核实测。
#   bash harness/exp_cublaslt_fallback.sh > harness/exp_cublaslt_fallback.out.txt
# 对应源码 exp_cublaslt_fallback.py；参考 ~/github/DeepGEMM 的
# csrc/apis/gemm.hpp、csrc/apis/einsum.hpp、csrc/runtime/runtime.hpp、
# csrc/jit_kernels/impls/smxx_cublaslt.hpp、csrc/jit_kernels/impls/epilogue_class.hpp。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
"$ROOT/scripts/lab.sh" exec "cd '$ROOT/harness' && nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader | head -1 && python exp_cublaslt_fallback.py"
