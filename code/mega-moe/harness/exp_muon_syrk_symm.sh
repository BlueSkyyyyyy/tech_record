#!/usr/bin/env bash
# P35-1 实验 driver：在 kernel_lab(H100) 上跑 DeepGEMM 的 Muon / 对称矩阵 batched 算子实测。
#   bash harness/exp_muon_syrk_symm.sh > harness/exp_muon_syrk_symm.out.txt
# 对应源码 exp_muon_syrk_symm.py；参考 ~/github/DeepGEMM 的
# csrc/apis/gemm.hpp、csrc/apis/einsum.hpp、csrc/jit_kernels/impls/smxx_cublaslt.hpp、
# csrc/utils/layout.hpp、tests/generators.py、tests/test_bf16.py。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
"$ROOT/scripts/lab.sh" exec "cd '$ROOT/harness' && nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader | head -1 && python exp_muon_syrk_symm.py"
