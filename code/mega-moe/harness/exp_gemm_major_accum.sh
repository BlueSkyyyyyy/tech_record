#!/usr/bin/env bash
# P36-1 实验 driver：在 kernel_lab(H100/sm90) 上跑 DeepGEMM GEMM API 的输入主序（nt/nn/tn/tt）与输出累积契约。
#   bash harness/exp_gemm_major_accum.sh > harness/exp_gemm_major_accum.out.txt
# 对应源码 exp_gemm_major_accum.py；参考 ~/github/DeepGEMM 的 csrc/apis/gemm.hpp、csrc/utils/layout.hpp。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
"$ROOT/scripts/lab.sh" exec "cd '$ROOT/harness' && nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader | head -1 && python exp_gemm_major_accum.py"
