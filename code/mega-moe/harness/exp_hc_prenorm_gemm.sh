#!/usr/bin/env bash
# P7-2 实验 driver：在 kernel_lab(H100) 上跑 HyperConnection prenorm GEMM 实测。
#   bash harness/exp_hc_prenorm_gemm.sh > harness/exp_hc_prenorm_gemm.out.txt
# 对应源码 exp_hc_prenorm_gemm.py；参考 ~/github/DeepGEMM 的
# csrc/apis/hyperconnection.hpp 与 sm90_tf32_hc_prenorm_gemm。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
"$ROOT/scripts/lab.sh" exec "cd '$ROOT/harness' && nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader | head -1 && python exp_hc_prenorm_gemm.py"
