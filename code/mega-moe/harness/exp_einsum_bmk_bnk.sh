#!/usr/bin/env bash
# P8-1 实验 driver：在 kernel_lab(H100) 上跑 DeepGEMM einsum `bmk,bnk->mn` 实测。
#   bash harness/exp_einsum_bmk_bnk.sh > harness/exp_einsum_bmk_bnk.out.txt
# 对应源码 exp_einsum_bmk_bnk.py；参考 ~/github/DeepGEMM 的
# csrc/apis/einsum.hpp、csrc/jit_kernels/impls/sm90_bmk_bnk_mn.hpp 与 impls/sm90_bmk_bnk_mn.cuh。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
"$ROOT/scripts/lab.sh" exec "cd '$ROOT/harness' && nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader | head -1 && python exp_einsum_bmk_bnk.py"
