#!/usr/bin/env bash
# P11-1 实验 driver：在 kernel_lab(H100/sm90) 上跑 DeepGEMM 的 FP8 einsum 投影
#   `bhr,hdr->bhd`（MLA 吸收投影的 FP8 版，per-token×per-block block-scale）。
#   bash harness/exp_fp8_einsum.sh > harness/exp_fp8_einsum.out.txt
# 对应源码 exp_fp8_einsum.py；参考 ~/github/DeepGEMM 的
# csrc/apis/einsum.hpp:181-298、csrc/jit_kernels/impls/sm90_fp8_gemm_1d2d.hpp:283-352
# 与 deep_gemm/include/deep_gemm/impls/sm90_fp8_gemm_1d2d.cuh 的 GemmType::Batched 分支。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
"$ROOT/scripts/lab.sh" exec "cd '$ROOT/harness' && nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader | head -1 && python exp_fp8_einsum.py"
