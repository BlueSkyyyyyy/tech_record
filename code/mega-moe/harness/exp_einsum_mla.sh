#!/usr/bin/env bash
# P8-2 实验 driver：在 kernel_lab(H100) 上跑 DeepGEMM einsum 的两个 MLA 投影
#   `bhr,hdr->bhd` / `bhd,hdr->bhr`（batched GEMM over heads）实测。
#   bash harness/exp_einsum_mla.sh > harness/exp_einsum_mla.out.txt
# 对应源码 exp_einsum_mla.py；参考 ~/github/DeepGEMM 的
# csrc/apis/einsum.hpp:60/:94、csrc/jit_kernels/impls/sm90_bf16_gemm.hpp:332/:386
# 与 deep_gemm/include/deep_gemm/impls/sm90_bf16_gemm.cuh 的 GemmType::Batched 分支。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
"$ROOT/scripts/lab.sh" exec "cd '$ROOT/harness' && nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader | head -1 && python exp_einsum_mla.py"
