#!/usr/bin/env bash
# P176-1 实验 driver：在 kernel_lab(H100) 真跑 DeepGEMM 的 BF16 MN-major K-grouped wgrad
#   (`k_grouped_bf16_gemm_tn_contiguous` -> sm90_bf16_k_grouped_gemm)，作为 SM100 FP8
#   `k_grouped_fp8_gemm_tn_contiguous`（MN-major，本机跑不了）的物理布局同构对照。
#   bash harness/exp_kgrouped_tn_mn_major.sh > harness/exp_kgrouped_tn_mn_major.out.txt
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
"$ROOT/scripts/lab.sh" exec "cd '$ROOT/harness' && \
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader | head -1 && \
python3 -u exp_kgrouped_tn_mn_major.py"
