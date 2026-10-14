#!/usr/bin/env bash
# P29-1 实验 driver：在 kernel_lab(H100) 上跑 DeepGEMM 的 legacy A100 Triton MoE grouped GEMM
#   家族的「显式索引 gather」语义与契约。
#   bash harness/exp_legacy_grouped_gemm.sh > harness/exp_legacy_grouped_gemm.out.txt
# 对应源码 exp_legacy_grouped_gemm.py；参考 ~/github/DeepGEMM 的
# deep_gemm/legacy/*.py、tests/test_legacy.py、tests/generators.py。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
"$ROOT/scripts/lab.sh" exec "cd '${HOME}/github/DeepGEMM/tests' && nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader | head -1 && python '$ROOT/harness/exp_legacy_grouped_gemm.py'"
