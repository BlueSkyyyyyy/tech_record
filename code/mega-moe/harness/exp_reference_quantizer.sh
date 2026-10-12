#!/usr/bin/env bash
# P27-1 实验 driver：在 kernel_lab(H100) 上跑 DeepGEMM 测试参考量化器
#   deep_gemm/utils/math.py 的语义与边界实测。
#   bash harness/exp_reference_quantizer.sh > harness/exp_reference_quantizer.out.txt
# 对应源码 exp_reference_quantizer.py；参考 ~/github/DeepGEMM 的
# deep_gemm/utils/math.py、tests/generators.py、tests/test_mega_moe.py。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
"$ROOT/scripts/lab.sh" exec "cd '$ROOT/harness' && nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader | head -1 && python exp_reference_quantizer.py"
