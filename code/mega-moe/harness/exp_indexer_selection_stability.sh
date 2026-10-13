#!/usr/bin/env bash
# P161-1 实验 driver：在 kernel_lab(H100/sm90) 上跑 lightning indexer 的 top-k KV 选择稳定性。
#   bash harness/exp_indexer_selection_stability.sh > harness/exp_indexer_selection_stability.out.txt
# 对应源码 exp_indexer_selection_stability.py；参考 ~/github/DeepGEMM
#   csrc/apis/attention.hpp、tests/test_attention.py、deep_gemm/utils/math.py。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
"$ROOT/scripts/lab.sh" exec "cd '$ROOT/harness' && nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader | head -1 && python exp_indexer_selection_stability.py"
