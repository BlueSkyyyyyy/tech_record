#!/usr/bin/env bash
# P162-1 实验 driver：在 kernel_lab(H100/sm90) 上跑 indexer top-k 选择翻转对下游稀疏注意力输出的影响。
#   bash harness/exp_indexer_downstream.sh > harness/exp_indexer_downstream.out.txt
# 对应源码 exp_indexer_downstream.py；参考 ~/github/DeepGEMM
#   csrc/apis/attention.hpp、tests/test_attention.py、README:88-110。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
"$ROOT/scripts/lab.sh" exec "cd '$ROOT/harness' && nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader | head -1 && python exp_indexer_downstream.py"
