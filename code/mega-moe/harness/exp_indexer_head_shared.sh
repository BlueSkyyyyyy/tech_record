#!/usr/bin/env bash
# P163-1 实验 driver：在 kernel_lab(H100/sm90) 上跑 indexer 选择翻转对真实 DSA 主注意力
#   （GQA/MQA 头共享 + per-head softmax + 1/sqrt(d)）前向与训练反向的联合影响。
#   bash harness/exp_indexer_head_shared.sh > harness/exp_indexer_head_shared.out.txt
# 对应源码 exp_indexer_head_shared.py；参考 ~/github/DeepGEMM
#   csrc/apis/attention.hpp、tests/test_attention.py、README:88-110。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
"$ROOT/scripts/lab.sh" exec "cd '$ROOT/harness' && nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader | head -1 && python exp_indexer_head_shared.py"
