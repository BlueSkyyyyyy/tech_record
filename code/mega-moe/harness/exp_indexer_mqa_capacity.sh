#!/usr/bin/env bash
# P169-1 实验 driver：真实 indexer 架构（MQA + ReLU）的可训练容量。
#   bash harness/exp_indexer_mqa_capacity.sh > harness/exp_indexer_mqa_capacity.out.txt
# 对应源码 exp_indexer_mqa_capacity.py；参考 ~/github/DeepGEMM README.md:88-110、
#   csrc/apis/attention.hpp、deep_gemm/utils/math.py、tests/test_attention.py。
# 本机是 8 卡共享节点：自动绑到当前利用率最低的 GPU，避免争用噪声。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
"$ROOT/scripts/lab.sh" exec "nvidia-smi >/dev/null 2>&1 || true; \
  IDLE=\$(nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader,nounits | sort -t, -k2 -n | head -1 | cut -d, -f1 | tr -d ' '); \
  echo \"[exp_indexer_mqa_capacity] pinned to GPU \$IDLE\"; \
  cd '$ROOT/harness' && nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader | head -1 && CUDA_VISIBLE_DEVICES=\$IDLE python exp_indexer_mqa_capacity.py"
