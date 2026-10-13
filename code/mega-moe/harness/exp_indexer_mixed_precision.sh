#!/usr/bin/env bash
# P164-1 实验 driver：indexer 打分的混合精度分配（Q x KV 精度矩阵 / 误差归因 / 带宽账 / 下游）。
#   bash harness/exp_indexer_mixed_precision.sh > harness/exp_indexer_mixed_precision.out.txt
# 对应源码 exp_indexer_mixed_precision.py；参考 ~/github/DeepGEMM
#   csrc/apis/attention.hpp、tests/test_attention.py、deep_gemm/utils/math.py。
# 本机是 8 卡共享节点：自动绑到当前利用率最低的 GPU，避免争用噪声。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
"$ROOT/scripts/lab.sh" exec "nvidia-smi >/dev/null 2>&1 || true; \
  IDLE=\$(nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader,nounits | sort -t, -k2 -n | head -1 | cut -d, -f1 | tr -d ' '); \
  echo \"[exp_indexer_mixed_precision] pinned to GPU \$IDLE\"; \
  cd '$ROOT/harness' && nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader | head -1 && CUDA_VISIBLE_DEVICES=\$IDLE python exp_indexer_mixed_precision.py"
