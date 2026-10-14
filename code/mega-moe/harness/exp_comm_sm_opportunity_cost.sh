#!/usr/bin/env bash
# P183-1 实验 driver：给通信让 SM 的机会成本（第 190/191/192 篇的合账）。
#   bash harness/exp_comm_sm_opportunity_cost.sh > harness/exp_comm_sm_opportunity_cost.out.txt
# 对应源码 exp_comm_sm_opportunity_cost.py；复用 P180-1 的 build/kernel_time 与 P63-1 的 ast 抽取。
# 本机是 8 卡共享节点：自动绑到当前利用率最低的 GPU。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
"$ROOT/scripts/lab.sh" exec "nvidia-smi >/dev/null 2>&1 || true; \
  IDLE=\$(nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader,nounits | sort -t, -k2 -n | head -1 | cut -d, -f1 | tr -d ' '); \
  echo \"[exp_comm_sm_opportunity_cost] pinned to GPU \$IDLE\"; \
  cd '$ROOT/harness' && nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader | head -1 && CUDA_VISIBLE_DEVICES=\$IDLE python exp_comm_sm_opportunity_cost.py"
