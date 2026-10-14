#!/usr/bin/env bash
# P182-1 实验 driver：通信腿在「多对多（all-to-all）」下的每 SM 带宽预算。
#   bash harness/exp_nvlink_alltoall_budget.sh > harness/exp_nvlink_alltoall_budget.out.txt
# 对应源码 src/nvlink_alltoall_budget.cu（Part A 单 peer 基线 / Part B 每块一个 peer /
#   Part C 每块轮询所有 peer / Part D 全部 rank 同时 all-to-all）+ exp_nvlink_alltoall_budget.py。
# 参考 ~/github/DeepEP：envs.py:193-204（get_sm_read_gbs=180 / get_sm_write_gbs=45，带 TODO 硬编码）、
#   buffers/ep.py:431-541（用这两个常量反推 dispatch/combine 的 SM 数）。
# 本机是 8 卡共享节点：排掉被别的任务吃满的卡，暴露剩余卡做 UVA peer all-to-all，
#   rank0 = 可见序号 0（最空闲的那张）。全部走真实 nvidia-smi 探测。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

"$ROOT/scripts/lab.sh" exec "set -e; \
  nvcc -arch=sm_90 -O3 -std=c++17 --expt-relaxed-constexpr -diag-suppress 2389 \
    '$ROOT/src/nvlink_alltoall_budget.cu' -o /tmp/nvlink_alltoall_budget; \
  IDLE=\$(nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader,nounits | sort -t, -k2 -n | head -7 | cut -d, -f1 | tr -d ' ' | paste -sd, -); \
  echo \"[exp_nvlink_alltoall_budget] pinned to GPUs \$IDLE\"; \
  CUDA_VISIBLE_DEVICES=\$IDLE /tmp/nvlink_alltoall_budget | tee /tmp/nvlink_alltoall_budget.raw.txt; \
  echo; echo '== Python 侧：把多对多膝点代回 DeepEP SM 模型 =='; \
  A2A_RAW=/tmp/nvlink_alltoall_budget.raw.txt DEEPEP_ROOT=${HOME}/github/DeepEP \
    python '$ROOT/harness/exp_nvlink_alltoall_budget.py'"
