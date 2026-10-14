#!/usr/bin/env bash
# P181-1 实验 driver：通信腿的「每 SM 带宽」预算。
#   bash harness/exp_sm_bandwidth_budget.sh > harness/exp_sm_bandwidth_budget.out.txt
# 对应源码 src/sm_bandwidth_budget.cu（Part A HBM / Part B NVLink P2P）+ exp_sm_bandwidth_budget.py
#   （Part C：复用 P63-1 的 ast 抽取跑真实 DeepEP get_theoretical_num_sms，换用实测常量）。
# 参考 ~/github/DeepEP：envs.py:193-204（get_sm_read_gbs=180 / get_sm_write_gbs=45，带 TODO 硬编码）、
#   buffers/ep.py:431-541（用这两个常量反推 SM 数）。
# 本机是 8 卡共享节点：Part A 绑到利用率最低的卡；Part B 需要两张卡做 NVLink P2P，
# 故进程用 CUDA_VISIBLE_DEVICES 暴露两张最空闲的卡（rank0/rank1）。两者都走真实 nvidia-smi 探测。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

"$ROOT/scripts/lab.sh" exec "set -e; \
  nvcc -arch=sm_90 -O3 -std=c++17 --expt-relaxed-constexpr -diag-suppress 2389 \
    '$ROOT/src/sm_bandwidth_budget.cu' -o /tmp/sm_bandwidth_budget; \
  IDLE=\$(nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader,nounits | sort -t, -k2 -n | head -2 | cut -d, -f1 | tr -d ' ' | paste -sd, -); \
  echo \"[exp_sm_bandwidth_budget] pinned to GPUs \$IDLE\"; \
  CUDA_VISIBLE_DEVICES=\$IDLE /tmp/sm_bandwidth_budget | tee /tmp/sm_bandwidth_budget.raw.txt; \
  echo; echo '== Python 侧：重构 DeepEP SM 模型 =='; \
  BANDWIDTH_RAW=/tmp/sm_bandwidth_budget.raw.txt DEEPEP_ROOT=~/github/DeepEP \
    python '$ROOT/harness/exp_sm_bandwidth_budget.py'"
