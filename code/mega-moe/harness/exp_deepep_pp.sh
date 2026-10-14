#!/usr/bin/env bash
# P58-1 实验 driver：DeepEP 的流水线并行 send/recv 原语（pp_send_recv）。
#
# 为什么 H100(sm90) 可跑：
#   * 数据面 `pp_tma_copy` 只调 `ptx::tma_load_1d` / `ptx::tma_store_1d` / mbarrier（sm90 起可用）。
#   * 寻址面 `get_pp_buffer_offset` 与 4 段缓冲 block / arrival / release 索引是纯整数映射。
#   * 只有数据面第二跳的 `NCCLGin::put`（push 到对端 recv 槽）与 `NCCLGin::signal`
#     （StrongVASignalAdd）用 nccl_device.h 的 GIN 设备 API（本机 NCCL 2.29.7 不提供）→ 仅源码分析。
#   故把 `get_pp_buffer_offset`（cuh:20-25）与 `pp_tma_copy`（cuh:27-102）逐行原样抽取，
#   直接 include 真实 DeepEP 头文件、跑真实设备代码；host 侧索引 / 缓冲几何 / 调度不变式由 Python
#   独立复刻并读设备 dump 交叉核对。
#   bash harness/exp_deepep_pp.sh > harness/exp_deepep_pp.out.txt
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$ROOT/src/deepep_pp_send_recv.cu"
PY="$ROOT/harness/exp_deepep_pp.py"
DEP=${HOME}/github/DeepEP
DEV_DUMP=/tmp/deepep_pp_dev.txt
"$ROOT/scripts/lab.sh" exec "nvcc -arch=sm_90a -O3 -std=c++17 --expt-relaxed-constexpr -w \
  -I$DEP/deep_ep/include -I/usr/local/cuda/include/cccl \
  '$SRC' -o /tmp/deepep_pp_send_recv \
  && nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader | head -1 \
  && /tmp/deepep_pp_send_recv '$DEV_DUMP' \
  && echo '' && echo '=== Python: host oracle（索引复刻 + 调度不变式 + 缓冲几何）===' \
  && python '$PY' '$DEV_DUMP'"
