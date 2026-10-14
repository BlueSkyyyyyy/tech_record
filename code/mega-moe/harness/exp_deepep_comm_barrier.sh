#!/usr/bin/env bash
# P53-1 实验 driver：DeepEP 的多 rank 屏障与 symmetric-memory 句柄层
#           （comm/barrier.cuh + comm/handle.cuh + layout/common/barrier.cuh）。
#
# 为什么 H100(sm90) 可跑：
#   * NVLink barrier 腿（`nvlink_barrier_wo_local_sync`）只用 `red.release.sys` / `ld.acquire.sys` /
#     `atomicAdd` 这些架构无关内联 asm；与 NCCL 相关的只有 `NCCLGin::get_sym_ptr`（纯地址翻译）。
#     故把该函数**逐行原样抽取**，把 NCCL 句柄换成等价的对称指针桩，在一个 GPU 上多 block 模拟多 rank
#     跑**真实设备代码**。scaleout 腿 `gin_barrier_wo_local_sync` 用 ncclGin，本机 NCCL 2.29.7 头
#     不提供 GIN 设备 API → 仅源码分析。
#   * host 侧域分解（scaleout/scaleup、is_scaleup_nvlink、ginSignalCount）与 get_qp_mode 由 Python
#     独立复刻，并读设备写出的结果交叉核对。
#   bash harness/exp_deepep_comm_barrier.sh > harness/exp_deepep_comm_barrier.out.txt
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$ROOT/src/deepep_comm_barrier.cu"
PY="$ROOT/harness/exp_deepep_comm_barrier.py"
DEP=${HOME}/github/DeepEP
DEV_DUMP=/tmp/deepep_comm_barrier_dev.txt
"$ROOT/scripts/lab.sh" exec "nvcc -arch=sm_90a -O3 -std=c++17 --expt-relaxed-constexpr -w \
  -I$DEP/deep_ep/include -I/usr/local/cuda/include/cccl \
  '$SRC' -o /tmp/deepep_comm_barrier \
  && nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader | head -1 \
  && /tmp/deepep_comm_barrier '$DEV_DUMP' \
  && echo '' && echo '=== Python: host 域分解 + get_qp_mode + 与设备结果交叉核对 ===' \
  && python '$PY' '$DEV_DUMP'"
