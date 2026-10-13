#!/usr/bin/env bash
# P61-1 实验 driver：DeepEP 两级（hybrid）EP dispatch/combine 的架构无关内核。
#
# 为什么 H100(sm90) 可跑（详见 src/deepep_hybrid.cu 头部）：
#   * `hybrid_dispatch.cuh` / `hybrid_combine.cuh` 的数据面（TMA / `gin.put` / 跨 rank barrier）
#     走 NCCL 设备 API，本机 NCCL 2.29.7 不提供 → 仅源码分析。
#   * 但计数就绪哨兵（`encode_decode_positive`）、tail 信号（`pack2/unpack2`）、
#     前缀和（`do_psum`）、round-robin 分片调度只踩 `common/{math,ptx}.cuh` + `layout/ep/token.cuh`，
#     纯 warp 内建/位运算，sm90 支持。
#   故 `src/deepep_hybrid.cu` **直接 include 真实头文件、编译运行真实设备代码**，再与独立 host
#   oracle 对拍（CUDA 侧 + Python 侧各一套）。
#   bash harness/exp_deepep_hybrid.sh > harness/exp_deepep_hybrid.out.txt
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$ROOT/src/deepep_hybrid.cu"
PY="$ROOT/harness/exp_deepep_hybrid.py"
DEP=~/github/DeepEP
"$ROOT/scripts/lab.sh" exec "nvcc -arch=sm_90a -O3 -std=c++17 --expt-relaxed-constexpr -w \
  -I$DEP/deep_ep/include -I/usr/local/cuda/include/cccl \
  '$SRC' -o /tmp/deepep_hybrid \
  && nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader | head -1 \
  && echo '' && echo '=== DeepEP commit: 93eb6eb ===' \
  && /tmp/deepep_hybrid \
  && echo '' && echo '=== Python: 独立 host oracle（哨兵/tail/前缀和/round-robin/布局） ===' \
  && python3 '$PY'"
