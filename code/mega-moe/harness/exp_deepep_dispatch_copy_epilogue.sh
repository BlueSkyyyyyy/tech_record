#!/usr/bin/env bash
# P51-1 实验 driver：DeepEP dispatch 的拷贝 epilogue（`dispatch_copy_epilogue_impl`）。
#
# 为什么 H100(sm90) 可跑：
#   * `dispatch_copy_epilogue.cuh` 整段**不碰 NCCL**（无 `ncclDevComm`/`NCCLGin`/`ncclTeam`），只用
#     `TokenLayout`/`BufferLayout`（`layout/ep/token.cuh`）+ `EPWorkspaceLayout`（`layout/ep/workspace.cuh`）
#     + `ptx::{tma_*,st_bulk,mbarrier_*}`（第 59 篇已逐条在 sm90 验证）+ 轻量 warp 内建 + PDL。
#   故 `src/deepep_dispatch_copy_epilogue.cu` **直接 include 真实头文件、编译运行真实设备代码**，
#   用真实 token 缓冲布局喂进去，再与独立 host oracle 对拍。
#   bash harness/exp_deepep_dispatch_copy_epilogue.sh > harness/exp_deepep_dispatch_copy_epilogue.out.txt
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$ROOT/src/deepep_dispatch_copy_epilogue.cu"
PY="$ROOT/harness/exp_deepep_dispatch_copy_epilogue.py"
DEP=${HOME}/github/DeepEP
"$ROOT/scripts/lab.sh" exec "nvcc -arch=sm_90a -O3 -std=c++17 --expt-relaxed-constexpr -w \
  -I$DEP/deep_ep/include -I/usr/local/cuda/include/cccl \
  '$SRC' -o /tmp/deepep_dispatch_copy_epilogue \
  && nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader | head -1 \
  && /tmp/deepep_dispatch_copy_epilogue \
  && echo '' && echo '=== Python: 独立 host oracle（TokenLayout 偏移 + expand 对齐分配公式） ===' \
  && python '$PY'"
