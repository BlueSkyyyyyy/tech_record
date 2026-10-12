#!/usr/bin/env bash
# P48-1 实验 driver：DeepEP combine 的归约 epilogue（combine_reduce / combine_reduce_epilogue_impl）。
#
# 为什么 H100(sm90) 可跑：
#   * `combine_utils.cuh` 的 host constexpr（use_rank_layout / get_num_tokens_in_layout /
#     get_max_unroll_factor）与 `CombineVecTraits`（sm90 选 int4）；`combine_reduce` 只用 warp 内建 +
#     `ptx::ldg_with_ge_pred` + `ptx::accumulate`（sm90 回退）。
#   * `combine_reduce_epilogue_impl` 只用 TMA store（sm90 支持）+ PDL + TokenLayout。
#   故 `src/deepep_combine_reduce.cu` **直接 include 真实头文件、编译运行真实设备代码**，
#   用真实对称 recv_buffer 布局喂进去，再与独立 host oracle 逐位对拍。
#   sm100 的融合 `add.rn.f32.bf16` 快路（ptx.cuh:593-602）在 sm90 自动回退。
#   bash harness/exp_deepep_combine_reduce.sh > harness/exp_deepep_combine_reduce.out.txt
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$ROOT/src/deepep_combine_reduce.cu"
PY="$ROOT/harness/exp_deepep_combine_reduce.py"
DEP=~/github/DeepEP
"$ROOT/scripts/lab.sh" exec "nvcc -arch=sm_90a -O3 -std=c++17 --expt-relaxed-constexpr -w \
  -I$DEP/deep_ep/include -I/usr/local/cuda/include/cccl \
  '$SRC' -o /tmp/deepep_combine_reduce \
  && nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader | head -1 \
  && /tmp/deepep_combine_reduce \
  && echo '' && echo '=== Python: 独立 host oracle（公式 + dedup 选择逻辑） ===' \
  && python '$PY'"
