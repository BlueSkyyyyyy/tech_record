#!/usr/bin/env bash
# P62-1 实验 driver：DeepEP bucket 的 all_reduce / reduce_scatter（架构无关部分）。
#
# 为什么 H100(sm90) 可跑（详见 src/deepep_bucket_collectives.cu 头部）：
#   * 六个 kernel 的数据面走 NCCL 设备 API（ncclDevComm/ncclGin/multimem），本机 NCCL 2.29.7 无 GIN
#     设备 API；bf16 路径的 `cvt.rs` 也是 SM100 专属 → 那两部分仅源码分析。
#   * 但信号/工作区布局、`ChunkIterator` 覆盖、两种分块映射、归约语义（fadd4/fmul4）、
#     bf16 随机舍入哈希只踩 common + layout，sm90 完全支持 → 直接 include 真实头文件跑真实设备代码。
#
#   bash harness/exp_deepep_bucket_collectives.sh > harness/exp_deepep_bucket_collectives.out.txt
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$ROOT/src/deepep_bucket_collectives.cu"
PY="$ROOT/harness/exp_deepep_bucket_collectives.py"
DEP=${HOME}/github/DeepEP

"$ROOT/scripts/lab.sh" exec "nvcc -arch=sm_90a -O3 -std=c++17 --expt-relaxed-constexpr -w \
  -I$DEP/deep_ep/include -I/usr/local/cuda/include/cccl \
  '$SRC' -o /tmp/deepep_bucket_collectives"

echo "DeepEP bucket all_reduce / reduce_scatter —— H100 (sm90a)，架构无关部分的真实设备代码"
"$ROOT/scripts/lab.sh" exec "nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader | head -1"
echo ""

"$ROOT/scripts/lab.sh" exec "timeout 90 /tmp/deepep_bucket_collectives /tmp/dbc_dump.txt; echo 'EXIT='\$?"
echo ""
echo "=== Python: 独立 host oracle（几何 / 映射 / 归约 / bf16 哈希 / host 配置算术）==="
"$ROOT/scripts/lab.sh" exec "python3 '$PY' /tmp/dbc_dump.txt"
