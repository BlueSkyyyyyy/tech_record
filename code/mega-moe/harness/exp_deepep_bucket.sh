#!/usr/bin/env bash
# P54-1 实验 driver：DeepEP bucket 集合通信的 copy-engine 路径与分块层。
#
# 为什么 H100(sm90) 可跑（真机真实代码路径）：
#   * bucket 的纯 NVLink all-gather **不用 SM kernel**，而是硬件 copy engine：
#       copy_engine_push = cudaMemcpyBatchAsync（CUDA 12.8+，本机 13.2）
#       copy_engine_sync = cuStreamBatchMemOp(WRITE_VALUE_64 + WAIT_VALUE_64)
#       copy_engine_wait = WAIT_VALUE_GEQ
#     全是驱动/运行时 API，与 SM 架构无关。本文件把它们逐行等价复刻，并：
#       Part1  D2D 批拷贝语义
#       Part2  mem-op write/wait 顺序 + 255 批拆分 + 跨 stream 握手
#       Part3  单 GPU 多逻辑 rank 模拟的纯 NVLink all-gather
#       Part5  真·多 GPU（每 rank 一块 H100）跨 NVLink peer 的 all-gather
#       Part4  真实 `layout::ChunkIterator`（分块布局，直接 include 真实头文件跑设备代码）
#   * all_reduce/reduce_scatter 用 `ncclDevComm`/`ncclGin`/`multimem` → 本机 NCCL 2.29.7 无 GIN
#     设备 API → 仅源码分析（见 docs）。
#   * host 算术层（get_theoretical_num_sms / chunk 自选择 / 布局大小）由 Python 抽真实源码对拍。
#
# 用法：bash harness/exp_deepep_bucket.sh > harness/exp_deepep_bucket.out.txt
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$ROOT/src/deepep_bucket_copy_engine.cu"
PY="$ROOT/harness/exp_deepep_bucket.py"
DEP=${HOME}/github/DeepEP

"$ROOT/scripts/lab.sh" exec "nvcc -arch=sm_90a -O3 -std=c++17 --expt-relaxed-constexpr -w \
  -I$DEP/deep_ep/include -I/usr/local/cuda/include/cccl \
  '$SRC' -o /tmp/deepep_bucket_ce -lcuda"

echo "DeepEP bucket 集合通信的 copy-engine 路径与分块层实验 —— H100 (sm90a)，真实代码路径"
"$ROOT/scripts/lab.sh" exec "nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader | head -1"

# Part 1/2/4（本进程不跑 Part3，避免单 GPU 模拟里长时互锁）
"$ROOT/scripts/lab.sh" exec "timeout 90 /tmp/deepep_bucket_ce /tmp/deepep_bucket_ce.txt; echo 'EXIT(parts124)='\$?"

# Part 3：单 GPU 模拟多 rank（R=2/4 通过；R>=8 会因单 GPU 无法真正并发而互锁 → 记录为模拟上限）
echo ""
echo "### Part 3：单 GPU 模拟纯 NVLink all-gather（copy engine，多逻辑 rank）"
for cfg in "2 4194304" "4 1048576" "8 262144" "16 131072"; do
  set -- $cfg
  "$ROOT/scripts/lab.sh" exec "timeout 30 /tmp/deepep_bucket_ce --part3 $1 $2 /tmp/p3_$1.txt; echo '  EXIT(R=$1)='\$?"
done

# Part 5：真·多 GPU all-gather（每 rank 一块 H100，跨 NVLink peer）
echo ""
echo "### Part 5：真·多 GPU 纯 NVLink all-gather（copy engine + peer 内存）"
for cfg in "2 4194304" "4 1048576" "8 262144"; do
  set -- $cfg
  "$ROOT/scripts/lab.sh" exec "timeout 60 /tmp/deepep_bucket_ce --multigpu $1 $2 /tmp/mg_$1.txt; echo '  EXIT(R=$1)='\$?"
done

echo ""
echo "=== Python: host 算术层（真实源码抽取 + 独立复刻 + 真实 nvidia-smi 读数）==="
"$ROOT/scripts/lab.sh" exec "python '$PY' /tmp/deepep_bucket_ce.txt /tmp/mg_2.txt /tmp/mg_4.txt /tmp/mg_8.txt /tmp/p3_2.txt /tmp/p3_4.txt"
