#!/usr/bin/env bash
# P59-1 实验 driver：DeepEP 的 Engram 远程抓取原语（engram_fetch / engram_fetch_wait）。
#
# 为什么 H100(sm90) 可跑：
#   * 布局层 `layout::EngramLayout`（字节偏移 / SF 行号与分片号）与
#     `layout::EngramWorkspace`（字段偏移）都是 constexpr 整数运算，真实头文件直接可用。
#   * 发射面 `EngramIssueWarpLayout`（issue warp↔(peer,gin_context,tile) 映射）与 issue 循环里
#     「哪些 entry 归本 peer、落到哪个 window/源偏移/目的偏移」是纯整数映射 + `__ballot_sync`/`__popc`。
#   * 只有 `gin.get`（RDMA 拉取）/ `gin.flush_async` / `gin.wait` / SF 的 `gin.get_sym_ptr<Lsa>`
#     用 `nccl_device.h` 的 GIN 设备 API（本机 NCCL 2.29.7 不提供）→ 仅源码分析。
#   故把 `EngramIssueWarpLayout`（cuh:13-39）与 issue 循环（cuh:99-171）/ SF 主体（cuh:208-254）
#   逐行原样抽取，`gin.get` 换成 recorder 记录参数，跑真实设备代码；host 侧公式由 Python 独立复刻。
#   bash harness/exp_deepep_engram.sh > harness/exp_deepep_engram.out.txt
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$ROOT/src/deepep_engram_fetch.cu"
PY="$ROOT/harness/exp_deepep_engram.py"
DEP=${HOME}/github/DeepEP
DEV_DUMP=/tmp/deepep_engram_dev.txt
"$ROOT/scripts/lab.sh" exec "nvcc -arch=sm_90a -O3 -std=c++17 --expt-relaxed-constexpr -w \
  -I$DEP/deep_ep/include -I/usr/local/cuda/include/cccl \
  '$SRC' -o /tmp/deepep_engram_fetch \
  && nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader | head -1 \
  && /tmp/deepep_engram_fetch '$DEV_DUMP' \
  && echo '' && echo '=== Python: host oracle（布局/发射映射/请求索引/SF 拷贝/host 公式）===' \
  && python '$PY' '$DEV_DUMP'"
