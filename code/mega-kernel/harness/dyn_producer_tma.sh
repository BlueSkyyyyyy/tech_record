#!/usr/bin/env bash
# P61-1 复现：**动态生产者的第三种写载体：TMA bulk store（SMEM→global）**——补齐 §49.7（P60-1）
# 明留的「最值得做的下一条尾巴」（uint4 是寄存器 store 上限；TMA store 理论可把「写」接上近 HBM
# 整机写带宽，但须先在 SMEM 造好）。
# 唯一新变量 = producer 写载体 gvar：0 标量 / 1 逐线程 uint4 / 2 协作 coalesced / 3 TMA bulk store。
# 另加 mode 6 sgen（只生成到 SMEM、不落 global）隔离 SMEM 暂存代价。
# 四个 Section：A 正确性（四载体逐位一致）/ B gen 相对 static 的代价 × P / C 纯写吞吐 × P /
#   C2 SMEM 暂存代价 / D 接数据面后「生成盖不住执行」是否进一步收窄。
# 真实锚点（mirage 快照 f9eb70c）：worker 取 task 用 TMA bulk 1D；scheduler 只做「事件→task id」映射
#   入队（persistent_kernel.cuh:1254-1523），payload 来自编译期预计算的 FullTaskDesc 数组（runtime.cc），
#   **不在设备侧现场生成**——本节量「现场生成即使接上最宽写载体（TMA store）也不值得」。
# 目标卡 H100 / sm90a，DEV 选空闲 device（默认 3）。
set -u
DEV=${DEV:-3}
SRC=~/proj/tech_record/code/mega-kernel/src/dyn_producer_tma.cu
BIN=/tmp/dyn_producer_tma
nvcc -O3 -arch=sm_90a "$SRC" -o "$BIN" || exit 1

CUDA_VISIBLE_DEVICES="$DEV" timeout 3600 "$BIN" bench
