#!/usr/bin/env bash
# P60-1 复现：**动态生产者的写入向量化**——把 §48 的标量 strided 写换成向量化/协作写后，「生成」的
# 代价能压回多少。补齐 §48.8 明留的尾巴②（gen 仍是标量 strided 写，向量化/协作写未单独做）。
# 唯一新变量 = producer 写入形态 gvar：0 标量 / 1 逐线程 uint4 / 2 协作 coalesced（payload 内容相同）。
# 四个 Section：A 正确性（三形态逐位一致）/ B gen 相对 static 的代价 × P / C 纯写吞吐（无 worker 消费）/
#   D 接数据面后「生成盖不住执行」是否进一步收窄。
# 真实锚点（mirage 快照 f9eb70c）：H100 128 worker/16 scheduler（utils.py:25-29）；scheduler 只做
#   「事件→task id」映射入队 persistent_kernel.cuh:1254-1523，payload 来自编译期预计算的 FullTaskDesc
#   数组（runtime.cc），**不在设备侧现场生成**——本节量「现场生成即使向量化也不值得」。
# 目标卡 H100 / sm90a，DEV 选空闲 device（默认 3）。
set -u
DEV=${DEV:-3}
SRC=${HOME}/proj/tech_record/code/mega-kernel/src/dyn_producer_vector.cu
BIN=/tmp/dyn_producer_vector
nvcc -O3 -arch=sm_90a "$SRC" -o "$BIN" || exit 1

CUDA_VISIBLE_DEVICES="$DEV" timeout 3600 "$BIN" bench
