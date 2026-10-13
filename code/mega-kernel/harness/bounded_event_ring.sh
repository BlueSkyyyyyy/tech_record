#!/usr/bin/env bash
# P46-1 复现：**事件驱动的 ready 队列从「无界」变「有界环形缓冲」后会发生什么**——补齐 §34.9
# 明确留的尾巴「event 的 ready 队列按 N+4 开数组、无界；真实实现要环形缓冲 + 容量上界」。
# 真实锚点：MPK 固定 per_worker_queue_len=8192 / per_sched_queue_len=4096（persistent_kernel.cuh:1790-1791，
# worker 侧靠 assert `:1027` 不溢出）；DeepGEMM Mega MoE get_num_max_live_pool_blocks()
# （scheduler/mega_moe.cuh:47-80）闭式给 ring 容量上界 + get_num_l1_warmup_waves()（`:15-45`）防死锁。
# 机制：单常驻 kernel，block0 播种源任务、其余 G 为 worker；每 task 一个 dep_left 计数器，归零即入队。
#   unb（=§34 event，无界） vs bnd（有界 CAP + 阻塞式背压：仅 tail-head<CAP 才 CAS 预留槽）。
# 目标卡 H100 / sm90，用 DEV 选择空闲 device（默认 3）。
#
#   bench : §A 容量 vs W_max / §A2 FIN,span / §B 独立扫 G / §C 容量→吞吐 / §D 正确性 / §E 规模标度。
#   run   : 单次（mode shape CAP G L W0 FIN span scramble iters）。mode: unb|bnd。
#
# 安全：设备侧无进度守门狗（clock64）检测死锁并置 flag，kernel 正常返回（host 不挂）。
set -u
DEV=${DEV:-3}
SRC=~/proj/tech_record/code/mega-kernel/src/bounded_event_ring.cu
BIN=/tmp/bounded_event_ring
nvcc -O3 -arch=sm_90 "$SRC" -o "$BIN" || exit 1

echo "===== P46-1 bounded_event_ring bench ====="
CUDA_VISIBLE_DEVICES="$DEV" timeout 1800 "$BIN" bench 64
echo

echo "===== 精扫 1：reg(L=5,W0=24,FIN=1,G=64) bnd —— 边界在小 CAP（CAP=1 死锁频发、CAP=2 稳） ====="
for cap in 1 2 3; do
  CUDA_VISIBLE_DEVICES="$DEV" timeout 60 "$BIN" run bnd reg "$cap" 64 5 24 1 1 0 20000 | sed 's/^/  /'
done
echo "--- 同图 unb（无界）恒活 ---"
CUDA_VISIBLE_DEVICES="$DEV" timeout 60 "$BIN" run unb reg 1 64 5 24 1 1 0 20000 | sed 's/^/  /'
echo

echo "===== 精扫 2：shrink(L=5,W0=40,FIN=2,G=1) bnd —— 单 worker 时边界贴近 W_max=40 ====="
for cap in 38 39 40 41; do
  CUDA_VISIBLE_DEVICES="$DEV" timeout 60 "$BIN" run bnd shrink "$cap" 1 5 40 2 1 0 20000 | sed 's/^/  /'
done
echo

echo "===== 精扫 3：同图同 CAP 下 G=1 vs G=64 —— 容量需求随 G 下降（CAP_safe+G ≈ W_max+1） ====="
for g in 1 16 64; do
  CUDA_VISIBLE_DEVICES="$DEV" timeout 60 "$BIN" run bnd shrink 26 "$g" 5 40 2 1 0 20000 | sed 's/^/  /'
done
echo
