#!/usr/bin/env bash
# P45-1 复现：**事件驱动的「就地入队」如何消除有界 ready 池的死锁**——把 P44-1（docs/04 §33）的
# 「依赖门控 gated」从**调度器扫描**改成 MPK §5.1 的真机制：每个 task 一个事件计数器（未完成前驱数），
# 由完成该前驱的 worker 直接把后继 push 进 ready 队列（计数归零即入队），调度器不参与准入。
# 于是 ready 队列只装就绪 task ⇒ B(π)≡0 ⇒ CAP*=G*=1，不需要 warmup、不需要拓扑序。
# 接续 DeepGEMM Mega MoE `get_num_l1_warmup_waves()`/`get_num_max_live_pool_blocks()`
# （scheduler/mega_moe.cuh:15-45/47-80）、MPK EventDesc + trigger_event（docs/06 §2/§4、docs/08 §7）。
# 目标卡 H100 / sm90，用 DEV 选择空闲 device（默认 3）。
#
#   bench : §A 三机制正确性 / §B 死锁边界 / §C 更大 N 标度 / §D 独立扫 G / §E 扇入敏感。
#   run   : 单次（mode shape CAP G L W0 FIN span scramble iters）。mode: scan|topo|event|raw。
#
# 安全：设备侧无进度守门狗（clock64）检测死锁并置 flag，kernel 正常返回（host 不挂）。
set -u
DEV=${DEV:-3}
SRC=~/proj/tech_record/code/mega-kernel/src/event_driven_pool.cu
BIN=/tmp/event_driven_pool
nvcc -O3 -arch=sm_90 "$SRC" -o "$BIN" || exit 1

echo "===== P45-1 event_driven_pool bench ====="
CUDA_VISIBLE_DEVICES="$DEV" timeout 900 "$BIN" bench 64
echo

echo "===== §B 精扫：raw(reg,scramble=1) B=16 —— CAP=1 死锁、CAP=17 才活；event CAP=1/G=1 即活 ====="
for cfg in "raw 1" "raw 17" "event 1"; do
  set -- $cfg
  CUDA_VISIBLE_DEVICES="$DEV" timeout 60 "$BIN" run "$1" reg "$2" 64 4 16 2 1 1 20000 | sed 's/^/  /'
done
echo

echo "===== §D 精扫：shrink(W0=32,scramble=1) raw B=29 —— 独立扫 G：G=28 死锁、G=30 活（须 G>B） ====="
for G in 28 29 30; do
  CUDA_VISIBLE_DEVICES="$DEV" timeout 60 "$BIN" run raw shrink 256 "$G" 4 32 2 1 1 20000 | sed 's/^/  /'
done
echo "--- 同图 scan(CAP=1)/event(CAP=1,G=1) 恒活 ---"
CUDA_VISIBLE_DEVICES="$DEV" timeout 60 "$BIN" run scan  shrink 1 64 4 32 2 1 1 20000 | sed 's/^/  /'
CUDA_VISIBLE_DEVICES="$DEV" timeout 60 "$BIN" run event shrink 1 1  4 32 2 1 1 20000 | sed 's/^/  /'
echo
