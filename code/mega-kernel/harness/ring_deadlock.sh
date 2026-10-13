#!/usr/bin/env bash
# P42-1 复现：有界 ready 池（ring/pool）+ warmup wave 的**防死锁**与**容量下界**。
# 模型对应 DeepGEMM Mega MoE scheduler 的 `get_num_l1_warmup_waves`
# （`scheduler/mega_moe.cuh:15-45`）与 `get_num_max_live_pool_blocks`（`:47-80`，ring 容量闭式上界）。
# 目标卡 H100 / sm90，用 DEV 选择空闲 device（默认 3）。
#
#   bench : §A 死锁边界 / §B warmup 宽度下界 / §C 吞吐 vs 池容量 / §D worker 数扫描。
#   run   : 单次（order CAP G M Kp Kc W iters），用于 §E 依赖宽度敏感性等补充点。
#
# 安全：设备侧无进度守门狗（clock64）检测死锁并置 flag，kernel 正常返回（host 不挂）。
set -u
DEV=${DEV:-3}
SRC=~/proj/tech_record/code/mega-kernel/src/ring_deadlock.cu
BIN=/tmp/ring_deadlock
nvcc -O3 -arch=sm_90 "$SRC" -o "$BIN" || exit 1

echo "===== P42-1 ring_deadlock bench ====="
CUDA_VISIBLE_DEVICES="$DEV" timeout 300 "$BIN" bench 30
echo

echo "===== §E 依赖宽度 Kc 扫描（naive 死锁阈值应 = Kc；M=16, Kp=8）====="
for KC in 2 4 8; do
  for CAP in 2 4 8 9; do
    CUDA_VISIBLE_DEVICES="$DEV" timeout 60 "$BIN" run naive "$CAP" 64 16 8 "$KC" 0 20000 | sed 's/^/  /'
  done
  echo
done

echo "===== §E2 同 Kc 下 warmup 全部脱困（W=M=16）====="
for KC in 2 4 8; do
  for CAP in 2 4 8; do
    CUDA_VISIBLE_DEVICES="$DEV" timeout 60 "$BIN" run warmup "$CAP" 64 16 8 "$KC" 16 20000 | sed 's/^/  /'
  done
  echo
done
