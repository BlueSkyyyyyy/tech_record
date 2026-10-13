#!/usr/bin/env bash
# P164-1 §89：cluster 跨 SM 交换——把 §88.5 的「完全无 cluster.sync 的纯 volatile 每槽
# release/done（ring）不可用」换成 **cluster 作用域 mbarrier** 的每槽 release/done（mbar）：
# 生产者用 `mbarrier.arrive.release.cluster.shared::cluster` 远端到达消费者的每槽 mbarrier、
# 消费者本地 `mbarrier.try_wait.parity` 等待。目标卡 H100 / sm90a。
# 关键实现坑（本轮踩到并修正，见 §89）：
#   (1) `.arrive.shared::cluster` 的默认 sem/scope 不是 cluster——必须显式
#       `.release.cluster`，否则数据可见性不成立（症状：所有 U 都 maxerr>0 但 launch 成功）；
#   (2) 循环末尾必须补一个 teardown `cluster.sync()`，否则先跑完的 block 提前退出释放 SMEM、
#       邻块仍在给它的槽远端 arrive → `unspecified launch failure`（症状：大 grid/大 U 必挂）；
#   (3) `try_wait` 用 CUTLASS 同款 suspend-time-hint，避免紧旋把远端 arrive 通路饿死。
# 每配置单独进程（一次 cluster launch failure 会 sticky 污染 CUDA context）；
# 计时项取 min-of-20、报 ok/20；边界/可行性用 probe（每配置 1 发/重复发）。
set -u
DEV=${DEV:-6}
SRC=~/proj/tech_record/code/mega-kernel/src/cluster_concurrency.cu
BIN=/tmp/cluster_mbar
nvcc -O3 -arch=sm_90a "$SRC" -o "$BIN" || exit 1

run(){ echo "\$ CUDA_VISIBLE_DEVICES=$DEV $BIN $*"; CUDA_VISIBLE_DEVICES="$DEV" timeout 120 "$BIN" "$@" 2>&1; }
probe(){ echo -n "  "; CUDA_VISIBLE_DEVICES="$DEV" timeout 60 "$BIN" "$@" 2>&1 | tail -1; }

SB=32   # 每缓冲区 32 KB
RR=64   # 计时 rounds

echo "== A. 资源表：pull/push/ring/ring1/mbar 各 U 的 numRegs / localB / 占用率（T=256）=="
run A

echo
echo "== B.（Q1 稳定性）mbar（每槽 mbarrier）对 ring1（每 round 单 cluster.sync）的 grid 标度 =="
echo "-- mbar（P，U=8, rounds=64，单发探针） --"
for G in 2 16 32 64 132 264 528 792 1056; do probe P 2 $G $SB 8 1 64; done
echo "-- ring1（L，同 grid，单发探针，对照） --"
for G in 2 16 32 64 132 264 528 792 1056; do probe L 2 $G $SB 8 1 64; done
echo "-- mbar 重复 3 发（P），grid=264/528/1056 --"
for G in 264 528 1056; do probe P 2 $G $SB 8 3 64; done
echo "-- ring（O，纯 volatile 握手，对照：§88.5 已知不可用） --"
for G in 2 16 264; do probe O 2 $G $SB 8 3 64; done

echo
echo "== C.（Q2 headline）mbar vs ring1 vs push 的吞吐：C=2/8，grid=264，扫 U（min-of-20）=="
for C in 2 8; do
  echo "-- C=$C mbar（B） --"
  for U in 1 2 4 8 16; do run B $C 264 $SB $RR $U; done
  echo "-- C=$C ring1（M） --"
  for U in 1 2 4 8 16; do run M $C 264 $SB $RR $U; done
  echo "-- C=$C push（W） --"
  for U in 1 2 4 8 16; do run W $C 264 $SB $RR $U; done
done

echo
echo "== D.（Q3）大 grid：C=2，grid=528/1056，U=8/16（min-of-20）=="
for G in 528 1056; do
  for U in 8 16; do run B 2 $G $SB $RR $U; done
  for U in 8 16; do run M 2 $G $SB $RR $U; done
done

echo
echo "== E.（Q4）环深 S 作为独立旋钮：mbar C=2 grid=264 U=1/U=8，S=1/2/4/8（min-of-20）=="
for U in 1 8; do
  for S in 1 2 4 8; do run B 2 264 $SB $RR $U $S; done
done
