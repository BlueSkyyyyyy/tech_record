#!/usr/bin/env bash
# P172-1 §97：cluster 跨 SM 交换——把 §89 `mbar` 系列的「ready/done = 软件 mbarrier arrive」
# 换成 **TMA 事务计数（expect_tx）表达每槽 done**（mtex）。
# 语义/拓扑/数据量/环深与 §89 `mbar` 逐字相同（每 block 把 snd 的 SEG 段推给 (rank+1)%C 的
# rcv 第 j=r%S 槽、读回后 ack 给上游），唯一变量 = 「数据交付 + done 信号」如何表达：
#   mbar  = 生产者全 block 每线程 `st.shared::cluster` 写数据，再每线程远端 arrive 对端 readyBar
#           （每 round 每 block 2·TH 次跨 SM arrive；done = 软件 arrive）；
#   mtex  = 生产者单线程发一笔 `cp.async.bulk.shared::cluster.shared::cta`（TMA 引擎搬 SEG*4 字节），
#           完成按 **TMA tx count** 落到对端 readyBar；登记用远端 `mbarrier.arrive.expect_tx`
#           （生产者侧、与 bulk 同程序序，消费者无需预 arm），done = 硬件记账的实际字节交付。
#           ack（背压）方向与 mbar 逐字相同（count=TH 每线程远端 arrive），以便单变量隔离。
# 目标卡 H100 / sm90a。每配置单独进程（一次 cluster launch failure 会 sticky 污染 CUDA context）；
# 计时项取 min-of-20、报 ok/20；边界/可行性用单发/重复发 probe。
set -u
DEV=${DEV:-6}
SRC=${HOME}/proj/tech_record/code/mega-kernel/src/cluster_concurrency.cu
BIN=/tmp/cluster_mtex
nvcc -O3 -arch=sm_90a "$SRC" -o "$BIN" || exit 1

run(){ echo "\$ CUDA_VISIBLE_DEVICES=$DEV $BIN $*"; CUDA_VISIBLE_DEVICES="$DEV" timeout 120 "$BIN" "$@" 2>&1; }
probe(){ echo -n "  "; CUDA_VISIBLE_DEVICES="$DEV" timeout 60 "$BIN" "$@" 2>&1 | tail -1; }

SB=32   # 每缓冲区 32 KB
RR=64   # 计时 rounds

echo "== A. 资源表：pull/push/ring/ring1/mbar/mbarb/mbarb0/mbarw/mbarw0/mtex 各 U 的 numRegs / localB / 占用率（T=256）=="
run A

echo
echo "== B.（Q1 稳定性）mtex（D）/ mbar（P）/ ring1（L）的 grid 标度（单发探针，U=8，rounds=64）=="
echo "-- mtex（D, 单发探针） --"
for G in 2 16 32 64 132 264 528 792 1056; do probe E 2 $G $SB 8 1 64; done
echo "-- mtex（D）重复 3 发，grid=264/528/1056 --"
for G in 264 528 1056; do probe E 2 $G $SB 8 3 64; done
echo "-- mbar（P, 对照, 单发） --"
for G in 2 16 132 264 528 1056; do probe P 2 $G $SB 8 1 64; done
echo "-- ring1（L, 对照, 单发） --"
for G in 2 16 132 264 528 1056; do probe L 2 $G $SB 8 1 64; done

echo
echo "== C.（Q2 headline）吞吐：C=2/8，grid=264，扫 U（min-of-20，rounds=64）=="
for C in 2 8; do
  echo "-- C=$C mtex（D） --"
  for U in 1 2 4 8 16; do run D $C 264 $SB $RR $U; done
  echo "-- C=$C mbar（B） --"
  for U in 1 2 4 8 16; do run B $C 264 $SB $RR $U; done
  echo "-- C=$C ring1（M） --"
  for U in 1 2 4 8 16; do run M $C 264 $SB $RR $U; done
done

echo
echo "== D.（Q3）大 grid：C=2，grid=528/1056，U=8/16（min-of-20）=="
for G in 528 1056; do
  for U in 8 16; do run D 2 $G $SB $RR $U; done
  for U in 8 16; do run B 2 $G $SB $RR $U; done
  for U in 8 16; do run M 2 $G $SB $RR $U; done
done

echo
echo "== E.（Q4）环深 S 作为独立旋钮：mtex C=2 grid=264 U=1/U=8，S=1/2/4/8（min-of-20）=="
for U in 1 8; do
  for S in 1 2 4 8; do run D 2 264 $SB $RR $U $S; done
done

echo
echo "== F.（Q5 确定性）同一配置连跑 6 次：mtex C=2 grid=264 U=8（min-of-20）=="
for i in 1 2 3 4 5 6; do run D 2 264 $SB $RR 8; done
