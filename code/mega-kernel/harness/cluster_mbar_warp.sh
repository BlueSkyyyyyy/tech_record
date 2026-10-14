#!/usr/bin/env bash
# P166-1 §91：cluster 跨 SM 交换——把 §90 的「每 block 一次 arrive（count=1）+ __syncthreads」与
# §89 的「每线程 arrive（count=TH）」之间，补上**每 warp 一次 arrive（count=NW=blockDim/32）+ __syncwarp**
# 的中间形态（mbarw），并加一条「无显式 cluster fence」的诊断变体（mbarw0）量清每-warp 发布慢在哪。
# 语义/拓扑/数据量与 §89 `mbar`/§90 `mbarb` 逐字相同，唯一设计变量 = arrive/等待/会合的粒度：
#   mbar   = 每线程写完后各自远端 arrive 一次（count=TH=256，每 round 每 block 512 次跨 SM arrive，0 会合）；
#   mbarw  = 每 warp 写完 __syncwarp → lane0 `fence.acq_rel.cluster` + 远端 arrive（count=NW=8，
#            每 round 每 block 16 次跨 SM arrive + 16 条 cluster fence，会合=__syncwarp）；
#   mbarw0 = 同 mbarw，但**去掉 lane0 的显式 cluster fence**，依赖 `__syncwarp` + arrive 自带 release；
#   mbarb  = 全 block 协作写 → __syncthreads → tid0 单点远端 arrive（count=1，每 round 2 次跨 SM arrive
#            + 2 条 cluster fence，会合=__syncthreads）。
# 目标卡 H100 / sm90a。每配置单独进程（一次 cluster launch failure 会 sticky 污染 CUDA context）；
# 计时项取 min-of-20、报 ok/20；边界/可行性用单发/重复发 probe。
set -u
DEV=${DEV:-6}
SRC=${HOME}/proj/tech_record/code/mega-kernel/src/cluster_concurrency.cu
BIN=/tmp/cluster_mbar_warp
nvcc -O3 -arch=sm_90a "$SRC" -o "$BIN" || exit 1

run(){ echo "\$ CUDA_VISIBLE_DEVICES=$DEV $BIN $*"; CUDA_VISIBLE_DEVICES="$DEV" timeout 120 "$BIN" "$@" 2>&1; }
probe(){ echo -n "  "; CUDA_VISIBLE_DEVICES="$DEV" timeout 60 "$BIN" "$@" 2>&1 | tail -1; }

SB=32   # 每缓冲区 32 KB
RR=64   # 计时 rounds

echo "== A. 资源表：pull/push/ring/ring1/mbar/mbarb/mbarb0/mbarw/mbarw0 各 U 的 numRegs / localB / 占用率（T=256）=="
run A

echo
echo "== B.（Q1 稳定性）mbarw（T）/ mbarw0（J）的 grid 标度；对照 mbar（P）/ ring1（L）=="
echo "-- mbarw（T, U=8, rounds=64, 单发探针） --"
for G in 2 16 32 64 132 264 528 792 1056; do probe T 2 $G $SB 8 1 64; done
echo "-- mbarw 重复 3 发（T），grid=264/528/1056 --"
for G in 264 528 1056; do probe T 2 $G $SB 8 3 64; done
echo "-- mbarw0（J, U=8, rounds=64, 单发探针） --"
for G in 2 16 32 64 132 264 528 792 1056; do probe J 2 $G $SB 8 1 64; done
echo "-- mbarw0 重复 3 发（J），grid=264/528/1056 --"
for G in 264 528 1056; do probe J 2 $G $SB 8 3 64; done
echo "-- mbar（P, 对照, 单发） --"
for G in 2 16 132 264 528 1056; do probe P 2 $G $SB 8 1 64; done
echo "-- ring1（L, 对照, 单发） --"
for G in 2 16 132 264 528 1056; do probe L 2 $G $SB 8 1 64; done

echo
echo "== C.（headline）mbarw0 vs mbarw vs mbar vs mbarb vs ring1 的吞吐：C=2/8，grid=264，扫 U（min-of-20）=="
for C in 2 8; do
  echo "-- C=$C mbar（B, count=TH 每线程 arrive） --"
  for U in 1 2 4 8 16; do run B $C 264 $SB $RR $U; done
  echo "-- C=$C mbarw0（U, 每 warp 一次 arrive, 无显式 fence） --"
  for U in 1 2 4 8 16; do run U $C 264 $SB $RR $U; done
  echo "-- C=$C mbarw（V, 每 warp 一次 arrive + 显式 cluster fence） --"
  for U in 1 2 4 8 16; do run V $C 264 $SB $RR $U; done
  echo "-- C=$C mbarb（Q, 每 block 一次 arrive） --"
  for U in 1 2 4 8 16; do run Q $C 264 $SB $RR $U; done
  echo "-- C=$C ring1（M, 单次 cluster.sync/round） --"
  for U in 1 2 4 8 16; do run M $C 264 $SB $RR $U; done
done

echo
echo "== D. 大 grid：C=2，grid=528/1056，U=8/16（min-of-20）=="
for G in 528 1056; do
  for U in 8 16; do run U 2 $G $SB $RR $U; done
  for U in 8 16; do run V 2 $G $SB $RR $U; done
  for U in 8 16; do run B 2 $G $SB $RR $U; done
  for U in 8 16; do run M 2 $G $SB $RR $U; done
done

echo
echo "== E. 环深 S 作为独立旋钮：mbarw0 C=2 grid=264 U=1/U=8，S=1/2/4/8（min-of-20）=="
for U in 1 8; do
  for S in 1 2 4 8; do run U 2 264 $SB $RR $U $S; done
done

echo
echo "== F.（fence 归因）同一配置连跑 6 次：mbarw（V, 带 cluster fence）vs mbarw0（U, 无 fence）：C=2 grid=264 U=8 =="
for i in 1 2 3 4 5 6; do run V 2 264 $SB $RR 8; done
for i in 1 2 3 4 5 6; do run U 2 264 $SB $RR 8; done

echo
echo "== G.（正确性压力）mbarw0（J）重复探针：C=2/4，grid=132/264/528，U=1/4/8/16，reps=8 =="
for C in 2 4; do
  for G in 132 264 528; do
    for U in 1 4 8 16; do probe J $C $G $SB $U 8 $RR; done
  done
done
