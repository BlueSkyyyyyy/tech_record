#!/usr/bin/env bash
# P165-1 §90：cluster 跨 SM 交换——把 §89 的「每槽每线程一对 mbarrier（count=TH）」换成
# **每 block 一次 arrive（count=1）+ `__syncthreads` 包裹**的「省 arrive」形态（mbarb）。
# 语义/拓扑/数据量与 §89 `mbar` 逐字相同，唯一变量 = arrive/等待的粒度：
#   mbar   = 每线程写完后各自远端 arrive 一次（每 round 每 block 2·TH 次跨 SM arrive）；
#   mbarb  = 全 block 协作写 → __syncthreads → tid0 单点远端 arrive，消费端 tid0 单点等 → __syncthreads
#            （每 round 每 block 2 次跨 SM arrive，但多出 3 个 __syncthreads）；
#   mbarb0 = 同 mbarb，但单线程等待用 try_wait hint=0：诊断出 mbarb 的性能双模态来自
#            「CUTLASS 式 10ms suspend-hint 作用在单个等待线程上」，换成持续轮询后稳定。
# 目标卡 H100 / sm90a。每配置单独进程（一次 cluster launch failure 会 sticky 污染 CUDA context）；
# 计时项取 min-of-20、报 ok/20；边界/可行性用单发/重复发 probe。
set -u
DEV=${DEV:-6}
SRC=~/proj/tech_record/code/mega-kernel/src/cluster_concurrency.cu
BIN=/tmp/cluster_mbar_bulk
nvcc -O3 -arch=sm_90a "$SRC" -o "$BIN" || exit 1

run(){ echo "\$ CUDA_VISIBLE_DEVICES=$DEV $BIN $*"; CUDA_VISIBLE_DEVICES="$DEV" timeout 120 "$BIN" "$@" 2>&1; }
probe(){ echo -n "  "; CUDA_VISIBLE_DEVICES="$DEV" timeout 60 "$BIN" "$@" 2>&1 | tail -1; }

SB=32   # 每缓冲区 32 KB
RR=64   # 计时 rounds

echo "== A. 资源表：pull/push/ring/ring1/mbar/mbarb/mbarb0 各 U 的 numRegs / localB / 占用率（T=256）=="
run A

echo
echo "== B.（Q1 稳定性）mbarb0（X）/ mbarb（Y）/ mbar（P）/ ring1（L）的 grid 标度 =="
echo "-- mbarb0（X, U=8, rounds=64, 单发探针） --"
for G in 2 16 32 64 132 264 528 792 1056; do probe X 2 $G $SB 8 1 64; done
echo "-- mbarb0 重复 3 发（X），grid=264/528/1056 --"
for G in 264 528 1056; do probe X 2 $G $SB 8 3 64; done
echo "-- mbarb（Y, 带 suspend-hint, 单发探针） --"
for G in 2 16 132 264 528 1056; do probe Y 2 $G $SB 8 1 64; done
echo "-- mbar（P, 对照, 单发） --"
for G in 2 16 132 264 528 1056; do probe P 2 $G $SB 8 1 64; done
echo "-- ring1（L, 对照, 单发） --"
for G in 2 16 132 264 528 1056; do probe L 2 $G $SB 8 1 64; done

echo
echo "== C.（Q2 headline）mbarb0 vs mbarb vs mbar vs ring1 vs push 的吞吐：C=2/8，grid=264，扫 U（min-of-20）=="
for C in 2 8; do
  echo "-- C=$C mbarb0（Z, hint=0） --"
  for U in 1 2 4 8 16; do run Z $C 264 $SB $RR $U; done
  echo "-- C=$C mbarb（Q, 带 hint） --"
  for U in 1 2 4 8 16; do run Q $C 264 $SB $RR $U; done
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
  for U in 8 16; do run Z 2 $G $SB $RR $U; done
  for U in 8 16; do run B 2 $G $SB $RR $U; done
  for U in 8 16; do run M 2 $G $SB $RR $U; done
done

echo
echo "== E.（Q4）环深 S 作为独立旋钮：mbarb0 C=2 grid=264 U=1/U=8，S=1/2/4/8（min-of-20）=="
for U in 1 8; do
  for S in 1 2 4 8; do run Z 2 264 $SB $RR $U $S; done
done

echo
echo "== F.（headline 二）性能双模态：同一配置连跑，mbarb（带 hint）vs mbarb0（hint=0）：C=2 grid=264 U=8 =="
for i in 1 2 3 4 5 6; do run Q 2 264 $SB $RR 8; done
for i in 1 2 3 4 5 6; do run Z 2 264 $SB $RR 8; done
