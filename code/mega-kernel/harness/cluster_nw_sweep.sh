#!/usr/bin/env bash
# P167-1 §92：把「每 warp 一次 arrive」（mbarw）形态对**每 block warp 数 NW=TH/32** 的依赖量出来
# （§91.8 明留的尾巴：mbarw/mbarw0 只在 T=256/NW=8 上测过）。
# 源码复用 src/cluster_concurrency.cu，唯一改动是 `TH` 变成编译期宏 `-DTH=<T>`；逐 T 编译一份二进制，
# 其余（拓扑/数据量/环形槽/唯一设计变量=arrive/等待/会合的粒度）与 §89–§91 逐字相同。
#   mbar   = 每线程 arrive（count=TH，零会合、零 fence）
#   mbarw  = 每 warp 一次 arrive（count=NW=TH/32）+ __syncwarp + lane0 显式 fence.acq_rel.cluster
#   mbarw0 = 同 mbarw，去掉 lane0 显式 cluster fence（诊断变量）
#   mbarb  = 每 block 一次 arrive（count=1）+ __syncthreads
#   ring1  = 单次 cluster.sync/round（对照天花板）
# 目标卡 H100 / sm90a。每配置单独进程（cluster launch failure 会 sticky 污染 context）；
# 计时项取 min-of-20、报 ok/20；可行性探针报 ok/N。
set -u
DEV=${DEV:-6}
SRC=${HOME}/proj/tech_record/code/mega-kernel/src/cluster_concurrency.cu
TS=(64 128 256 512 1024)

for T in "${TS[@]}"; do
  nvcc -O3 -arch=sm_90a -DTH=$T "$SRC" -o /tmp/ccnw_$T || exit 1
done
echo "built: ${TS[*]}"

run(){ local T=$1; shift; echo "\$ T=$T CUDA_VISIBLE_DEVICES=$DEV /tmp/ccnw_$T $*"; \
      CUDA_VISIBLE_DEVICES="$DEV" timeout 120 /tmp/ccnw_$T "$@" 2>&1; }
# 可行性探针：共享机器上 cluster launch 偶发 `unspecified launch failure`（§86/§88 记的重并发
# DSM 脆弱性），故对未跑满 ok 的配置**重试至多 3 次**、逐次打印（不掩盖瞬时失败，只取干净会话）。
probe(){ local T=$1; shift; local reps=$6; \
  for _try in 1 2 3; do \
    local out; out=$(CUDA_VISIBLE_DEVICES="$DEV" timeout 60 /tmp/ccnw_$T "$@" 2>&1 | tail -1); \
    echo "  $out"; \
    case "$out" in *"$reps/$reps ok"*) return 0;; esac; \
  done; }

SB=32   # 每缓冲区 32 KB
RR=64   # 计时 rounds

echo
echo "== A. 资源表：各 T 下 pull/push/ring/ring1/mbar/mbarb/mbarb0/mbarw/mbarw0 的 numRegs/localB/占用率（U=1/2/4/8/16）=="
for T in "${TS[@]}"; do echo "-- T=$T (NW=$((T/32))) --"; run $T A; done

echo
echo "== B.（稳定性）mbarw（探针 T）/ mbarw0（探针 J）扫 grid；对照 mbar（P）/ mbarb（Y）/ ring1（L），C=2 U=8 =="
for T in "${TS[@]}"; do
  echo "-- T=$T mbarw（T, U=8, 单发探针） --"
  for G in 2 16 132 264 528 1056; do probe $T T 2 $G $SB 8 1 $RR; done
  echo "-- T=$T mbarw0（J, U=8, 单发探针） --"
  for G in 2 16 132 264 528 1056; do probe $T J 2 $G $SB 8 1 $RR; done
  echo "-- T=$T mbar（P）/ mbarb（Y）/ ring1（L） 单发探针 --"
  for G in 132 264 528 1056; do probe $T P 2 $G $SB 8 1 $RR; done
  for G in 132 264 528 1056; do probe $T Y 2 $G $SB 8 1 $RR; done
  for G in 132 264 528 1056; do probe $T L 2 $G $SB 8 1 $RR; done
  echo "-- T=$T mbarw/mbarw0 重复 3 发，grid=264/528 --"
  for G in 264 528; do probe $T T 2 $G $SB 8 3 $RR; done
  for G in 264 528; do probe $T J 2 $G $SB 8 3 $RR; done
done

echo
echo "== C.（headline 一）固定每线程 ILP U=8：扫 T（NW），五载体吞吐（C=2 grid=264 min-of-20） =="
for C in 2 8; do
  echo "-- C=$C --"
  for T in "${TS[@]}"; do
    for c in V U B Q M; do run $T $c $C 264 $SB $RR 8; done
  done
done

echo
echo "== D.（headline 二）固定每 round 字节数 SEG=TH*U=2048（同时固定环深 S=4）：配 (T,U) 使 SEG 恒定 =="
echo "-- 对 (T,U) = (128,16) (256,8) (512,4) (1024,2)，C=2 grid=264 min-of-20 --"
for pair in "128 16" "256 8" "512 4" "1024 2"; do
  set -- $pair; T=$1; U=$2
  echo "-- T=$T U=$U SEG=$(($T*$U)) --"
  for c in V U B Q M; do run $T $c 2 264 $SB $RR $U; done
done

echo
echo "== E.（fence 归因）固定 U=8：mbarw（V, 带 fence）vs mbarw0（U, 无 fence）逐 T 连跑 6 次（grid=264） =="
for T in "${TS[@]}"; do
  echo "-- T=$T mbarw（V） --"; for i in 1 2 3 4 5 6; do run $T V 2 264 $SB $RR 8; done
  echo "-- T=$T mbarw0（U） --"; for i in 1 2 3 4 5 6; do run $T U 2 264 $SB $RR 8; done
done

echo
echo "== F.（正确性压力）逐 T：mbarw(T)/mbarw0(J)/mbar(P)/mbarb(Y) 在 C=2/4、grid=132/264/528、U=1/4/8/16 重复探针 reps=8 =="
for T in "${TS[@]}"; do
  echo "-- T=$T (NW=$((T/32))) --"
  for C in 2 4; do
    for G in 132 264 528; do
      for U in 1 4 8 16; do
        if [ $((T*U)) -gt $((SB*1024/4)) ]; then echo "  (skip T=$T U=$U: SEG=TH*U=$(($T*U)) > sb=$(($SB*1024/4)))"; continue; fi
        probe $T T $C $G $SB $U 8 $RR
        probe $T J $C $G $SB $U 8 $RR
        probe $T P $C $G $SB $U 8 $RR
        probe $T Y $C $G $SB $U 8 $RR
      done
    done
  done
done

echo
echo "== done =="
