#!/usr/bin/env bash
# P163-1 §88：cluster 跨 SM 交换——把「每 round 双 cluster.sync 的远端写 push」换成
# 「环形槽 + 单次 cluster.sync/round（ring1）」与「完全无 cluster.sync 的每槽
# release/done（ring）」。目标卡 H100 / sm90a。
# 每配置单独进程（一次 cluster launch failure 会 sticky 污染 CUDA context）；
# 计时项取 min-of-20、报 ok/20；边界/可行性用 probe（每配置 1 发）。
set -u
DEV=${DEV:-1}
SRC=~/proj/tech_record/code/mega-kernel/src/cluster_concurrency.cu
BIN=/tmp/cluster_ring
nvcc -O3 -arch=sm_90a "$SRC" -o "$BIN" || exit 1

run(){ echo "\$ CUDA_VISIBLE_DEVICES=$DEV $BIN $*"; CUDA_VISIBLE_DEVICES="$DEV" timeout 120 "$BIN" "$@" 2>&1; }
probe(){ echo -n "  "; CUDA_VISIBLE_DEVICES="$DEV" timeout 60 "$BIN" "$@" 2>&1 | tail -1; }

SB=32   # 每缓冲区 32 KB（报文字节量 sb*4=128KB？高速缓存：snd+rcv=64 KB）
RR=64   # 计时 rounds

echo "== A. 资源表：pull/push/pulls/ring/ring1 各 U 的 numRegs / localB / 占用率（T=256）=="
run A

echo
echo "== B.（Q1 稳定性）ring1（单次 sync/round）对 push（双次 sync/round）的 grid 标度 =="
echo "-- ring1（L）：grid 2..1056，U=8, rounds=64（单发探针） --"
for G in 2 16 32 64 132 264 528 1056; do probe L 2 $G $SB 8 1 64; done
echo "-- push（G）：同 grid（单发探针，对照） --"
for G in 2 16 32 64 132 264 528 1056; do probe G 2 $G $SB 8 1 64; done

echo
echo "== C.（Q2 headline）ring1 vs push 的吞吐：C=2/4/8，grid=264，扫 U（min-of-20）=="
for C in 2 4 8; do
  echo "-- C=$C push（W） --"
  for U in 1 2 4 8 16; do run W $C 264 $SB $RR $U; done
  echo "-- C=$C ring1（M） --"
  for U in 1 2 4 8 16; do run M $C 264 $SB $RR $U; done
done

echo
echo "== D.（Q3 headline 二）push vs ring1 在大 grid：C=2，grid=528，扫 U（min-of-20）=="
for U in 1 8 16; do run W 2 528 $SB $RR $U; done
for U in 1 8 16; do run M 2 528 $SB $RR $U; done

echo
echo "== E.（Q4）完全无 cluster.sync 的 ring（每槽 release/done，N/O）：可靠性 =="
echo "-- 单发探针（O）：grid 2..48，U=8, rounds=64 --"
for G in 2 16 24 32 48; do probe O 2 $G $SB 8 1 64; done
echo "-- 重复 20 发（N）：grid=2/8/16，U=8, rounds=64 --"
for G in 2 8 16; do run N 2 $G $SB $RR 8; done
