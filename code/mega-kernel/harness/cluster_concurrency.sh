#!/usr/bin/env bash
# P162-1 §87：cluster 跨 SM 交换的「远端读 vs 远端写+本地读」（目标卡 H100 / sm90a）。
# 每个配置单独一次进程调用——一次 cluster launch failure 会污染 CUDA context（sticky）。
# 计时项 try_run 重试（最多 5 次）；边界项 probe 每配置起 3 个独立进程各 launch 1 次、报 ok/3。
set -u
DEV=${DEV:-3}
SRC=${HOME}/proj/tech_record/code/mega-kernel/src/cluster_concurrency.cu
BIN=/tmp/cluster_concurrency
nvcc -O3 -arch=sm_90a "$SRC" -o "$BIN" || exit 1

run(){ echo "\$ CUDA_VISIBLE_DEVICES=$DEV $BIN $*"; CUDA_VISIBLE_DEVICES="$DEV" timeout 120 "$BIN" "$@"; }

try_run(){
  local i out
  for ((i=1;i<=5;i++)); do
    out=$(CUDA_VISIBLE_DEVICES="$DEV" timeout 120 "$BIN" "$@" 2>&1)
    if [ $? -eq 0 ]; then
      echo "\$ CUDA_VISIBLE_DEVICES=$DEV $BIN $*   (try $i)"
      echo "$out"
      return 0
    fi
  done
  echo "\$ CUDA_VISIBLE_DEVICES=$DEV $BIN $*   !! ALLFAIL x5"
  echo "$out"
  return 1
}

probe(){
  local i ok=0 out last=""
  for ((i=1;i<=3;i++)); do
    out=$(CUDA_VISIBLE_DEVICES="$DEV" timeout 120 "$BIN" "$@" 2>&1)
    if [ $? -eq 0 ]; then ok=$((ok+1)); last="$out"; fi
  done
  echo "$(printf '%s\n' "$last" | grep -E '^[RWS] ' | head -1)   -> ok $ok/3"
}

SB=32   # 每缓冲区 32 KB（snd+rcv=64 KB）
RR=64   # 计时 rounds

echo "== A. 资源表：pull/push/pulls 各 U 的 numRegs / localB / 占用率（T=256，SMEM=64KB）=="
run A

echo
echo "== B.（Q1）单次 launch 可行性 vs 远端读/写路径与轮数：C=2, grid=396/528, U=8, rounds=4 vs 64 =="
echo "-- 远端读 pull（F） --"
for G in 396 528; do for R in 4 64; do probe F 2 $G $SB 8 1 $R; done; done
echo "-- 远端读·外层串行 pulls（K） --"
for G in 396 528; do for R in 4 64; do probe K 2 $G $SB 8 1 $R; done; done
echo "-- 远端写 push（G） --"
for G in 396 528; do for R in 4 64; do probe G 2 $G $SB 8 1 $R; done; done

echo
echo "== C.（Q2 headline）同一进程内重复 launch 20 次的可靠性：C=2, grid=264, rounds=$RR =="
echo "-- 远端读 pull：U=1（编译器可 unroll 成多笔在飞）/ U=8 / U=16 --"
for U in 1 8 16; do try_run R 2 264 $SB $RR $U; done
echo "-- 远端读·外层串行 pulls：U=1 / U=8 / U=16 --"
for U in 1 8 16; do try_run S 2 264 $SB $RR $U; done
echo "-- 远端写 push：U=1 / U=8 / U=16 --"
for U in 1 8 16; do try_run W 2 264 $SB $RR $U; done

echo
echo "== D.（Q3）远端写 push 吞吐随 U 标度：C=2, grid=264 / 528, rounds=$RR =="
for U in 1 2 4 8 16; do try_run W 2 264 $SB $RR $U; done
for U in 1 2 4 8 16; do try_run W 2 528 $SB $RR $U; done

echo
echo "== E. 更大 cluster：C=4，grid 132/264，U=8，三条路（单次 launch 探针 ok/3）=="
for G in 132 264; do probe F 4 $G $SB 8 1 64; done
for G in 132 264; do probe K 4 $G $SB 8 1 64; done
for G in 132 264; do probe G 4 $G $SB 8 1 64; done

echo
echo "== F. 轻通道对照：U=1 扫 grid（读/写，rounds=4 单次 launch）=="
for G in 264 528 1056; do probe F 2 $G $SB 1 1 4; done
for G in 264 528 1056; do probe G 2 $G $SB 1 1 4; done
