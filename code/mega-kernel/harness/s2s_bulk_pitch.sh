#!/usr/bin/env bash
# P179-1 §104：S2S bulk 的「源/目的 pitch 组合」+「整段连搬 vs 逐行拆搬」。
# §103 只测了「源带行距 spitch、目的打包 dpitch=cols」。本节把**目的行距 dpitch** 也变成变量
# （源打包→目的带距 S0D1、两端都带距 S1D1），并加关键对照 **full**：tile 就住在一条带行距的缓冲
# 里时，**把 padding 一起当一整段连搬（1 笔、payload=rows*spitch）** vs **只搬有效列逐行拆搬
# （rows 笔、payload=rows*cols）**——同一份有效数据两种搬法。slotpitch = full?spitch:dpitch。
# 目标卡 H100 / sm90a。每配置单独进程（cluster launch failure 会 sticky 污染 CUDA context）。
set -u
if [ -z "${DEV:-}" ]; then
  DEV=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
        | sort -t, -k2 -n | head -1 | cut -d, -f1)
fi
SRC=${HOME}/proj/tech_record/code/mega-kernel/src/s2s_bulk_nd.cu
BIN=/tmp/s2s_bulk_nd
nvcc -O3 -arch=sm_90a "$SRC" -o "$BIN" || exit 1

run(){ echo "\$ CUDA_VISIBLE_DEVICES=$DEV $BIN $*"; CUDA_VISIBLE_DEVICES="$DEV" timeout 180 "$BIN" "$@" 2>&1; }
probe(){ echo -n "  "; CUDA_VISIBLE_DEVICES="$DEV" timeout 60 "$BIN" "$@" 2>&1 | tail -1; }

COLS=256
GAP=64
PITCH=$((COLS+GAP))
ROUNDS=64
C=2
GRID=132

echo "# §104 S2S bulk pitch 组合（DEV=$DEV, TH=256, SLOTS=4, cols=$COLS gap=$GAP C=$C grid=$GRID rounds=$ROUNDS）"
echo

echo "== A. 资源表（numRegs / 占用率 / SMEM；X 内核）=="
run Y
echo

echo "== B.（headline 之一）源/目的 pitch 组合矩阵（rows=16）=="
echo "-- p1  = 源打包→目的打包，1 笔（天花板） --"
run X 2 132 16 $COLS $COLS $COLS 1 $ROUNDS 0
echo "-- r_pd = 源带距→目的打包，rows 笔（§103 形态） --"
run X 2 132 16 $COLS $PITCH $COLS 1 $ROUNDS 0
echo "-- r_dp = 源打包→目的带距，rows 笔（新） --"
run X 2 132 16 $COLS $COLS $PITCH 1 $ROUNDS 0
echo "-- r_dd = 两端带距，rows 笔（新） --"
run X 2 132 16 $COLS $PITCH $PITCH 1 $ROUNDS 0
echo "-- f_dd = 两端带距，整段连搬（含 padding），1 笔（新 headline） --"
run X 2 132 16 $COLS $PITCH $PITCH 1 $ROUNDS 1
echo

echo "== C.（headline 之二）扫 gap：逐行拆搬 r_dd vs 整段连搬 f_dd（rows=16）=="
for G in 0 32 64 128 256; do
  run X 2 132 16 $COLS $((COLS+G)) $((COLS+G)) 1 $ROUNDS 0
done
for G in 32 64 128 256; do
  run X 2 132 16 $COLS $((COLS+G)) $((COLS+G)) 1 $ROUNDS 1
done
echo

echo "== D. 扫 rows：逐行拆搬 r_dd vs 整段连搬 f_dd（gap=$GAP）=="
for R in 2 4 8 16 32; do run X 2 132 $R $COLS $PITCH $PITCH 1 $ROUNDS 0; done
for R in 2 4 8 16 32; do run X 2 132 $R $COLS $PITCH $PITCH 1 $ROUNDS 1; done
echo

echo "== E. 固定 rows=16 扫拆分 K（r_dd，两端带距；每行再切 K 段 ⇒ rows*K 笔）=="
for K in 1 2 4 8 16; do run X 2 132 16 $COLS $PITCH $PITCH $K $ROUNDS 0; done
echo

echo "== F. grid / C 敏感与稳定性（rows=16 gap=$GAP）=="
run X 4 132 16 $COLS $PITCH $PITCH 1 $ROUNDS 0
run X 2 264 16 $COLS $PITCH $PITCH 1 $ROUNDS 0
run X 2 264 16 $COLS $PITCH $PITCH 1 $ROUNDS 1
echo

echo "== G. 多 grid 稳定性（r_dd 与 f_dd；每档 20 次 launch 全 ok）=="
for GG in 132 264 528 1056; do run X 2 $GG 16 $COLS $PITCH $PITCH 1 $ROUNDS 0; done
for GG in 132 264 528 1056; do run X 2 $GG 16 $COLS $PITCH $PITCH 1 $ROUNDS 1; done
