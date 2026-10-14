#!/usr/bin/env bash
# P178-1 §103：S2S bulk 的「非连续（2D/3D）搬运」——把 §97 `mtex`/§98 `mtex2` 的
# `cp.async.bulk.shared::cluster.shared::cta`（SMEM→对端 SMEM 的大块搬运）从**单笔连续段**
# 推广到**多笔线性搬运拼出的非连续 2D tile**（PTX 里 S2S bulk 只有线性形式，无 tensor/2D）。
# 语义：C 个 CTA 成环，每 round 把本 block 的 2D tile（rows×cols 字，源行距 pitch≥cols）搬到
# (rank+1) 的打包 rcv 槽，读回 ack。唯一变量 = 一笔 tile 被拆成几笔 bulk：
#   pitch==cols,K=1 ⇒ 1 笔（连续，吞吐天花板）；pitch>cols,K=1 ⇒ rows 笔；pitch>cols,K>1 ⇒ rows*K 笔。
# ready/done 一律 tid0 `arrive.expect_tx(payload 总字节)` + N 笔 bulk 落同一 mbarrier（tx 累加）。
# nd_sw 是软件基线（全 block st.shared::cluster + __syncthreads + tid0 软件 arrive）。
# 目标卡 H100 / sm90a。每配置单独进程（cluster launch failure 会 sticky 污染 CUDA context）。
set -u
DEV=${DEV:-3}
SRC=${HOME}/proj/tech_record/code/mega-kernel/src/s2s_bulk_nd.cu
BIN=/tmp/s2s_bulk_nd
nvcc -O3 -arch=sm_90a "$SRC" -o "$BIN" || exit 1

run(){ echo "\$ CUDA_VISIBLE_DEVICES=$DEV $BIN $*"; CUDA_VISIBLE_DEVICES="$DEV" timeout 180 "$BIN" "$@" 2>&1; }
probe(){ echo -n "  "; CUDA_VISIBLE_DEVICES="$DEV" timeout 60 "$BIN" "$@" 2>&1 | tail -1; }

COLS=256
GAP=64
PITCH=320
ROUNDS=64

echo "== A. 资源表（numRegs / 占用率）=="
run A

echo
echo "== B.（headline 之一）同一 2D tile：连续 1 笔 vs 非连续 rows 笔（C=2 grid=132 rounds=$ROUNDS）=="
echo "-- 连续（pitch=cols, K=1 = 1 笔） --"
for R in 2 4 8 16; do run T 2 132 $R $COLS $COLS 1 $ROUNDS; done
echo "-- 非连续（pitch=$PITCH, K=1 = rows 笔） --"
for R in 2 4 8 16; do run T 2 132 $R $COLS $PITCH 1 $ROUNDS; done
echo "-- 软件基线 nd_sw（pitch=$PITCH, 逐元素 st.shared::cluster） --"
for R in 2 4 8 16; do run W 2 132 $R $COLS $PITCH $ROUNDS; done

echo
echo "== C.（headline 之二）固定 rows=16 扫拆分 K（每行再切 K 段 ⇒ rows*K 笔；cols=$COLS pitch=$PITCH）=="
for K in 1 2 4 8 16 32 64; do run T 2 132 16 $COLS $PITCH $K $ROUNDS; done

echo
echo "== D. 段大小扫描：固定 rows=8 扫 cols（pitch=cols+$GAP，K=1 ⇒ 8 笔）=="
for CO in 64 128 256 512; do run T 2 132 8 $CO $((CO+GAP)) 1 $ROUNDS; done

echo
echo "== E. grid / C 敏感（rows=16 pitch=$PITCH K=1）=="
for C in 2 4 8; do run T $C 132 16 $COLS $PITCH 1 $ROUNDS; done
for G in 132 264; do run T 2 $G 16 $COLS $PITCH 1 $ROUNDS; done

echo
echo "== F. 可行性探针（非连续 rows 笔 / 拆分笔 在市面 grid 上是否稳）=="
for G in 132 264 528 1056; do probe F 2 $G 16 $COLS $PITCH 1 3 $ROUNDS; done
for G in 132 264 528 1056; do probe F 2 $G 16 $COLS $PITCH 4 3 $ROUNDS; done
