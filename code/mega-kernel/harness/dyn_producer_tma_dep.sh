#!/usr/bin/env bash
# P63-1 复现：把 §51（P62-1）留下的**开放问题**钉到「内存序 / 依赖链」一级——
#   §51.6 量出「只拆背压（skipbp）或只拆发布（skiphd）都回 664 GB/s，两者同时存在才 454」，
#   并自陈未到 SASS/依赖链一级给出因果。本节用**正交的显式位编码**（WA 完成等待 / BP 背压 /
#   HD 发布 / REL head release vs relaxed / ACQ tail acquire vs relaxed / FENCE）重跑同一骨架：
#     - 复现 §51 的 ser/nowait/nobp；
#     - 发现 §51 的 skipbp/skiphd 口径**退化**（win<=-3 使两者都等于 nobp），其「只拆一个就够快」是编码假象；
#     - 用正交编码给出真实分解，并用 relaxed/fence 哨兵把「内存序语义」与「每批会合」分开；
#     - 逐相位 clock64（§D）与 SASS 指令计数定位依赖链。
# 载体固定 TMA bulk store（gvar 3）；其余与 §47–§51 同骨架。
# 目标卡 H100 / sm90a，DEV 选空闲 device（默认 3）。
set -u
DEV=${DEV:-3}
DIR=${HOME}/proj/tech_record/code/mega-kernel
SRC=$DIR/src/dyn_producer_tma_dep.cu
BIN=/tmp/dyn_producer_tma_dep
nvcc -O3 -arch=sm_90a "$SRC" -o "$BIN" || exit 1

echo "########## §bench：正交归因矩阵 + 逐相位计时 ##########"
CUDA_VISIBLE_DEVICES="$DEV" timeout 3600 "$BIN" bench

echo
echo "########## SASS：producer gvar3 主循环的强序访存/屏障指令计数 ##########"
# 反汇编并在 dptd_kernel 段内统计关键指令（stride/依赖链证据）。
cuobjdump -sass -arch sm_90a "$BIN" 2>/dev/null | awk '
  /Function : .*dptd_kernel/ {infun=1}
  infun {print}
' > /tmp/dptd_kernel.sass
echo "sass lines(dptd_kernel) = $(wc -l < /tmp/dptd_kernel.sass)"
echo "--- 强序访存 / 屏障 / async 指令计数（整个 dptd_kernel）---"
grep -oE "BAR\.SYNC|MEMBAR\.[A-Z.]+|LD\.E\.STRONG\.[A-Z.]+|ST\.E\.STRONG\.[A-Z.]+|LD\.E\.[A-Z0-9.]+|ST\.E\.[A-Z0-9.]+|CP\.ASYNC\.BULK[A-Z._:]*|FENCE[A-Z._:]*" /tmp/dptd_kernel.sass | sort | uniq -c | sort -rn
echo "--- 含 acquire/release/strong 的原样行（前 40 行）---"
grep -nE "STRONG|ACQUIRE|RELEASE|MEMBAR|FENCE|CP\.ASYNC\.BULK" /tmp/dptd_kernel.sass | head -40
