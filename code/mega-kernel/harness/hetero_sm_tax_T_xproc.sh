#!/usr/bin/env bash
# P97-1（§85）复现：补 P96-1（§84）**§84.4 明留的尾巴**——「bootstrap CI 基于同一进程内 S=30
#   单次实验，**不含跨进程 / 跨 device 的系统差**」。
# 做法：复用 §82/§83/§84 同一内核（src/hetero_sm_tax_T.cu，**新增 `xproc <S> <seed>` 子命令**）：
#   §95A 每进程 `xproc S=30`（round 内 Fisher-Yates 打乱访问顺序），把**原始样本**逐行输出；
#   §85B 在 4 块空闲 H100 上各起 6 个独立进程（共 24 进程），跨进程 / 跨 device 汇总
#        T512/T128、T256/T128 的 median 比分布 + 对「进程」做 bootstrap 的 95% CI；
#   §85C 方差分解：进程内样本 std vs 跨进程 median std；§85D 设备间一致性；§85E 逐进程显著性计数。
# 目标卡 H100 / sm90a。REPS / S / DEVICES 可用环境变量覆盖。
set -u
DIR=~/proj/tech_record/code/mega-kernel
SRC=$DIR/src/hetero_sm_tax_T.cu
PY=$DIR/harness/hetero_sm_tax_T_xproc.py
BIN=/tmp/hetero_sm_tax_T_xproc
REPS=${REPS:-6}
S=${S:-30}
# 自动挑 4 块最闲 device（按 memory.used 升序），可 DEVICES="1 3 5 7" 覆盖。
DEVICES=${DEVICES:-$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
        | sort -t, -k2 -n | head -4 | cut -d, -f1 | tr '\n' ' ')}
RAW=${RAW:-/tmp/hetero_sm_tax_T_xproc_raw.txt}
NDEV=$(echo "$DEVICES" | wc -w)

echo "########## §85 块粒度 T 趋势的跨进程 / 跨 device 稳健性（P97-1，补 §84.4 尾巴） ##########"
echo "# nvcc -O3 -arch=sm_90a；U=8 R=32 map=bind（与 §82/§83/§84 同配置同内核）"
echo "# DEVICES='$DEVICES'  REPS=$REPS  S=$S"

echo
echo "===== 构建（--ptxas-options=-v，与 §82 同内核）====="
nvcc -O3 -arch=sm_90a --ptxas-options=-v "$SRC" -o "$BIN" 2>/tmp/hetero_sm_tax_T_xproc_ptxas.txt || exit 1
grep -E "Function properties for _Z5pat_T|Used [0-9]+ registers|bytes spill" /tmp/hetero_sm_tax_T_xproc_ptxas.txt | head -60

: > "$RAW"
echo
echo "===== 跨进程采样：$REPS 进程 × $NDEV device（每进程 S=$S，打乱访问顺序）====="
for dev in $DEVICES; do
  for r in $(seq 1 "$REPS"); do
    seed=$((dev * 1000 + r))
    echo "----- dev=$dev rep=$r seed=$seed -----"
    CUDA_VISIBLE_DEVICES="$dev" timeout 400 "$BIN" xproc "$S" "$seed" 2>&1 \
      | sed "s/^/P $dev $r /" | tee -a "$RAW" | grep -E "META|RES|MED 0 |MED 2 "
  done
done

echo
echo "===== 汇总分析（跨进程 / 跨 device）====="
python3 "$PY" "$RAW"

echo
echo "########## §85 完成 ##########"
