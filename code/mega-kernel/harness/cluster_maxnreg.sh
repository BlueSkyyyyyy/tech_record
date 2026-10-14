#!/usr/bin/env bash
# P161-1 §86：`__maxnreg__` 寄存器预算 × Thread Block Cluster（目标卡 H100 / sm90a）。
# 每个配置单独一次进程调用——一次 cluster launch failure 会污染 CUDA context（sticky）。
# 共享卡上并发 context 会让 cluster launch 间歇性失败，故计时项 try_run 重试（最多 5 次）；
# 边界项 §G 用 probe（每配置试 3 次，报 ok/3）。
set -u
DEV=${DEV:-3}
SRC=${HOME}/proj/tech_record/code/mega-kernel/src/cluster_maxnreg.cu
BIN=/tmp/cluster_maxnreg
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
  echo "$(printf '%s\n' "$last" | grep -E '^[A-Z] ' | head -1)   -> ok $ok/3"
}

echo "== A. 资源表：__maxnreg__(R) → numRegs / 占用率（T=256）；hreadp=同 body 无属性 =="
run A

echo
echo "== B.（headline）重 DSM 读（同一 body）在 1–2 块/SM 区间的 R 敏感性：grid=132/264，R=32(occ8)/R=48(occ5)/hreadp(occ8) =="
for G in 132 264; do try_run H 2 $G 4096 2048 32; done
for G in 132 264; do try_run H 2 $G 4096 2048 48; done
for G in 132 264; do try_run P 2 $G 4096 2048; done

echo
echo "== C. __maxnreg__ 对轻 DSM kernel（自然 13 regs）是 no-op：R=32 vs R=255 =="
try_run R 2 132 4096 2048 32
try_run R 2 132 4096 2048 255
try_run R 2 528 4096 2048 32
try_run R 2 528 4096 2048 255

echo
echo "== D. 占用率旋钮 = SMEM 预算（grid=528, C=2, R=32, 轻 kernel）：16/48/96/224 KB/block =="
for SB in 4096 12288 24576 57344; do try_run R 2 528 $SB 2048 32; done

echo
echo "== E. cluster.sync() 成本 vs cluster 大小 C 与 R（grid=128, reps=200）=="
for C in 2 4 8 16; do try_run C $C 128 200 32; done
for C in 2 4 8 16; do try_run C $C 128 200 255; done

echo
echo "== F. 可行性：cluster launch vs (C, grid=512, T, R)（确定性，无重试）=="
for C in 2 4 8 16; do run F $C 512 256 32; done
echo "-- 硬边界复现：C=32 非法；grid 非 C 倍数非法 --"
run F 32 32 256 32
run F 8 132 256 32
run F 16 64 256 32

echo
echo "== G.（headline 二）重 DSM + cluster 的可靠区边界：C=1 无限制；C≥2 在中等 grid 稳定失败 =="
echo "-- C=1（无 cluster）--"
for G in 264 528 1056; do probe H 1 $G 4096 2048 32; done
echo "-- C=2 --"
for G in 132 264 396 528; do probe H 2 $G 4096 2048 32; done
echo "-- C=4 --"
for G in 132 264; do probe H 4 $G 4096 2048 32; done
echo "-- 轻 DSM 对照（C=2）--"
for G in 264 528 1056; do probe R 2 $G 4096 2048 32; done
