#!/usr/bin/env bash
# P39-1 复现：多租户共存隔离的第二条路线 —— MPS（Multi-Process Service）
#   目标卡 H100 / sm90, CUDA 13.2。与 P38-1（Green Contexts SM 张数分区）对照：
#   P38 在同一进程内按 SM 张数分区；本实验是**跨进程**，MPS 是让不同进程真正并发 + 按
#   ACTIVE_THREAD_PERCENTAGE 给每个 client 设「线程容量 / 可见 SM 上限」的服务。
# 用 DEV 选择空闲 device（默认 3）；MPS server 只占该 device。
set -u
DEV=${DEV:-3}
MB=${MB:-256}
VPASS=${VPASS:-20}
ROUNDS=${ROUNDS:-40000}
SRC=~/proj/tech_record/code/mega-kernel/src/mps_isolation.cu
BIN=/tmp/mps_isolation
nvcc -O2 -arch=sm_90 "$SRC" -o "$BIN" || exit 1

echo "=== P39-1 跨进程共存隔离：MPS vs 无 MPS（目标卡 H100 / sm90）==="
echo "device=$DEV  buf=${MB}MB  victim_passes=$VPASS  persist_rounds=$ROUNDS"

# ---------------- 无 MPS ----------------
echo
echo "[Section A] 无 MPS：不同进程的 kernel 不能真正共驻（由驱动时分/抢占）"
echo -n "  A0 victim alone (no MPS)      : "; CUDA_VISIBLE_DEVICES=$DEV "$BIN" victim $MB $VPASS
echo -n "  A1 persist alone (no MPS)     : "; CUDA_VISIBLE_DEVICES=$DEV "$BIN" persist $MB $ROUNDS /dev/null | grep PERSIST_DONE
CUDA_VISIBLE_DEVICES=$DEV "$BIN" persist $MB $ROUNDS /dev/null > /tmp/pA.txt 2>&1 & PA=$!
sleep 2
echo -n "  A2 victim during persist      : "; CUDA_VISIBLE_DEVICES=$DEV "$BIN" victim $MB $VPASS
wait $PA
echo -n "  A2 persist (concurrent victim): "; grep PERSIST_DONE /tmp/pA.txt

# ---------------- MPS ----------------
D=$(mktemp -d /tmp/mps_p39.XXXXXX)
mkdir -p "$D/pipe" "$D/log"
export CUDA_MPS_PIPE_DIRECTORY="$D/pipe"
export CUDA_MPS_LOG_DIRECTORY="$D/log"
CUDA_VISIBLE_DEVICES=$DEV nvidia-cuda-mps-control -d
sleep 1
unset CUDA_VISIBLE_DEVICES   # MPS client 经 pipe 连到 server（只暴露 DEV）
cleanup() { echo quit | nvidia-cuda-mps-control >/dev/null 2>&1; sleep 1; rm -rf "$D"; }
trap cleanup EXIT

echo
echo "[Section B] MPS 默认：两 client 各 100%（真并发，共享占用/带宽）"
echo -n "  B0 victim alone (in MPS)       : "; "$BIN" victim $MB $VPASS
"$BIN" persist $MB $ROUNDS /dev/null > /tmp/pB.txt 2>&1 & PB=$!
sleep 2
echo -n "  B1 victim during persist       : "; "$BIN" victim $MB $VPASS
wait $PB
echo -n "  B1 persist (concurrent victim) : "; grep PERSIST_DONE /tmp/pB.txt

echo
echo "[Section C] MPS 给 persistent 设 ACTIVE_THREAD_PERCENTAGE=P（线程容量上限）"
for P in 100 90 75 60 50 40 33 25 10; do
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$P "$BIN" persist $MB $ROUNDS /dev/null > /tmp/pC_$P.txt 2>&1 & PC=$!
  sleep 2
  printf "  C P=%-3s victim during persist : " "$P"; "$BIN" victim $MB $VPASS
  wait $PC
  printf "  C P=%-3s persist rate (≈solo)  : " "$P"; grep PERSIST_DONE /tmp/pC_$P.txt
done

echo
echo "OK"
