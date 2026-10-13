#!/usr/bin/env bash
# P177-1（§102）复现：cluster **形状**（1D/2D/3D）敏感性——固定 C（每 cluster 块数），
#   把 cluster 形状在 C 的各种因子分解间变化，测「放置墙」与「multicast deliver」是否随形状变。
#   补「下一步候选」两条：「2D/3D cluster × …」与「C 放置墙的 SM 数/形状敏感性」。
#   复用既有源码：src/multicast_occupancy.cu（§93 的 mcast_body / §100 的 P / §101 的 T），
#   本节新增 **形状版 kernel** `mcast_geo`（mcast_body 逐字 + 线性块号由 blockIdx.(x,y,z) 合成）
#   与两个**只读/加测**子命令：`Y`（形状→cudaOccupancyMaxActiveClusters 只读读数）、
#   `G`（该形状跑 §93 广播 + 计时 + maxerr）。**不改任何既有语义。**
#   目标卡 H100 / sm90a，nvcc 13.2。DEV 选空闲 device（可 DEV=n 覆盖）。
#   每配置一次独立进程调用：一次 cluster launch failure 会 sticky 污染 CUDA context。
set -u
DEV=${DEV:-$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
        | sort -t, -k2 -n | head -1 | cut -d, -f1)}
DIR=~/proj/tech_record/code/mega-kernel
MO=$DIR/src/multicast_occupancy.cu
BIN=/tmp/mcast_shape

echo "########## §102 cluster 形状（1D/2D/3D）敏感性（P177-1） ##########"
echo "# DEV=$DEV  nvcc -O3 -arch=sm_90a  (CUDA 13.2)"
echo "# Y = 形状→cluster 感知占用率 API 只读读数；G = 该形状跑 §93 广播（KB=8 L=2 mode 可变）"
echo "# 形状记法 clu=(cx,cy,cz)，C=cx·cy·cz；grid=(Gx,Gy,Gz)，总块数 tot=Gx·Gy·Gz"
echo
echo "===== 编译（复用 src/multicast_occupancy.cu，新增形状版 kernel mcast_geo + 只读子命令 Y + 加测子命令 G） ====="
nvcc -O3 -arch=sm_90a "$MO" -o "$BIN" -lcuda 2>/dev/null || { echo "compile failed"; exit 1; }
echo "ok: $BIN"

Y(){ echo "\$ Y $*"; CUDA_VISIBLE_DEVICES="$DEV" "$BIN" Y "$@"; }
G(){ echo "\$ G $*"; CUDA_VISIBLE_DEVICES="$DEV" "$BIN" G "$@"; }

NSM=$(CUDA_VISIBLE_DEVICES="$DEV" python3 -c 'import torch; print(torch.cuda.get_device_properties(0).multi_processor_count)' 2>/dev/null)
[ -z "${NSM:-}" ] && NSM=132
echo "# numSM=$NSM"

echo
echo "===== §A（headline）API 读数：同一 C 的各种形状，maxActiveClusters 是否只由 C 决定？ ====="
echo "#   C=2 : (2,1,1) | C=4 : (4,1,1),(2,2,1),(1,2,2) | C=8 : (8,1,1),(4,2,1),(2,4,1),(2,2,2)"
Y 2 1 1 8 2 0
Y 4 1 1 8 2 0
Y 2 2 1 8 2 0
Y 1 2 2 8 2 0
Y 8 1 1 8 2 0
Y 4 2 1 8 2 0
Y 2 4 1 8 2 0
Y 2 2 2 8 2 0
echo "#   C=6 : (6,1,1),(2,3,1),(3,2,1),(1,2,3) | C=16 : (16,1,1),(4,4,1),(2,8,1),(2,2,4)"
Y 6 1 1 8 2 0
Y 2 3 1 8 2 0
Y 3 2 1 8 2 0
Y 1 2 3 8 2 0
Y 16 1 1 8 2 0
Y 4 4 1 8 2 0
Y 2 8 1 8 2 0
Y 2 2 4 8 2 0

echo
echo "===== §B deliver 吞吐：固定 tot=cap=992 扫形状（C=4 / C=8），mode0 ====="
echo "#   cap_blocks(C=4)=992, cap_blocks(C=8)=992 —— 所有形状都在 cap 上"
G 4 1 1 992 1 1 8 300 0 2 0
G 2 2 1 62 16 1 8 300 0 2 0
G 2 2 1 124 8 1 8 300 0 2 0
G 2 2 1 496 2 1 8 300 0 2 0
G 1 2 2 248 2 2 8 300 0 2 0
G 8 1 1 992 1 1 8 300 0 2 0
G 4 2 1 124 8 1 8 300 0 2 0
G 2 4 1 124 8 1 8 300 0 2 0
G 2 2 2 62 8 2 8 300 0 2 0

echo
echo "===== §C（headline）放置墙：cap（=992）vs cap+1 cluster（=996 / 1000），扫形状 ====="
echo "#   C=4 : (4,1,1) vs (2,2,1) | C=8 : (8,1,1),(4,2,1),(2,2,2)"
echo "#   ——若形状不影响墙，则同 C 的每个形状都应在同一处从 ~992 块崩塌"
echo "# C=4 cap:"
G 4 1 1 992 1 1 8 300 0 2 0
G 2 2 1 62 16 1 8 300 0 2 0
echo "# C=4 +1 cluster (996 = 249 cluster):"
G 4 1 1 996 1 1 8 300 0 2 0
G 2 2 1 498 2 1 8 300 0 2 0
echo "# C=8 cap:"
G 8 1 1 992 1 1 8 300 0 2 0
G 4 2 1 124 8 1 8 300 0 2 0
G 2 2 2 62 8 2 8 300 0 2 0
echo "# C=8 +1 cluster (1000 = 125 cluster):"
G 8 1 1 1000 1 1 8 300 0 2 0
G 4 2 1 500 2 1 8 300 0 2 0
G 2 2 2 50 10 2 8 300 0 2 0

echo
echo "===== §D multicast（mode1，仅 rank0 发 .multicast::cluster）在 2D/3D 形状下是否成立、是否同速 ====="
G 4 1 1 992 1 1 8 300 1 2 0
G 2 2 1 62 16 1 8 300 1 2 0
G 8 1 1 992 1 1 8 300 1 2 0
G 2 2 2 62 8 2 8 300 1 2 0

echo
echo "===== §E C=16 形状族：cap=928 vs +1 cluster=944（大 C 既更早触墙，形状是否仍是无关变量） ====="
G 16 1 1 928 1 1 8 300 0 2 0
G 4 4 1 116 8 1 8 300 0 2 0
G 2 2 4 116 2 4 8 300 0 2 0
G 16 1 1 944 1 1 8 300 0 2 0
G 4 4 1 236 4 1 8 300 0 2 0
G 2 2 4 118 2 4 8 300 0 2 0

echo
echo "===== §F C=2 对照（cap=1056，grid≤cap 内无墙）：1D (2,1,1) / 2D (1,2,1) 形状 ====="
G 2 1 1 1056 1 1 8 300 0 2 0
G 1 2 1 528 2 1 8 300 0 2 0
G 2 1 1 992 1 1 8 300 0 2 0

echo
echo "########## 完 ##########"
