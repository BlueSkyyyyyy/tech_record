#!/usr/bin/env bash
# P173-1 实验 driver：把第 182 篇（P172-1）留下的口子做成**单 kernel 融合**实验——
#   「重算（compute-bound，吃张量核）能不能被 shadow 的访存受限段（wgrad）吸收、却被 shadow 的
#   计算受限段（dgrad）挡住」。
#   bash harness/exp_recompute_shadow_fused.sh > harness/exp_recompute_shadow_fused.out.txt
# 对应源码 src/recompute_shadow_fused.cu（真实 WGMMA 计算腿 + 访存 RMW 腿，单 kernel 两 warpgroup、
#   setmaxnreg 分区）；参考 ~/github/DeepGEMM 的
#     deep_gemm/include/deep_gemm/mma/sm90.cuh（make_gmma_desc / BF16MMASelector）
#     deep_gemm/include/deep_gemm/ptx/wgmma.cuh（warpgroup_arrive/commit/wait）
#   与 deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:326-335/344/948（寄存器预算分区）。
# 复合第 5 篇 / 第 167 篇（P157-1）的单 kernel warp specialization 原型：把「通信腿」换成「访存受限腿」，
#   再加一条「第二个计算腿」以隔离**张量核竞争**。
#
# 为什么 H100(sm90a) 可跑：setmaxnreg / warp specialization / WGMMA 都是架构无关原语。
#   Mega MoE 融合 kernel 本体是 SM100 专属 → **仅分析**。
#
# 本机 8 卡共享，GPU0 常被其它自驱任务占满；用 CUDA_VISIBLE_DEVICES=1（或 2）避开它。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DG=~/github/DeepGEMM
INC=$DG/deep_gemm/include

echo "重算 × shadow 的单 kernel 融合资源账（H100 真机；避开被占用的 GPU0）"
"$ROOT/scripts/lab.sh" exec "nvidia-smi --query-gpu=index,name,compute_cap,driver_version --format=csv,noheader | sed -n '1,2p'"
"$ROOT/scripts/lab.sh" exec "nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader"
echo ""

echo "===== 编译（ptxas -v 看寄存器分配） ====="
"$ROOT/scripts/lab.sh" exec "nvcc -gencode=arch=compute_90a,code=sm_90a -O3 -std=c++17 --expt-relaxed-constexpr -diag-suppress 20012 -Xptxas -v -I'$INC' -I'$INC/cutlass' -I'$INC/cute' '$ROOT/src/recompute_shadow_fused.cu' -c -o /tmp/rs.o 2>&1 | grep -E 'entry function|Used|spill|registers' | head -20"
echo ""

echo "===== 头文件指纹 ====="
"$ROOT/scripts/lab.sh" exec "sha256sum $INC/deep_gemm/mma/sm90.cuh $INC/deep_gemm/ptx/wgmma.cuh | cut -c1-24,66-"
echo ""

echo "===== 主实验 ====="
"$ROOT/scripts/lab.sh" exec "nvcc -gencode=arch=compute_90a,code=sm_90a -O3 -std=c++17 --expt-relaxed-constexpr -diag-suppress 20012 -I'$INC' -I'$INC/cutlass' -I'$INC/cute' '$ROOT/src/recompute_shadow_fused.cu' -o /tmp/recompute_shadow_fused && CUDA_VISIBLE_DEVICES=1 /tmp/recompute_shadow_fused"
