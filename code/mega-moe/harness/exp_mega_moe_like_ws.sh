#!/usr/bin/env bash
# P157-1 实验 driver：把 Mega MoE 的「单 kernel 内 通信 warp + 计算 warp 按寄存器预算分区」搬到 H100 上做原型。
#   bash harness/exp_mega_moe_like_ws.sh > harness/exp_mega_moe_like_ws.out.txt
# 对应源码 src/mega_moe_like_ws.cu（A 部分，CUDA-core FMA 计算腿）与 src/mega_moe_like_ws_tc.cu（B 部分，
# WGMMA 异步张量核计算腿）；参考 ~/github/DeepGEMM 的
#   deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh（:326-335 预算、:344/:948 dealloc/alloc）
#   ~/github/DeepEP 的 deep_ep/include/deep_ep/impls/ep/hybrid_combine.cuh（:106-112/:353-359）
#   与 deep_ep/include/deep_ep/common/ptx.cuh:456-469（setmaxnreg）。
# 补齐 ROADMAP「下一步」③：用「限块通信」的思想在 8×H100 上做一个单进程「Mega-MoE-like 单 kernel」原型，
#   量它 vs「两 kernel co-residency」的重叠差，并验证「异步张量核是融合前提」。
#
# 为什么 H100(sm90a) 可跑：setmaxnreg / warp specialization / NVLink peer copy / WGMMA 都是架构无关原语。
#   Mega MoE 融合 kernel 本体是 SM100 专属 → **仅分析**。
#
# 本机 8 卡共享，GPU0 常被其它自驱任务占满；本脚本用 -DWORLD=7 + CUDA_VISIBLE_DEVICES=1..7 避开它。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DG=~/github/DeepGEMM
INC=$DG/deep_gemm/include

echo "Mega-MoE-like 单 kernel warp specialization 原型（H100 真机，7×H100 避开被占用的 GPU0）"
"$ROOT/scripts/lab.sh" exec "nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader | head -1"
"$ROOT/scripts/lab.sh" exec "nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader"
echo ""

echo "===== A) CUDA-core FMA 计算腿：寄存器分配 + 重叠对照 ====="
echo "--- ptxas 寄存器分配（setmaxnreg 让计算 warpgroup 拿到 > 均匀份额） ---"
"$ROOT/scripts/lab.sh" exec "nvcc -gencode=arch=compute_90a,code=sm_90a -O3 -std=c++17 --expt-relaxed-constexpr -Xptxas -v -DWORLD=7 '$ROOT/src/mega_moe_like_ws.cu' -c -o /tmp/x.o 2>&1 | grep -E 'entry function|Used|spill stores'"
echo "--- 主实验 ---"
"$ROOT/scripts/lab.sh" exec "nvcc -gencode=arch=compute_90a,code=sm_90a -O3 -std=c++17 --expt-relaxed-constexpr -DWORLD=7 '$ROOT/src/mega_moe_like_ws.cu' -o /tmp/mega_moe_like_ws && CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 /tmp/mega_moe_like_ws"
echo ""

echo "===== B) 异步张量核（WGMMA）vs CUDA-core（FMA）计算腿的重叠对照 ====="
"$ROOT/scripts/lab.sh" exec "sha256sum $INC/deep_gemm/mma/sm90.cuh $INC/deep_gemm/ptx/wgmma.cuh"
"$ROOT/scripts/lab.sh" exec "nvcc -gencode=arch=compute_90a,code=sm_90a -O3 -std=c++17 --expt-relaxed-constexpr -diag-suppress 20012 -I'$INC' -I'$INC/cutlass' -I'$INC/cute' '$ROOT/src/mega_moe_like_ws_tc.cu' -o /tmp/mega_moe_like_ws_tc && CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 /tmp/mega_moe_like_ws_tc"
