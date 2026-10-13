#!/usr/bin/env bash
# P132-1 实验 driver：在 kernel_lab(H100) 给 Mega MoE **前向**的两段 m-grouped GEMM 做 roofline 归因。
#   bash harness/exp_forward_roofline.sh > harness/exp_forward_roofline.out.txt
# 对应 exp_forward_roofline.py（GEMM 本体）与 src/moe_forward_epilogue.cu（SwiGLU+量化 epilogue 预算）。
#   第一段：event 计时（single-shot 热启动 / sustained 持续负载）+ 字节/AI/roofline 分析。
#   第二段：ncu 单次 launch 采五个 section + 定向 metric（时长/DRAM 读写/GMMA 指令/SM active）。
#   第三段：编译运行 src/moe_forward_epilogue.cu（前向 epilogue 的数值核对 + 定时）。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
H="$ROOT/harness"
SECTIONS="--section SpeedOfLight --section Occupancy --section LaunchStats --section MemoryWorkloadAnalysis --section ComputeWorkloadAnalysis"
METRICS="gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum,sm__inst_executed_pipe_tensor_op_gmma.sum,sm__cycles_active.avg"
KNAME="--target-processes all --launch-count 1 --kernel-name 'regex:sm90_fp8_gemm'"
"$ROOT/scripts/lab.sh" exec "export CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES:-1}; cd '$H' && nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader | head -1 && \
echo '##### roofline + event timing: single-shot (热启动 3 次后测 1 次, 高时钟) #####' && \
for k in l1 l2; do python exp_forward_roofline.py \$k --warmup 3 --iters 1; done && \
echo '##### event timing: sustained (热启动 10 次后测 100 次, 持续负载/降频) #####' && \
for k in l1 l2; do python exp_forward_roofline.py \$k --warmup 10 --iters 100 2>/dev/null | grep event_time; done && \
for k in l1 l2; do echo \"##### KERNEL \$k (ncu 单次 launch, 热 cache) #####\"; ncu $KNAME $SECTIONS python exp_forward_roofline.py \$k --report; echo '--- directed metrics ---'; ncu $KNAME --metrics $METRICS python exp_forward_roofline.py \$k --report 2>/dev/null | grep -E 'gpu__time|dram__bytes|tensor_op_gmma|cycles_active'; done && \
echo '##### forward epilogue: L1 in-place SwiGLU(gate/up RNE->bf16) + topk weight + per-token UE8M0 quant #####' && \
nvcc -arch=sm_90 -O3 -std=c++17 '$ROOT/src/moe_forward_epilogue.cu' -o /tmp/moe_forward_epilogue && /tmp/moe_forward_epilogue"
