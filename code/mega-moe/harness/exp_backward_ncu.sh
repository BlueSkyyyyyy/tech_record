#!/usr/bin/env bash
# 反向性能归因实验 driver：在 kernel_lab(H100) 用 ncu 采集四个 grouped GEMM kernel 的硬指标。
#   bash harness/exp_backward_ncu.sh > harness/exp_backward_ncu.out.txt
# 对应 exp_backward_ncu.py。
#   第一段：五个 section（SpeedOfLight/Occupancy/LaunchStats/Memory/Compute）。
#   第二段：定向 metric —— 单次 kernel 时长、DRAM 读/写字节、GMMA 指令数、SM active 周期。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
H="$ROOT/harness"
SECTIONS="--section SpeedOfLight --section Occupancy --section LaunchStats --section MemoryWorkloadAnalysis --section ComputeWorkloadAnalysis"
METRICS="gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum,sm__inst_executed_pipe_tensor_op_gmma.sum,sm__cycles_active.avg"
KNAME="--target-processes all --launch-count 1 --kernel-name 'regex:sm90_fp8_gemm'"
"$ROOT/scripts/lab.sh" exec "cd '$H' && nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader | head -1 && \
echo '##### event timing: single-shot (热启动 3 次后测 1 次, 高时钟) #####' && \
for k in w2g w1g d2g d1g; do python exp_backward_ncu.py \$k --warmup 3 --iters 1; done && \
echo '##### event timing: sustained (热启动 10 次后测 100 次, 持续负载/降频) #####' && \
for k in w2g w1g d2g d1g; do python exp_backward_ncu.py \$k --warmup 10 --iters 100; done && \
for k in w2g w1g d2g d1g; do echo \"##### KERNEL \$k (ncu 单次 launch, 热 cache) #####\"; SKIP_EVENT_PRINT=1 ncu $KNAME $SECTIONS python exp_backward_ncu.py \$k; echo '--- directed metrics ---'; SKIP_EVENT_PRINT=1 ncu $KNAME --metrics $METRICS python exp_backward_ncu.py \$k 2>/dev/null | grep -E 'gpu__time|dram__bytes|tensor_op_gmma|cycles_active'; done"
