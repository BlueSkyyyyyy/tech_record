#!/usr/bin/env bash
# P135-1 实验 driver：在 kernel_lab(H100) 量化 MoE wgrad 的「强制累积（fp32 D 回读）」代价。
#   bash harness/exp_wgrad_accum_cost.sh > harness/exp_wgrad_accum_cost.out.txt
#   第一段：exp_wgrad_accum_cost.py —— 正确性 sanity + wgrad/dgrad 对照 + 两因子分解 + K 扫描 + 访存账。
#   第二段：ncu 单次 launch，采 DRAM 读/写字节 与时长，核对 Part E 的解析访存账（k 真实路径 vs m 等价）。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
H="$ROOT/harness"
SECTIONS="--section SpeedOfLight --section MemoryWorkloadAnalysis --section ComputeWorkloadAnalysis"
METRICS="gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum,sm__cycles_active.avg"
KNAME="--target-processes all --launch-count 1 --kernel-name 'regex:sm90_fp8_gemm'"
"$ROOT/scripts/lab.sh" exec "export CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES:-1}; cd '$H' && \
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader | head -1 && \
python exp_wgrad_accum_cost.py && \
echo '##### ncu: 真实 wgrad k_grouped (fp32/强制累积) #####' && \
ncu $KNAME --metrics $METRICS python exp_wgrad_accum_cost.py --ncu k 2>&1 | grep -E 'Profiling \"|gpu__time|dram__bytes|sm__cycles' ; \
echo '##### ncu: 等价 dgrad m_grouped (bf16/直写) #####' && \
ncu $KNAME --metrics $METRICS python exp_wgrad_accum_cost.py --ncu m 2>&1 | grep -E 'Profiling \"|gpu__time|dram__bytes|sm__cycles' ; \
echo '##### ncu: 稠密 1D1D fp32/acc #####' && \
ncu $KNAME --metrics $METRICS python exp_wgrad_accum_cost.py --ncu d1 2>&1 | grep -E 'Profiling \"|gpu__time|dram__bytes|sm__cycles' ; \
echo '##### ncu: 稠密 1D2D bf16/直写 #####' && \
ncu $KNAME --metrics $METRICS python exp_wgrad_accum_cost.py --ncu d2 2>&1 | grep -E 'Profiling \"|gpu__time|dram__bytes|sm__cycles'"
