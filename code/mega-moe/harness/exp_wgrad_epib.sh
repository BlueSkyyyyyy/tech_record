#!/usr/bin/env bash
# P139-1 实验 driver：把 MoE wgrad 的 EPI-B（bf16 累加器 RMW）落到真实 SM90 bf16 GEMM 上。
#   bash harness/exp_wgrad_epib.sh > harness/exp_wgrad_epib.out.txt
#   第一段：exp_wgrad_epib.py —— Part A 载体正确性 / Part B 端到端三协议 / Part C 算术强度对照。
#   第二段：ncu 单次 launch —— 对比 fresh 与 EPI-B 的 DRAM 读/写字节（证明多读一份 D）。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
H="$ROOT/harness"
METRICS="gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum"
KNAME="--target-processes all --launch-count 1 --kernel-name 'regex:sm90_bf16_gemm'"
"$ROOT/scripts/lab.sh" exec "export CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES:-1}; cd '$H' && \
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader | head -1 && \
echo '##### Part A/B/C #####' && \
python3 -u exp_wgrad_epib.py && \
echo && echo '##### ncu: fresh（只写）#####' && \
ncu $KNAME --metrics $METRICS python3 exp_wgrad_epib.py --ncu fresh 2>/dev/null | \
  grep -E 'gpu__time|dram__bytes|Duration|dram__' || true && \
echo '##### ncu: EPI-B（读+写 RMW）#####' && \
ncu $KNAME --metrics $METRICS python3 exp_wgrad_epib.py --ncu epib 2>/dev/null | \
  grep -E 'gpu__time|dram__bytes|Duration|dram__' || true"
