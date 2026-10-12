#!/usr/bin/env bash
# P7-6: decompose the host-side overhead of FLA chunk_kda vs FlashKDA forward.
#
# For each (mode, case) it:
#   1. runs probe_nsys.py in `profile` mode -> wall (event-pair), dev (CUPTI),
#      and the torch.profiler CPU self-time ranking (where the CPU goes);
#   2. runs probe_nsys.py in `loop` mode under `nsys profile` with a CUDA-profiler
#      capture range (so warmup / Triton JIT is excluded) -> the steady-state
#      CUDA-API and CUDA-kernel summaries (how many launches, how much driver time).
#
# Everything is tee'd to probe_nsys.out.txt.
#
# Usage (inside kernel_lab):
#   bash ~/proj/tech_record/code/flash-kda/harness/probe_nsys.sh
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$HERE/probe_nsys.out.txt"
N="${N:-200}"

exec > >(tee "$OUT") 2>&1
echo "==== P7-6 host-tax decomposition: FLA chunk_kda vs FlashKDA fwd ===="
echo "date: $(date -Is)   host: $(hostname)"
python - <<'PY'
import torch
print("torch", torch.__version__, "| GPU", torch.cuda.get_device_name())
try:
    import triton, fla
    print("triton", triton.__version__, "| fla", fla.__version__)
except Exception as e:
    print("ver err", e)
PY
echo
echo "==== microbench: per-launch host overhead (tiny GPU work) ===="
python "$HERE/probe_nsys.py" micro

echo
echo "==== deterministic host-CPU decomposition (process_time) ===="
echo "-- FLA (dispatch ON) --"
python "$HERE/probe_nsys.py" fla cpu
echo "-- FLA (dispatch OFF, FLA_DISABLE_BACKEND_DISPATCH=1) --"
FLA_DISABLE_BACKEND_DISPATCH=1 python "$HERE/probe_nsys.py" fla cpu
echo "-- FlashKDA (contrast) --"
python "$HERE/probe_nsys.py" flash cpu

for case in small big bh; do
  for mode in fla flash; do
    echo
    echo "================================================================"
    echo "#### mode=$mode case=$case"
    echo "---- wall/dev (torch.profiler) ----"
    python "$HERE/probe_nsys.py" "$mode" "$case" profile 2>/dev/null | grep -vE 'UserWarning|_warn_once' | head -6
  done
done

for case in small big bh; do
  for mode in fla flash; do
    echo
    echo "================================================================"
    echo "#### nsys steady state (capture-range, N=$N): mode=$mode case=$case"
    echo "---- wall in-loop ----"
    nsys profile -o "/tmp/nsys_${mode}_${case}" --force-overwrite=true \
        --capture-range=cudaProfilerApi --capture-range-end=stop --trace=cuda \
        python "$HERE/probe_nsys.py" "$mode" "$case" loop "$N" 2>&1 | grep -E 'wall=|Error|error' || true
    echo "---- CUDA API summary ----"
    nsys stats --force-export=true --report cuda_api_sum --format table "/tmp/nsys_${mode}_${case}.nsys-rep" 2>&1 \
        | grep -E 'Time|cuLaunch|cudaLaunch|cudaMalloc|cudaDeviceSynch|cudaMemcpy|cuKernel|cudaFuncSet' || true
    echo "---- CUDA kernel summary ----"
    nsys stats --force-export=true --report cuda_gpu_kern_sum --format table "/tmp/nsys_${mode}_${case}.nsys-rep" 2>&1 \
        | grep -E 'Time \(|_fwd|_kernel|flash_kda|elementwise_kernel|fill|recompute|l2norm|gate|sigmoid' || true
  done
done
echo
echo "DONE"
