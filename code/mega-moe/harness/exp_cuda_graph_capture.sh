#!/usr/bin/env bash
# P99-1 实验 driver：在 kernel_lab(H100/sm90) 上跑 DeepGEMM 的 CUDA Graph 兼容性面。
#   bash harness/exp_cuda_graph_capture.sh > harness/exp_cuda_graph_capture.out.txt
# 对应 exp_cuda_graph_capture.py；参考 ~/github/DeepGEMM 的 csrc/apis/{mega_mhc,mega_gate,attention}.hpp、
# csrc/runtime/runtime.hpp、third-party/deep_jit/.../backend/cuda/kernel.hpp、tests/test_mega_mhc.py。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DG=~/github/DeepGEMM
CMD=$(cat <<EOS
set -e
echo "== 环境坐标 =="
( cd $DG && git -c safe.directory='*' log --oneline -1 )
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader | head -1
python3 - <<'PY'
import torch, os
print("torch", torch.__version__, "cuda", torch.version.cuda)
h = os.path.join(os.path.dirname(torch.__file__), "include/c10/cuda/CUDAGraphsC10Utils.h")
print("CurrentStreamCaptureStatusMayInitCtx header sha256:")
import hashlib; print(" ", hashlib.sha256(open(h,'rb').read()).hexdigest(), h)
PY
echo
echo "== 实验 =="
python3 '$ROOT/harness/exp_cuda_graph_capture.py'
EOS
)
"$ROOT/scripts/lab.sh" exec "$CMD"
