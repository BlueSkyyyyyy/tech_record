#!/usr/bin/env bash
# P105-1 实验 driver：DeepEP `refs.dispatch` 的【FP8 分支】——per-128 block scale 的传递契约，
#   并与设备端 token 缓冲布局 `layout/ep/token.cuh` 的 `sf_pack_t` / `TokenLayout` 对照。
#
# 为什么 H100(sm90) 可跑：
#   * `refs.py` / `math.py` 是纯 PyTorch，只用 torch 自带 NCCL 的 `all_to_all_single`；
#   * `layout/ep/token.cuh` 的 `TokenLayout` / `BufferLayout` 方法全带 `__host__ __device__`，
#     **host 可直接调用**——`src/deepep_token_layout_sf.cu` include 真实头文件后不启动任何 kernel，
#     只把 SF 布局整数几何打印出来，交给 Python 独立公式对拍；
#   * `import deep_ep` 会触发 `check_nccl_so()`（本机 NCCL 2.29.7 不匹配），故 Python 侧用
#     sys.modules 注入合成包绕过（同第 65/114 篇）。
#
#   bash harness/exp_deepep_refs_fp8.sh > harness/exp_deepep_refs_fp8.out.txt
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$ROOT/src/deepep_token_layout_sf.cu"
PY="$ROOT/harness/exp_deepep_refs_fp8.py"
DEP=${HOME}/github/DeepEP

echo "DeepEP refs.dispatch 的 FP8 分支 + 设备端 token SF 布局 —— H100 真机"
"$ROOT/scripts/lab.sh" exec "nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader | head -1"
"$ROOT/scripts/lab.sh" exec "python -c \"import torch; print('torch', torch.__version__, 'nccl', torch.cuda.nccl.version())\""
echo "DeepEP 源码：$(git -C "$DEP" rev-parse --short HEAD 2>/dev/null || echo '?')"
echo ""
echo "=== C++: include layout/ep/token.cuh 真实头文件、host 调用 TokenLayout ==="
"$ROOT/scripts/lab.sh" exec "nvcc -arch=sm_90a -O2 -std=c++17 --expt-relaxed-constexpr -w \
  -I$DEP/deep_ep/include '$SRC' -o /tmp/deepep_token_layout_sf && /tmp/deepep_token_layout_sf"
echo ""
"$ROOT/scripts/lab.sh" exec "cd '$ROOT/harness' && TORCHDYNAMO_DISABLE=1 DEEPEP_ROOT='$DEP' python '$PY'"
