#!/usr/bin/env bash
# P83-1 实验 driver：DeepEP BucketBuffer 的「会话（BucketSession）」机制与临时张量注册。
#
# 为什么 H100(sm90) 可跑：
#   * BucketSession / BufferAllocator 是纯 host Python，不碰任何 CUDA device API。
#   * `import deep_ep` 会触发 `deep_ep/__init__.py check_nccl_so()`（本机 NCCL 2.29.7 与编译期
#     不匹配 -> abort）。故脚本用 sys.modules 注入合成包 + stub `deep_ep._C` / `deep_ep.comm` /
#     `deep_ep.utils.envs`，再按文件路径 exec 真实 bucket.py / allocator.py / base.py / event.py。
#   * 真正需要 NCCL GIN 的 `_C.BucketBuffer` kernel 侧本机不可用，用一个同名方法签名的
#     FakeBucket 承接 `_call_original`，逐字段核对会话的注册/暂存/越界/组一致性/结果回拷语义；
#     BufferAllocator 用真实 torch.utils.swap_tensors + 真实 CUDA storage。
#
# 用法：bash harness/exp_deepep_bucket_session.sh > harness/exp_deepep_bucket_session.out.txt
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT/harness/exp_deepep_bucket_session.py"
DEEPEP=~/github/DeepEP

echo "DeepEP BucketSession（会话式临时张量注册 + 生命周期守卫）—— H100 真机"
"$ROOT/scripts/lab.sh" exec "nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader | head -1"
"$ROOT/scripts/lab.sh" exec "python -c \"import torch; print('torch', torch.__version__, 'nccl', torch.cuda.nccl.version())\""
echo "DeepEP 源码：$(git -C "$DEEPEP" rev-parse --short HEAD 2>/dev/null || echo '?')"
echo ""

"$ROOT/scripts/lab.sh" exec "DEEPEP_ROOT=$DEEPEP python '$PY'"
