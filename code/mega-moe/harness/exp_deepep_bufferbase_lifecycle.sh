#!/usr/bin/env bash
# P111-1 实验 driver：DeepEP 四个 Buffer 家族共享的生命周期契约 BufferBase。
#
# 为什么 H100(sm90) 可跑：
#   * Python 侧 deep_ep/buffers/base.py 只 import `abc`，不碰 CUDA/NCCL；
#   * C++ 侧 csrc/buffers/base.hpp 全是 host 代码，driver 在临时目录搭"影子 include 树"
#     （真实 base.hpp 原样拷入 + 最小 stub 顶掉 pybind11 / comm::api.hpp）后 g++ 编译运行。
#   * `import deep_ep` 会触发 check_nccl_so()（本机 NCCL 2.29.7 与编译期不符 -> abort），
#     故 Python 侧按文件路径直接加载 base.py；需要真实 NCCL GIN 的四个家族 C++ 构造函数
#     本机不可用，用一个同构的 DerivedBuffer 承接 base.hpp 的真实逻辑。
#
# 用法：bash harness/exp_deepep_bufferbase_lifecycle.sh > harness/exp_deepep_bufferbase_lifecycle.out.txt
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT/harness/exp_deepep_bufferbase_lifecycle.py"
DEEPEP=~/github/DeepEP

echo "DeepEP BufferBase（四个 Buffer 家族共享的生命周期契约）—— H100 真机"
"$ROOT/scripts/lab.sh" exec "nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader | head -1"
"$ROOT/scripts/lab.sh" exec "python -c \"import torch; print('torch', torch.__version__, 'nccl', torch.cuda.nccl.version())\""
"$ROOT/scripts/lab.sh" exec "g++ --version | head -1"
echo "DeepEP 源码：$(git -C "$DEEPEP" rev-parse --short HEAD 2>/dev/null || echo '?')"
echo ""

"$ROOT/scripts/lab.sh" exec "DEEPEP_ROOT=$DEEPEP python '$PY'"
