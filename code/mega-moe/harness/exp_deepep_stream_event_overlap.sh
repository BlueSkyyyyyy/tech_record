#!/usr/bin/env bash
# P65-1 实验 driver：DeepEP 的【通信-计算重叠 Python 调度地基】——事件 / 流 / buffer 生命周期。
#
# 为什么 H100(sm90) 可跑：
#   * `deep_ep/utils/event.py`(EventOverlap) / `deep_ep/comm/stream.py`(get_comm_stream) /
#     `deep_ep/buffers/base.py`(BufferBase) 都是**纯 host / Python**；底层是 torch::Event /
#     torch::Stream。`import deep_ep` 会触发 NCCL so 版本校验 abort，故脚本用 sys.modules 注入
#     合成 `deep_ep._C`（提供 EventHandle/get_comm_stream 桩）后按文件路径 exec 真实文件。
#   * 事件/流原语用真实 `torch.cuda.Event/Stream` 驱动；并额外编译一个 CUDA 程序镜像
#     `csrc/utils/event.hpp` / `csrc/comm/stream.hpp` 的语义，验证 ordering 与重叠收益。
#
# 用法：bash harness/exp_deepep_stream_event_overlap.sh > harness/exp_deepep_stream_event_overlap.out.txt
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$ROOT/src/deepep_stream_event.cu"
PY="$ROOT/harness/exp_deepep_stream_event_overlap.py"
DEP=${HOME}/github/DeepEP

echo "DeepEP 通信-计算重叠 Python 调度地基（事件 / 流 / BufferBase）—— H100 真机"
"$ROOT/scripts/lab.sh" exec "nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader | head -1"
"$ROOT/scripts/lab.sh" exec "python -c \"import torch; print('torch', torch.__version__, 'nccl', torch.cuda.nccl.version(), 'sms', torch.cuda.get_device_properties(0).multi_processor_count)\""
echo "DeepEP 源码：$(git -C "$DEP" rev-parse --short HEAD 2>/dev/null || echo '?')  ($(git -C "$DEP" log -1 --format='%ci' 2>/dev/null || echo '?'))"
echo ""

echo "===== C++ 镜像（csrc/utils/event.hpp + csrc/comm/stream.hpp 语义，真实 CUDA event/stream） ====="
"$ROOT/scripts/lab.sh" exec "nvcc -arch=sm_90a -O3 -std=c++17 -w '$SRC' -o /tmp/deepep_stream_event && /tmp/deepep_stream_event"

echo ""
echo "===== Python（exec 真实 deep_ep/utils/event.py + comm/stream.py + buffers/base.py） ====="
"$ROOT/scripts/lab.sh" exec "DEEPEP_ROOT=$DEP python '$PY'"
