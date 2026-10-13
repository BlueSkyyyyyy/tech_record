#!/usr/bin/env bash
# P52-1 实验 driver：DeepEP 的确定性 dispatch 排序 `EPHandle.deterministic_sort`。
#
# 为什么 H100(sm90) 可跑真实代码：
#   * `deterministic_sort`（`deep_ep/buffers/ep.py`）是纯 Python + `torch.sort`/`index_select`，
#     **完全不碰 NCCL / 对称显存 / kernel**；它作用在 dispatch 已落地的本地张量上。
#   * 本机 `import deep_ep` 会在包 `__init__` 的 `check_nccl_so()` 被拒（NCCL 2.29.7 vs 期望
#     2.32.3，见 ROADMAP「阻塞」）。故实验**不 import 包**，用 `importlib.util.find_spec` 定位
#     已安装 `ep.py`，`ast` **逐行原样抽取真实方法**（打印 sha256）后运行；独立 host oracle 对拍。
#   bash harness/exp_deepep_deterministic_sort.sh > harness/exp_deepep_deterministic_sort.out.txt
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT/harness/exp_deepep_deterministic_sort.py"
"$ROOT/scripts/lab.sh" exec "nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader | head -1 \
  && python '$PY'"
