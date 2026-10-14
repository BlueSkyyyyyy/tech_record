#!/usr/bin/env bash
# P112-1 实验 driver：DeepEP 的两类非 all-to-all 通信原语（PP send/recv 与 Engram fetch）
# 的验收测试即正确性规范。
#   * tests/pp/test_pp.py      （152 行）：无死锁收发调度 generate_stress_ops + torch.equal 判据
#   * tests/engram/test_engram.py（155 行）：索引远程抓取的参考 gather oracle + FP8 SF 复制通路
#
# 为什么 H100(sm90) 只能跑「测试侧架构无关部分」：
#   * 两个端到端测试都要构造 buffer（PPBuffer / EngramBuffer），依赖 C++/NCCL 设备 API
#     （ncclDevComm/ncclGin）；本机 NCCL 2.29.7 与编译期不匹配 → deep_ep/__init__.py:55
#     check_nccl_so() abort；真正跑 kernel 还需 ≥2.31 的 NCCL。故 buffer 端到端**仅分析**。
#   * 测试侧纯 Python/张量逻辑可原样抽取/复刻运行：PP 调度生成、Engram 参考 gather oracle。
#
# 用法：bash harness/exp_deepep_pp_engram_test.sh > harness/exp_deepep_pp_engram_test.out.txt
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT/harness/exp_deepep_pp_engram_test.py"
DEEPEP=${HOME}/github/DeepEP

echo "DeepEP PP send/recv 与 Engram fetch 的验收测试即正确性规范 —— H100 真机（测试侧）"
"$ROOT/scripts/lab.sh" exec "nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader | head -1"
"$ROOT/scripts/lab.sh" exec "python -c \"import torch; print('torch', torch.__version__, 'nccl', torch.cuda.nccl.version())\""
echo "DeepEP 源码：$(git -C "$DEEPEP" rev-parse --short HEAD 2>/dev/null || echo '?')"
echo "tests/pp/test_pp.py        sha256[:16] $(sha256sum "$DEEPEP/tests/pp/test_pp.py" | cut -c1-16)"
echo "tests/engram/test_engram.py sha256[:16] $(sha256sum "$DEEPEP/tests/engram/test_engram.py" | cut -c1-16)"
echo ""

"$ROOT/scripts/lab.sh" exec "DEEPEP_ROOT=$DEEPEP TORCHDYNAMO_DISABLE=1 python '$PY'"
