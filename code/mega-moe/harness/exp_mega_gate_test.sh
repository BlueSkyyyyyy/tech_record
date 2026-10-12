#!/usr/bin/env bash
# P47-1 实验 driver：Mega Gate 的端到端验收测试与路由 oracle（tests/test_mega_gate.py）。
#
# 为什么 H100(sm90) 可跑：
#   * 路由 oracle（sqrt-softplus 打分、带 bias 排名、top-k 权重归一化、logical->physical
#     逆映射、crossing 容忍、mask/fix_routing/force_random）全是纯 torch，架构无关；
#   * 本脚本直接 import 真实 tests/test_mega_gate.py 复用它的 helper，
#     用一个「完全正确的 torch 代理 gate」喂 check_topk_weights，再用扰动反例验证它能抓错。
# 真正的主体 kernel `sm100_bf16_mega_gate` 是 SM100 专属（mega_gate.hpp:159 arch==10），不涉及。
#   bash harness/exp_mega_gate_test.sh > harness/exp_mega_gate_test.out.txt
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT/harness/exp_mega_gate_test.py"
"$ROOT/scripts/lab.sh" exec "nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader | head -1 \
  && python '$PY'"
