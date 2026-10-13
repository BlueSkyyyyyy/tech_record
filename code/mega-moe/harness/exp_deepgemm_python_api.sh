#!/usr/bin/env bash
# P74-1 实验 driver：DeepGEMM 的【Python 绑定面】（csrc/python_api.cpp + 11 个 register_apis）。
#
# 为什么 H100(sm90) 可跑：
#   * 整层是 pybind11 的 host 绑定，系统里 deep_gemm 是 ~/github/DeepGEMM 的 editable 安装，
#     `import deep_gemm` 直接拿到真实 `_C` 扩展；不触发任何 SM100 kernel。
#   * 实验把 Python 侧 `_C` 暴露的名字逐项对回 C++ 源码的 m.def / m.attr / submodule.def，
#     并核对 compiled_dims/recipe 默认值、别名同一性、子模块工厂、隐藏入口，再跑一个
#     架构无关的 bf16_gemm_nt 冒烟验证 alpha 简写 == epilogue.Alpha。
#
# 用法：bash harness/exp_deepgemm_python_api.sh > harness/exp_deepgemm_python_api.out.txt
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT/harness/exp_deepgemm_python_api.py"
DG=~/github/DeepGEMM

echo "DeepGEMM Python 绑定面 —— H100 真机"
"$ROOT/scripts/lab.sh" exec "nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader | head -1"
"$ROOT/scripts/lab.sh" exec "python -c \"import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)\""
echo "DeepGEMM 源码：$(git -C "$DG" rev-parse --short HEAD 2>/dev/null || echo '?')  ($(git -C "$DG" log -1 --format='%ci' 2>/dev/null || echo '?'))"
echo ""

"$ROOT/scripts/lab.sh" exec "DEEPGEMM_ROOT=$DG python '$PY'"
