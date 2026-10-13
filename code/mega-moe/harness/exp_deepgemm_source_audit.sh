#!/usr/bin/env bash
# P96-1 实验 driver：DeepGEMM / DeepEP 源码引用行号健康检查 + 算子 / API 全索引。
#   bash harness/exp_deepgemm_source_audit.sh > harness/exp_deepgemm_source_audit.out.txt
# 纯文件解析，不加载 GPU；在 kernel_lab(H100) 里跑以与专题其它实验同环境。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
"$ROOT/scripts/lab.sh" exec "cd '$ROOT/harness' && python exp_deepgemm_source_audit.py"
