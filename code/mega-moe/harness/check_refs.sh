#!/usr/bin/env bash
# P97-1 driver：在 kernel_lab(H100) 里跑引用行号 linter，产出 check_refs.out.txt。
#   bash harness/check_refs.sh > harness/check_refs.out.txt
# 纯文件解析，不加载 GPU；在 lab 里跑只为与专题其它实验同环境（仓库挂载方式一致）。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
"$ROOT/scripts/lab.sh" exec "python '$ROOT/harness/check_refs.py'"
