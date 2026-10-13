#!/usr/bin/env bash
# P123-1 driver：在 kernel_lab(H100) 里跑 near_miss 定点分类，产出 triage_refs.out.txt。
#   bash harness/triage_refs.sh > harness/triage_refs.out.txt
# 纯文件解析，不加载 GPU；在 lab 里跑只为与专题其它实验同环境（仓库挂载方式一致）。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
"$ROOT/scripts/lab.sh" exec "python '$ROOT/harness/check_refs.py' > /dev/null; python '$ROOT/harness/triage_refs.py'"
