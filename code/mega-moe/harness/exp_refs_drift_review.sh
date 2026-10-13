#!/usr/bin/env bash
# P143-1 driver：跑「引用行号 linter + near_miss 分类 + 复核表」三件套，产出 exp_refs_drift_review.out.txt。
#   bash harness/exp_refs_drift_review.sh > harness/exp_refs_drift_review.out.txt
# 纯文件解析，不加载 GPU；在 lab 里跑只为与专题其它实验同环境。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
"$ROOT/scripts/lab.sh" exec "
  python3 '$ROOT/harness/check_refs.py'   ;
  echo                                                                       ;
  echo '########## triage_refs.py ##########'                                ;
  python3 '$ROOT/harness/triage_refs.py'                                     ;
  echo                                                                       ;
  echo '########## fix_refs.py ##########'                                   ;
  python3 '$ROOT/harness/fix_refs.py'
"
