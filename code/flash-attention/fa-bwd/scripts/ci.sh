#!/usr/bin/env bash
# fa-bwd 端到端回归的单一入口（P3-4d，ROADMAP 第一百一十七轮）。
#
# 一条命令串起：跑 ours（单/两文件）→ 数值汇总 → 单/两文件一致性 gate（按 dtype ulp 容差）
# → docs/04 内嵌数值表校验（陈旧即非零退出）。可选：--hopper 走 Hopper 快路构建；
# --perf-baseline <dtype> 额外跑纯反向基线（FA2/FA3/TE）。
#
# 用法（宿主机，任意目录）：
#   scripts/ci.sh                       # 全量扫 + gate + docs 校验（默认 sm_90/mma）
#   scripts/ci.sh --no-run              # 不用 GPU：只复核已有 npy + gate + docs 校验
#   scripts/ci.sh --dtype fp8 --hopper  # 只跑 fp8、走 TMA+wgmma 快路
#   scripts/ci.sh --perf-baseline fp16  # 另跑纯反向基线落盘
# 环境变量：LAB_NAME（默认 kernel_lab）。
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
cd "$ROOT"
exec python harness/fa_bwd_run.py --ci "$@"
