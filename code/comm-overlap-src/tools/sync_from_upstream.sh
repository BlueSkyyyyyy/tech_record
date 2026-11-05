#!/usr/bin/env bash
# 从上游重新拉取所有副本（== 丢弃全部注释，回到「纯上游」状态）。
#
# 用途：上游换 commit 后重建基线；或者某次注释改坏了想从头再来。
# 注意：这会**覆盖**本目录下所有带注释副本，注释不会保留。
#      正常流程是「改注释」而不是「重拉」，重拉只在升级上游时用。
#
# 用法：
#   ./tools/sync_from_upstream.sh            # 覆盖全部
#   ./tools/sync_from_upstream.sh uniep/     # 只覆盖路径含 uniep/ 的
set -euo pipefail

BASE="$(cd "$(dirname "$0")/.." && pwd)"
ROOT="${UPSTREAM_ROOT:-$HOME/github}"
FILTER="${1:-}"

# 仓库标记 -> 相对 $ROOT 的目录名（与 verify_annotations.py 的 REPO_DIRS 保持一致）
declare -A REPO_DIRS=(
  [triton-distributed]="Triton-distributed"
  [flux]="flux"
)

while IFS=$'\t' read -r repo upstream dest; do
  case "$repo" in \#*|"") continue ;; esac
  if [[ -n "$FILTER" && "$dest" != *"$FILTER"* ]]; then continue; fi

  src="$ROOT/${REPO_DIRS[$repo]:-}/$upstream"
  if [[ ! -f "$src" ]]; then
    echo "跳过（上游不存在）: $src" >&2
    continue
  fi
  mkdir -p "$BASE/$(dirname "$dest")"
  cp "$src" "$BASE/$dest"
  echo "同步 $dest"
done < "$BASE/tools/MANIFEST.tsv"

echo
echo "完成。现在跑一次校验（应当全部 OK，且注释行数为上游原有值）："
echo "  python3 $BASE/tools/verify_annotations.py"
