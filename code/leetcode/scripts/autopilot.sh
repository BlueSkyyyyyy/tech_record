#!/usr/bin/env bash
# leetcode 专题的无人值守自驱循环。用法：
#   scripts/autopilot.sh {start|status|stop|run}
# 环境变量：MAX_ROUNDS(默认20) SLEEP_BETWEEN(20) TIMEOUT_PER_ROUND(3600)
set -uo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # code/leetcode
REPO="$(cd "$DIR/../.." && pwd)"                          # 仓库根 tech_record
LOG="$DIR/autopilot.log"
STOP="$DIR/AUTOPILOT_STOP"
LOCK="$DIR/autopilot.lock"
PIDFILE="$DIR/autopilot.pid"
OPENCODE="/home/xieminglin/.opencode/bin/opencode"
MODEL="local-vllm//ssd/models/DeepSeek-V4.1-Flash"

MAX_ROUNDS="${MAX_ROUNDS:-20}"
SLEEP_BETWEEN="${SLEEP_BETWEEN:-20}"
TIMEOUT_PER_ROUND="${TIMEOUT_PER_ROUND:-3600}"

PROMPT='你是 tech_record 仓库的长期自驱 agent，负责推进「LeetCode 题解精讲」专题。**一轮只做一个增量**，做完就停，宁可少而精。
1. 先读 code/leetcode/ROADMAP.md 的「任务清单」「每轮增量流程」「已知坑」「验证标准」，选**一个未完成子项**（优先把进行中的分类补满到 ≥10 题，再开新分类）。
2. 先写代码 code/leetcode/src/<category>/<problem>.py 与 .cpp，各自带 assert 自测；运行
   `python3 code/leetcode/scripts/run_all.py <filter>` 直到全绿（含 C++ 编译运行）。
3. 把**与 src 逐字一致**的解法粘贴进 code/leetcode/docs/<NN>-<category>.md，按固定档案补全：
   题目 → 思路（讲清“为什么”）→ 代码 → 复杂度 → 易错点 → 相似题；篇末写「规律总结」。相似题放在一起并交叉引用。
4. 若是新分类：同时创建 content/posts/leetcode-NN-<category>/index.md（frontmatter：draft: false、
   series: ["leetcode"]、weight 递增、categories: ["算法"]、tags 含分类名+"系列"），正文仅一行
   `{{< lc_include "code/leetcode/docs/<NN>-<category>.md" >}}`；并在 content/posts/leetcode-00-index/index.md 的分类表里把它改成 relref 链接。
5. 用 `/tmp/hugo_bin/hugo --gc --minify -d /tmp/lc_hugo_check` 验证构建无 ERROR。
6. git 只 add 自己改的路径（见 ROADMAP「git 提交边界」），**不要 git add -A**；commit 用 `leetcode:` 前缀；
   `git pull --rebase` 后 push。网络失败就只本地 commit 并在 ROADMAP 记「待推送」，不要阻塞、不要反复重试。
**质量要求（重要）**：逻辑清晰、通俗易懂；发布前至少通读一遍并多改几遍措辞；**不要写“上一篇/下一篇”式闲聊**；
站内链接只在 content/posts 文章里用 relref；被内联的 docs 里**绝不能**出现 {{< >}} 短代码。C++ assert 不要把 `{1,2}` 直接写进 assert 实参（逗号会被当宏参数）。
**不要创建 AUTOPILOT_STOP（只有用户能停）。** 遇到无法解决的阻塞，写进 ROADMAP「阻塞」并结束本轮。
全程不要请求人工确认、不要用 question 类工具。网络：git/opencode 直连（已 unset 代理）。'

start() {
  if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    echo "autopilot already running (pid $(cat "$PIDFILE"))"; return 0
  fi
  rm -f "$STOP"
  setsid nohup bash "$0" run >>"$LOG" 2>&1 < /dev/null &
  echo $! > "$PIDFILE"
  sleep 1
  echo "autopilot started (pid $(cat "$PIDFILE")), log: $LOG"
}

status() {
  if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    echo "RUNNING (pid $(cat "$PIDFILE"))"; else echo "NOT running"; fi
  echo "--- last 20 log lines ---"; tail -n 20 "$LOG" 2>/dev/null || echo "(no log)"
}

stop() {
  touch "$STOP"
  if [ -f "$PIDFILE" ]; then
    pid="$(cat "$PIDFILE")"; pkill -TERM -P "$pid" 2>/dev/null || true; kill -TERM "$pid" 2>/dev/null || true
    echo "stopped (pid $pid); stop file at $STOP"
  else echo "no pidfile; created stop file"; fi
}

run_loop() {
  if [ -f "$LOCK" ] && kill -0 "$(cat "$LOCK" 2>/dev/null)" 2>/dev/null; then
    echo "[autopilot] another loop holds lock; exit"; exit 0; fi
  echo $$ > "$LOCK"; trap 'rm -f "$LOCK"' EXIT
  export NO_PROXY="192.168.36.3,localhost,127.0.0.1" no_proxy="192.168.36.3,localhost,127.0.0.1"
  export OPENAI_BASE_URL="http://192.168.36.3:8005/v1" OPENAI_API_KEY="EMPTY"
  unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
  cd "$REPO" || exit 1
  local fail=0
  for round in $(seq 1 "$MAX_ROUNDS"); do
    if [ -f "$STOP" ]; then echo "[autopilot] stop file present; exit before round $round"; break; fi
    echo "==================== ROUND $round / $MAX_ROUNDS @ $(date '+%F %T') ===================="
    if timeout "$TIMEOUT_PER_ROUND" "$OPENCODE" run --auto --model "$MODEL" --dir "$REPO" "$PROMPT"; then fail=0; else
      rc=$?; fail=$((fail+1)); echo "[autopilot] round $round rc=$rc (fail=$fail)"
      if [ "$fail" -ge 3 ]; then echo "[autopilot] 连续失败 3 次，停止"; break; fi
    fi
    echo "==================== ROUND $round done @ $(date '+%F %T') ===================="
    sleep "$SLEEP_BETWEEN"
  done
  echo "[autopilot] loop finished @ $(date '+%F %T')"
}

case "${1:-}" in
  start) start ;; status) status ;; stop) stop ;; run) run_loop ;;
  *) echo "usage: $0 {start|status|stop|run}"; exit 1 ;;
esac
