#!/usr/bin/env bash
# 无人值守自驱：反复启动 opencode 无头会话，每次推进「CUDA 算子调优」路线图的一步，
# 直到达到停止条件。适合用户离开电脑后让 agent 持续工作。
#
# 用法：
#   scripts/autopilot.sh start     # 后台启动（setsid 脱离终端，关终端也不停）
#   scripts/autopilot.sh status    # 查看是否在跑 / 最近日志
#   scripts/autopilot.sh stop      # 停止
#   scripts/autopilot.sh run       # 前台跑（调试用）
#
# 环境变量：
#   MAX_ROUNDS=40        最多跑多少轮（每轮=一篇文章增量）
#   SLEEP_BETWEEN=30     每轮之间休息秒数
#   TIMEOUT_PER_ROUND=5400  单轮超时（秒），防止卡死
set -uo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # code/kernel-opt
REPO="$(cd "$DIR/../.." && pwd)"                          # 仓库根
LOG="$DIR/autopilot.log"
STOP="$DIR/AUTOPILOT_STOP"
LOCK="$DIR/autopilot.lock"
PIDFILE="$DIR/autopilot.pid"
OPENCODE="/home/xieminglin/.opencode/bin/opencode"
MODEL="local-vllm//ssd/models/DeepSeek-V4.1-Flash"

MAX_ROUNDS="${MAX_ROUNDS:-40}"
SLEEP_BETWEEN="${SLEEP_BETWEEN:-30}"
TIMEOUT_PER_ROUND="${TIMEOUT_PER_ROUND:-5400}"

PROMPT='你是 tech_record 仓库的长期自驱 agent。按以下步骤**只完成一个增量**，做完就停，不要贪多。
1. 先读 code/kernel-opt/ROADMAP.md 的「下一步」和「当前进度」，再读 agent_guide.md 与 agent_skills/kernel-opt.md。
2. 挑其中**一篇**（或一个明确的增量），新增/修改代码到 code/kernel-opt/NN-*/；用 code/kernel-opt/scripts/run.sh 在 kernel_lab 容器里编译运行、与参考实现对拍，用 scripts/ncu.sh 采集关键指标；实测输出存到该目录 *.out.txt。
3. 按 agent_skills/write-post.md 写文章 content/posts/cuda-kernel-opt-NN-<slug>/index.md（draft:false，带 weight 和系列 tag），数字必须来自实测，讲清楚优化思路与过程，配 ASCII/表格图示优先。
4. 按 agent_skills/publish.md：/tmp/hugo_bin/hugo --gc --minify 确认无 ERROR → git add/commit（post:/bench:/skill:）→ push → 验证线上页面 HTTP 200。
5. 更新 code/kernel-opt/ROADMAP.md（状态、当前进度、下一步）、README.md、code/README.md 索引。
6. 如果整个系列已完成（ROADMAP 无未完成项），执行 `touch code/kernel-opt/AUTOPILOT_STOP` 后结束。
若遇到无法自行解决的阻塞，把原因写进 ROADMAP 的「阻塞」小节并结束本轮，不要反复重试同一件事。
全程不要请求人工确认、不要使用 question/交互类工具，自主决策。
注意：GPU 用 kernel_lab 容器；ncu 必须在该容器里跑；git push 需要代理（NO_PROXY 已包含本地 vLLM 地址）。'

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
    echo "RUNNING (pid $(cat "$PIDFILE"))"
  else
    echo "NOT running"
  fi
  echo "--- last 20 log lines ---"
  tail -n 20 "$LOG" 2>/dev/null || echo "(no log yet)"
}

stop() {
  touch "$STOP"
  if [ -f "$PIDFILE" ]; then
    pid="$(cat "$PIDFILE")"
    pkill -TERM -P "$pid" 2>/dev/null || true
    kill -TERM "$pid" 2>/dev/null || true
    echo "stopped (pid $pid); stop file at $STOP"
  else
    echo "no pidfile; created stop file"
  fi
}

run_loop() {
  if [ -f "$LOCK" ] && kill -0 "$(cat "$LOCK" 2>/dev/null)" 2>/dev/null; then
    echo "[autopilot] another loop holds lock ($(cat "$LOCK")); exit"; exit 0
  fi
  echo $$ > "$LOCK"
  trap 'rm -f "$LOCK"' EXIT

  # 本地 vLLM 不走代理；git / 外网仍走 socks 代理
  export NO_PROXY="192.168.36.3,localhost,127.0.0.1"
  export no_proxy="192.168.36.3,localhost,127.0.0.1"
  export OPENAI_BASE_URL="http://192.168.36.3:8005/v1"
  export OPENAI_API_KEY="EMPTY"
  export http_proxy="socks5://127.0.0.1:12345"
  export https_proxy="socks5://127.0.0.1:12345"

  cd "$REPO" || exit 1
  local fail=0
  for round in $(seq 1 "$MAX_ROUNDS"); do
    if [ -f "$STOP" ]; then echo "[autopilot] stop file present; exit before round $round"; break; fi
    echo "==================== ROUND $round / $MAX_ROUNDS @ $(date '+%F %T') ===================="
    if timeout "$TIMEOUT_PER_ROUND" "$OPENCODE" run --auto --model "$MODEL" --dir "$REPO" "$PROMPT"; then
      fail=0
    else
      rc=$?
      fail=$((fail+1))
      echo "[autopilot] round $round exited rc=$rc (consecutive failures=$fail)"
      if [ "$fail" -ge 3 ]; then echo "[autopilot] 连续失败 3 次，停止"; break; fi
    fi
    echo "==================== ROUND $round done @ $(date '+%F %T') ===================="
    sleep "$SLEEP_BETWEEN"
  done
  echo "[autopilot] loop finished @ $(date '+%F %T')"
}

case "${1:-}" in
  start) start ;;
  status) status ;;
  stop) stop ;;
  run) run_loop ;;
  *) echo "usage: $0 {start|status|stop|run}"; exit 1 ;;
esac
