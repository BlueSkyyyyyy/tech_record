#!/usr/bin/env bash
# mega-moe 专题的无人值守自驱循环。用法：
#   scripts/autopilot.sh {start|status|stop|run}
# 环境变量：MAX_ROUNDS(默认200) SLEEP_BETWEEN(30) TIMEOUT_PER_ROUND(5400)
set -uo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # code/mega-moe
REPO="$(cd "$DIR/../.." && pwd)"                          # 仓库根 tech_record
LOG="$DIR/autopilot.log"
STOP="$DIR/AUTOPILOT_STOP"
LOCK="$DIR/autopilot.lock"
PIDFILE="$DIR/autopilot.pid"
OPENCODE="${HOME}/.opencode/bin/opencode"
MODEL="local-vllm//ssd/models/DeepSeek-V4.1-Flash"

MAX_ROUNDS="${MAX_ROUNDS:-200}"
SLEEP_BETWEEN="${SLEEP_BETWEEN:-30}"
TIMEOUT_PER_ROUND="${TIMEOUT_PER_ROUND:-5400}"

PROMPT='你是 tech_record 仓库的长期自驱 agent，负责推进「DeepSeek Mega MoE 前向 / 反向算子实现解析」专题。**一轮只做一个增量**，做完就停。
1. 先读 code/mega-moe/ROADMAP.md 的「任务清单」「当前进度/下一步」「已知坑」，选**一个未完成子项**（优先 P0→P1→…）。
2. 主要参考：~/github/DeepGEMM（Mega MoE：csrc/apis/mega_moe.hpp、deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh、
   scheduler/mega_moe.cuh、layout/mega_moe.cuh、tests/test_mega_moe.py；grouped GEMM 与反向 wgrad k_grouped_fp8_gemm_tn_contiguous；
   Mega Gate / Mega mHC）与 ~/github/DeepEP（dispatch/combine、symmetric memory）。可结合线上资料（DeepGEMM README/release、
   PR #316/#95）但**不要编造 URL**，给真实链接。
3. **重要约束**：Mega MoE 主实现是 SM100（Blackwell），本机是 H100（sm90），端到端**跑不了**——这部分只做源码精读与线上资料分析，
   明确标注「SM100 专属，仅分析」。H100 上可行的数值实验（grouped GEMM / backward grouped GEMM / 通信原语）用
   `code/mega-moe/scripts/lab.sh exec "..."` 跑真实数字留 *.out.txt。
4. 分析文档写到 code/mega-moe/docs/<NN>-*.md（会被 doc_include 内联到博客，**里面不要写 {{< >}} 短代码**）；
   实验代码写到 code/mega-moe/src/、harness/。精读源码贴**真实核对过的行号**，讲清「为什么」。
5. 更新 code/mega-moe/ROADMAP.md（状态/当前进度/下一步），git 只 add 本专题路径（不要 git add -A），
   commit 用 `megamoe:` 前缀，git pull --rebase 后 push；网络失败就只本地 commit 并在 ROADMAP 记「待推送」，不要阻塞、不要反复重试。
   用 `/tmp/hugo_bin/hugo --gc --minify -d /tmp/megamoe_hugo_check` 验证构建无 ERROR。
**安全**：不要在仓库任何文件（文章/代码/注释/commit message）里出现目标 AI 卡的实际代号，一律用「目标卡」指代。
**不要创建 AUTOPILOT_STOP（只有用户能停）。** 遇到无法自行解决的阻塞，写进 ROADMAP「阻塞」并结束本轮。
全程不要请求人工确认、不要用 question/交互类工具，自主决策。网络：git/opencode 直连（已 unset 代理）。'

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
