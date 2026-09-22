#!/usr/bin/env bash
# fa-bwd 项目的无人值守自驱循环（时限较短）。用法：
#   scripts/autopilot.sh {start|status|stop|run}
# 环境变量：MAX_ROUNDS(默认14) SLEEP_BETWEEN(30) TIMEOUT_PER_ROUND(5400)
set -uo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # code/flash-attention/fa-bwd
REPO="$(cd "$DIR/../../.." && pwd)"                       # 仓库根 tech_record
LOG="$DIR/autopilot.log"
STOP="$DIR/AUTOPILOT_STOP"
LOCK="$DIR/autopilot.lock"
PIDFILE="$DIR/autopilot.pid"
OPENCODE="/home/xieminglin/.opencode/bin/opencode"
MODEL="local-vllm//ssd/models/DeepSeek-V4.1-Flash"

MAX_ROUNDS="${MAX_ROUNDS:-14}"
SLEEP_BETWEEN="${SLEEP_BETWEEN:-30}"
TIMEOUT_PER_ROUND="${TIMEOUT_PER_ROUND:-5400}"

PROMPT='你是 tech_record 仓库的长期自驱 agent，负责推进「flash-attention 反向（fa-bwd）」项目。按以下步骤**只完成一个增量**，做完就停。
1. 先读 code/flash-attention/fa-bwd/ROADMAP.md 的「下一步」「当前进度」，再读 docs/00-fa-bwd-optimization-catalog.md、code/kernel-opt/agent_skills/kernel-opt.md（环境/ncu 权限/峰值常量）。
2. 完成 ROADMAP 里的**一个 P 项**（例如 P1-1 单文件 fp16 反向），代码写到 code/flash-attention/fa-bwd/src/<dtype>/。
3. 用 code/flash-attention/fa-bwd/scripts/run.sh 在 kernel_lab 容器里编译运行；用 scripts/ncu.sh 剖析；把实测原始输出留在同目录 *.out.txt。
4. 数值对拍：用自己的 kernel 读 /home/xieminglin/proj/output/fa-bwd/<case>/ 的 q,k,v,do.npy，算 dq,dk,dv，
   与 ref_*.npy（以及 TE/FA）比对，报告 max abs/rel diff；必要时用 harness/fa_bwd_bench.py dump 新 shape。
5. 性能对标：用 harness/fa_bwd_bench.py bench 或自己写计时，与 FA2/TE 同 shape 对比，给出 TFLOPS 与峰值占比。
6. 更新 docs/ 分析文档（实现与优化逐条说明、ncu bound 结论）与 ROADMAP（状态/进度/下一步），git add/commit/push，必要时验证线上（本项目文档不上站点，可只 push 代码与 md）。
关键要求：**单文件 + 两文件**两种形式；fp16、bf16、fp8（fp8 最重点，参考 TE 的 E4M3/E5M2 + rowwise scaling）。
**不要创建 AUTOPILOT_STOP（只有用户能停）。** 若遇无法解决的阻塞，写入 ROADMAP「阻塞」并结束本轮。
全程不要请求人工确认、不要用 question 类工具。GPU 用 kernel_lab 容器，ncu 必须在该容器里跑，git push 需要代理（NO_PROXY 已含本地 vLLM）。
**安全**：不要在仓库任何文件（文章/代码/注释/commit message）里出现目标 AI 卡的实际代号，一律用「目标卡」指代。
**博客**：code/flash-attention/fa-bwd/docs/*.md 会通过 fa_include shortcode 自动内联到博客专题，改 docs 即等于更新博客。'

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
  export http_proxy="socks5://127.0.0.1:12345" https_proxy="socks5://127.0.0.1:12345"
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
