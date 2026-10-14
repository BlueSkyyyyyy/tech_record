#!/usr/bin/env bash
# flash-kda 专题的无人值守自驱循环。用法：
#   scripts/autopilot.sh {start|status|stop|run}
# 环境变量：MAX_ROUNDS(默认200) SLEEP_BETWEEN(30) TIMEOUT_PER_ROUND(5400)
set -uo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # code/flash-kda
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

PROMPT='你是 tech_record 仓库的长期自驱 agent，负责推进「Kimi Delta Attention（KDA）前向 / 反向算子实现解析」专题。**一轮只做一个增量**，做完就停。
1. 先读 code/flash-kda/ROADMAP.md 的「任务清单」「当前进度/下一步」「已知坑」，选**一个未完成子项**（优先 P0→P1→…顺序）。
2. 参考实现：forward 看 ~/github/FlashKDA（csrc/flash_kda.cpp、fwd.h、csrc/smxx/fwd_kernel1.cuh、fwd_kernel2.cuh、
   utils.cuh，以及 docs/20260420-flashkda-v1-deep-dive.md）。反向参照容器内 `pip install flash-linear-attention`
   的 fla.ops.kda.chunk_kda（用 torch.autograd 穿过去拿反向），必要时自行推导。
3. 分析文档写到 code/flash-kda/docs/<NN>-*.md（会被 doc_include 内联到博客，**里面不要写 {{< >}} 短代码**）；
   可运行参考实现写到 code/flash-kda/src/（纯 PyTorch，自带 if __name__ 自测）；探测/dump/对拍脚本写到 code/flash-kda/harness/。
4. 需要 GPU 的实验用 `code/flash-kda/scripts/lab.sh exec "..."` 在 kernel_lab 容器里跑（绝对路径），结果留同目录 *.out.txt。
   数值对拍要有真实数字（max abs/rel），性能用 CUPTI/自有计时；不确定的结论标明「待验证」。
5. 精读源码时贴**真实核对过的行号**（宁可不标也不标错）；讲清「为什么」而不是只描述「是什么」。
6. 更新 code/flash-kda/ROADMAP.md（状态/当前进度/下一步），git 只 add 本专题路径（不要 git add -A），
   commit 用 `kda:` 前缀，git pull --rebase 后 push；网络失败就只本地 commit 并在 ROADMAP 记「待推送」，不要阻塞、不要反复重试。
   用 `/tmp/hugo_bin/hugo --gc --minify -d /tmp/kda_hugo_check` 验证构建无 ERROR。
**安全**：不要在仓库任何文件（文章/代码/注释/commit message）里出现目标 AI 卡的实际代号，一律用「目标卡」指代。
**不要创建 AUTOPILOT_STOP（只有用户能停）。** 遇到无法自行解决的阻塞，写进 ROADMAP「阻塞」并结束本轮，不要反复重试同一件事。
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
