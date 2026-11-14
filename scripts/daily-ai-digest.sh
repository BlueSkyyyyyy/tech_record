#!/usr/bin/env bash
# 每天 21:00 的 AI 日报任务：抓 HF Daily Papers 当天论文 → 写速览 → 构建 → 提交推送。
#
# 由 cron 调用（见 README「定时任务」）：
#   0 21 * * *  $HOME/proj/tech_record/scripts/daily-ai-digest.sh >> $HOME/log/ai-daily/cron.log 2>&1
#
# 手动补跑某一天：
#   FORCE_DATE=2026-11-06 $HOME/proj/tech_record/scripts/daily-ai-digest.sh
#
# 退出码：0 成功/今天已有（跳过）；非 0 失败（看 ~/log/ai-daily/<date>.log）
set -uo pipefail

REPO="${HOME}/proj/tech_record"
LOG_DIR="${HOME}/log/ai-daily"
DSH_BIN="${DSH_BIN:-${HOME}/dsh-run/node_modules/.bin/dsh}"
# 本地 vLLM 端点不校验 key，但 dsh 的 provider 走了 apiKeyEnv，必须有个非空值。
export DS41_API_KEY="${DS41_API_KEY:-local-no-auth}"
# cron 的 PATH 很短，把 hugo/git/python 的位置补上。
export PATH="/usr/local/bin:/usr/bin:/bin:${HOME}/.local/bin:${PATH}"

# 目标日期：显式 FORCE_DATE 优先；否则中午前当作「补昨天的」，中午后才是「写今天」。
# （HF 的 daily 列表有时当天 21:00 还没换榜，次日早上补跑才有货；见 agent_skills/ai-daily.md）
if [[ -n "${FORCE_DATE:-}" ]]; then
  TARGET_DATE="${FORCE_DATE}"
elif (( 10#$(date +%H) < 12 )); then
  TARGET_DATE="$(date -d 'yesterday' +%F)"
else
  TARGET_DATE="$(date +%F)"
fi
STAMP="$(date +%Y%m%d-%H%M%S)"
LOG="${LOG_DIR}/${TARGET_DATE}.log"

mkdir -p "${LOG_DIR}" "${HOME}/tmp/ai-daily"
# 只保留最近 30 天日志
find "${LOG_DIR}" -name '*.log' -type f -mtime +30 -delete 2>/dev/null

log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "${LOG}"; }

cd "${REPO}" || { log "FATAL: repo 不存在 ${REPO}"; exit 1; }
log "=== 开始：target_date=${TARGET_DATE} ==="

# 今天已经有文章就跳过（防重跑）
if [[ -d "content/posts/ai-daily-${TARGET_DATE}" ]]; then
  log "已存在 content/posts/ai-daily-${TARGET_DATE}/，跳过"
  exit 0
fi

# 先抓数据，确认当天真的有论文（周末/假日可能是 0 篇）
FETCH_ERR="${HOME}/tmp/ai-daily/.fetch-err"
if ! python3 code/ai-daily/fetch_papers.py --date "${TARGET_DATE}" \
      --out "${HOME}/tmp/ai-daily/${TARGET_DATE}.json" \
      --brief "${HOME}/tmp/ai-daily/${TARGET_DATE}.md" >"${FETCH_ERR}" 2>&1; then
  cat "${FETCH_ERR}" >>"${LOG}"
  # 「还没换榜」不是故障：日期页会回吐前一天的列表，fetch 会明确拒绝并说明。
  # 这时按「本次跳过」处理（exit 0），等 23:30 / 次日 09:00 的补跑。
  if grep -q "还没换榜" "${FETCH_ERR}"; then
    log "daily 还没换榜，本次跳过（等补跑）"
    exit 0
  fi
  log "抓取失败，退出"
  exit 2
fi
cat "${FETCH_ERR}" >>"${LOG}"
COUNT="$(python3 -c "import json;print(json.load(open('${HOME}/tmp/ai-daily/${TARGET_DATE}.json'))['count'])" 2>/dev/null || echo 0)"
log "抓到 ${COUNT} 篇"
if [[ "${COUNT}" -lt 5 ]]; then
  log "论文数 ${COUNT} < 5（周末/假日 daily 可能没更新），本次跳过；下次跑会自然补上"
  exit 0
fi

TASK="$(cat <<EOF
你是 tech_record 博客的维护 agent。现在是北京时间 $(date '+%F %T')。

任务：为 Hugging Face Daily Papers 的 ${TARGET_DATE} 这一期写一篇「AI 日报」，发布到站点专题 ai-daily。

严格按仓库里的技能文档执行，先读再动手：
- agent_guide.md（仓库约定与已知坑）
- agent_skills/ai-daily.md（本任务的完整步骤与验收标准）
- agent_skills/publish.md（构建、提交、推送、验证）

关键约束（违反会导致返工）：
1. 数据已经抓好了：${HOME}/tmp/ai-daily/${TARGET_DATE}.json 与 .md。不要重复联网抓；直接读 brief。
2. 正文写 code/ai-daily/docs/${TARGET_DATE}.md，文章壳写 content/posts/ai-daily-${TARGET_DATE}/index.md（frontmatter 的 date 写裸日期 ${TARGET_DATE}，weight 按「日期倒序」：导读恒 1，日报最新一天是 2，把比今天旧的所有篇 weight 各 +1）。
3. 论文摘要必须自己重写并保留具体数字，禁止直接粘贴 brief 或 ai_summary；分组内按 upvotes 降序；Top 10 表放最前；结尾写「今天的三个观察」。
4. 同步更新两处索引：content/posts/ai-daily-00-index/index.md 的「文章索引」（最新在上）、根 README.md 的「AI 日报」小节。
5. 先修加粗：python3 code/ai-daily/fix_bold.py --fix code/ai-daily/docs/${TARGET_DATE}.md
   （**加粗** 以「」（）等 CJK 标点收尾会在 doc_include 里退化成字面星号；这个脚本会自动把标点挪到 ** 外面。必须做到它输出 bold ok。）
6. 构建验证：/tmp/hugo_bin/hugo --gc --minify，确认 public/posts/ai-daily-${TARGET_DATE}/ 存在、无 ERROR。
7. 提交并推送：只 add 本任务相关文件（content/posts/ai-daily-${TARGET_DATE}、code/ai-daily/docs/${TARGET_DATE}.md、content/posts/ai-daily-00-index/index.md、README.md），然后 git commit -m "post: AI 日报 ${TARGET_DATE}"、git push。不要 git add -A（仓库里有别的专题的半成品改动）；push 失败不要 force，把错误写进最终输出即可。
7. 全程不要向用户提问：遇到歧义按 agent_skills/ai-daily.md 的默认口径自行决定，并在最终输出里写明你的选择。

完成标准：线上 https://blueskyyyyyy.github.io/tech_record/posts/ai-daily-${TARGET_DATE}/ 返回 200（推送后等 90 秒再查）。若 Actions 部署失败，把失败原因写进最终输出，不要反复重试。
EOF
)"

log "调用 dsh headless ..."
timeout 4200 "${DSH_BIN}" headless "${TASK}" >>"${LOG}" 2>&1
RC=$?
log "dsh 退出码 ${RC}"

if [[ ${RC} -ne 0 ]]; then
  log "任务失败，见 ${LOG}"
  exit ${RC}
fi

# 交付自检：文章壳在不在、正文里有没有退化的字面 **、构建产物有没有
FAIL=0
[[ -f "content/posts/ai-daily-${TARGET_DATE}/index.md" ]] || { log "自检失败：缺 content/posts/ai-daily-${TARGET_DATE}/index.md"; FAIL=1; }
[[ -f "code/ai-daily/docs/${TARGET_DATE}.md" ]] || { log "自检失败：缺 code/ai-daily/docs/${TARGET_DATE}.md"; FAIL=1; }
# **…** 以 CJK 标点开头/结尾会退化成字面星号（agent_guide.md 坑 #8）；用 python 判，grep 的字节级正则对多字节不可靠
if [[ -f "code/ai-daily/docs/${TARGET_DATE}.md" ]] && ! python3 - "code/ai-daily/docs/${TARGET_DATE}.md" <<'PY'
import re, sys
BAD_START = set('「『（【《〈“‘([<')
BAD_END = set('」』）】》〉”’)]>、，。：；！？…—·')
text = open(sys.argv[1], encoding='utf-8').read()
bad = [m.group(1) for m in re.finditer(r'\*\*(.+?)\*\*', text, re.S)
       if m.group(1) and (m.group(1)[0] in BAD_START or m.group(1)[-1] in BAD_END)]
if bad:
    print('   退化的加粗:', bad[:5])
sys.exit(1 if bad else 0)
PY
then
  log "自检失败：正文里有以 CJK 标点收尾的 ** 加粗，会退化成字面星号（见 agent_guide.md 坑 #8）"
  FAIL=1
fi
[[ -f "public/posts/ai-daily-${TARGET_DATE}/index.html" ]] || { log "自检失败：public/ 里没有构建产物（多为 frontmatter date 落到未来）"; FAIL=1; }
if [[ -f "public/posts/ai-daily-${TARGET_DATE}/index.html" ]] && grep -q '\*\*' "public/posts/ai-daily-${TARGET_DATE}/index.html"; then
  log "自检失败：渲染后的页面里还有裸 **（markdownify 退化）"
  FAIL=1
fi

if [[ ${FAIL} -ne 0 ]]; then
  log "=== 交接完成但自检未过：${TARGET_DATE}（见上方说明）==="
  exit 3
fi
log "=== 完成并自检通过：${TARGET_DATE} ==="
