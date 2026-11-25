#!/usr/bin/env bash
# AI 日报任务：抓 HF Daily Papers 某天论文 → 写速览 → 构建 → 提交推送。
#
# 由 systemd user timer 调用（21:00 主跑、23:30 与次日 09:00 补跑）。
#
# 手动补跑某一天：
#   FORCE_DATE=2026-11-06 $HOME/proj/tech_record/scripts/daily-ai-digest.sh
#
# 退出码：0 成功/跳过；2 抓取失败；3 自检未过；4 推送失败；5 线上未生效
set -uo pipefail

REPO="${HOME}/proj/tech_record"
LOG_DIR="${HOME}/log/ai-daily"
DSH_BIN="${DSH_BIN:-${HOME}/dsh-run/node_modules/.bin/dsh}"
# 本地 vLLM 端点不校验 key，但 dsh 的 provider 走了 apiKeyEnv，必须有个非空值。
export DS41_API_KEY="${DS41_API_KEY:-local-no-auth}"
# cron 的 PATH 很短，把 hugo/git/python 的位置补上。
export PATH="/usr/local/bin:/usr/bin:/bin:${HOME}/.local/bin:${PATH}"

# 目标日期：默认写「昨天」。HF 的 daily 榜在当天会持续进人（2026-11-19 实测：当晚
# 23:30 只有 15 篇，次日 21:00 已经 52 篇），当天写必然只是快照，之后要补遗几十篇。
# 隔一天写昨天，拿到的就是稳定终值，一次写对。FORCE_DATE=<日期> 可覆盖。
#
# 晚上写昨天（D-1，已有 21 小时积累）；早上那次写前天（D-2），
# 只在前一天漏跑时才有活干（已存在会直接跳过）。
if [[ -n "${FORCE_DATE:-}" ]]; then
  TARGET_DATE="${FORCE_DATE}"
elif (( 10#$(date +%H) < 12 )); then
  TARGET_DATE="$(date -d '2 days ago' +%F)"
else
  TARGET_DATE="$(date -d 'yesterday' +%F)"
fi
LOG="${LOG_DIR}/${TARGET_DATE}.log"

mkdir -p "${LOG_DIR}" "${HOME}/tmp/ai-daily"
# 只保留最近 30 天日志
find "${LOG_DIR}" -name '*.log' -type f -mtime +30 -delete 2>/dev/null

log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "${LOG}"; }

cd "${REPO}" || { log "FATAL: repo 不存在 ${REPO}"; exit 1; }
log "=== 开始：target_date=${TARGET_DATE} ==="

# 已经有文章就跳过（防重跑）
if [[ -d "content/posts/ai-daily-${TARGET_DATE}" ]]; then
  log "已存在 content/posts/ai-daily-${TARGET_DATE}/，跳过"
  exit 0
fi

# 先抓数据，确认那天真的有论文（周末/假日可能是 0 篇）
FETCH_ERR="${HOME}/tmp/ai-daily/.fetch-err"
if ! python3 code/ai-daily/fetch_papers.py --date "${TARGET_DATE}" \
      --out "${HOME}/tmp/ai-daily/${TARGET_DATE}.json" \
      --brief "${HOME}/tmp/ai-daily/${TARGET_DATE}.md" >"${FETCH_ERR}" 2>&1; then
  cat "${FETCH_ERR}" >>"${LOG}"
  # 「还没换榜」不是故障：日期页会回吐前一天的列表，fetch 会明确拒绝并说明。
  if grep -q "还没换榜" "${FETCH_ERR}"; then
    log "daily 还没换榜，本次跳过（等补跑）"
    exit 0
  fi
  log "抓取失败，退出"
  exit 2
fi
cat "${FETCH_ERR}" >>"${LOG}"
DATA_JSON="${HOME}/tmp/ai-daily/${TARGET_DATE}.json"
DOC="code/ai-daily/docs/${TARGET_DATE}.md"
COUNT="$(python3 -c "import json;print(json.load(open('${DATA_JSON}'))['count'])" 2>/dev/null || echo 0)"
log "抓到 ${COUNT} 篇"
# 门槛：>=8 篇才写。周末/假日 daily 可能为空，直接跳过。
MIN="${MIN_PAPERS:-8}"
if [[ "${COUNT}" -lt "${MIN}" ]]; then
  log "论文数 ${COUNT} < ${MIN}（daily 还没铺开/周末），本次跳过；下次跑会自然补上"
  exit 0
fi

TASK="$(cat <<EOF
你是 tech_record 博客的维护 agent。现在是北京时间 $(date '+%F %T')。

任务：为 Hugging Face Daily Papers 的 ${TARGET_DATE} 这一期写一篇「AI 日报」，发布到站点专题 ai-daily。
先读仓库里的约定再动手：agent_guide.md、agent_skills/ai-daily.md。

## 你只负责写作，构建与推送由外层脚本做
不要执行 hugo 构建、不要 git commit / push、不要联网抓数据（数据已备好）。

## 数据
${DATA_JSON}（字段 id/title/abstract/organization/upvotes/published_at/github/project）
${HOME}/tmp/ai-daily/${TARGET_DATE}.md（brief，罗列用）
直接读 JSON，摘要自己重写并保留具体数字，禁止粘贴原文。

## 产出 1：正文 ${DOC}
- 开头 H1：\`# ${TARGET_DATE} · N 篇：<副标题>\`，一段导语（总量、上双、集中度、最值得看的线）。
- 紧跟 \`## 本日 Top 10\` 表格（排名 / 论文 / 票数 / 一句话）。
- 主体按主题分组（H2），组内按 upvotes 降序；每篇 H3 \`### <序号>. <短名>（<upvotes>↑）\`，
  下接 4~6 条 \`- 问题\` \`- 方法\` \`- 数字\` \`- 为什么值得看\`。
  **必须覆盖全部论文**：篇数以 JSON 的 count 为准，一篇都不能漏；票数 <3 的可压到 2 条，但不能省。
- 结尾 \`## 今天的三个观察\` 与 \`## 数据与方法\`。

## 产出 2：文章壳 content/posts/ai-daily-${TARGET_DATE}/index.md
frontmatter：date 写裸日期 ${TARGET_DATE}、draft: false、series: ["ai-daily"]、categories: ["LLM算法"]，
weight 按「日期倒序」（导读恒 1，日报最新一天是 2，把比它旧的所有篇 weight 各 +1），
summary 一句带篇数与 upvotes；正文只留一两段导语 + {{< doc_include "${DOC}" >}}。

## 产出 3：两处索引
content/posts/ai-daily-00-index/index.md 的「文章索引」（最新在上）、根 README.md 的「AI 日报」小节。

## 输出预算（重要：上一版就是这里挂的）
单次输出有上限，几十篇一次写完会中断。所以：
- **分节追加**：用 bash \`cat >> ${DOC} <<'MD'\` 一次写 6~8 篇；
  **禁止**用 write 工具重写整个文件（每次重写都把全文重新输出一遍，O(n²) 直接撞上限）。
- 先写正文，再写壳与索引。本次调用没写完没关系，落盘即可，外层会再调用你续写。
- 少写长篇推理，直接干活。

## 自检（写完必须跑）
python3 -c "import json,re;d=json.load(open('${DATA_JSON}'));ids={p['id'] for p in d['papers']};have=set(re.findall(r'papers/(\d{4}\.\d{4,5})',open('${DOC}',encoding='utf-8').read()));print('缺',sorted(ids-have))"
输出「缺 []」才算写完。
EOF
)"

# ---- 写作循环：大日子单轮写不完，按覆盖度续写，最多 4 轮 ----
COVER_FILE="$(mktemp)"
cover() {
  python3 - "$DOC" "${DATA_JSON}" >"${COVER_FILE}" <<'PY'
import json, re, sys
doc, js = sys.argv[1:3]
ids = {p['id'] for p in json.load(open(js, encoding='utf-8'))['papers']}
try:
    txt = open(doc, encoding='utf-8').read()
except FileNotFoundError:
    txt = ''
have = set(re.findall(r'papers/(\d{4}\.\d{4,5})', txt))
miss = sorted(ids - have)
print(len(ids & have), len(ids), ' '.join(miss))
PY
}
PROMPT="${TASK}"
for ATTEMPT in 1 2 3 4; do
  log "调用 dsh headless（第 ${ATTEMPT} 轮）..."
  timeout 5400 "${DSH_BIN}" headless "${PROMPT}" >>"${LOG}" 2>&1
  RC=$?
  cover; read -r HAVE TOTAL MISS <"${COVER_FILE}"
  log "第 ${ATTEMPT} 轮结束：dsh 退出码 ${RC}，覆盖 ${HAVE:-0}/${TOTAL:-?} 篇"
  [[ -z "${MISS}" && "${HAVE:-0}" != "0" ]] && break
  [[ ${ATTEMPT} -eq 4 ]] && break
  PROMPT="【续写 · 第 $((ATTEMPT + 1)) 轮】上一轮因为单轮输出上限中断了。
${DOC} 目前覆盖 ${HAVE}/${TOTAL} 篇，还缺这些 id：
${MISS}

请继续完成原任务，只做两件事：
1. 先读现有文件，用 bash \`cat >> ${DOC}\` 追加缺的论文（一次 6~8 篇），禁止重写整个文件；
2. 确认「缺 []」后，补文章壳与两处索引。
不要做构建与推送。原任务要求如下：

${TASK}"
done
rm -f "${COVER_FILE}"
cover; read -r HAVE TOTAL MISS <"${COVER_FILE}" 2>/dev/null || true
log "最终覆盖：${HAVE:-0}/${TOTAL:-?} 篇${MISS:+（缺 ${MISS}）}"

# ---- 收尾由脚本做：修加粗 → 构建 → 自检 → 提交推送 → 验线上 ----
if [[ -f "${DOC}" ]]; then
  python3 code/ai-daily/fix_bold.py --fix "${DOC}" >>"${LOG}" 2>&1 || true
fi

HUGO_BIN="/tmp/hugo_bin/hugo"
[[ -x "${HUGO_BIN}" ]] || HUGO_BIN="$(command -v hugo || true)"
if [[ -n "${HUGO_BIN}" ]]; then
  rm -rf public
  "${HUGO_BIN}" --gc --minify --baseURL "https://blueskyyyyyy.github.io/tech_record/" >>"${LOG}" 2>&1
  log "构建退出码 $?"
else
  log "找不到 hugo，跳过构建"
fi

FAIL=0
[[ -f "content/posts/ai-daily-${TARGET_DATE}/index.md" ]] || { log "自检失败：缺文章壳"; FAIL=1; }
[[ -f "${DOC}" ]] || { log "自检失败：缺正文"; FAIL=1; }
if [[ -n "${MISS}" ]]; then
  log "自检失败：正文还缺 ${MISS}"
  FAIL=1
fi
if [[ -f "public/posts/ai-daily-${TARGET_DATE}/index.html" ]]; then
  # 排除 <code>…</code>：正文里会合法地出现反引号包住的 `**`（讲加粗退化那一段），不能误判
  if ! python3 -c '
import re, sys
html = open(sys.argv[1], encoding="utf-8").read()
body = html[html.find("<article"):html.find("</article")] if "<article" in html else html
body = re.sub(r"<code>.*?</code>", "", body, flags=re.S)
n = body.count("**")
print("   裸 ** 计数:", n)
sys.exit(1 if n else 0)
' "public/posts/ai-daily-${TARGET_DATE}/index.html" >>"${LOG}" 2>&1; then
    log "自检失败：渲染页面里还有裸 **（已排除 <code> 内合法用法）"
    FAIL=1
  fi
else
  log "自检失败：public/ 里没有产物（多为 frontmatter date 落到未来）"
  FAIL=1
fi

if [[ ${FAIL} -ne 0 ]]; then
  log "=== 自检未过：${TARGET_DATE}（未提交）==="
  exit 3
fi

# 注意要带上全系列：新一期会把比它旧的每一篇 weight 各 +1，只 add 当天目录会漏掉这些改动
git add content/posts/ai-daily-* "${DOC}" README.md >>"${LOG}" 2>&1
git -c user.name="BlueSkyyyyyy" -c user.email="BlueSkyyyyyy@users.noreply.github.com" \
    commit -q -m "post: AI 日报 ${TARGET_DATE}" >>"${LOG}" 2>&1
git push -q origin main >>"${LOG}" 2>&1 || { log "push 失败"; exit 4; }
log "已提交并推送"

sleep 90
CODE="$(curl -sL --max-time 25 -o /dev/null -w '%{http_code}' \
  "https://blueskyyyyyy.github.io/tech_record/posts/ai-daily-${TARGET_DATE}/")"
log "线上 HTTP ${CODE}"
[[ "${CODE}" == "200" ]] || { log "线上未生效"; exit 5; }
log "=== 完成并自检通过：${TARGET_DATE} ==="
