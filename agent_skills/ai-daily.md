# ai-daily — 每日 HF Daily Papers 速览

## 触发场景

- 定时任务 / 用户说「做今天的 AI 日报」「总结今天的 AI 进展」；
- 补写某一天的日报（给定 `YYYY-MM-DD`）；
- 需要抓 HF Daily Papers 某天全部论文的清单（只要数据时用同一脚本）。

## 前置条件

- 已读根 `agent_guide.md`（尤其坑 #5 的日期语义——**这里最容易翻车**）。
- 目标文章目录：`content/posts/ai-daily-<date>/index.md`，正文文件：`code/ai-daily/docs/<date>.md`（由 `{{< doc_include >}}` 内联）。
- 系列参数已在 `hugo.toml` 注册：`seriesOrder` 与 `seriesTitles` 都有 `ai-daily`。

## 步骤

1. **抓数据**（在仓库根目录）：
   ```bash
   python3 code/ai-daily/fetch_papers.py --date <YYYY-MM-DD> \
     --out ~/tmp/ai-daily/<date>.json --brief ~/tmp/ai-daily/<date>.md
   ```
   失败会依次尝试 HF 官方 API → hf-mirror → 页面内嵌 JSON；全挂了再排查网络。**论文数为 0 就不写**（周末常见）。
2. **读 brief**，先看 `## 速览` 那张表定 Top 10，再逐篇读摘要。brief 里的 `ai_summary` 是机器生成的，只用来快速定位，**不要照抄**。
3. **写 `code/ai-daily/docs/<date>.md`**，格式固定（见 `code/ai-daily/README.md` 的「写作口径」）：
   - 第一行 `# <date> · N 篇：<当天三个主题词>`；
   - 导语 1–2 段（当天主线，不要复述标题）；
   - `## 本日 Top 10` 表（论文 + upvotes + 一句话）；
   - 主题分组（一、二、三…），组内 upvotes 降序，每篇 H3 `### n. 短名（N↑）` + 一行出处 + 3–5 bullet（问题 / 方法 / 数字 / 为什么值得看）；
   - `## 今天的三个观察`；
   - `## 数据与方法`（抓取命令、排序口径、daily 日 vs arXiv 日）。
4. **建文章壳** `content/posts/ai-daily-<date>/index.md`：
   ```yaml
   ---
   title: "AI 日报 <date>：N 篇 HF Daily Papers — <副题>"
   subtitle: "<一句话看点>"
   date: <date>          # 裸日期，见坑
   draft: false
   weight: <上一篇 + 1>
   series: ["ai-daily"]
   tags: ["AI 日报", "系列", "论文速览", ...]
   categories: ["LLM算法"]
   summary: "..."
   ---
   ```
   正文 = 2–3 段导语 + `{{< doc_include "code/ai-daily/docs/<date>.md" >}}`。
5. **更新两处索引**：`content/posts/ai-daily-00-index/index.md` 的「文章索引」（最新在上）、根 `README.md` 的「AI 日报」小节。
6. **构建验证**：`/tmp/hugo_bin/hugo --gc --minify`，确认 `public/posts/ai-daily-<date>/` 存在（不存在多半是日期落到未来，见坑）。
7. **发布**：走 `publish.md`（commit message：`post: AI 日报 <date>`）。
   **只提交本任务相关文件**，不要 `git add -A`——仓库里常有别的专题的半成品改动：
   ```bash
   git add content/posts/ai-daily-<date> code/ai-daily/docs/<date>.md \
           content/posts/ai-daily-00-index/index.md README.md
   ```

## 验收标准

- [ ] 论文数与当天 daily 实际条数一致（脚本输出里核对）
- [ ] 每篇 bullet 里的数字都能在摘要里找到出处
- [ ] `hugo` 构建无 ERROR，`public/posts/ai-daily-<date>/index.html` 存在
- [ ] 系列页 `/series/ai-daily/` 能看到新篇，且顺序正确（weight 递增）
- [ ] 两处索引已更新

## 已知坑

- **日期语义**：Hugo 对 `date: 2026-11-09T21:00:00+08:00` 的处理与本机 `date` 不一致（实测按 UTC 墙钟比较），21:00 的任务在 21:00 前后构建会被判成「未来」而**静默不输出**。**一律写裸日期** `date: <daily 日期>`，它 ≤ 今天，永远安全。
- **`weight` 必须递增**：系列内排序是 `weight` + slug 数字前缀；新篇 weight = 当前最大 + 1，否则会插到中间。
- **同一天不要重复跑同一个日期**：先看 `content/posts/ai-daily-<date>/` 是否已存在，存在则只增量修订，不要新建第二篇。
- **upvotes 会变**：文中注明「截至抓取时间」，同一篇隔天重抓数字会不同，不要为此改旧文。
- **不要把 brief 直接贴进正文**：brief 是素材，正文必须重写、必须有观点。
- **`**加粗**` 不要以「」（）收尾**：正文走 `doc_include`（`readFile|markdownify`），这类加粗会退化成字面星号（见 `agent_guide.md` 坑 #8）。写完 grep 一下 `\*\*「` 和 `\*\*（`，或构建后确认页面里没有裸 `**`。
- **frontmatter 的 `weight`**：`ai-daily-00-index` 是 1，之后按期递增（2、3、4…）；不要用日期当 weight。
