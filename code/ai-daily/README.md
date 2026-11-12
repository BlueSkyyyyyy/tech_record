# ai-daily — Hugging Face Daily Papers 日报

每天 21:00 抓取 [HF Daily Papers](https://huggingface.co/papers) 当天的论文，按主题分组写一篇中文速览，发到本站的新专题「AI 日报」。

## 目录

```
code/ai-daily/
├── fetch_papers.py            # 抓取脚本（标准库，API + 镜像 + 页面兜底）
├── docs/
│   └── YYYY-MM-DD.md          # 每天一篇速览正文，由 content/posts/ai-daily-YYYY-MM-DD/index.md 内联渲染
└── README.md
```

文章页与 `docs/<date>.md` 是「壳 + 内容」的关系：`index.md` 只放 frontmatter 和一段导语，正文用 `{{< doc_include >}}` 内联 `docs/` 里的文件。改内容只改 `docs/`，博客下次构建自动同步。

## 每天怎么做

```bash
# 1. 抓数据（默认昨天；也可显式给日期）
python3 code/ai-daily/fetch_papers.py --date 2026-11-06 \
  --out ~/tmp/ai-daily/2026-11-06.json \
  --brief ~/tmp/ai-daily/2026-11-06.md

# 2. 读 brief，挑重点写 docs/<date>.md（不要直接贴 brief，摘要必须自己重写并保留数字）

# 3. 建文章壳 + 构建
mkdir -p content/posts/ai-daily-<date>
#    index.md 的 frontmatter：series: ["ai-daily"]、weight = 上一篇 + 1、draft: false
/tmp/hugo_bin/hugo --gc --minify --baseURL "https://blueskyyyyyy.github.io/tech_record/"
```

## 写作口径（保持每天一致才有可比性）

- **分组**：Agent / 具身 / 图像生成 / LLM 结构与可解释性 / 评测 / 表示与安全 / 其它。组内按 upvotes 降序。
- **每篇 3–5 个 bullet**：问题 → 方法（一句话能说清的关键取舍）→ 数字（**必须带具体数值**，如 25×、70.2% vs 62.6%）→ 一句「为什么值得看 / 对本站主干的意义」。
- **Top 10 表**置顶，给「一句话」。
- **结尾三个观察**：当天真正的新意在哪、哪些是包装、和前几天比有什么变化。
- **7 天内的日期不重复写同一篇**：同一 arXiv id 在近期日报里出现过就只加一句「已在 X 日提过」。
- **只写摘要里有的数字**，不猜；机构与 upvotes 取自接口原始字段。
- 与本站主线（训练系统、kernel、优化器、agent 工程）相关的论文，bullet 里要点明「能落到哪个工程问题上」。

## 数据源与坑

- 官方 API：`https://huggingface.co/api/daily_papers?date=YYYY-MM-DD`（首选，返回 JSON 数组）。
- 本机直连 huggingface.co 会被解析到不可用地址，脚本自动退到 `hf-mirror.com` 的同一 API；再不行就解析 `/papers/date/<date>` 页面里 Svelte 的 `data-props` JSON。
- `submittedOnDailyAt` 才是「登上 daily 的日期」，`publishedAt` 是 arXiv 首发日，两者常差 1–3 天，标题里用前者。
- 周末/节假日 daily 可能只有很少几篇，甚至为空；空的话当天不发文，不要硬凑。
- upvotes 会变（发布后几小时还在涨），写稿时以抓取那一刻为准，文中注明「截至抓取时间」。

## 自测

```bash
python3 code/ai-daily/fetch_papers.py --test   # 离线：页面解析 / 去重 / 排序 / 摘要截断
```

## 定时任务（每天 21:00）

入口脚本：`scripts/daily-ai-digest.sh`（抓数据 → 少于 5 篇就跳过 → 调 `dsh headless` 写稿/构建/推送）。
日志：`~/log/ai-daily/<date>.log`，保留 30 天。

已安装 **systemd user timer**（本机 `crontab` 不可写，所以用 systemd）：

```bash
systemctl --user list-timers ai-daily-digest.timer   # 看下次触发时间
systemctl --user start  ai-daily-digest.service      # 立刻手动跑一次
systemctl --user status ai-daily-digest.service      # 上次运行结果
journalctl --user -u ai-daily-digest -n 50           # 服务日志（脚本日志仍在 ~/log/ai-daily/）

# 卸载
systemctl --user disable --now ai-daily-digest.timer
```

unit 文件在 `~/.config/systemd/user/ai-daily-digest.{service,timer}`，已 `enable-linger`（注销后仍会触发）。

备选：如果你有可写的 crontab，等价写法是

```cron
0 21 * * *  $HOME/proj/tech_record/scripts/daily-ai-digest.sh >> $HOME/log/ai-daily/cron.log 2>&1
```

手动补某一天（不会覆盖已有文章）：

```bash
FORCE_DATE=2026-11-06 ~/proj/tech_record/scripts/daily-ai-digest.sh
```

### 为什么 21:00 要允许「跳过」

HF 的 daily 列表在 UTC 白天才逐步补齐，且**周末经常只有 0–3 篇**。脚本的规则是「当天 <5 篇就不写」，第二天自然补新的一期；连续几天都空说明上游没更新，不是脚本坏了。

### 抓取口径（2026-11 实测，都是坑）

- 上游接口会**部分返回**：`?date=` 在批次没收完时常返回 0 或只给几篇，`/papers/date/` 页面也不全。所以 `fetch_papers.py` 现在是**多源按 paper id 取并集**（最新一批 × 多个 limit、`?date=`、日期页）。
- **「最新一批」不同 limit 返回不同切片**：2026-11-12 实测 `limit=100` 是旧快照（没有当天），`limit=30` 才有 17 篇。脚本因此固定按 30/100/50 各取一次再合并。
- **官方域名在本机不可达**（DNS 污染 / connection reset），脚本对连不上的域名做熔断，不会为每个接口白等一个超时。
- **daily 列表当天晚上还会继续进人**：21:00 抓到的通常只有最终量的 60%~80%。标题里的篇数是「抓取时点快照」，不是终值；要终值得第二天再抓一次。
- 写完正文先跑 `python3 code/ai-daily/fix_bold.py --fix code/ai-daily/docs/<日期>.md`：`**…**` 以 CJK 标点收尾会在 `doc_include` 里退化成字面星号。

### 漂移体检（`reconcile.py`）

因为 daily 列表当天晚上还在进人，隔天再查会发现「发布时 23 篇、现在 39 篇」。体检：

```bash
python3 code/ai-daily/reconcile.py --since 2026-11-09
```

它逐个已发布日期重抓，输出「发布 N 篇 / 上游 M 篇 | 缺 K 篇」，并列出新增的论文号。有缺口就照「补遗」套路补：在 `## 今天的三个观察` 前加一节 `## 补遗：…篇`，同步 H1 篇数、首段数字、`数据与方法`，再跑 `fix_bold.py`。
