# publish — 构建、推送、验证部署

## 触发场景

文章写完/改完，需要发布上线；或任何推送到 main 之后确认部署成功。

## 前置条件

- 环境：`/tmp/hugo_bin/hugo` 存在（`hugo version` 应输出 0.140.2 extended）。不存在则重建：
  ```bash
  cd /tmp && curl -sL -o hugo.tar.gz \
    https://github.com/gohugoio/hugo/releases/download/v0.140.2/hugo_extended_0.140.2_linux-amd64.tar.gz \
    && mkdir -p hugo_bin && tar -xzf hugo.tar.gz -C hugo_bin hugo
  ```
  （release 资产下载偶发超时，重试即可；不要因此去升级主题——见 `agent_guide.md` 坑 #2。）

## 步骤

1. **本地构建**（仓库根目录）：
   ```bash
   cd /home/xieminglin/proj/tech_record
   /tmp/hugo_bin/hugo --gc --minify --baseURL "https://blueskyyyyyy.github.io/tech_record/"
   ```
   要求无 ERROR；新文章的目录应出现在 `public/posts/<slug>/`。
2. **公式自检**（文章含公式时）：跑 `math-check.md` 的脚本。
3. **提交推送**：
   ```bash
   git add -A
   git -c user.name="BlueSkyyyyyy" -c user.email="BlueSkyyyyyy@users.noreply.github.com" \
     commit -m "post: <主题>"   # 或 fix: / skill:
   git push
   ```
4. **验证 Actions**（推送后约 1~2 分钟）：
   ```bash
   sleep 90
   curl -sL "https://api.github.com/repos/BlueSkyyyyyy/tech_record/actions/runs?per_page=1" \
     | grep -E '"(status|conclusion)"' | head -2
   ```
   期望 `status: completed` + `conclusion: success`。失败时查哪个 job/step：
   ```bash
   curl -sL "https://api.github.com/repos/BlueSkyyyyyy/tech_record/actions/runs/<run_id>/jobs" \
     | grep -E '"(name|conclusion)"'
   ```
5. **验证线上页面**：
   ```bash
   curl -sL -o /dev/null -w "%{http_code}\n" \
     "https://blueskyyyyyy.github.io/tech_record/posts/<slug>/"
   ```
   每个新页面都要 200。注意 CDN 缓存：刚部署完偶尔 404，等 30s 重试再判定失败。

## 验收标准

- [ ] Actions 最近一次 run 为 success
- [ ] 所有新增/改动页面线上 HTTP 200
- [ ] 含公式页面已做 `math-check.md` 线上抽查

## 已知坑

- **Pages source 必须是 "GitHub Actions"**（仓库 Settings → Pages）。曾出现过 Setup Pages 步骤失败，原因就是这个设置没开；开错成 "Deploy from a branch" 会导致部署内容不对。
- Actions run 页面/状态通过未认证 API 查询即可（公开仓库），无需 gh CLI（本机未安装）。
- 不要 push 空提交来"碰运气"重试——先查失败 step 的日志再动手。
