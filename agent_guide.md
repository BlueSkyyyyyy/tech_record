# Agent Guide — tech_record

AI agent 在本仓库工作的入口文档。先读本文件，再按任务查 `agent_skills/` 下的具体技能。

## 这个仓库是什么

个人技术博客 + 配套代码。AI 训练系统方向（Megatron-LM、Kernel、优化器、量化）。中文为主。

- 在线站点：https://blueskyyyyyy.github.io/tech_record/
- Hugo 0.140.2 extended + PaperMod v8.0（vendored 于 `themes/PaperMod/`，非 submodule）
- Hugo 二进制：`/tmp/hugo_bin/hugo`（临时目录，机器重启后需重新下载；见 `agent_skills/publish.md` §环境）

## 目录约定（最重要）

```
content/posts/<slug>/index.md   # 文章。slug 全小写连字符，建议带日期前缀或主题前缀
code/<slug>/                    # 配套可运行代码，目录名与文章 slug 一一对应
agent_skills/                   # agent 技能（本文件指向的那些）
static/katex/                   # 自托管 KaTeX，勿删
.github/workflows/deploy.yml    # push 到 main 自动部署，无需手动操作
```

**写新文章的完整流程见 `agent_skills/write-post.md`，发布验证见 `agent_skills/publish.md`。**

## 必须知道的坑（踩过，别再踩）

1. **数学公式**：`hugo.toml` 里的 `markup.goldmark.extensions.passthrough` 配置**不可删除**——没有它 Goldmark 会把 LaTeX 的 `_` 当斜体标记吃掉。KaTeX 不支持 `\boxed{\begin{aligned}...}` 嵌套（用裸 aligned）。自检方法见 `agent_skills/math-check.md`。
2. **PaperMod 是 v8.0 不是最新版**：最新版要求 Hugo ≥ 0.146。且主题打过两处补丁（`themes/PaperMod/layouts/partials/templates/opengraph.html` 和 `twitter_cards.html` 删除了废弃的 `.Site.Social` 回退分支）。**升级主题或 Hugo 前必须重测构建**。
3. **本机网络下载 GitHub release 资产不稳定**，`codeload.github.com` 的 tar.gz 通常可用；npmmirror 可作 npm 包镜像。
4. **frontmatter 的 `draft: true` 必须改成 `false`**，否则文章不会发布（本地 `hugo server -D` 能看到但线上没有，极易误判）。

## 内容组织惯例

- 系列文章用统一 slug 前缀（如 `flash-attention-01-theory` … `-05-dsl-zoo`），篇间用 `{{</* relref */>}}` 互链，第一篇开头放全系列目录。
- 写作模板：**问题背景 → 踩坑现象 → 根因（贴源码行号链接）→ 解法 → 最小复现代码**。
- 行号引用必须真实核对过（读文件确认），宁可不标行号也不标错。
- 配套代码必须能跑：自带 `if __name__ == "__main__"` 自测，CPU 可跑优先。

## Agent 工作守则

- 改完文章后本地构建验证：`/tmp/hugo_bin/hugo --gc --minify`（在仓库根目录执行），确认无 ERROR。
- 提交信息格式：`post: <主题>` / `fix: <问题>` / `skill: <技能变更>`。
- 推送后必须验证线上状态（Actions run + 页面 HTTP 200），步骤见 `agent_skills/publish.md` §验证。
- 遇到新坑：修完后把"坑 + 修法"追加到本文件的「必须知道的坑」一节，并视情况沉淀为 `agent_skills/` 新技能。

## 技能索引

| 技能 | 何时用 |
|---|---|
| [write-post](agent_skills/write-post.md) | 写/改任何文章 |
| [math-check](agent_skills/math-check.md) | 文章含 LaTeX 公式，或公式显示异常时 |
| [code-dive](agent_skills/code-dive.md) | 深读某个 kernel/框架仓库，写源码分析文章 |
| [publish](agent_skills/publish.md) | 构建、推送、验证线上部署 |
