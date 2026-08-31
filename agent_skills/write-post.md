# write-post — 写一篇新文章

## 触发场景

用户要求写博客文章、整理学习笔记、把某个技术主题沉淀成文。

## 前置条件

- 已读根目录 `agent_guide.md`（目录约定、已知坑）。
- 若文章基于某仓库的源码分析，先走 `code-dive.md` 流程产出分析素材。

## 步骤

1. **定 slug**：全小写连字符，系列文章带序号前缀（`主题-序号-子题`，如 `flash-attention-02-triton-fwd`）。
2. **创建文章**：
   ```bash
   mkdir -p content/posts/<slug>
   # 复制 archetypes/posts.md 的 frontmatter 起手
   ```
   frontmatter 必填：`title`、`date`（当天）、`draft: false`（**写完后必须确认是 false**）、`tags`（含主题标签，系列文加 `"系列"`）、`categories`。
   - `categories` 只能从下面的**固定分类清单**里选（每篇文章一般一个）；不要把系列名当分类。系列归属用 tags 表达（如 `"flash-attention"` + `"系列"`）。
   - **固定分类清单**：`算子开发`、`训练框架`、`推理框架`、`LLM算法`、`多模态算子`、`强化学习`、`Linux`、`C++`、`Python`、`随笔`。确有需要新增分类时，先与用户确认并把新分类加回本清单。
   - **系列文章必须加 `weight`**（序号从 1 起，按阅读顺序）。Hugo 默认按 Weight → Date → 标题排序；同日期不加 weight 时标题里的中文数字（一二三四五）会按 Unicode 码点排序，顺序错乱（一三二五四）。
3. **正文结构**（技术文章默认模板）：
   - 开头一段说明本篇在系列/主题中的位置，给前置阅读链接（`{{</* relref "slug" */>}}`）。
   - 主体按「问题背景 → 现象 → 根因（源码行号引用）→ 解法 → 复现」组织。
   - 代码引用格式：`文件名:行号`，并附 GitHub 永久链接（带 commit hash）。
   - 结尾：小结（bullet）+ 下一篇预告。
4. **配套代码**（如需要）：在 `code/<slug>/` 下建同名目录，代码必须带 `if __name__ == "__main__"` 自测且能跑通（CPU 优先）；登记到 `code/README.md` 的索引表。
5. **索引登记**：更新根目录 `README.md` 的「文章索引」。
6. **公式自检**：若含 LaTeX，走 `math-check.md`。
7. **构建验证 + 发布**：走 `publish.md`。

## 验收标准

- [ ] `draft: false`；tags/categories 已填
- [ ] 本地 `hugo --gc --minify` 无 ERROR
- [ ] 行号引用抽查过至少 3 处属实
- [ ] 配套代码实跑通过（粘贴运行输出到提交说明或注释）
- [ ] README.md 和 code/README.md 索引已更新
- [ ] 线上页面 HTTP 200 且公式渲染正常

## 已知坑

- `draft: true` 时本地 `hugo server -D` 可见、线上不可见——发布后必须以线上 200 为准。
- markdown 里写 `$` 公式前确认 passthrough 配置还在（见 `agent_guide.md` 坑 #1）。
- 中文段落与公式/代码块之间留空行，避免渲染粘连。
