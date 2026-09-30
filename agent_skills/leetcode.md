# leetcode — LeetCode 题解精讲专题的自驱流程

## 触发场景

用户说「继续 LeetCode 专题」「推进 leetcode」。本仓库长期自驱任务之一。

## 前置条件

- 读根目录 `agent_guide.md`（目录约定/已知坑）。
- 读控制面板 `code/leetcode/ROADMAP.md`：每次先读，从「任务清单」里挑一个未完成子项，做完更新。
- 无需 GPU / 容器；本机 `python3` 与 `g++` 即可。

## 关键事实

- 题解文档 `code/leetcode/docs/<NN>-<category>.md` 通过 `layouts/shortcodes/lc_include.html`
  **实时内联**到博客 `content/posts/leetcode-NN-<category>/index.md`，改 docs 即等于改博客。
- **被内联的 docs 里不能写 `{{< >}}` 短代码**（`readFile`+`markdownify` 不二次解析），否则页面会原样输出。
  站内 `relref` 链接只能写在 `content/posts/*/index.md` 里。
- 代码 `code/leetcode/src/<category>/<problem>.py|.cpp`，每个文件自带自测；
  统一入口 `scripts/run_all.py`。**以 src 为准**，docs 里的代码是粘贴副本，必须逐字一致。
- 固定分类新增了「算法」，已同步 `agent_skills/write-post.md`。

## 步骤（每轮一个增量）

1. 读 `code/leetcode/ROADMAP.md`，选一个未完成子项。
2. 写 `src/<category>/`，`python3 scripts/run_all.py <filter>` 全绿。
3. 粘贴进 `docs/<NN>-<category>.md`（固定档案：题目→思路→代码→复杂度→易错点→相似题；篇末规律总结）。
4. 新分类要建对应博客文章，并在 index 的分类表里改成 `relref` 链接。
5. `/tmp/hugo_bin/hugo --gc --minify -d /tmp/lc_hugo_check` 无 ERROR。
6. 更新 ROADMAP；git 只 add 自己改的路径（见 ROADMAP「git 提交边界」），commit（`leetcode:`）→ `git pull --rebase` → push。

## 验收标准

- [ ] `run_all.py` 全绿（Python + C++ 编译运行）
- [ ] docs 无短代码；新分类博客 `draft: false`、weight 正确、relref 无 404
- [ ] `hugo` 构建无 ERROR
- [ ] 逻辑清晰、语言通顺；无「上一篇/下一篇」式闲聊

## 已知坑

- C++ `assert(v == std::vector<int>{1, 2})`：`{}` 里的逗号会被当成宏参数 → 编译失败。
  先把期望值存变量再断言。
- 全角/半角、中文标点与代码块之间留空行。
- 仓库有 fa-bwd 等并行自驱任务：**绝不 `git add -A`**，只提交 `code/leetcode/`、`content/posts/leetcode-*/` 等自己那部分。
