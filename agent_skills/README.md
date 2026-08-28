# Agent Skills

本目录是 tech_record 仓库的 agent 技能集：每个文件是一个可执行的任务流程。入口与全局约定见根目录 `agent_guide.md`。

| 技能 | 何时用 |
|---|---|
| [write-post.md](write-post.md) | 写/改任何文章 |
| [math-check.md](math-check.md) | 文章含 LaTeX 公式，或公式显示异常时 |
| [code-dive.md](code-dive.md) | 深读 kernel/框架仓库，写源码分析文章 |
| [publish.md](publish.md) | 构建、推送、验证线上部署 |

新增技能的格式约定：

```markdown
# <技能名>

## 触发场景
## 前置条件
## 步骤（编号，可执行）
## 验收标准
## 已知坑
```
