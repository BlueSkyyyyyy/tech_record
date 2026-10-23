# Mirage / MPK 项目分析 · 说明

本目录是「Mirage 项目分析」专题的配套材料。

- 专题正文在 `docs/*.md`，由博客 `content/posts/mirage-*` 实时内联渲染。
- 本机参考快照：`~/github/mirage-f9eb70c`（`mirage-project/mirage` 的 `mpk` 分支，commit `f9eb70c`）。
  正文里所有 `文件:行号` 都对着这份快照核对。
- 构建与实跑记录：见 `docs/04-mirage-build.md`；原始日志放 `harness/*.log.txt`
  （仓库 `.gitignore` 忽略 `*.txt`，只在本机留存）。

## 目录

| 文件 | 内容 |
|---|---|
| `docs/01-mirage-overview.md` | 项目定位、两条产品线、仓库地图、五分钟上手 |
| `docs/02-mirage-endtoend.md` | 从 HF 模型到 megakernel：`demo/qwen3` 与 `compile()` 全链路 |
| `docs/03-mirage-runtime.md` | 生成的 megakernel 怎么跑：worker / scheduler / 事件队列 |
| `docs/04-mirage-build.md` | 在 H100 容器里构建与实跑：真实报错、耗时与结论 |
