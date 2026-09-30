# LeetCode 题解精讲（配套代码）

把 LeetCode 高频经典题按**数据结构 / 算法模式**分类，写成由易到难、相似题成组的题解，
每道题配 Python 与 C++ 两版可运行代码，并总结可迁移的通用规律。

- 专题控制面板：[ROADMAP.md](ROADMAP.md)（分类规划、进度、每轮增量流程）
- 题解文档：`docs/<NN>-<category>.md`（通过 `lc_include` 短代码实时内联到博客）
- 代码：`src/<category>/<problem>.py|.cpp`，每个文件自带自测

## 快速开始

```bash
cd code/leetcode

# 全部自测（Python + C++）
python3 scripts/run_all.py

# 只跑某一类，或只跑一种语言
python3 scripts/run_all.py array
python3 scripts/run_all.py --lang py
python3 scripts/run_all.py --lang cpp

# 查看会被执行的文件
python3 scripts/run_all.py --list
```

## 目录

```
leetcode/
  README.md
  ROADMAP.md                 # 分类规划 / 进度 / 写作规范
  docs/<NN>-<category>.md    # 题解正文（内联到博客）
  src/<category>/            # Python / C++ 实现 + 自测
  scripts/run_all.py         # 统一自测入口
  scripts/autopilot.sh       # 自驱循环
```

## 分类进度

见 [ROADMAP.md](ROADMAP.md) 的「任务清单」。当前：
`array`（进行中，已有 5 题样题），其余分类陆续补充。
