---
title: "LeetCode 题解精讲（专题导读）"
date: 2026-09-30
draft: false
weight: 1
series: ["leetcode"]
tags: ["LeetCode", "算法", "题解", "Python", "C++", "系列"]
categories: ["算法"]
---

这个专题做一件事：**把 LeetCode 上高频、经典的题目，按数据结构与算法模式分类整理，讲清「怎么想到」和「通用规律」，并给出可直接运行的 Python 与 C++ 实现。**

它和刷题记录不一样：每道题不只给答案，还要回答三个问题——**为什么这样做、复杂度是多少、以后遇到什么题能用同一招**。相似的题目会被放在一起，形成可迁移的「模板」。

## 怎么读

- 每个大类一篇（动态规划这类大专题会拆成多篇），篇内按**由易到难**排列。
- 每道题的固定档案：**题目 → 思路（含为什么）→ 代码 → 复杂度 → 易错点 → 相似题**。
- 每篇结尾有一节「规律总结」，把本篇的通用招式提炼成可背可套用的模板。
- 解法代码与仓库 `code/leetcode/src/<分类>/` 一一对应，均可独立运行自测。
- 只想知道「这题用哪招」时，直接翻[模板速查与易错清单]({{< relref "leetcode-36-cheatsheet" >}})。

### 三条读法

不同目标不必从头读到尾：

| 读法 | 适合谁 | 建议路线 |
|---|---|---|
| **面试冲刺** | 一两周后要面，想快速补齐高频题 | 数组 → 哈希 → 滑动窗口 → 二分 → 链表 → 栈 → 二叉树 → 回溯 → DP（一）→ 图与搜索；每篇只做「模式一 + 中等题」，最后过一遍附录的易错清单 |
| **系统补基础** | 想真正把题型吃透 | 按下面的编号顺序读；每篇先读「适用信号」，再遮住代码自己写一遍，最后对照「易错点」和「规律总结」 |
| **复习速查** | 已经刷过，临场回忆 | 只看每篇的「本篇题目」表和篇末「规律总结」，卡住再回看具体题；配合[附录]({{< relref "leetcode-36-cheatsheet" >}})的 16 张模板卡 |

一条经验：**同一篇里的题按顺序做，比跳着刷有效**。相邻题往往只差一个条件，那个「只差的一点」就是模式本身。

## 分类目录

按学习顺序排列，大专题会拆成多篇。题数按篇计：同一道题若在不同篇从不同角度出现（如 307、493），会在各篇分别计数。

| # | 分类 | 核心模式 | 题数 |
|---|---|---|---|
| 1 | [数组与双指针]({{< relref "leetcode-01-array" >}}) | 对撞双指针、快慢指针、原地数组 | 13 |
| 2 | [哈希表]({{< relref "leetcode-02-hash" >}}) | 用空间换时间、去重、计数 | 10 |
| 3 | [滑动窗口]({{< relref "leetcode-03-sliding-window" >}}) | 连续子数组/子串的最优解 | 10 |
| 4 | [前缀和与差分]({{< relref "leetcode-04-prefix-sum" >}}) | 区间和、区间修改 | 10 |
| 5 | [二分查找]({{< relref "leetcode-05-binary-search" >}}) | 有序查找、答案二分、边界处理 | 10 |
| 6 | [链表]({{< relref "leetcode-06-linked-list" >}}) | 虚拟头结点、快慢指针、反转 | 11 |
| 7 | [栈与单调栈]({{< relref "leetcode-07-stack" >}}) | 括号匹配、下一个更大元素 | 11 |
| 8 | [堆与 Top-K]({{< relref "leetcode-08-heap" >}}) | 优先队列、多路归并 | 10 |
| 9 | [二叉树]({{< relref "leetcode-09-binary-tree" >}}) | 遍历、递归返回值、镜像、层序、构造 | 22 |
| 10 | [图与搜索]({{< relref "leetcode-10-graph" >}}) | BFS/DFS、并查集、拓扑排序 | 11 |
| 11 | [回溯与递归]({{< relref "leetcode-11-backtracking" >}}) | 决策树、剪枝、排列组合 | 12 |
| 12 | [分治法]({{< relref "leetcode-12-divide-conquer" >}}) | 归并、划分、递归分解 | 10 |
| 13 | [动态规划（一）]({{< relref "leetcode-13-dynamic-programming" >}}) | 状态设计、转移、背包/序列/区间/状态机 | 35 |
| 14 | [贪心]({{< relref "leetcode-14-greedy" >}}) | 局部最优、交换论证 | 10 |
| 15 | [字符串]({{< relref "leetcode-15-string" >}}) | 反转、匹配、编辑距离、回文 | 10 |
| 16 | [位运算]({{< relref "leetcode-16-bit" >}}) | 异或抵消、清最低位 1、位递推、位枚举 | 10 |
| 17 | [前缀树]({{< relref "leetcode-17-trie" >}}) | 共享前缀、通配符搜索、前缀聚合 | 10 |
| 18 | [数学]({{< relref "leetcode-18-math" >}}) | 逐位处理、进制映射、数论结论 | 10 |
| 19 | [设计题]({{< relref "leetcode-19-design" >}}) | 组合基础结构、哈希+链表、O(1) 容器 | 10 |
| 20 | [矩阵与二维数组]({{< relref "leetcode-20-matrix" >}}) | 原地变换、螺旋/对角遍历、结构化查找 | 10 |
| 21 | [排序]({{< relref "leetcode-21-sort" >}}) | 自定义顺序/比较器、频率排序、桶排序 | 10 |
| 22 | [最短路径]({{< relref "leetcode-22-shortest-path" >}}) | BFS/Dijkstra/0-1 BFS/Bellman-Ford/Floyd、二分答案 | 10 |
| 23 | [并查集]({{< relref "leetcode-23-union-find" >}}) | 连通分量、判环、等价约束、关系建模 | 10 |
| 24 | [树状数组]({{< relref "leetcode-24-fenwick" >}}) | 单点修改+区间和、值域计数、差分区间加 | 10 |
| 25 | [线段树]({{< relref "leetcode-25-segment-tree" >}}) | 单点/区间修改、懒标记、动态开点、区间合并信息 | 10 |
| 26 | [博弈论]({{< relref "leetcode-26-game-theory" >}}) | 必胜必败态、分差 DP、minimax、状态压缩 | 10 |
| 27 | [随机化与采样]({{< relref "leetcode-27-randomized" >}}) | 洗牌、加权抽样、水塘抽样、拒绝采样 | 10 |
| 28 | [计算几何]({{< relref "leetcode-28-geometry" >}}) | 矩形相交、叉积、凸包、三维形体、最小矩形 | 10 |
| 29 | [区间与扫描线]({{< relref "leetcode-29-interval" >}}) | 区间合并/求交/相减、扫描线、矩形面积覆盖 | 10 |
| 30 | [队列与双端队列]({{< relref "leetcode-30-queue" >}}) | 队列模拟、双端队列设计、单调队列 | 10 |
| 31 | [数位 DP]({{< relref "leetcode-31-digit-dp" >}}) | 按位枚举、tight/前导零、掩码、相邻位、数位和 | 10 |
| 32 | [状态压缩]({{< relref "leetcode-32-bitmask" >}}) | 位掩码当集合、枚举子集、覆盖集合、TSP 状压 | 10 |
| 33 | [字符串匹配]({{< relref "leetcode-33-string-matching" >}}) | KMP 前缀函数、滚动哈希、Manacher 回文 | 10 |
| 34 | [树形 DP]({{< relref "leetcode-34-tree-dp" >}}) | 后序返回状态、选/不选、净流量、单臂、覆盖状态机 | 8 |
| 35 | [动态规划（二）DP 优化]({{< relref "leetcode-35-dp-optimization" >}}) | 前缀和、哈希表、二分、单调队列、滚动数组 | 8 |

**附录**：[模板速查与易错清单]({{< relref "leetcode-36-cheatsheet" >}}) —— 16 张可直接抄的模板卡、按数据范围选复杂度、15 条跨题型易错点。动手前扫一眼，比重新翻一遍正文快。

当前共 **35 篇正文 + 1 篇附录**，按篇累计 **391 道·次**，每道题都有 Python 与 C++ 两份可运行实现。

> 各篇正文由仓库 `code/leetcode/docs/` 下的 Markdown **实时内联**渲染，随配套代码一起更新。

## 配套代码

所有代码位于 [`code/leetcode/`](https://github.com/BlueSkyyyyyy/tech_record/tree/main/code/leetcode)，按分类分子目录。统一自测：

```bash
cd code/leetcode
python3 scripts/run_all.py          # 全部（Python + C++）
python3 scripts/run_all.py array    # 只跑路径含 array 的
```

## 还没覆盖的部分

这个专题不是题库全集，下面几块目前是空白，按价值和难度排在后面：

- **最小生成树**（Kruskal / Prim）：树状数组、线段树都有专篇，MST 却还没开——图论里最后一块常用拼图。
- **概率与期望 DP**：688、837 这类题介于「随机化」与「DP」之间，暂时没有合适归属。
- **AC 自动机 / 后缀数组**：字符串匹配篇只到 KMP、滚动哈希、Manacher，多模式匹配还没展开。
- **矩阵快速幂与线性递推**：`O(log n)` 求斐波那契这类，属于「数学 + DP」的交叉。
- **树形 DP 与 DP 优化继续补题**：这两篇已从 6 题补到 8 题，仍低于其它篇的 10 题，后续各再补 2 题。

有新题想加，按 `code/leetcode/ROADMAP.md` 的流程走：先写 `src/` 自测，再粘进 `docs/`，最后在对应博客页确认渲染。
