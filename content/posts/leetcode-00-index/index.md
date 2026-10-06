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

## 分类目录

按学习顺序排列，大专题会拆成多篇。

| # | 分类 | 核心模式 | 状态 |
|---|---|---|---|
| 1 | [数组与双指针]({{< relref "leetcode-01-array" >}}) | 对撞双指针、快慢指针、原地数组 | 连载中 |
| 2 | [哈希表]({{< relref "leetcode-02-hash" >}}) | 用空间换时间、去重、计数 | 连载中 |
| 3 | [滑动窗口]({{< relref "leetcode-03-sliding-window" >}}) | 连续子数组/子串的最优解 | 连载中 |
| 4 | [前缀和与差分]({{< relref "leetcode-04-prefix-sum" >}}) | 区间和、区间修改 | 连载中 |
| 5 | [二分查找]({{< relref "leetcode-05-binary-search" >}}) | 有序查找、答案二分、边界处理 | 连载中 |
| 6 | [链表]({{< relref "leetcode-06-linked-list" >}}) | 虚拟头结点、快慢指针、反转 | 连载中 |
| 7 | [栈与单调栈]({{< relref "leetcode-07-stack" >}}) | 括号匹配、下一个更大元素 | 连载中 |
| 8 | [堆与 Top-K]({{< relref "leetcode-08-heap" >}}) | 优先队列、多路归并 | 连载中 |
| 9 | [二叉树]({{< relref "leetcode-09-binary-tree" >}}) | 遍历、递归返回值、镜像、层序、构造 | 连载中 |
| 10 | [图与搜索]({{< relref "leetcode-10-graph" >}}) | BFS/DFS、并查集、拓扑排序 | 连载中 |
| 11 | [回溯与递归]({{< relref "leetcode-11-backtracking" >}}) | 决策树、剪枝、排列组合 | 连载中 |
| 12 | [分治法]({{< relref "leetcode-12-divide-conquer" >}}) | 归并、划分、递归分解 | 连载中 |
| 13 | [动态规划]({{< relref "leetcode-13-dynamic-programming" >}}) | 状态设计、转移、背包/序列/区间 | 连载中 |
| 14 | [贪心]({{< relref "leetcode-14-greedy" >}}) | 局部最优、交换论证 | 连载中 |
| 15 | 字符串 | 匹配、编辑距离、回文 | 计划中 |
| 16 | 位运算 | 位技巧、状态压缩 | 计划中 |

> 各篇正文由仓库 `code/leetcode/docs/` 下的 Markdown **实时内联**渲染，随配套代码一起更新。

## 配套代码

所有代码位于 [`code/leetcode/`](https://github.com/BlueSkyyyyyy/tech_record/tree/main/code/leetcode)，按分类分子目录。统一自测：

```bash
cd code/leetcode
python3 scripts/run_all.py          # 全部（Python + C++）
python3 scripts/run_all.py array    # 只跑路径含 array 的
```
