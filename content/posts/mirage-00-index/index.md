---
title: "Mirage / MPK 项目分析（专题）"
date: 2026-10-23T09:10:00+08:00
draft: false
weight: 1
series: ["mirage"]
tags: ["Mirage", "MPK", "megakernel", "编译器", "推理优化", "系列"]
categories: ["算子开发"]
---

这个专题讲清楚一个开源项目：[**mirage-project/mirage**](https://github.com/mirage-project/mirage)。

它有两副面孔：**Mirage** 是「自动搜 kernel 的编译器」，**MPK**（Mirage Persistent Kernel）是
「把整个 LLM 推理编成一颗常驻 kernel」的运行时。两副面孔共用一个仓库、一张图 IR、一套 codegen。

- 配套材料：[`code/mirage/`](https://github.com/BlueSkyyyyyy/tech_record/tree/main/code/mirage)
- 参考快照：`mpk` 分支，commit `f9eb70c`；所有 `文件:行号` 都对着它核对

## 一条主线

> **Mirage 卖的是「不用手写 kernel」，MPK 卖的是「不用手写调度」。**

前者用搜索替代手写优化，后者用设备内调度替代 host 派发。理解这两句话，这个项目就没有秘密了。

## 怎么读

| # | 篇目 | 回答什么问题 | 适合谁 |
|---|---|---|---|
| 1 | [项目全景]({{< relref "mirage-01-overview" >}}) | 两条产品线什么关系？仓库怎么认路？怎么装？ | 第一次接触 |
| 2 | [从 HF 模型到 megakernel]({{< relref "mirage-02-endtoend" >}}) | `compile()` 生成了什么？为什么一次调用能跑完 decode？ | 想用它 |
| 3 | [生成的 megakernel 怎么跑]({{< relref "mirage-03-runtime" >}}) | worker/scheduler 怎么协作？依赖怎么保证？ | 想改它 |
| 4 | [构建与实跑记录]({{< relref "mirage-04-build" >}}) | 在本机 H100 上到底能不能跑起来？踩了什么坑？ | 想复现 |

如果你还不熟 megakernel 这个概念，先读姊妹专题
[MegaKernel 资料与实现分析]({{< relref "mega-kernel-00-index" >}}) 的第 1 篇——那里讲清了
「为什么要把整张图融进一个 kernel」。

## 两个专题的分工

| | [MegaKernel 专题]({{< relref "mega-kernel-00-index" >}}) | 本专题 |
|---|---|---|
| 关注 | **范式与论文**：三笔账、五条判据、MPK 论文 | **项目与代码**：API、编译链路、运行时、构建 |
| 粒度 | 逐文件、逐 task 的源码走读（75 篇） | 端到端主链路（4 篇） |
| 适合 | 研究/对比不同 megakernel 方案 | 上手、改代码、复现实验 |

本专题不重复源码走读，只在关键处给出「深入阅读」的链接。

## 可信度说明

- 原理与行号：对着快照 `f9eb70c` 逐条核对。
- 实跑：在 H100（sm90）容器里真实构建（`pip install -e .` 成功）、真实跑测试；跑不通的地方
  如实写原因，不粉饰。
- 论文数字（1.2×–6.7×）来自项目 README 与论文，正文标明出处。
