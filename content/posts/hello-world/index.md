---
title: "开篇：为什么写这个博客"
date: 2026-08-28
draft: false
tags: ["meta"]
categories: ["随笔"]
---

这是本博客的第一篇文章。

## 这个博客写什么

主要记录 AI 训练系统工程方向的学习和实践：

1. **Megatron-LM 实战踩坑** —— 大模型分布式训练框架的配置、源码分析与疑难杂症
2. **Kernel / 算子** —— Triton、Liger-Kernel 等算子源码阅读与性能分析
3. **优化器前沿** —— Muon、MuonClip 等新兴优化器的原理与源码
4. **量化与推理** —— 量化方案细节、推理部署

## 文章与代码的关系

每篇文章在 `code/` 目录下有一个同名目录，包含可运行的最小复现代码：

```
content/posts/<article-name>/index.md   # 文章
code/<article-name>/                    # 配套代码
```

固定写作模板：**问题背景 → 踩坑现象 → 根因分析（贴源码） → 解法 → 最小复现**。

## Hello, world

```python
print("Hello, tech_record!")
```
