---
title: "源码精读（三）：UniEP 的确定性从哪来——M-contiguous 调度 + 固定顺序归约"
date: 2026-11-05
draft: false
weight: 23
series: ["comm-overlap-src"]
tags: ["通算并行", "源码精读", "UniEP", "确定性", "调度器", "MoE"]
categories: ["训练框架"]
---

> 本页正文来自 `code/comm-overlap-src/docs/03-uniep-determinism.md`。
> 两个源码（调度器 + top-k 归约）都在页尾（折叠）。

{{< ov_doc "code/comm-overlap-src/docs/03-uniep-determinism.md" >}}

## 源码（带注释副本）

{{< ov_src "code/comm-overlap-src/uniep/kernels/m_contig_group_tile_scheduler.py" >}}

{{< ov_src "code/comm-overlap-src/uniep/kernels/cutedsl_topk_reduce.py" >}}
