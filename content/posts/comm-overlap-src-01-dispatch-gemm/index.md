---
title: "源码精读（一）：UniEP 的 dispatch + FC1 GroupGEMM，把等待粒度压到一个 expert"
date: 2026-11-05
draft: false
weight: 21
series: ["comm-overlap-src"]
tags: ["通算并行", "源码精读", "UniEP", "CuTe DSL", "MegaKernel", "MoE", "TMA"]
categories: ["训练框架"]
---

> 本页正文来自 `code/comm-overlap-src/docs/01-uniep-dispatch-gemm.md`，源码在页尾（折叠）。
> 文中「搜 `symbol`」的锚点、以及文末行号表，都对应页尾那份带注释副本。

{{< ov_doc "code/comm-overlap-src/docs/01-uniep-dispatch-gemm.md" >}}

## 源码（带注释副本）

{{< ov_src "code/comm-overlap-src/uniep/kernels/cutedsl_dispatch_group_gemm.py" >}}
