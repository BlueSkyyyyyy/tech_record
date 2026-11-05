---
title: "源码精读（二）：UniEP 的 FC2 GroupGEMM + Combine，在 epilogue 里把结果推出去"
date: 2026-11-05
draft: false
weight: 22
series: ["comm-overlap-src"]
tags: ["通算并行", "源码精读", "UniEP", "CuTe DSL", "MegaKernel", "MoE", "TMEM"]
categories: ["训练框架"]
---

> 本页正文来自 `code/comm-overlap-src/docs/02-uniep-gemm-combine.md`，源码在页尾（折叠）。

{{< ov_doc "code/comm-overlap-src/docs/02-uniep-gemm-combine.md" >}}

## 源码（带注释副本）

{{< ov_src "code/comm-overlap-src/uniep/kernels/cutedsl_group_gemm_combine.py" >}}
