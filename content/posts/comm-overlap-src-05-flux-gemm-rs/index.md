---
title: "源码精读（五）：Flux 的 GEMM + ReduceScatter——把 RS 塞进 epilogue"
date: 2026-11-05
draft: false
weight: 25
series: ["comm-overlap-src"]
tags: ["通算并行", "源码精读", "Flux", "CUTLASS", "NVSHMEM", "ReduceScatter", "内存序"]
categories: ["训练框架"]
---

> 本页正文来自 `code/comm-overlap-src/docs/05-flux-gemm-rs.md`。
> 三个源码（sys-scope barrier、flag 布局、NVSHMEM epilogue）都在页尾（折叠）。

{{< ov_doc "code/comm-overlap-src/docs/05-flux-gemm-rs.md" >}}

## 源码（带注释副本）

{{< ov_src "code/comm-overlap-src/flux/include/system_barrier.hpp" >}}

{{< ov_src "code/comm-overlap-src/flux/src/reduce_scatter_barrier_struct.hpp" >}}

{{< ov_src "code/comm-overlap-src/flux/src/epilogue_evt_nvshmem.hpp" >}}
