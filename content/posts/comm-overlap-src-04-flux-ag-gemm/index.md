---
title: "源码精读（四）：Flux 的 AllGather + GEMM——两条 stream，按 chunk 等 flag"
date: 2026-11-05
draft: false
weight: 24
series: ["comm-overlap-src"]
tags: ["通算并行", "源码精读", "Flux", "CUTLASS", "Hopper", "NVSHMEM", "Tile Scheduler"]
categories: ["训练框架"]
---

> 本页正文来自 `code/comm-overlap-src/docs/04-flux-ag-gemm.md`，源码在页尾（折叠）。

{{< ov_doc "code/comm-overlap-src/docs/04-flux-ag-gemm.md" >}}

## 源码（带注释副本）

{{< ov_src "code/comm-overlap-src/flux/src/sm90_all_gather_gemm_tma_warpspecialized_cooperative.hpp" >}}

{{< ov_src "code/comm-overlap-src/flux/src/sm90_all_gather_gemm_tile_scheduler.hpp" >}}
