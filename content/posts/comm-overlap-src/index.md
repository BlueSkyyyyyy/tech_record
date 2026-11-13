---
title: "通算并行源码精读：UniEP 与 Flux，同一个「等」的两种切法"
date: 2026-11-05
draft: false
weight: 20
series: ["comm-overlap-src"]
tags: ["通算并行", "系列", "源码精读", "UniEP", "Flux", "CuTe DSL", "CUTLASS", "NVSHMEM", "MoE"]
categories: ["训练框架"]
---

同一份「等」，UniEP 把它切到「一个 expert 的数据齐了」，Flux 把它切到「一个 M-chunk 落地」。
**前者把通信塞进计算 kernel、换来更细的等待，代价是同一批 SM 要排开多组 warp；后者用另一条 stream 解耦、实现更简单，代价是多一条 stream 和一批 flag**。

读完你会知道：

- 同样叫 overlap，为什么 MoE 和 TP 会长成这两种完全不同的写法；
- 每个 flag 到底覆盖多少数据，这个粒度又是被调度器怎么选出来的；
- 一套「等待粒度 vs 开销」的判断框架，拿去读 DualPipe / Weave / MegaFlux。

纯 Python 那套性能模型算的是**账**；这里看的是**账本是怎么被写出来的**——真实 kernel 用哪些原语、在哪个粒度上等、把谁和谁叠在一起。见 [通算并行研读](../comm-overlap-00-index/)。

{{< ov_doc "code/comm-overlap-src/docs/00-overview.md" >}}

## 六篇正文

这套副本共 12 个文件：4536 行代码（与上游逐字节一致）+ 2932 行中文注释，只插入整行注释、不动一行代码。

| # | 内容 | 源码 |
|---|---|---|
| 一 | [UniEP 的 dispatch + FC1 GroupGEMM](../comm-overlap-src-01-dispatch-gemm/) | `cutedsl_dispatch_group_gemm.py`（上游 1314 行 → 副本 1879 行，1222 行代码） |
| 二 | [UniEP 的 FC2 GroupGEMM + Combine](../comm-overlap-src-02-gemm-combine/) | `cutedsl_group_gemm_combine.py`（上游 1067 行 → 副本 1411 行，972 行代码） |
| 三 | [UniEP 的确定性从哪来](../comm-overlap-src-03-determinism/) | `m_contig_group_tile_scheduler.py` + `cutedsl_topk_reduce.py` |
| 四 | [Flux 的 AllGather + GEMM](../comm-overlap-src-04-flux-ag-gemm/) | `sm90_all_gather_gemm_*.hpp` |
| 五 | [Flux 的 GEMM + ReduceScatter](../comm-overlap-src-05-flux-gemm-rs/) | `system_barrier.hpp` + `epilogue_evt_nvshmem.hpp` |
| 六 | [对照：各自把「等」切到多细](../comm-overlap-src-06-compare/) | — |

> 脚注：`tools/verify_annotations.py` 校验全部副本剥掉注释后与上游逐字节一致（12/12）；`check_links.py` 查文档链接；`sync_from_upstream.sh` 在上游换版时重拉，等于丢弃注释、回到基线。

## 附：host 侧三个 op

三篇正文讲的是 device 侧 kernel；真正把它们拼起来的 host 侧代码在这里。
三段式结构（`_compile` 收变体键 → `_build` 编 geometry → `run` 只封送指针）三份文件完全同构，区别只在「输出是不是本地张量」。

{{< ov_src "code/comm-overlap-src/uniep/_ops/dispatch_group_gemm.py" >}}

{{< ov_src "code/comm-overlap-src/uniep/_ops/group_gemm_combine.py" >}}

{{< ov_src "code/comm-overlap-src/uniep/_ops/topk_reduce.py" >}}

> 口径说明：这台机器没有 SM100、也没有多卡 NVLink，所以**没有任何实测数字**。
> UniEP 是 Blackwell（sm100）+ Hopper（sm90）代码，Flux 是 Hopper（sm90）；
> 这里只做源码精读。论文里的加速比见 [comm-overlap-13-uniep](../comm-overlap-13-uniep/) / [-14](../comm-overlap-14-megaflux/)。
