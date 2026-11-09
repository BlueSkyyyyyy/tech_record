---
title: "通算并行源码精读：UniEP 与 Flux（12 个带注释副本 + 6 篇精读 + 1 篇导读）"
date: 2026-11-05
draft: false
weight: 20
series: ["comm-overlap-src"]
tags: ["通算并行", "系列", "源码精读", "UniEP", "Flux", "CuTe DSL", "CUTLASS", "NVSHMEM", "MoE"]
categories: ["训练框架"]
---

[通算并行研读](../comm-overlap-00-index/) 那一套是纯 Python 的性能模型，算的是**账**；
这一套看的是**账本是怎么被写出来的** —— 真实 kernel 用哪些原语、在哪个粒度上等、把谁和谁叠在一起。

做法是：从上游仓库逐行复制 12 个关键文件，**只插入整行中文注释，不动一行代码**，
再用 `tools/verify_annotations.py` 卡住这条约束（剥掉注释行后与上游逐字节 diff）。

规模：**4536 行代码（与上游逐字节一致）+ 2932 行中文注释**。

计数口径：**12 个带注释副本 + 6 篇正文 + 1 篇总览**。

## 上游在哪

| 仓库 | commit | 说明 |
|---|---|---|
| [Triton-distributed](https://github.com/ByteDance-Seed/Triton-distributed) | `7908e4e`（main） | **UniEP 没有独立仓库**，代码在 `FlashComm/python/flash_comm/ep_overlap/` |
| [flux](https://github.com/bytedance/flux) | `7f98c8a`（**comet** 分支） | Comet 没有公开仓库，用同源同族的 flux comet 分支；AG-GEMM / GEMM-RS 都在里面 |

## 六篇正文

| # | 内容 | 源码 |
|---|---|---|
| 一 | [UniEP 的 dispatch + FC1 GroupGEMM](../comm-overlap-src-01-dispatch-gemm/) | `cutedsl_dispatch_group_gemm.py`（上游 1314 行 → 副本 1879 行，1222 行代码） |
| 二 | [UniEP 的 FC2 GroupGEMM + Combine](../comm-overlap-src-02-gemm-combine/) | `cutedsl_group_gemm_combine.py`（上游 1067 行 → 副本 1411 行，972 行代码） |
| 三 | [UniEP 的确定性从哪来](../comm-overlap-src-03-determinism/) | `m_contig_group_tile_scheduler.py` + `cutedsl_topk_reduce.py` |
| 四 | [Flux 的 AllGather + GEMM](../comm-overlap-src-04-flux-ag-gemm/) | `sm90_all_gather_gemm_*.hpp` |
| 五 | [Flux 的 GEMM + ReduceScatter](../comm-overlap-src-05-flux-gemm-rs/) | `system_barrier.hpp` + `epilogue_evt_nvshmem.hpp` |
| 六 | [对照：各自把「等」切到多细](../comm-overlap-src-06-compare/) | — |

## 自带三个工具

```bash
python3 tools/verify_annotations.py   # 12/12：剥掉注释行后与上游逐字节一致
python3 tools/check_links.py          # docs 里的相对链接是否都有效
./tools/sync_from_upstream.sh         # 上游换版本时重拉（== 丢弃注释，回到基线）
```

## 附：host 侧三个 op

三篇正文讲的是 device 侧 kernel；真正把它们拼起来的 host 侧代码在这里。
三段式结构（`_compile` 收变体键 → `_build` 编 geometry → `run` 只封送指针）三份文件完全同构，
区别只在「输出是不是本地张量」。

{{< ov_src "code/comm-overlap-src/uniep/_ops/dispatch_group_gemm.py" >}}

{{< ov_src "code/comm-overlap-src/uniep/_ops/group_gemm_combine.py" >}}

{{< ov_src "code/comm-overlap-src/uniep/_ops/topk_reduce.py" >}}

> 口径说明：这台机器没有 SM100、也没有多卡 NVLink，所以**没有任何实测数字**。
> UniEP 是 Blackwell（sm100）+ Hopper（sm90）代码，Flux 是 Hopper（sm90）；
> 这里只做源码精读。论文里的加速比见 [comm-overlap-13-uniep](../comm-overlap-13-uniep/) / [-14](../comm-overlap-14-megaflux/)。
