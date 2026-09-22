---
title: "FlashAttention 反向（九）：用 SM90 的 FA3 重测——更正「FA 比 TE 慢」的结论"
date: 2026-09-22
draft: false
weight: 10
series: ["fa-bwd"]
tags: ["flash-attention", "flash-attention-3", "transformer-engine", "反向", "ncu", "算子优化", "系列"]
categories: ["算子开发"]
---

上一篇用 `flash_attn` **2.7.4（FA2，SM80 内核）**对标 TE，得出「FA 比 TE 慢」，并归因到代际差异。
这篇把 **FA3（`flash_attn_3`，SM90，wgmma+TMA）**从源码编译出来重测：**结论反转——FA3 比 TE 更快**
（GQA/MQA 355–438 TF vs 307–356 TF；MHA S=4096 **850 TF vs 618 TF**）。附编译方法、ncu/SASS 证据与对标方法论。

{{< fa_include "code/flash-attention/fa-bwd/docs/07-fa3-sm90-comparison.md" >}}
