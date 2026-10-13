---
title: "Mega MoE 实现解析（一百五十七）：MQA logits（lightning indexer）的训练反向——KL 蒸馏梯度 + 打分算子反向（H100 实测）"
date: 2026-10-14T03:44:00+08:00
draft: false
weight: 157
series: ["mega-moe"]
tags: ["DeepGEMM", "DeepSeek", "indexer", "稀疏注意力", "反向", "KL", "FP8", "系列"]
categories: ["算子开发"]
---

{{< doc_include "code/mega-moe/docs/157-mqa-logits-indexer-backward.md" >}}
