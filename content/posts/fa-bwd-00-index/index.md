---
title: "FlashAttention 反向工程（专题）"
date: 2026-09-22
draft: false
weight: 1
series: ["fa-bwd"]
tags: ["flash-attention", "反向", "CUDA", "算子优化", "系列"]
categories: ["算子开发"]
---

这个专题做一件事：**从零把 FlashAttention 的「反向」写出来、测准、调快，并面向目标 AI 卡做移植**。

参考对象是 flash-attention（FA2/FA3）和 TransformerEngine（TE）；我们自己的实现（下称 **ours**）支持 fp16 / bf16 / **fp8**，每种都给**单文件**（好读）和**两文件**（工程拆分）两个版本。每个关键结论都有本机 H100 的实测数字和 `ncu` 证据，负结果也照记。

- 配套代码：[`code/flash-attention/fa-bwd/`](https://github.com/BlueSkyyyyyy/tech_record/tree/main/code/flash-attention/fa-bwd)
- 数值 I/O dump：`~/proj/output/fa-bwd/`（CPU npy，便于 load 比对）

## 先记住一个画面

把反向 kernel 想成一家**有三道工序的小厂**：

1. **备料**（preprocess）：算出 $\Delta = \mathrm{rowsum}(dO \circ O)$ 和每行 LSE，把主工序要用的"半成品"先备好；
2. **主工序**（main）：沿 KV 块循环，边重算 $P$、边算 $dP/dS$、边把 $dK/dV$ 攒下来，$dQ$ 则要跨块汇总；
3. **包装**（convert/postprocess）：把 fp32 累加缓冲转成目标精度、按 scale 缩放、写回。

调优的全部故事，就是**这三道工序里谁先成为瓶颈、打掉之后瓶颈又搬到哪**。第八篇（调优历程）把每次"墙迁移"都记了下来。

> **类比的边界**：工厂可以加班、可以加人；GPU 上 SM 数、smem、寄存器、L2 带宽都是固定的。所以"打掉一个瓶颈"通常不是"做得更快"，而是**把压力从一个硬件单元挪到另一个**——理解这一点，才不会把某次 1.2× 的收益当成"优化方法"本身。

读完这个专题，你应该能回答：

1. FlashAttention 反向的三段式结构是什么？为什么 $dQ$ 必须跨块归约，而 $dK/dV$ 不用？
2. 为什么 fp8 反向比 fp16 难？E4M3/E5M2 怎么分工，rowwise scale 怎么折回？
3. 一个能跑的 kernel 离"快"有多远？该按什么顺序找瓶颈、验证收益、记录负结果？

## 先给结论（赶时间看这里）

- **正确性**：MHA 形状上，fp16 与 fp32 参考的 `max_abs` 为 ~1.5–2.3e-3、bf16 为 ~7e-3–1.5e-2，与 FA2/TE 同量级；GQA/MQA 的 `dk/dv` 误差随 KV 头减少而变大（梯度要在更多 Q 头上叠加）。fp8 的误差在 O(0.24–0.94)（TE 同量级），这是 FP8 量化本身的代价，不是累加精度问题。
- **性能**：fp16 端到端从标量版的 **138.33 ms** 降到 **1.95 ms**（S=4096，70.9×）；fp8 从 **80.63 ms** 降到 **3.40 ms**（23.7×）。纯反向口径下，当前约为 FA3 的 **1/6** 时间（~8–22% TFLOPS）。
- **覆盖**：MHA / GQA / MQA / MLA（`head_dim=512`）/ 变长（`cu_seqlens`）/ 非 causal；`head_dim` 已覆盖 128 / 256 / 512。
- **确定性**：提供 opt-in 的 `--det`（dK/dV 分片 partial 加固定次序二次归约），fp16/bf16/fp8 都能做到两次跑逐位相同，代价是放弃 split-K 的并行度。
- **当前主瓶颈**：fp8 主 kernel 是 **L2 原子/归约流量**（dK/dV 的 `red` 占 L2 扇区约 74%）；fp16/bf16 的墙在寄存器/occupancy 与 `red` 之间迁移。再往下走要靠 **tile 调度与工作划分**，而不是换一条指令。
- **移植**：与硬件强相关的只有 L3（计算指令）与 L4（smem 布局）两层；L0–L2、L6 是可复用骨架。清单见第九篇。

## 怎么读（按目标选路线）

### 路线 A：只想弄懂反向的数学与结构

读 **一**（实现结构）→ **四**（FP8 设计，其中也讲了反向的四段式）→ **六**（数值/性能汇总，看结论）。

不需要先读源码；每篇都能独立看懂，看源码时也能一一对上。

### 路线 B：想照着写一个自己的反向

读 **一** → **二**（fp16 单/两文件）→ **三**（bf16：dtype 参数化与 bank conflict）→ **四/五**（fp8）。每篇都是"先说为什么慢，再说怎么改"，代码路径与行号都给到。

### 路线 C：想做性能调优

读 **八**（调优历程与方法论，先看"墙迁移总表"）→ **七**（对标：和谁比、差在哪）→ **六**（数据汇总）。重点不是记住某个 1.2×，而是学会**用 ncu 定量当前 bound，再选手段**。

### 路线 D：想移植到目标卡

读 **九**（移植清单）→ **一**（结构）→ **四**（量化口径）。L0–L2 的数学与数据流可以整套搬走，只换 L3/L4。

## 目录

1. [实现结构与优化手段梳理（FA/TE 对照）]({{< relref "fa-bwd-01-catalog" >}}) —— 反向的数学、FA 的三段式、优化手段清单
2. [fp16 反向实现与优化]({{< relref "fa-bwd-02-fp16" >}}) —— 单/两文件，preprocess + 主 kernel
3. [bf16 反向实现与优化]({{< relref "fa-bwd-03-bf16" >}}) —— dtype 参数化 + bank conflict 那些坑
4. [FP8 反向设计]({{< relref "fa-bwd-04-fp8-design" >}}) —— E4M3/E5M2 + rowwise scaling 怎么定
5. [FP8 反向实现与优化]({{< relref "fa-bwd-05-fp8-impl" >}}) —— 张量核 mma 与逐轮优化（O1–O11）
6. [数值与性能汇总（GQA/MQA/MLA）]({{< relref "fa-bwd-06-summary" >}}) —— 对拍表 + 性能表 + ncu 结论
7. [对标分析：FA2 / FA3 / TE 与目标形状]({{< relref "fa-bwd-07-analysis" >}}) —— 纯反向口径、kernel/SASS/SOL 对照
8. [性能调优历程与方法论：从 138 ms 到 1.95 ms]({{< relref "fa-bwd-08-optimization-journey" >}}) —— 逐轮记录、墙迁移总表
9. [目标卡移植注意事项]({{< relref "fa-bwd-09-porting" >}}) —— 可替换层与移植 checklist

> 说明：以上各篇正文由仓库 `code/flash-attention/fa-bwd/docs/` 下的 Markdown **实时内联**渲染，随代码与实测一起更新，不另行复制。

## 读数字之前：三条口径约定

这个专题里数字很多，先统一口径，避免误读：

1. **主指标是时间**（ms/µs，越小越好）。`TF`/`TFLOPS` 是"每秒 Tera 次浮点运算"，是吞吐、不是时间；它由 $\text{FLOPs} \div \text{时间}$ 换算，只作参考。
2. **"端到端" vs "main"**：端到端 = preprocess + main + convert（fp8 还含 quant）；main 单指反向主 kernel。**两个数不能混着比**——很多"优化"只动了 main，端到端收益会被其他段稀释。
3. **FLOPs 口径**：`4·B·S²·H·D`，causal **不折半**。所以所有实现的 TFLOPS 都是下界，但横比是公平的（ours/FA/TE 同口径）。

还有两个常用缩写：

- **S / B / H / D**：序列长度 / batch / head 数 / head_dim；`S=4096 H16` 指 `(B,S,H,D)=(1,4096,16,128)`、causal、MHA。
- **kv=4 / kv=1**：GQA 的 KV head 数；kv=1 即 MQA。

## 这个专题不做什么

- 不重复推导前向的 online softmax——那是[另一个系列]({{< relref "flash-attention-01-theory" >}})的事，这里只从反向的 $\Delta$、LSE 说起。
- 不追求"打败 FA3"。当前目标是**正确性打通 + 张量核化 + 找到 bound**；和 FA3 的差距是已知的、可解释的（见第七篇）。
- 不隐藏失败。变慢的改动、被证伪的假设、因为 ISA 限制走不通的路，都写在第八篇里——**负结果和正结果一样值钱**。
