# 对照：UniEP 与 Flux 各自把「等」切到多细

**UniEP 把「等」切到一个 expert，Flux 切到一个 M-chunk：前者等得更细，代价是一块 SMEM 要在通信和计算之间分账；后者对 GEMM 主体零侵入，代价是粒度被拷贝单位锁死**。

读完你会知道：
- 为什么一份能把等待压到 expert，另一份只能到 chunk；
- 两份代码的代价各自记在哪本账上；
- 什么形状下该选哪一种。

## 差异只有六行，其余都是细节

| 维度 | UniEP（EP / MoE） | Flux（TP / AG-GEMM） | 这一差异带来什么 |
|---|---|---|---|
| 等待粒度 | 一个 expert 的 token 全部落地 | 一个 M-chunk，`M / (world_size × SPLIT)` 行 | 细的那边，元数据也跟着细 |
| 通信跑在哪 | 同 kernel 的 dispatch / epilogue warp（TMA 与 SIMT 128b store） | 另一条 stream 的 `cudaMemcpyAsync` / NVSHMEM put | 一个吃 SM 的调度自由度，一个吃 HBM/L2 带宽 |
| 同步原语 | 每个 expert 一个 int32 计数器 + mbarrier + gmem release-acquire | 每个 chunk 一个 flag + `wait_eq`（`ld.global.acquire.sys` 自旋） | 可见范围决定能跨多远 |
| SM 归属 | 共享：8 warp 角色分工，等待写进 TMA warp 的职责 | 分离：拷贝在另一条 stream，kernel 内只读 flag | 等待藏进别的 warp，还是藏进别的数据 |
| 确定性 | 显式保证（M-contiguous 调度 + 固定顺序归约） | 不保证（tile 顺序随调度变） | 要逐位复现，只有前者那条路（见 [03](03-uniep-determinism.md)） |
| 代价 | SMEM 分账：ring 深一档，A/B stage 就浅一档 | 等待粒度锁死：再细就是更多 flag 和更碎的 DMA | 两边的上限都在资源账上；grid 超过物理驻留能力会死锁或长尾 |

## 暴露时长是可以算出来的

一个 chunk 的暴露时长，就是它到达的时刻比计算侧需要它的时刻晚了多少——早到的时间不算，晚到的才算。设 $c$ 是一个等待单元（一个 expert 或一个 M-chunk），$t^{\text{arrive}}_c$ 是它到达的时刻，$t^{\text{need}}_c$ 是计算侧最早需要它的时刻：

$$
T_{\text{exposed}} = \sum_c \max\bigl(0,\; t^{\text{arrive}}_c - t^{\text{need}}_c\bigr)
$$

所以优化只有两条路：让 $t^{\text{arrive}}_c$ 早点到，或者让 $t^{\text{need}}_c$ 晚点来。粒度决定求和里有多少个 $c$——越细，$\max(0,\cdot)$ 越容易为 0，但每细一档都要多付一份元数据。

## 为什么 UniEP 能等到 expert

【粒度】这个粒度不是挑出来的，是捡来的：GroupGEMM 的调度单元本就是 expert，同一个 expert 的 A 行在缓冲里连续，signal 挂在它身上，等一次正好覆盖这一组全部 tile。【代价】也出在同一处：共用一批 SM 和同一块 SMEM，ring 深一档，A/B stage 就浅一档；expert 越小 dispatch 越快，这份等待换来的收益越接近零。

## 为什么 Flux 只能等到 chunk

【切法】Flux 的通信在另一条 stream 上，kernel 里能看见的只有内存 flag，而 flag 由拷贝方写：一次 DMA 搬一个 chunk，flag 就只能一个 chunk 一个。想细到 tile，得让拷贝引擎按 tile 切 DMA，flag 数量和 `CUStreamWriteValue` 次数一起乘上去，小 DMA 的搬运效率也会垮。所以 `SPLIT` 默认 1，宁可让跨 chunk 的 tile 多等一次。

## 所以适用场景是分开的

等得细只在「等待能被别的活盖住」时才值钱：EP 的流量按 expert 倾斜、消息小，切到 expert，每个 group 才能尽早开算；TP 的数据连续、单次量大，一次搬整段最划算，chunk 粗一档换来 GEMM 主体一行不用改。这就是分区型与融合型的分别。

如果只记一件事：粒度不是越细越好，细到「等待小于填充这份等待所需的工作量」就够；单卡或没有 NVLink 时，这套对照整体不成立。
跨工程的对照（Triton vs CuTeDSL）在[下一篇](../comm-overlap-src-07-triton-vs-cutedsl/)。
同一个问题换一种语言写，取舍只是换了个地方收。
