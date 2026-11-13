---
title: "通算并行源码精读（七）：同一个 overlap，Triton 和 CuTeDSL 交给了谁"
date: 2026-11-13
draft: false
weight: 27
series: ["comm-overlap-src"]
tags: ["通算并行", "系列", "Triton", "CuTeDSL", "overlap", "NVSHMEM", "MoE"]
categories: ["训练框架"]
---

**Triton 的 overlap 单位是「任务」，CuTeDSL 的是「warp 角色 + mbarrier 事务」；分水岭是谁决定一个 warp 干什么**。

读完你会知道：

- 同一段 dispatch 加 group GEMM，两边各把并发切在哪一层；
- 为什么一边靠 `ld.global.acquire.gpu` 自旋，另一边靠 mbarrier 的期望字节数；
- 选型只看一件事：你愿不愿意把 8 个 warp 的分工写死。

## 1. 并发单位不同，代码就不像在解同一道题

两边吃同一个问题：dispatch 把别的 rank 的 token 拉过来，第一段 group GEMM 紧接着拿它算；要 overlap 的是 GEMM 算到一半时，要用的行还在路上。

Triton 侧写成两个 kernel，各占一个 grid，host 开两条 stream 用 event 串依赖（`gemm_reduce_scatter.py:650`、`:695`）。并发靠 SM 切分：GEMM 拿 `NUM_SMS - num_rs_sms`，多机时剩下的给通信。

CuTeDSL 侧写成一个 kernel：8 warp，0-3 做 epilogue、4-5 做 dispatch、6 做 MMA、7 做 TMA。通信不占额外 SM，就在同一个 CTA 的两个 warp 里（`cutedsl_dispatch_group_gemm.py:46-58`）。

## 2. Triton 侧：overlap 是任务编排，三种形态别混

第一种是双 kernel 加双 stream，dense GEMM 与 ReduceScatter 的默认路径（`gemm_reduce_scatter.py:649-700`）。GEMM 当 producer，通信当 consumer，同时跑靠 SM 切分加全局内存信号量。

```python
rs_stream.wait_stream(current_stream)         # 650
if persistent:
    gemm_rs_producer_persistent(A, B, ...)    # 684 当前 stream
if not fuse_scatter:
    with torch.cuda.stream(rs_stream):        # 692 另一条 stream
        reduce_scatter_2d_op(gemm_out, ctx.rs_ctx, output)
    current_stream.wait_stream(rs_stream)     # 695
```

第二种是单 kernel 加 CTA 分区，MoE EP 的两级 dispatch 走这条（`ep_all2all_fused.py:201-298`）：头部 CTA 用 `putmem_warp` 推 token，尾部 CTA 重排再 release flag（`:206`、`:262`）。

第三种是单 kernel 加全局任务队列：`:907` 用 `tl.atomic_add` 抢号，抢到 dispatch 就 dispatch，抢到 tile 就 GEMM。

三种形态都停在 CTA：同一个 CTA 里所有 warp 干同一件事，没有通信 warp、MMA warp 之分。

`dl.wait` 就是一条 `ld.global.acquire.gpu` 自旋：每 lane 用 `tid.x` 对 32 取模选 slot，读不到期望值就跳回去（`DistributedOpToLLVM.cpp:189-201`）。一个 warp 最多等 32 个 barrier，超了行为未定义（`:193-196`）。

`dl.notify` 是对偶：只有 0 号线程做一次 release store，机内 scope 取 `gpu`、跨节点取 `sys`（`DistributedOpToLLVM.cpp:319-333`），再往外走 `nvshmemx_signal_op`（同文件 `:364`）。远端写更粗：拿 peer 指针后普通 `tl.store`（`gemm_reduce_scatter.py:505-510`），或走 descriptor 的 `atomic_add`（`:218-229`）——**后者落到哪条 PTX，未确认**。

跨 stream 不靠 host 轮询，靠 `cuStreamWaitValue/WriteValue`（`common_ops.py:364-392`）。单机时 reduce-scatter 不占 SM（`reduce_scatter.py:130-136`）：每个 peer 的整段 $M/W \times N$ 一次 `cudaMemcpyAsync` 交给 copy engine（`:570-579`），GEMM 独占 SM。

swizzle 不是分区，效果是把 tile 的 rank 顺序旋转到从 `rank+1` 起（`gemm_rs_threadblock_swizzle.py:146-153`），与 scatter 循环从 `local_rank+1` 起一致（`reduce_scatter.py:566-568`），先到的段先算。

等待粒度看消费端能等多细：AG-GEMM 一个 tile 等一次，可一次等多个 rank（`allgather_gemm.py:276-279`）；EP 分 tile、expert、源 rank 三档（`ep_all2all_fused.py:658-676`）；GEMM-RS 最粗，等一个 rank 段 $M/W \times N$（`reduce_scatter.py:566-579`）——消费端是一次整段拷贝，等待粒度不可能比段更细。

**Triton 侧是不是完全没用 cluster，未确认**：`kernels/nvidia` 下 grep 不到 `cluster`，并发走 cooperative launch（`triton_dist/utils.py:828-833`），看着是单 CTA 语义；编译器内部会不会自己起，查不到。

## 3. CuTeDSL 侧：overlap 是 8 个 warp 的固定分工

角色写死在六个常量里（`cutedsl_dispatch_group_gemm.py:86-91`）：

```python
self.epilog_warp_id = (0, 1, 2, 3)
self.dispatch_warp_id_base = 4
self.num_dispatch_warps = 2
self.mma_warp_id = 6
self.tma_warp_id = 7
self.cta_num_warps = 8
```

这不是配置，是契约：0-3 号只写 epilogue，6 号只发 `cute.gemm`，7 号只发 A/B 的 TMA 并先 acquire per-expert 信号。combine 侧是 6 warp 版（`cutedsl_group_gemm_combine.py:47-56`）。

CTA 内的交接全走 mbarrier：TMA 发起时把这次要搬的字节数写进 `expect_tx`，搬完硬件自己 arrive，消费者阻塞在 full barrier 上（`:556-608`）。另有 named barrier，注释特意说不能算进 dispatch warp（`:99-108`）。

只有「expert 的行齐了没」这种跨 rank 事件才退回 gmem：每个 CTA 对计数器做一次 release `atomic_add`，第 `num_blocks` 个到的人 release store 信号，GEMM 的 TMA warp 用单 lane `acquire` 自旋等（`:894-910`、`:943-963`）。

粒度是 per-expert，不是 per-tile，也不是 Triton 那种整个 rank 段：dispatch 按 expert 一段段写（`cutedsl_dispatch_group_gemm.py:755-840`），信号也按 expert 发（`:894-910`）；按 per-expert 信号的用法推断，GEMM 的 TMA warp 轮到哪个 group 才等哪个（`:943-963`），等齐一个 expert 就能开算它那段 M。内存序全用 `scope="gpu"`，没有 `membar.sys`：机内 peer 在同一张对称堆上直连。

它也不切 SM：`gemm_num_sm` 与 `comm_num_sm` 只影响 grid 预算（`ep_overlap_kernels.py:160-191`）。grid 是 persistent cluster，数量取 $\min(\text{hw\_max}, \lfloor \text{num\_sm}/(c_m c_n) \rfloor)$（`_ops/dispatch_group_gemm.py:108-112`）。

sm100 用 `cluster=(2,2)`、tile 256×256、两 CTA 的 tcgen05；sm90 降到 `(2,1)` 与 `(128,256)`（`:87-102`）。

combine 更省：epilogue 里 16 个 lane 从 leader 取地址，直接 SIMT store 到 peer（`cutedsl_group_gemm_combine.py:847-932`）。机内 peer 不用 NVSHMEM，用 VMM 或 CUDA IPC 的对称堆指针表（`buffer/symmetric.py:232`），跨机才走 NCCL GIN。

SMEM 还得手工分账：dispatch 的 buffer 先从容量里扣，剩下的才给 GEMM 流水线（`cutedsl_dispatch_group_gemm.py:112-118`）。扣多了就掉一个 stage。

## 4. 十行对照，第一行就是谁决定 warp 分工

| 维度 | Triton | CuTeDSL |
|---|---|---|
| overlap 落点 | 多 kernel 多 stream；或单 kernel 内 CTA 分区 | 单 kernel 内的 warp specialization |
| warp 角色 | 无；分工要 patch codegen | 写死：0-3 epilogue、4-5 dispatch、6 MMA、7 TMA |
| grid | persistent，按 SM 数取；机内 RS 不占 SM | persistent cluster，按 `num_sm/cluster` |
| 片上同步 | 交给 `num_stages` | mbarrier 加 expect-tx |
| 跨 rank 同步 | `acquire.gpu` 自旋；多机 `signal_op` | `gpu` 域 release/acquire |

| 维度 | Triton | CuTeDSL |
|---|---|---|
| 跨 stream | `cuStreamWaitValue/WriteValue` | 无，单 stream |
| 等待粒度 | tile 级、per-expert、rank 段 | per-expert、per-token 事务 |
| 搬运 | TMA descriptor、`tl.load`、`putmem`、DMA | TMA multicast、bulk copy、直写 peer |
| GEMM | `tl.dot`，落点**未确认** | 手写 tcgen05 加 TMEM |
| 代码量 | dense 六文件 5417；DSL 8632；C++ 5854 | 2381；kernels 目录 13757 |

**`tl.dot` 落到 wgmma 还是 tcgen05，未确认**：仓库里没有显式指定，`3rdparty/triton` 空着，只能看到 `tl.dot` 这层。

## 5. 代价：复杂度没消失，只是换了地方

CuTeDSL 把复杂度压进两个文件：1314 加 1067，合计 2381 行。读着难受，但 bug 翻得到。

Triton 的 kernel 短得多，dense 六个文件（`kernels/nvidia/{gemm,gemm_reduce_scatter,reduce_scatter,allgather_gemm,allgather_group_gemm,group_gemm}.py`）合计 5417 行，复杂度却外溢到 `python/triton_dist/language/` 的 8632 行 Python 与 `lib/`+`include/` 的 5854 行 C++/TableGen（均按 `wc -l`，含注释与空行）。你写十行 `tl.dot`，背后垫着近两万行。

分工更是没有出口。想在 Triton 里说「这个 warp 只做 TMA」，只能靠给 codegen 打补丁的 `simt_exec_region`（`simt_ops.py:317-340`），一升级就碎。

CuTeDSL 反过来：分工是你的，编译代价也是你的。`cute.compile` 出 `.so`，variant 一变就重编，产线得配 AOT 缓存（`utils/cutedsl_aot.py:52-95`）；opt-level 还因 CUDA 版本从 3 降到 2（`_ops/_base.py:160-175`）。

## 6. 怎么选：四个处境，一边一个答案

| 你的处境 | 选 | 因为 |
|---|---|---|
| 只想通信原语进 DSL、kernel 保持短 | Triton | `dl.wait` 像普通 op |
| 单机榨干 SM、接受手写 | CuTeDSL | 8 warp 与 SMEM 归你 |
| 没人愿意维护 1314 行单文件 | Triton | 升级由上游扛 |
| 只跑 Hopper 以上、要 2CTA | CuTeDSL | sm100 与 sm90 分两套写好 |

## 7. 三个坑

Triton 侧：TMA descriptor 要全局 allocator，`set_allocator` 忘了就崩（`gemm_reduce_scatter.py:677-681`）；32 barrier 上限超了没有检查，注释写的是行为未定义（`DistributedOpToLLVM.cpp:193-196`）；改流水深度只能调 `num_stages`，层数被编译器吃掉。

CuTeDSL 侧：named barrier 的线程数写错就死锁（`cutedsl_dispatch_group_gemm.py:99-104`）；每次 variant 变化都要重编（AOT 缓存就是为这个存在的），cache key 还敏感；只上 Hopper 以上（同文件 `:87-104`）。

## 8. 带走三条

第一，细不细别看是不是单 kernel，看等待单位是 CTA、expert 还是一行 token。

第二，同步原语被并发单位逼出来：单位是 CTA，同步只能落到全局内存信号量；单位是 warp 加事务，才用得上 mbarrier 的期望字节数。

第三，谁决定一个 warp 干什么，决定了你要为这点性能付多少工程债。Triton 交给编译器，省事但插不上手；CuTeDSL 交给你，自由但 SMEM 与 named barrier 都得自己管。

拆过细节：[六]({{< relref "comm-overlap-src-06-compare" >}}) 是 UniEP 与 Flux 的对照，[四]({{< relref "comm-overlap-src-04-flux-ag-gemm" >}}) 的 AG-GEMM 正好对着第二节读，目录在[系列]({{< relref "comm-overlap-src" >}})。想看「把顺序写进 kernel」的另一条路，读[补遗一]({{< relref "comm-overlap-13-uniep" >}})。
