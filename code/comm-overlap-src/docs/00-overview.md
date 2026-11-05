# 通算并行源码精读：UniEP 与 Flux

这个目录放的是**从上游仓库复制过来、只加注释不改代码**的精读副本，配套 [`comm-overlap`](../../comm-overlap/) 那套纯 Python 性能模型。
模型算的是「账」，这里看的是「账本是怎么被写出来的」——真实的 kernel 用哪些原语、在哪个粒度上等、把谁和谁叠在一起。

> 口径说明：本目录**不产生任何实测数字**。本机没有 SM100、也没有多卡 NVLink 环境，
> UniEP 的 kernel 是 Blackwell（sm100）+ Hopper（sm90）代码，Flux 是 Hopper（sm90）；
> 这里只做源码精读。凡是论文里的加速比，正文一律不给，要看数字回 [comm-overlap-13-uniep](../../../content/posts/comm-overlap-13-uniep/index.md) / [-14](../../../content/posts/comm-overlap-14-megaflux/index.md)。

规模：**12 个文件、4536 行代码（与上游逐行一致）+ 2932 行中文注释**。
重点不是行数，而是每个副本开头那 10 行导读，和正文里标了【】的那些结论。

## 1. 为什么是这两个仓库

| | UniEP | Flux |
|---|---|---|
| 上游 | `ByteDance-Seed/Triton-distributed`，`FlashComm/python/flash_comm/ep_overlap/` | `bytedance/flux`，`comet` 分支 |
| 版本 | commit `7908e4e`（2026-09-18） | commit `7f98c8a`（2024-10-16） |
| kernel 语言 | CuTe DSL（Python 描述 GPU kernel，JIT 出 cubin） | CUTLASS 3.x / CuTe C++ 模板 + NVSHMEM |
| 场景 | MoE 的 expert parallelism（EP） | 张量并行（TP）的 AllGather-GEMM / GEMM-ReduceScatter |
| 通信在哪跑 | **同一个 kernel 里**，由专门的 dispatch/combine warp 发 TMA / SIMT push | **另一条 stream** 上的拷贝（P2P / NVSHMEM），GEMM kernel 只等 flag |
| 等待粒度 | 每个 expert 一份 signal（该专家的 token 全部落地） | 每个 M-chunk 一份 flag（`world_size × SPLIT` 份） |
| 确定性 | 显式设计：M-contiguous 调度 + 固定顺序 top-k 归约 | 不保证（浮点归约顺序随 tile 调度变化） |

两份代码刚好代表「通算并行」的两条主流路线：

- **融合型（Flux）**：把通信和计算放进同一个 GPU 生命周期，但用**两条 stream** 解耦；
  计算侧只对「自己当前这块 tile 依赖的数据」设一个 flag 等待。通信 SM/拷贝引擎与计算 SM 天然分离，实现简单、对 GEMM 主体侵入小。
- **分区型（UniEP）**：通信和计算在**同一个 kernel、同一批 SM** 上，靠 warp 角色分工
  （epilogue / dispatch / MMA / TMA）在 SM 内部叠起来；等待粒度细到「一个 expert 的数据齐了」。

看懂这两条路线，再看 DualPipe / Weave / MegaFlux 那些调度花活，就只是在问同一个问题的不同答案。

## 2. 所有通算并行都在回答三个问题

读任何一份 overlap 代码，都可以按这三问拆：

1. **谁等谁**：等待关系由什么原语表达？（mbarrier / `ld.global.acquire.sys` 自旋 / `cudaStreamWaitValue` / 事件）
2. **等多大一块**：一个 flag 覆盖多少数据？（整个 all-to-all / 一个 expert / 一个 M-chunk / 一个 tile 的一次 k-iteration）
3. **等的时候另一个引擎在干什么**：SM 在跑别人的 MMA？拷贝引擎在搬下一块？还是整个 SM 空转？

粒度越细 → 暴露的等待越少，但 flag 数量、原子操作、调度复杂度上升。
本目录两份代码正好是「细到什么程度」的两个相邻档位。

## 3. 文件地图

`uniep/`（只注释了 7 个文件，不是全仓库）：

| 副本 | 上游路径（`FlashComm/python/flash_comm/ep_overlap/`） | 讲什么 |
|---|---|---|
| [`kernels/cutedsl_dispatch_group_gemm.py`](../uniep/kernels/cutedsl_dispatch_group_gemm.py) | `kernels/cutedsl_dispatch_group_gemm.py` | MegaKernel ①：Dispatch + FC1 GroupGEMM，8 warp 角色分工、per-expert signal |
| [`kernels/cutedsl_group_gemm_combine.py`](../uniep/kernels/cutedsl_group_gemm_combine.py) | `kernels/cutedsl_group_gemm_combine.py` | MegaKernel ②：FC2 GroupGEMM + push-combine，epilogue 里直接推结果 |
| [`kernels/m_contig_group_tile_scheduler.py`](../uniep/kernels/m_contig_group_tile_scheduler.py) | `kernels/m_contig_group_tile_scheduler.py` | M-contiguous tile 调度：确定性 + 信号单调性的地基 |
| [`kernels/cutedsl_topk_reduce.py`](../uniep/kernels/cutedsl_topk_reduce.py) | `kernels/cutedsl_topk_reduce.py` | 确定性 top-k 归约：一个 CTA 一个 token，固定求和顺序 |
| [`_ops/dispatch_group_gemm.py`](../uniep/_ops/dispatch_group_gemm.py) | `_ops/dispatch_group_gemm.py` | host 侧：编译期/运行期解耦、dummy tensor、SM 预算 |
| [`_ops/group_gemm_combine.py`](../uniep/_ops/group_gemm_combine.py) | `_ops/group_gemm_combine.py` | host 侧：combine 的元数据与 buffer 前置条件 |
| [`_ops/topk_reduce.py`](../uniep/_ops/topk_reduce.py) | `_ops/topk_reduce.py` | host 侧：top-k 归约的启动封装 |

`flux/`（只注释了 5 个文件）：

| 副本 | 上游路径（`bytedance/flux`） | 讲什么 |
|---|---|---|
| [`include/system_barrier.hpp`](../flux/include/system_barrier.hpp) | `include/flux/cuda/system_barrier.hpp` | `arrive_inc` / `wait_eq`：sys-scope 的 flag 协议 |
| [`src/sm90_all_gather_gemm_tile_scheduler.hpp`](../flux/src/sm90_all_gather_gemm_tile_scheduler.hpp) | `src/all_gather/sm90_all_gather_gemm_tile_scheduler.hpp` | M tile 重映射：先算本地数据、按 node 打散 |
| [`src/sm90_all_gather_gemm_tma_warpspecialized_cooperative.hpp`](../flux/src/sm90_all_gather_gemm_tma_warpspecialized_cooperative.hpp) | `src/all_gather/sm90_all_gather_gemm_tma_warpspecialized_cooperative.hpp` | AG+GEMM 主 kernel：按 chunk 等 flag 的 producer 循环 |
| [`src/reduce_scatter_barrier_struct.hpp`](../flux/src/reduce_scatter_barrier_struct.hpp) | `src/reduce_scatter/reduce_scatter_barrier_struct.hpp` | GEMM-RS 的跨 rank 归约 barrier |
| [`src/epilogue_evt_nvshmem.hpp`](../flux/src/epilogue_evt_nvshmem.hpp) | `src/reduce_scatter/epilogue_evt_nvshmem.hpp` | epilogue visitor：算完一个 tile 直接 NVSHMEM 归约+散射 |

## 4. 建议阅读顺序

1. 先读 [`docs/01-uniep-dispatch-gemm.md`](01-uniep-dispatch-gemm.md)，跟着注释看 `uniep/kernels/cutedsl_dispatch_group_gemm.py`。
   这是整个目录里信息密度最高的一段代码：**等待粒度被压到「一个 expert」**，并且通信与计算共享同一批 SM。
2. [`docs/02-uniep-gemm-combine.md`](02-uniep-gemm-combine.md)：反方向的那一半（计算 → 通信），看 push-combine 怎么在 epilogue 里发生。
3. [`docs/03-uniep-determinism.md`](03-uniep-determinism.md)：为什么 UniEP 敢说「和串行逐位一致」。
4. [`docs/04-flux-ag-gemm.md`](04-flux-ag-gemm.md)：换到 Flux，看「两条 stream + chunk flag」这种更朴素的写法能到什么程度。
5. [`docs/05-flux-gemm-rs.md`](05-flux-gemm-rs.md)：Flux 的另一半——把 RS 塞进 GEMM epilogue。
6. [`docs/06-compare.md`](06-compare.md)：两套实现的对照，以及一个统一的「等待粒度 vs 开销」框架。

## 5. 注释副本的规矩（以及怎么校验）

本目录的副本有一条硬约束：**只允许插入整行注释**，不允许改动/删除/重排任何一行代码。
所以：

- 你看到的每一行代码都和上游一字不差，可以直接拿去和上游 diff；
- 校验脚本会把「整行注释」滤掉再逐行比对：

```bash
cd code/comm-overlap-src
python3 tools/verify_annotations.py            # 全部校验
python3 tools/verify_annotations.py --list     # 看清单（tools/MANIFEST.tsv）
python3 tools/verify_annotations.py --only uniep/   # 只看 UniEP 侧
```

`tools/MANIFEST.tsv` 记录「仓库标记 + 上游相对路径 → 副本路径」，上游默认在 `~/github/<repo>`（可用 `--root` 改）。

另外两个小工具：

```bash
python3 tools/check_links.py        # 检查 docs 和 README 里的相对链接是否都有效
./tools/sync_from_upstream.sh       # 从上游重新复制（== 丢弃全部注释，只在升级上游时用）
```

真实流程：注释副本是**改了就不 pull** 的。上游换版本时，先 `sync_from_upstream.sh` 回到基线，
再重跑一遍注释（或者用 `git diff` 把注释块搬过去）——直接 `git pull` 会和注释冲突。

## 6. 副本的注释版式

每个副本开头（license 之后）都有一段固定格式的注释块，按这个顺序看最快：

```text
【上游】仓库 / 分支 / commit / 相对路径
【一句话】这个文件在通算并行里干什么
【本注释只讲三件事】……
【阅读顺序】先看哪段、再看哪段（带行号或函数名）
【术语】本文件用到的缩写（EVT / nbi / PE / TMEM / mbarrier …）
```

正文里的注释以**块级**为主：一段代码一段说明，回答「在干什么 / 为什么必须这样 / 去掉会怎样 / 代价是什么」，
而不是逐行翻译英文注释。上游本来就有英文注释的地方，中文注释只补它没说的那一层（为什么、代价、失效方式）。

## 7. 注释里用的记号

- 【】：关键结论，扫一眼就能抓住这段代码的 purpose。
- →：数据流方向（谁搬到谁）。
- 角色名一律用上游的英文词：dispatch（分发）、combine（合并）、GroupGEMM、mbarrier、TMA、TMEM、persistent CTA。
- 「第 N 行」类引用一律指**注释副本**的行号，不是上游行号（副本多了注释，行号会偏移）。
