# 通算并行源码精读：UniEP 与 Flux

## 1. 同样是 overlap，两份代码选了相反的路

| | UniEP | Flux |
|---|---|---|
| 上游 | [`ByteDance-Seed/Triton-distributed`](https://github.com/ByteDance-Seed/Triton-distributed)（UniEP 无独立仓库），`FlashComm/python/flash_comm/ep_overlap/` | [`bytedance/flux`](https://github.com/bytedance/flux)，`comet` 分支（Comet 无公开仓库，同源同族） |
| 版本 | commit `7908e4e`（main，2026-09-18） | commit `7f98c8a`（2024-10-16） |
| kernel 语言 | CuTe DSL（Python 描述 GPU kernel，JIT 出 cubin） | CUTLASS 3.x / CuTe C++ 模板 + NVSHMEM |
| 场景 | MoE 的 expert parallelism（EP） | 张量并行（TP）的 AllGather-GEMM / GEMM-ReduceScatter |
| 通信在哪跑 | 同一个 kernel 里，由专门的 dispatch/combine warp 发 TMA / SIMT push | 另一条 stream 上的拷贝（P2P / NVSHMEM），GEMM kernel 只等 flag |
| 等待粒度 | 每个 expert 一份 signal（该专家的 token 全部落地） | 每个 M-chunk 一份 flag（`world_size × SPLIT` 份） |
| 确定性 | 显式设计：M-contiguous 调度 + 固定顺序 top-k 归约 | 不保证（浮点归约顺序随 tile 调度变化） |

两份代码代表两条主流路线。**融合型**（Flux）把通信和计算放进同一个 GPU 生命周期，用两条 stream 解耦，计算侧只等自己当前 tile 依赖的数据；**分区型**（UniEP）让通信和计算共享同一个 kernel 与同一批 SM，靠 warp 角色分工（epilogue / dispatch / MMA / TMA）在 SM 内部叠起来。看懂这两条，再看 DualPipe / Weave / MegaFlux，都只是同一问题的不同答案。

## 2. 所有通算并行都在回答三个问题

读任何一份 overlap 代码，按这三问拆：

1. **谁等谁**：等待关系由什么原语表达？（mbarrier / `ld.global.acquire.sys` 自旋 / `cudaStreamWaitValue` / 事件）
2. **等多大一块**：一个 flag 覆盖多少数据？（整个 all-to-all / 一个 expert / 一个 M-chunk / 一个 tile 的一次 k-iteration）
3. **等的时候另一个引擎在干什么**：SM 在跑别人的 MMA？拷贝引擎在搬下一块？还是整个 SM 空转？

粒度越细，暴露的等待越少，但 flag 数量、原子操作、调度复杂度上升。两份代码正好相邻。

## 3. 文件地图：源码怎么组织

哪篇读哪份代码，见下面的「六篇正文」表；这里只讲文件怎么组织。

`uniep/` 只注释了 7 个文件，副本路径与上游 `FlashComm/python/flash_comm/ep_overlap/` 下完全一致：

| 副本 | 讲什么 |
|---|---|
| [`kernels/cutedsl_dispatch_group_gemm.py`](../uniep/kernels/cutedsl_dispatch_group_gemm.py) | MegaKernel ①：Dispatch + FC1 GroupGEMM，8 warp 角色分工、per-expert signal |
| [`kernels/cutedsl_group_gemm_combine.py`](../uniep/kernels/cutedsl_group_gemm_combine.py) | MegaKernel ②：FC2 GroupGEMM + push-combine，epilogue 里直接推结果 |
| [`kernels/m_contig_group_tile_scheduler.py`](../uniep/kernels/m_contig_group_tile_scheduler.py) | M-contiguous tile 调度：确定性 + 信号单调性的地基 |
| [`kernels/cutedsl_topk_reduce.py`](../uniep/kernels/cutedsl_topk_reduce.py) | 确定性 top-k 归约：一个 CTA 一个 token，固定求和顺序 |
| [`_ops/dispatch_group_gemm.py`](../uniep/_ops/dispatch_group_gemm.py) | host 侧：编译期/运行期解耦、dummy tensor、SM 预算 |
| [`_ops/group_gemm_combine.py`](../uniep/_ops/group_gemm_combine.py) | host 侧：combine 的元数据与 buffer 前置条件 |
| [`_ops/topk_reduce.py`](../uniep/_ops/topk_reduce.py) | host 侧：top-k 归约的启动封装 |

`flux/` 只注释了 5 个文件，上游路径都多一层子目录：

| 副本 | 上游路径（`bytedance/flux`） | 讲什么 |
|---|---|---|
| [`include/system_barrier.hpp`](../flux/include/system_barrier.hpp) | `include/flux/cuda/system_barrier.hpp` | `arrive_inc` / `wait_eq`：sys-scope 的 flag 协议 |
| [`src/sm90_all_gather_gemm_tile_scheduler.hpp`](../flux/src/sm90_all_gather_gemm_tile_scheduler.hpp) | `src/all_gather/sm90_all_gather_gemm_tile_scheduler.hpp` | M tile 重映射：先算本地数据、按 node 打散 |
| [`src/sm90_all_gather_gemm_tma_warpspecialized_cooperative.hpp`](../flux/src/sm90_all_gather_gemm_tma_warpspecialized_cooperative.hpp) | `src/all_gather/sm90_all_gather_gemm_tma_warpspecialized_cooperative.hpp` | AG+GEMM 主 kernel：按 chunk 等 flag 的 producer 循环 |
| [`src/reduce_scatter_barrier_struct.hpp`](../flux/src/reduce_scatter_barrier_struct.hpp) | `src/reduce_scatter/reduce_scatter_barrier_struct.hpp` | GEMM-RS 的跨 rank 归约 barrier |
| [`src/epilogue_evt_nvshmem.hpp`](../flux/src/epilogue_evt_nvshmem.hpp) | `src/reduce_scatter/epilogue_evt_nvshmem.hpp` | epilogue visitor：算完一个 tile 直接 NVSHMEM 归约+散射 |

## 4. 建议的阅读顺序

1. [UniEP：dispatch 与第一段 GEMM](01-uniep-dispatch-gemm.md) → `uniep/kernels/cutedsl_dispatch_group_gemm.py`：信息密度最高，等待粒度压到「一个 expert」，通信与计算共享同一批 SM。
2. [UniEP：第二段 GEMM 与 combine](02-uniep-gemm-combine.md) → 反方向：push-combine 怎么在 epilogue 里发生。
3. [UniEP：确定性从哪来](03-uniep-determinism.md) → 为什么它敢说「和串行逐位一致」。
4. [Flux：AllGather 与 GEMM](04-flux-ag-gemm.md) → 换到 Flux，看「两条 stream + chunk flag」能到什么程度。
5. [Flux：GEMM 与 ReduceScatter](05-flux-gemm-rs.md) → Flux 的另一半：把 RS 塞进 GEMM epilogue。
6. [两套实现的对照](06-compare.md) → 一张表加一个「等待粒度 vs 开销」的框架，也是全系列的收口。
7. [跨工程的对照：Triton 对 CuTeDSL](../comm-overlap-src-07-triton-vs-cutedsl/) → 把镜头拉到工程之间。这一篇不读源码副本，读的是两个上游仓库本身。

## 5. 注释规矩：只加整行，校验脚本守住

副本的硬约束：**只允许插入整行注释**，不允许改动、删除、重排任何一行代码。所以每一行代码都和上游一字不差，校验脚本滤掉整行注释再逐行比对：

```bash
cd code/comm-overlap-src
python3 tools/verify_annotations.py                  # 全部校验
python3 tools/verify_annotations.py --list           # 看清单
python3 tools/verify_annotations.py --only uniep/    # 只看 UniEP
```

`tools/MANIFEST.tsv` 记录「仓库标记 + 上游相对路径 → 副本路径」，上游默认在 `~/github/<repo>`（`--root` 可改）。另两个工具：`check_links.py` 查 docs 和 README 里的相对链接；`sync_from_upstream.sh` 从上游重拉，等于丢弃全部注释，只在升级上游时用。

注释副本是改了就不 pull 的。上游换版本时，先 `sync_from_upstream.sh` 回到基线，再重跑注释（或用 `git diff` 搬注释块）——直接 `git pull` 会和注释冲突。

## 6. 注释版式与记号

每个副本开头（license 后）有一段固定格式的注释块，按这个顺序看：

```text
【上游】仓库 / 分支 / commit / 相对路径
【一句话】这个文件在通算并行里干什么
【本注释只讲三件事】……
【阅读顺序】先看哪段、再看哪段（带行号或函数名）
【术语】本文件用到的缩写（EVT / nbi / PE / TMEM / mbarrier …）
```

正文注释以块级为主：一段代码一段说明，回答「在干什么 / 为什么必须这样 / 去掉会怎样 / 代价是什么」，不逐行翻译英文注释；上游已有英文注释的地方，只补它没说的那一层（为什么、代价、失效方式）。

记号：【】标关键结论；→ 标数据流方向；角色名一律用上游英文词（dispatch / combine / GroupGEMM / mbarrier / TMA / TMEM / persistent CTA）；「第 N 行」类引用一律指注释副本的行号，不是上游行号。
