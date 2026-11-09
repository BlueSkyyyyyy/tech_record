# （一）UniEP 的 Dispatch + FC1 GroupGEMM：把等待粒度压到一个 expert

对应副本：[`uniep/kernels/cutedsl_dispatch_group_gemm.py`](../uniep/kernels/cutedsl_dispatch_group_gemm.py)（上游 1314 行 → 注释副本 1879 行，其中 1222 行代码，Blackwell sm100 版）。
文中用「搜 `xxx`」定位代码，因为副本里插入的注释会让行号漂移；文末给了一张**以当前副本为准**的行号表（已逐条核对）。

## 1. 融的是什么

MoE 的一层前向大致是：

```text
gate → dispatch（按 expert 把 token 发到 owner rank）→ GroupGEMM(FC1) → SwiGLU → GroupGEMM(FC2) → combine
```

传统实现里 dispatch 是一个独立的 all-to-all kernel，GroupGEMM 是另一个 kernel。
两个 kernel 之间有硬边界：**必须等 dispatch 全部完成，GEMM 才能启动**——哪怕第一个 expert 的数据早就到了。

UniEP 的做法是把 `dispatch + FC1` 融成**一个 persistent kernel**：

- 一部分 warp 持续把远端 token 行拉进本地 A buffer；
- 另一部分 warp 对**已经齐了的 expert** 直接开算；
- 两者之间只用「每个 expert 一个 signal」沟通。

于是「等待」从「整轮 dispatch」缩小到「当前这个 expert 的数据」。这就是全部收益的来源。

## 2. 三条不变量（读这份代码前必须记住）

1. **A buffer 是共享的**：dispatch 的落点就是 GEMM 的输入。没有第二份 staging 拷贝，
   所以「expert e 的行是否全部写完整」必须有一个精确的判断依据。
2. **M-contiguous 的 tile 顺序**：调度器（[`m_contig_group_tile_scheduler.py`](../uniep/kernels/m_contig_group_tile_scheduler.py)）
   保证同一个 group（expert）的所有 M tile 连续被消费，expert 序号单调不减。
   有了这条，GEMM 侧只需要记住「上次等过哪个 expert」，不必每个 tile 都重新等。
3. **signal 是单向的**：dispatch 只置 1，GEMM 只读；每次 launch 前由 host 侧清零。
   没有「用完归还」的语义——因为 A buffer 在整个 kernel 生命周期内不会被复用给别的 expert。

## 3. 角色分工

8 个 warp / 256 线程 / 2 个 warp group：

```text
warp 0-3  epilogue    TMEM → RMEM → SMEM → TMA store 到全局 C
warp 4    dispatch 生产者   TMA G2S：对端 symmetric buffer → SMEM ring
warp 5    dispatch 消费者   TMA S2G：SMEM ring → 本地 A 行；发 per-expert 信号
warp 6    MMA          tcgen05 UMMA 主循环
warp 7    TMA          A/B 的 GMEM → SMEM（GEMM 的 load pipeline）
```

搜 `self.dispatch_warp_id_base = 4` 附近，能看到这套 id 是怎么定下来的；再往下搜
`if warp_idx >= self.dispatch_warp_id_base and warp_idx < (...)` 就是 dispatch 的两个 warp 的入口。

为什么 dispatch 要两个 warp？因为 TMA 的 G2S 和 S2G 是两个方向的操作，中间必须有一个 SMEM 环
（`dispatch_tma_buffer`，`dispatch_num_stages` 行 × K 个元素）做中转：

```text
对端 A 行 --TMA G2S--> SMEM ring[i] --TMA S2G--> 本地 A[row]
             (warp 4)                  (warp 5)
```

**TMA 不支持 GMEM→GMEM 直搬**，这是这个环存在的唯一原因。环深 `dispatch_num_stages`（1–5）
决定了「已经拉进来但还没写出去」的行数——它同时也是「dispatch 能领先 GEMM 多远」的上限。

## 4. per-expert signal：整个 overlap 的开关

### dispatch 侧（搜 `expert_signal_counters`）

每个 CTA 只负责自己那一份分片（`linear_dispatch_block` 把 grid 展平后取模）。
一个 expert 的所有行搬完后：

```python
old = atomic_add(expert_signal_counters + expert_idx, 1, sem="release", scope="gpu")
if old + 1 == num_blocks:           # 我是最后一个完成该 expert 分片的 CTA
    store(expert_signals + expert_idx, 1, sem="release", scope="gpu")
```

- 用 `num_blocks`（grid 里的 CTA 总数）而不是「每个 expert 的 CTA 数」：grid 是 persistent 的，
  所有 CTA 都会遍历所有 expert，所以每个 expert 都由全部 CTA 各贡献一份。
- `sem="release"` 保证「我之前写的 A 行」在 signal 置 1 之前对别的 SM 可见。
- 只有最后一个 CTA 写 signal：把「一个 expert 的 N 个分片」压成「一次写」，避免 N 次无效写。

### GEMM 侧（搜 `last_waited_group`）

TMA warp 在每个 work tile 开始前：

```python
if cur_group_idx != last_waited_group:
    with elect_one():
        v = load(expert_signals + cur_group_idx, sem="acquire", scope="gpu")
        while v == 0: v = load(..., acquire)
    last_waited_group = cur_group_idx
```

三个细节值得停一下：

1. **只让一个 lane 自旋**（`elect_one`），其余 lane 在 `sync_warp` 处等。
   否则 32 个 lane 一起打同一个全局地址，只会把 L2 打爆。
2. **等待放在 `producer_acquire` 之前**。注释里写得很清楚：如果先拿到 SMEM stage 再等 signal，
   这个 stage 会一直被占着，而 dispatch 可能还要很久——等于用 SMEM 换了一个空转。
3. **只等「换 group」的那一次**。这是第 2 节第 2 条不变量的兑现：M-contiguous 保证 expert 单调，
   所以每个 expert 只需等一次，而且等的时候前一个 expert 的 MMA 还在跑。

## 5. 时间线

```text
         expert 0        expert 1        expert 2
disp  ████████████████░░░░████████████░░░░████████
                 ↓signal0         ↓signal1
gemm           ░░░████████████████████████████████
               ↑等 signal0（此时 dispatch 已经在搬 expert 1）
```

- `░` = 等待，`█` = 干活。GEMM 只在开头等一次；之后每进入一个新 expert，等的时间已经被
  「上一个 expert 的计算」覆盖掉了。
- 如果 dispatch 比 GEMM 快（小 K、小 expert），`░` 会消失，剩下纯计算；
  如果 dispatch 慢（跨机、token 倾斜），GEMM 会持续卡在 signal 上——此时问题不在 kernel，而在带宽。

## 6. SMEM 预算：先给通信留，再给计算分

搜 `_dispatch_smem_bytes` 与 `_compute_stages`：

```text
SMEM 总容量
  − reserved（1 KB）
  − dispatch ring buffer（dispatch_num_stages × K × 2 B）
  − dispatch 的 mbarrier / 指针表
  = 留给 GEMM 的容量 → 决定 A/B stage 数、epilogue stage 数
```

这段代码解释了一个常被忽略的事实：**融合 kernel 里通信和计算抢的是同一块 SMEM**。
ring 深一档，GEMM 的 A/B pipeline 就可能少一档；而 A/B 少一档，MMA 更容易被 TMA 饿死。
`dispatch_num_stages` 这个看起来无关紧要的参数，实际上是在两者之间调平衡。

顺带：`expert_alignment` 会被 `max(user, GEMM_CLUSTER_TILE_M=256)` 抬到 256
（搜 `expert_alignment = max(`），因为 cluster tile 的 M 是 256，调度器和 signal 的粒度都按它对齐。

## 7. 通算并行要点

- 【粒度】等待单位是 **expert**，不是「整轮 dispatch」，也不是「单个 token」——前者太粗、后者元数据爆炸。
- 【位置】signal 的等待放在 GEMM 的**生产端**（TMA warp），消费端（MMA）只见 pipeline，不被阻塞。
- 【可见性】`release` 写 / `acquire` 读 + `scope="gpu"`；intranode 用 gpu scope 就够，跨机要走 `_inter` 变体。
- 【复用】dispatch 的落点就是 GEMM 的输入 buffer；SMEM ring 只是为了让 TMA 合法而存在的中转。
- 【代价】SMEM 要在通信 ring 与计算 stage 之间分账；per-expert 计数器每次 launch 都要 host 侧清零。

## 8. 读这份代码时容易踩的坑

- **dropped token**：`decode_token_src_rank_topk_and_indices` 会返回 `rank = -1` 表示这个 top-k 槽被丢弃。
  代码里到处是 `if (src_rank + 1) != 0` 和 `safe_*` 变量，就是为了不在指针表上越界读。
  看到这些 `safe_` 前缀不要以为是冗余。
- **`problem_sizes_m[expert]` 是真实 token 数**，但 A 的行是按 `expert_alignment` 补齐的；
  两者混用会导致行偏移错位（`expert_start += aligned_tokens` 这行就是处理这个）。
- **`num_recv_tokens + world_size`** 这种「算了不用」的语句没有语义；副本注释的说法是「这次读没有被任何计算使用，可以当上游遗留的占位读」。它读的是别的 rank 写的 recv 计数，不影响每个专家的行区间——行区间由 `problem_sizes_m` 前缀和决定。
- 这个文件是 **intranode** 版本（对端指针直接 TMA）。跨机看同目录的 `*_inter.py`，走 NCCL GIN，机制不同。

## 9. 行号表（副本 `uniep/kernels/cutedsl_dispatch_group_gemm.py`，已核对）

| 内容 | 行号 |
|---|---|
| warp 角色常量、两个 NamedBarrier 的同步域 | 209、213 |
| SMEM 预算：先扣 dispatch ring（`_dispatch_smem_bytes`） | 231 |
| dispatch 生产者（warp 4）专家循环入口 | 1084 |
| `acquire_and_advance` 拿 SMEM stage → 发 TMA G2S | 1158 |
| `bulk_s2g_atom` 把行写回本地 A | 1255 |
| 释放 stage（满/空两种路径） | 1266、1280 |
| per-expert 计数器 `atomic_add(release, gpu)` | 1300 |
| 最后一个 CTA 写 `expert_signals`（release） | 1312 |
| GEMM TMA warp 等 signal（acquire，`last_waited_group` 去重） | 1384 |
| `producer_acquire`（**在等 signal 之后**） | 1423 |
| MMA 主循环（`acc_pipeline.producer_acquire` / `cute.gemm`） | 1456、1499 |
| epilogue（`tmem.allocate` 起） | 1554、1567 |
| SMEM 第二笔账：`_compute_stages` 结算 A/B、acc、epi stage | 1780 |
