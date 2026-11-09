# （二）UniEP 的 FC2 GroupGEMM + Combine：在 epilogue 里把结果推出去

对应副本：[`uniep/kernels/cutedsl_group_gemm_combine.py`](../uniep/kernels/cutedsl_group_gemm_combine.py)（上游 1067 行 → 注释副本 1411 行，其中 972 行代码）。
前半篇（dispatch + FC1）见 [01](01-uniep-dispatch-gemm.md)。

## 1. 为什么 combine 不能照抄 dispatch 的写法

dispatch 是**数据的进场**：一个 expert 的行散落在所有 rank 上，GEMM 必须等「这个 expert 的行齐了」，
所以需要 pull + per-expert signal。
combine 是**结果的出场**：FC2 算出的每一行（token × topk 槽）**只属于一个目标 rank**，
写过去就完事，不需要目标侧再收集，也不需要「某专家齐了」这种聚合条件。

于是第二个 MegaKernel 的形态完全不同：

| | MegaKernel ①（dispatch + FC1） | MegaKernel ②（FC2 + combine） |
|---|---|---|
| warp 数 | 8（epilogue 0-3 / dispatch 4-5 / MMA 6 / TMA 7） | **6**（epilogue 0-3 / MMA 4 / TMA 5） |
| 通信方向 | pull（TMA 从对端读） | push（SIMT 128-bit store 写到对端） |
| 同步点 | 每个 expert 一个 signal | **没有 per-expert 同步**，只有一个 kernel 后的 group barrier |
| SMEM 花在哪 | dispatch ring + GEMM stages | 只花在 GEMM stages（单槽 epilogue） |

少掉的两个 warp 不是"省资源"，而是 push 模式的副产品：通信被塞进了**已经持有累加器的那批 epilogue lane**，
它们本来就要把 TMEM 里的结果搬出来，顺手写去对端即可，不需要第三个角色。
也不需要中转环：对端 buffer 可直接写（GPU scope 的 128b store），不用先落到本地 SMEM 再 S2G。

## 2. 每个 lane 拥有一个输出行

搜 `my_global_row`：

```python
my_global_row = global_m_base + epi_warp_idx * cpy_m_per_warp + epi_lane
encoded = token_src_rank_topk_and_indices[my_global_row]
src_token_idx, my_dst_rank, src_topk_idx = decode_token_src_rank_topk_and_indices(encoded)
my_dst_slot = src_token_idx * topk + src_topk_idx
```

这条链是理解 combine 的关键：**A 的第 `my_global_row` 行，是「源 rank 的 token `src_token_idx` 的第 `src_topk_idx` 个专家槽」**。
它在目标 rank 的 staging buffer 里的位置就是 `src_token_idx * topk + src_topk_idx`——一个与专家无关、只与 token 有关的固定槽位。

所以 combine 不需要知道「专家路由表」，只需要在 dispatch 时把这条编码 `(src_token, src_rank, topk_idx)` 跟着行一起搬过来。
这就是为什么 dispatch 阶段要维护 `token_src_rank_topk_and_indices`：**它是给 combine 用的回程地址**。

几个工程细节：

- `lane ↔ row` 一一对应（`cpy_m_per_warp == 32`、`epi_tile_m == cta_tile_m`），
  所以不需要在 epilogue 里再做一次 cross-lane 归约，每行独立寻址。
- 真正搬数据时是 **16 个 lane 合作搬一行**（`lanes_per_row = atoms_per_row`，
  每个 lane 128 bit = 8 个 bf16），所以一行的 N 方向被切成 `N/8` 个 atom，刚好一组 lane 铺满。
- 行首地址通过 `shuffle_sync(my_row_base, leader_lane)` 广播给同组的 lane——只有 leader 算地址，其他人复用。
- `my_pred` / `safe_dst_rank` / `bk_row_base != 0` 这一套是**丢弃槽的保护**：
  `dst_rank == -1` 的槽不写，地址算成 0，最后靠「0 地址不写」过滤掉。写 0 地址会踩空页，所以恒假但必须存在。

## 3. 单槽 SMEM epilogue：用省下的 SMEM 换一个 A/B stage

搜 `single-slot SMEM optimisation` 那段注释。常规 epilogue 会开 `num_epi_stage = 2+` 个 SMEM 槽，
让 R2S 和 TMA store 交替。这里强制 `num_epi_stage == 1`，然后把 R2S 与 push 按 subtile 交错：

```text
for each work tile:
  for s in subtile_cnt:
      TMEM → RMEM (cast bf16)
      if s == last: release acc stage      ← 关键
      RMEM → SMEM[0]
      SMEM[0] → peer GMEM (cooperative push)
```

- WAR 冲突（下一轮 R2S 覆盖 SMEM[0] 时 push 是否已读完）**靠 per-warp program order + 一次 `sync_warp`** 保证
  （搜 `cute.arch.sync_warp()`）：写者和读者是同一个 warp 里的同一批 lane，不存在跨 warp 竞争，所以不需要块级 barrier。
- 省下来的 SMEM = `(subtile_cnt - 1) × epi_bytes_per_stage`，正好够在 M=128/N=256/K=2560 的 bf16 主负载上
  多开一个 A/B stage。这是「用调度复杂度换流水深度」的典型交易。

## 4. 重叠发生在哪：提前释放 acc stage

```python
if subtile_idx == subtile_cnt_const - 1:
    with elect_one():
        acc_pipeline.consumer_release(acc_consumer_state)   # 在 push 之前
```

- 累加器只在**最后一个 subtile 读完 TMEM** 时才还给 MMA，而不是等 push 结束。
  于是 MMA 可以在 epilogue 还在往对端推数据的时候，就开始下一个 tile 的 GEMM。
- 这就是这个 kernel 里唯一的「计算-通信重叠」：**push 与下一个 tile 的 MMA 重叠**，
  而不是像 MegaKernel ① 那样「dispatch 与当前 tile 的 MMA 重叠」。
- 代价：acc stage 只有 `num_acc_stage=2` 个，提前释放意味着最多只能领先一个 tile；
  push 太慢仍然会把 MMA 堵住。**这个 2 不是随便定的**：`(128,256)` 的 fp32 累加器一级就占满 256 列 TMEM，
  两级正好吃满 512 列——重叠深度在这里是被 **TMEM 容量**卡住的，不是 SMEM。

## 5. 权重也要跟着走

MoE 的 top-k 加权（`topk_weights`）作用在 combine 的归约上。UniEP 让 dispatch 阶段顺带把每个 token-slot 的标量权重
搬到本地（`dispatched_weights[my_global_row]`），combine 时再按 `dst_slot` 推到目标 rank
（搜 `n_tile_idx == 0` 的门控：多 N-tile 的 epilogue 只在该 tile 的第一次写，避免重复推同一个标量）。

这个标量 store 被特意放在 `acc_pipeline.consumer_wait` **之前**——注释写得很直白：
让它和「等 acc stage 的停顿」共处，等于白捡。

## 6. 唯一不能省的那次同步

push 完不等于目标 rank 能看到：`topk_reduce` 要读回 staging 做 top-k 求和，必须知道「所有 rank 都推完了」。
这一步在 host 侧（`ep_overlap_kernels.py` 的 `group_gemm_combine`）用一次 `ep_group_barrier()` 收口，
然后才启动 `topk_reduce`。所以：

> UniEP 的 overlap 边界止于 combine 的 push。**最后的归约是一段串行区**，
> 它前面的 barrier 长度 ∝ 最慢那个 rank 的 combine 时间。

这也是为什么论文里"确定性"值得单独讲：这段串行区里每个 token 的 k 份贡献必须按固定顺序求和，
否则每次跑出来的低若干位都不一样（见 [03](03-uniep-determinism.md)）。

## 7. 通算并行要点

- 【方向】进场 pull、出场 push，是两份实现（UniEP 与 Flux）共同的选择，不是巧合。
- 【寻址】combine 的地址信息来自 dispatch 随行搬运的 `(src_token, src_rank, topk_idx)` 编码，回程不查表。
- 【重叠】epilogue 的 push 与**下一个 tile 的 MMA** 重叠（靠提前释放 acc stage），而不是与当前 tile 重叠。
- 【代价】单槽 epilogue 换来一个额外 A/B stage；`acc` 只有 2 档，push 慢会反压 MMA。
- 【边界】combine 之后必须有一次 group barrier + 独立的 topk reduce，这段不 overlap。

## 8. 坑

- `num_epi_stage == 1` 是 assert 出来的（`single-slot SMEM optimisation requires num_epi_stage==1`），
  改 stage 数会直接触发断言，不是性能回退而是不合法。
- `my_row_base` 对丢弃槽乘了 `my_pred`（0/1），依赖「地址 0 永远不可能是合法分配」这个假设。
  在有 UVA/统一寻址的机器上这个假设成立；移植时要注意。
- push 用的是 `memory_scope=GPU` + `l1c_evict_priority=NO_ALLOCATE`：
  数据只写一次、不会被读，绕 L1 是省带宽而不是省延迟。
- 权重 side-load 的 `n_tile_idx == 0` 门控依赖「N 方向 tile 数是编译期常量」，改 tile 形状时要一起看。

## 9. 行号表（副本 `uniep/kernels/cutedsl_group_gemm_combine.py`，已核对）

| 内容 | 行号 |
|---|---|
| 类 docstring：TMEM→RMEM→单槽 SMEM→16 lane 协作推 | 80 |
| 单槽 SMEM 的两阶段交错（注释） | 968 |
| `epi_tile_m == cta_tile_m`（一行一 lane 的前提） | 994 |
| `store_atom`：`memory_scope=GPU` + `NO_ALLOCATE` | 1006 |
| `num_epi_stage == 1` 的断言 | 1070 |
| 路由三段：`dst_slot = src_token_idx * topk + src_topk_idx` | 1114 |
| 提前释放 acc stage（最后一个 subtile） | 1195 |
| 单槽 WAR：`sync_warp()` | 1209 |
| 16 lane 协作推：`shuffle_sync` 广播行基址 | 1226 |
| `num_acc_stage = 2`（受 TMEM 512 列限制） | 1345 |
