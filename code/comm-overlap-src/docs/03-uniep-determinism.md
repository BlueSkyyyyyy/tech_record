# （三）UniEP 的确定性从哪来：M-contiguous 调度 + 固定顺序归约

对应副本：[`uniep/kernels/m_contig_group_tile_scheduler.py`](../uniep/kernels/m_contig_group_tile_scheduler.py)、
[`uniep/kernels/cutedsl_topk_reduce.py`](../uniep/kernels/cutedsl_topk_reduce.py)。

## 1. 问题：overlap 天生会打乱顺序

overlap 的本质是「让 B 在 A 还没完全结束时开始」。一旦允许乱序，就有三个地方会变：

| 会被打乱的东西 | 后果 |
|---|---|
| 数据**到达**的先后 | 如果按到达顺序决定落点，A buffer 的行序每次不同 |
| 输出元素被**写入**的次数/顺序 | 如果多个 CTA 对同一个输出累加，求和顺序不定 |
| 归约的**求和顺序** | 浮点不满足结合律，低若干位每次都不同 |

UniEP 的思路是把这三件事**分别掐死**，而不是靠"精度看起来没掉"。

## 2. 到达顺序不影响结果：落点是算出来的，不是抢来的

dispatch 的行落点由 `problem_sizes_m` 的前缀和决定：

```text
expert e 的行区间 = [Σ_{i<e} align(M_i), Σ_{i≤e} align(M_i))
```

也就是说，**第几个 expert 的第几行写到 A 的哪个位置，只取决于路由结果，与谁先到无关**。
到达顺序只影响「什么时候能开始算」，不影响「算出来的东西是什么」。

对比一下常见的另一类实现：用 `atomicAdd(offset[e], 1)` 抢一个槽位再写。
那种做法的行序是运行时的产物，同一份输入两次跑可以不一样——数值上其实等价（行序不影响 GEMM 结果），
但一旦后面有「按行序做归约」的东西，就不可复现了。

## 3. 每个输出元素只写一次：不做 split-K

搜 `MContiguousGroupTileScheduler` 的 docstring 里那句
`Host invariant: padded_m is a multiple of Cm`，以及父类 `StaticPersistentGroupTileScheduler` 的用法。

关键事实：这个调度器**不做 split-K**。一个输出 tile 的 K 方向由同一个 CTA 从头算到尾，
因此：

> 每个输出元素只会被一个 CTA 写一次，累加顺序 = K 方向 for 循环顺序（编译期固定）。
> 无论 tile 被哪个 SM、在第几个 wave 执行，结果都逐位相同。

这条比"调度顺序固定"更强，也更便宜：**你不需要让 tile 的执行顺序确定，只需要让每个输出元素的求和顺序确定**。

## 4. M-contiguous 改了什么

相对基类 `StaticPersistentGroupTileScheduler`，副本里注释列了四处改动。逐条说人话：

1. **`problem_sizes_m` 从 2D 变 1D、单次 `Int32` load**（`_get_problem_for_group`）。
   每个 group 的 N/K 都相同（同一层 MoE 的 FC1/FC2 形状固定），只有 M 在变，所以只需要存 M。
   基类为了通用性按 vec-4 读 4 个维度，这里省成一次 load——省的是**每个 tile 一次**的访存。
2. **N 折进 constexpr**（`ncluster_n`）。N 不变，`ceil_div(N, Cm_n)` 就没必要每次都算。
3. **`delinearize_z` 从 search state 反推 `cluster_count_m`**，而不是重新算一遍 `ceil_div`。
   二分搜索已经把「当前 group 之前有多少 tile」算出来了，直接减一下就是当前 group 的 tile 数。
4. **`ceil_div(real_m, Cm) == ceil_div(padded_m, Cm)`** 这条 host 不变量的意义：
   kernel 只认 `padded_m`（对齐后），但 signal / 归约只关心 `real_m`（真实行数）。
   两者在 `Cm` 对齐下等价，所以 kernel 里可以放心用其中一个。

### 为什么是 M 连续，不是 N 连续

`delinearize_z` 里先做一次 group 搜索（在扁平的 tile 序号上二分，找到「这个 tile 属于哪个 group」），
再用 `_compute_cta_tile_coord` 把组内序号拆成 `(cta_tile_idx_m, cta_tile_idx_n)`。

**「M-contiguous」的准确含义**：每个 expert 在 A 上占一段连续的 M 行，因此也占**一段连续的线性 tile 区间**；
expert 的区间按 M 顺序首尾相接。（组内是 m 快还是 n 快由基类决定，通常是 n 快——
这样同一个 M tile 的 A 行会被连续复用 `ncluster_n` 次，L2 更友好。但无论哪种，**group 区间连续且单调**这条不变。）

这样做的收益全在「行」这一侧：

- A 的行按 expert 连续 → 一张 TMA 描述符覆盖整个 monolithic A，不需要按 expert 建多张；
- 一个 expert 的 M tile 在时间上连续 → per-expert signal **只等一次**（MegaKernel ① 里的 `last_waited_group` 优化成立）；
- C 的行也连续 → combine 的 push 是「一行一个 lane」，寻址简单。

如果换成 N 连续，同一个 expert 的 M tile 会被 N 方向的 tile 隔开，signal 就得反复等，
`last_waited_group` 这个优化直接失效。

## 5. 归约这一侧：一个 CTA 一个 token，k 升序

搜 `MoETopkReduceBlockPerToken`。它的结构异常简单：

```text
grid = (num_tokens, 1, 1)      # 一个 CTA 负责一个 token
每个 thread 负责 8 个 half（128 bit）= 一个 atom
for k in range_constexpr(topk):        # k 从小到大，编译期展开
    load staging[token*topk + k, atom]
    acc += value                        # FP32 累加
store output[token, atom]
```

确定性的来源就三行：

1. **一个 CTA 独占一个输出 token**：不存在跨 CTA 对同一输出累加，也就没有 atomic 的乱序；
2. **`k` 的循环是 `range_constexpr`**：展开顺序 = 源码顺序，编译器不能重排浮点加法（除非 `-ffast-math`，CUDA 默认不开）；
3. **FP32 累加**：与参考实现的 `x.sum(dim=0)` 顺序一致（k 维升序）。

对比「用 `atomicAdd` 把 top-k 贡献加到输出上」：那种写法在 k=8 时只有 8 次加法，
看起来误差微不足道，但顺序不定意味着**两次跑的结果可能差 1 ulp**，梯度检查/对拍会随机失败。

## 6. 这套设计换来了什么，代价是什么

**换来**：

- 同一份输入、同一套并行配置，两次运行逐位相同（可以直接和串行参考对拍，误差 0）；
- 训练侧的梯度累积顺序固定，debug 时「昨天能复现、今天不能」的玄学问题消失。

**代价**：

- dispatch 必须先把 `problem_sizes_m` 算出来（需要一次全局的 token 计数与 all-reduce），
  这引入了一个 **host/device 之间的同步点**；UniEP 用 `_poll_local_recv_count` 之类的轮询把它藏起来，
  但它是真实存在的（这也是「确定性不免费」的地方）。
- M-contiguous 让负载不均更难摊平：如果某个 expert 的行特别多，它的 M tile 会集中在一段时间里，
  无法像 N 连续那样把不同 expert 的 tile 交错来抹平。
- 不做 split-K 意味着 K 很大时单 CTA 的 K 循环很长，长尾比 split-K 方案更明显。

## 7. 通算并行要点

- 【确定性】来自「每个输出元素只被写一次 + 求和顺序编译期固定」，而不是「让执行顺序确定」。
- 【映射】token → A 行的落点由前缀和算出，与到达顺序解耦；这是 overlap 敢乱序的前提。
- 【调度】M-contiguous 同时服务三件事：signal 只等一次、TMA 描述符单张覆盖、combine 寻址简单。
- 【归约】top-k 归约一个 CTA 一个 token、k 升序、FP32——三行代码，决定能不能 bitwise 对拍。
- 【代价】确定性要求先算出全局 expert 计数，引入一次不该被忽视的同步；也牺牲了负载均衡的灵活性。

## 8. 坑

- 「不做 split-K」是这套确定性的地基。如果哪天为了小 M 的利用率引入 split-K，
  就必须同时引入确定的归约树（比如按 rank 固定顺序的两阶段归约），否则确定性立刻破功。
- `padded_m` 与 `real_m` 只在 `Cm` 对齐时等价。手工构造不规则 expert 计数做测试时，
  这条不变量最容易破——破了之后 kernel 不报错，只是末尾几行读到垃圾。
- topk_reduce 的 `hidden_size % atom_v == 0`（atom_v=8）是硬约束，hidden 不是 8 的倍数会直接 assert。

## 9. 行号表（以当前副本为准，已核对）

`uniep/kernels/m_contig_group_tile_scheduler.py`：

| 内容 | 行号 |
|---|---|
| 类定义（docstring 列了相对基类的 4 处改动） | 80 |
| `ncluster_n` 折成 constexpr | 172 |
| `__new_from_mlir_values__`（CuTe DSL 跨 MLIR 边界传对象） | 194 |
| `create()`：用 `bidz` 当持久化 cluster 的初始线性序号 | 227 |
| `_get_problem_for_group`：1D 单次 Int32 读 | 285 |
| `_get_cluster_tile_count_mn`：N 因子 constexpr | 290 |
| `delinearize_z`：一次 group 搜索定出所有坐标 | 304 |
| 从 search state 反推 `cluster_count_m` | 346 |

`uniep/kernels/cutedsl_topk_reduce.py`：

| 内容 | 行号 |
|---|---|
| `MoETopkReduceBlockPerToken`（一个 CTA 一个 token = 确定性的一半） | 52 |
| `grid=(num_tokens,1,1)` | 112 |
| `my_topk[k] = topk_indices[token_idx, k]`（索引缓存进寄存器） | 164–165 |
| FP32 定序累加循环 | 204 |
| 累加并转回输出 dtype | 239 |
