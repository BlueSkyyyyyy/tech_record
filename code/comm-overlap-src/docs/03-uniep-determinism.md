# （三）UniEP 的确定性从哪来：M-contiguous 调度 + 固定顺序归约

**这方案把「谁先到谁先写」换成了「落点先算好、每个输出只写一次、按固定顺序累加」，代价是一次全局计数同步和被牺牲的调度自由度**。

读完你会知道：

- 到达顺序乱了，结果为什么还能逐位一样；
- 一个 tile 的落点怎么算出来，为什么不用抢；
- 它在什么形状下更慢、哪里会静默算错。

副本：[`uniep/kernels/m_contig_group_tile_scheduler.py`](../uniep/kernels/m_contig_group_tile_scheduler.py)、
[`uniep/kernels/cutedsl_topk_reduce.py`](../uniep/kernels/cutedsl_topk_reduce.py)。行号会漂，按函数名搜。

## 1. 乱序只从三个口子进来

overlap 让 B 在 A 还没结束时开始。收益是更早开算，副作用是顺序不由程序控制。乱序只从三处进来：

| 哪里乱 | 后果 |
|---|---|
| 数据**到达**的先后 | 按到达顺序定落点，A 的行序每次不同 |
| 输出元素被**写入**的次数 | 多个 CTA 对同一元素累加，求和顺序不定 |
| 归约的**求和顺序** | 浮点不满足结合律，低位每次都不同 |

UniEP 不靠「精度看起来没掉」，而是把三个口子分别堵死。

## 2. 落点是算出来的，不是抢来的

如果不这么做：常见实现用 `atomicAdd(offset[e], 1)` 抢一个槽位再写。行序成了运行时的产物，同一份输入两次跑可以不一样。数值上等价（行序不影响 GEMM 结果），但后面只要有按行序归约，就不可复现。

看什么：落点只由 `problem_sizes_m` 的前缀和算出，不含时间。

```text
expert e 的行区间 = [Σ_{i<e} align(M_i), Σ_{i≤e} align(M_i))
```

所以呢：第几个 expert 的第几行写到 A 的哪里，只取决于路由结果，与谁先到无关；到达顺序只决定何时能开算，不决定算出什么。

这套落点要求每个 expert 在 A 上占一段连续 M 行，因而也占一段连续线性 tile 区间，组间按 expert 顺序拼接。组内序号按 M-major（`AlongM`）展开：同一个 M tile 的 A 行在 N 方向连续复用，所以前缀和只需要对 M 求。

连续买到三件事：A 的行按 expert 连续，一张 TMA 描述符覆盖整个 monolithic A；per-expert signal 只等一次（`last_waited_group`）；C 的行也连续，combine 的 push 一行一个 lane，寻址简单。

换成 N 连续，同一 expert 的 M tile 会被 N 方向的 tile 隔开，signal 要反复等，`last_waited_group` 直接失效。

`MContiguousGroupTileScheduler` 相对基类的四处改动，三处在取数与计算上：

- `problem_sizes_m` 从 2D 压成 1D（同一层 MoE 的 N/K 固定，只有 M 随流量变），`_get_problem_for_group` 单次 `Int32` load（基类按 vec-4 读 4 维，省每个 tile 一次的访存）；
- N 折进 constexpr `ncluster_n`，`ceil_div(N, Cm_n)` 不必每 tile 现算；
- `delinearize_z` 从 search state 反推 `cluster_count_m`，不再二次加载。

第四处是 host 不变量：`padded_m` 是 `Cm` 的倍数时，`ceil_div(real_m, Cm) == ceil_div(padded_m, Cm)`。

于是 kernel 只认 `padded_m`、signal 和归约只关心 `real_m`，host 不必再传 offsets。

## 3. 每个输出元素只写一次，比锁住执行顺序便宜

如果做 split-K：一个输出 tile 的 K 被切成几段交给不同 CTA，最后加回部分和——加的次序又由谁先完成决定，确定性当场破功。

UniEP 的选择是这个调度器不做 split-K：一个输出 tile 的 K 方向由同一个 CTA 从头算到尾。于是每个输出元素只会被一个 CTA 写一次，累加顺序就是 K 方向 for 循环的顺序，编译期固定。无论 tile 在哪个 SM、第几个 wave 执行，结果都逐位相同。

这比「调度顺序固定」更强，也更便宜：你不需要让 tile 的执行顺序确定，只需要让每个输出元素的求和顺序确定。

## 4. 归约侧：一个 CTA 一个 token，k 只能升序

如果不这么做：用 `atomicAdd` 把 top-k 贡献加到输出上。k=8 时只有 8 次加法，看起来误差微不足道，但顺序不定意味着两次跑的结果可能差 1 ulp，梯度检查/对拍会随机失败。

看什么：`MoETopkReduceBlockPerToken` 里，一个 CTA 独占一个 token，k 循环编译期展开。

```text
grid = (num_tokens, 1, 1)      # 一个 CTA 负责一个 token
每个 thread 负责 8 个 half（128 bit）= 一个 atom
for k in range_constexpr(topk):        # k 从小到大，编译期展开
    load staging[token*topk + k, atom]
    acc += value                        # FP32 累加
store output[token, atom]
```

所以呢：确定性只靠三件事——不存在跨 CTA 对同一输出累加，也就没有 atomic 的乱序；展开顺序等于源码顺序，编译器不能重排浮点加法（除非 `-ffast-math`，CUDA 默认不开）；FP32 累加与参考实现的 `x.sum(dim=0)` 同序（k 维升序）。

## 5. 确定性不免费：一处同步、两处变慢

换来：同一份输入、同一套并行配置，两次运行逐位相同，能和串行参考对拍，误差 0；训练侧梯度累积顺序固定，debug 时「昨天能复现、今天不能」的玄学问题消失。

代价：

- dispatch 必须先把 `problem_sizes_m` 算出来（一次全局 token 计数与 all-reduce），这引入 host/device 同步。UniEP 用 `_poll_local_recv_count` 之类的轮询把它藏起来，但它是真实存在的——确定性不免费。
- M-contiguous 让负载更难摊平：某个 expert 的行特别多时，它的 M tile 会集中在一段时间里，无法像 N 连续那样把不同 expert 的 tile 交错来抹平。
- 不做 split-K，K 很大时单 CTA 的 K 循环很长，长尾比 split-K 方案更明显。

## 6. 会静默算错的坑

- 「不做 split-K」是地基。哪天为了小 M 的利用率引入 split-K，必须同时引入确定的归约树（比如按 rank 固定顺序的两阶段归约），否则确定性立刻破功。
- `padded_m` 与 `real_m` 只在 `Cm` 对齐时等价。手工构造不规则 expert 计数做测试时这条最容易破；破了 kernel 不报错，只是末尾几行读到垃圾。

如果只记一件事：确定性买的是求和顺序，不是执行顺序。
