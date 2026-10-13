# DeepEP 的 dispatch/combine 参考实现（语义规范）与流量/基准生成器

> 第 7/48/51/52/53/58/59/61/63/64 篇一直在拆 DeepEP 的 **kernel**：dispatch、combine、
> combine 归约 epilogue、dispatch 落盘 epilogue、EPLB 反向、ptx 公共层、多 rank 屏障、bucket 集合通信……
> 但它们**「算什么才对」的定义**从没系统讲过。本篇补上这层地基：`deep_ep/utils/` 里三个**纯 PyTorch**
> 模块——
> * `refs.py`（243 行）：dispatch / `generate_pre_combine_data` / `ordered_accumulate` / combine 的
>   **参考实现**，是所有 DeepEP kernel 与测试的**语义规范**；
> * `gate.py`（180 行）：负载不均衡的**流量生成器**（构造指定的 imbalance ratio）；
> * `testing.py`（239 行）：`flush_l2_cache` / `bench` / `bench_kineto` / `parse_num_bytes` 等**基座工具**。
>
> **本机可跑**：三者都是纯 PyTorch / numpy，`refs.dispatch` 只用 `dist.get_rank/get_world_size` +
> `all_to_all_single`（torch 自带 NCCL），**不碰 DeepEP 的 C++/NCCL 设备 API**。所以 H100 真机
> 8 卡 `mp.spawn` 跑真实 all_to_all 做独立 oracle 对拍完全可行；唯一障碍是 `import deep_ep` 会触发
> `check_nccl_so()`（本机 NCCL 2.29.7 不匹配）——用 `sys.modules` 注入合成包 + stub `envs` 即可绕过。
> 与第 30/31/40/41 篇（DeepGEMM 的 `testing/`/`generators.py`）成对照：那是「GEMM 怎么造操作数、
> 怎么判对」，本篇是「EP 通信怎么定义、怎么造流量」。

---

## 1. 三层测试栈里的「规范层」

DeepEP 的测试栈自下而上三层：

| 层 | 文件 | 作用 |
|---|---|---|
| 参考实现（规范） | `deep_ep/utils/refs.py` | 纯 PyTorch 版 dispatch/combine，**定义语义** |
| 流量/负载生成 | `deep_ep/utils/gate.py` | 造指定 imbalance 的 `topk_idx`/scores |
| 基座工具 | `deep_ep/utils/testing.py`、`math.py` | 计时（bench/bench_kineto）、`calc_diff`、FP8 cast、计数 |

被它服务的 kernel：`impls/ep/dispatch.cuh`（第 51 篇）、`combine.cuh`/`combine_reduce_epilogue.cuh`
（第 48 篇）、`dispatch_copy_epilogue.cuh`（第 51 篇）、`reduce_grads.cuh`（第 43 篇）、
`comm/barrier.cuh`（第 63 篇）。测试 `tests/ep/test_ep.py:76-77` 就是先用
`gate.get_unbalanced_scores(...)` 造 score、再拿来喂 dispatch/combine。

`math.py` 里有一个和 DeepGEMM **同名但不同定义**的判据：`calc_diff`（`math.py:5-9`）先把两侧
`+1` 再算，`sim = 2·Σxy /(Σx²+Σy²)`，返回 `1-sim` ——是「余弦相似度型」，与 DeepGEMM
`testing/numeric.py` 的归一化平方误差 `‖x−y‖²/(‖x‖²+‖y‖²)` 不是一回事（见第 30 篇）。

---

## 2. `refs.dispatch`：dispatch 的语义规范

函数签名 `refs.py:10-12`：输入 `x [T,H]`（或 FP8 的 `(x, sf)` 对）、`topk_idx [T,k]`、
`topk_weights [T,k]`、`num_max_tokens_per_rank`、`num_experts`，返回
`(recv_x, recv_topk_idx, recv_topk_weights, recv_src_token_idx, num_recv_tokens_per_rank)`。

它**不做 ragged all-to-all，而是用「mask + 定长 all_to_all_single」精确复刻**——这是理解 DeepEP
「token 按 (来源 rank, 全局索引) 排序」这一约定的关键。逐段：

1. **SF 布局硬约束**（`:45-52`）：FP8 时 `hidden_sf == ceil_div(hidden, 128)`，即 per-128 block scale。
2. **逐目标 rank 收集要发的 token**（`:63-81`）：对每个 `dst_rank`，用
   `mask_to_send = ((expert_start ≤ topk_idx) & (topk_idx < expert_end)).any(dim=1)`（`:67`）选出
   「至少有一个 top-k 专家落在该 rank」的 token；把该 token 整行发过去，并把**不属于该 rank 的
   top-k 槽改成 -1**（`masked_topk_idx`，`:76-78`）——这解释了：**同一 token 会被发给多个 dst rank
   （其 top-k 专家分散时），每份只保留本 dst 的专家，其余为 -1**。
3. **标注来源全局索引**（`:88-90`）：`send_src_token_idx += rank_idx * num_max_tokens_per_rank`，
   其中 `src_token_global_idx = src_rank * num_max_tokens_per_rank + local_idx`。这是全 DeepEP 的坐标
   约定（第 51 篇的 `dispatch_copy_epilogue` 也用它）。
4. **交换计数 → 交换数据**（`:92-110`）：先 `all_to_all_single` 交换每 rank 发/收的 **token 数**，
   再按该计数做**定长** `all_to_all_single`（x / sf / topk_idx / topk_weights / src_idx 各一次）。
   因为发送侧按 `dst_rank` 升序、dst 内按 token 升序拼接，接收侧的结果自然按
   `src_token_global_idx` **升序**。
5. **接收侧再掩码**（`:112-117`）：`recv_topk_idx -= expert_start`（转成本 rank 本地专家号），
   越界槽（不属于本 rank）`masked_fill_(~mask, -1)`——于是 **`recv_topk_idx ∈ [-1, E_per_rank)`**。

**文档声明的三大不变量**（`:14-25` 的 docstring）：
* *Not expanded*（同一 token 的多份副本分列在不同目的地，不展开成 top-k 行）；
* *Sorted by rank and then by token*（= 按 `src_token_global_idx` 升序）；
* 越界 `recv_topk_idx` 置 `-1`。

---

## 3. `generate_pre_combine_data` 与 `ordered_accumulate`

combine 的**输入** `y [T, k, H]`（每 token 的每个 top-k 专家输出）在参考里由
`generate_pre_combine_data`（`:126-153`）合成：

```
token_seeds = src_token_global_idx[i] * k + j
max_seed    = num_ranks * num_max_tokens_per_rank * k
y[j,:]      = sin( ((token_seeds*131071 % max_seed + 1)/max_seed) * (1..H) + sin(seed) )   → bf16
```

`P = 131071` 是大素数（`:133` 的 docstring），乘以素数再取模是为了让不同 token 的「种子」充分散开，
从而 combine 的归约结果有可辨识的结构。`seed` 来自 `envs.get_global_seed()`（`:151`）——所以
**同一 seed 下数据可复现**，这就是各篇里 DeepEP 测试能逐位复现的根。

`ordered_accumulate`（`:156-175`）把事情挑明：**沿 top-k 维按顺序、以 fp32 逐项累加**，最后
cast 回原 dtype。`combine` 的最后一步（`:242`）就是 `ordered_accumulate(y, bias_sum)`。
注意它**没有**用 `torch.sum`——累加顺序被固定，保证确定性（对照第 52 篇的 `deterministic_sort`）。

---

## 4. `refs.combine`：多级归约的语义

`combine`（`:177-243`）的签名里有三个「反直觉」参数：
`num_scaleout_ranks` / `num_scaleup_ranks`（两级物理域：scale-out RDMA × scale-up NVLink）与
`reduce_in_local` / `reduce_in_scaleup`（在哪一级做归约）。核心是内部 `grouped_reduce`（`:201-236`）：

* `group_id` 先 **stable sort**（`:212`），组内保持原始 top-k 顺序；
* 沿 top-k **严格升序**累加到 fp32 缓冲（`:221-223`）；
* 当遇到「段断点」（`group_id[:,i] != group_id[:,i+1]`，或最后一列，`:222`）时，把累计值写回**该组最右
  的槽**（`:226-227`），并把**组内其它槽清零**（`:230`）——所以归约结果的「落点」是每组最大的 top-k 下标。

三种配置（docstring `:188-193`）：

| `(reduce_in_local, reduce_in_scaleup)` | 含义 | 组号 |
|---|---|---|
| `(True, True)` | **hybrid**：先 rank 内、再 scale-up 组内归约 | `topk_idx // E_per_rank` → `topk_idx // (E_per_rank·scaleup)` |
| `(True, False)` | **non-hybrid**：只 rank 内归约 | 同上第一级 |
| `(False, False)` | 等价于 `allow_multiple_reduction=False` | 无 |

`:237-240` 就是这两步。**`bias` 的处理**（`:241`）：
`bias_sum = bias[0]+bias[1]`（tuple）或 `bias`（单张量），shape 需可广播到 `[T,H]`。
docstring 特别警告：**即使 `num_scaleout_ranks == 1`，`reduce_in_scaleup=True/False` 也不等价**
——因为 bias 是最后统一加一次，归约层级会改变中间舍入边界。实测坐实了这一点（见 §7）。

---

## 5. `gate.py`：造出「指定不均衡比」的流量

四个函数构成一条链：

* `generate_rank_count(num_tokens, num_experts, num_ranks, num_topk, ratio)`（`:32-113`）：
  造 `[num_tokens, num_ranks]` 的「每 token 在每个 rank 上命中几个 top-k 专家」矩阵，使
  **rank 0 是特殊 rank**、流量约为其余的 `ratio` 倍。要点：
  - 先随机取 top-k 专家、算每 token 命中的**不同 rank 数** `a`（`:46-51`），再 clamp 到上界
    `upper_bound_per_token = int((num_ranks-1)/ratio)+1`（`:54-57`）——这是该 **approximate** 算法的
    能力上限；
  - `special_token_count`（`:60-63`）决定 rank 0 要多接多少 token；`must_mask = (a == num_ranks)`（`:66`）
    是「top-k 覆盖了全部 rank、rank 0 必然可选」的 token，必须全部计入（`:67-69` 的 assert）；
  - 用 `rank_perm`（`:78`，值域 `1..num_ranks-1`）+ 特殊位置留 `0`（`:85-91`）拼接，最后
    `scatter_add_` 出 `rank_count`（`:112`）。
* `generate_topk_idx(rank_count, ...)`（`:4-30`）：把「每 token 每 rank 命中数」还原成具体
  `topk_idx`；开着 `assert torch.equal(rank_count.sum(1), num_topk)`（`:7-8`）。
* `get_precise_unbalanced_scores(...)`（`:116-137`）：造 scores，使 **top-k 恰好**等于
  `generate_rank_count` 给出的分布（低分布 `U(0,0.9)`、选中项高分布 `U(0.9,1)`，`:127-136`）。
  这是「**精确**指定 ratio」。
* `get_scores_by_factor`（`:140-145`）/ `map_unbalanced_ratio_to_factor`（`:148-164`）/
  `get_random_unbalanced_scores`（`:167-173`）/ `get_unbalanced_scores`（`:176-180`）：后两种是
  **随机**近似比例——二分一个 factor，让 rank 0 的覆盖数达到其余均值的 `ratio` 倍。

判据（`tests/utils/test_gate.py:39-40`）：`practical_ratio = count[0] / max(count[1:].min(), 1)`、
`inequality = count[1:].max() / max(count[1:].min(), 1)`；**只在 precise 模式**校验
`|practical-ratio|/ratio < 0.1` 且 `inequality < 1.02`（`test_gate.py:48-52`）。

---

## 6. `testing.py`：计时与工具

与 DeepGEMM `testing/bench.py` **同名不同实现**（DeepEP 版多 numpy、多 distributed、多 tempfile）：

| 函数 | 行号 | 要点 |
|---|---|---|
| `flush_l2_cache(enabled)` | `:14-23` | 写 256 MB 零张量刷 L2 |
| `bench(fn, ...)` | `:26-62` | event 计时；`num_warmups=30/num_tests=30`；每轮可 `flush_l2`；返回 `(avg,min,max)`，**丢掉第 0 个样本**（`:61`） |
| `bench_kineto(fn, kernel_names, ...)` | `:113-221` | profiler `schedule(wait=0,warmup=1,active=1,repeat=1)`（`:156`）、两批取第二批、按 kernel 名子串匹配 + `ms/us` 单位换算（`:190-204`）、`num_kernels_per_period>1` 时导出 chrome trace 再按周期聚合（`:207-218`）；`EP_USE_NVIDIA_TOOLS=1` 直接旁路返回 `1`（`:145-146`）；`barrier_comm_profiling` 用 `cuda._sleep(2e7)` + `all_reduce` 削 CPU 发射抖动（`:166-173`） |
| `empty_suppress` / `suppress_stdout_stderr` | `:65-110` | 后者用 `dup2` 把 stdout/stderr 接到 `/dev/null`，退出恢复 |
| `parse_num_bytes(text)` | `:224-239` | `1G/64M/512k/1tb/2 GiB` → 字节；正则 `([0-9.]+)\s*([kmgt])?(?:i?b)?`；非正数/不可解析报 `ValueError` |

一个易忽略点：`bench` 里 `times = ...[1:]`（`:61`）——**明丢弃第 0 个测量**（首轮常含冷启动）。

---

## 7. H100 实测证据

脚本 `harness/exp_deepep_refs.py` + `harness/exp_deepep_refs.sh`（输出 `harness/exp_deepep_refs.out.txt`）。
**真实源码路径加载**（`sys.modules` 注入合成包 + stub `envs`，按文件路径 exec），打印 sha256：

```
refs.py    sha256=c5f551b49e55b448…   gate.py sha256=5372e45604c53b0b…
testing.py sha256=9a6f140744f53dc8…   math.py sha256=ed0dfb4bf9893769…
DeepEP 源码：93eb6eb
```

**Part B —— `refs.dispatch`（8×H100 NCCL all_to_all，T=512、E=32、k=4）** 与**独立 oracle**
（不调用 refs 任何代码，纯 Python 按 `(src_rank, local_idx)` 升序重建应到行）逐项对拍：

```
本 rank 收到 1753 行；每来源 rank 计数 = [207,225,206,232,210,222,230,221]
recv_src 前 8 个全局索引 = [0,4,10,11,13,21,29,30]   → 确认升序
[PASS] recv_src 升序 / recv_src==oracle / recv_topk_idx==oracle
[PASS] recv_x==oracle（bf16 逐位）/ num_recv_tokens_per_rank==oracle / 越界专家置 -1
```

即：**参考实现的四大不变量全部被独立复现**（排序、本地专家号、-1 掩码、计数分解）。

**Part C —— `refs.combine`（T=64、k=4、scaleout×scaleup=2×4、E=32）** 与独立 oracle 逐位相等：

```
[PASS] hybrid(T,T) / non-hybrid(T,F) / multi-red-disabled(F,F) / +bias / +bias_pair 全部逐位相等
scaleout=1 下 (T,T) vs (T,F) 不同 = True        （坐实 docstring 的 bias 效应）
[PASS] (F,T) 非法组合被 assert 拒绝
```

**Part D —— `gate.py`（判据镜像 `tests/utils/test_gate.py`，T=4096）**：

```
generate_topk_idx 反推 rank_count 逐元素一致；非法 rank_count（和 != k）被 assert 拒绝
precise 主扫 E/r∈{4,8} × R∈{8,16} × k∈{1,2,4,6} × ratio∈{1,2,4}：
  全部 practical_ratio 与目标差 <10%、inequality <1.02（例 R=8 k=2 ratio=2.0 → 2.000/1.000）
→ Part D: 98/98 PASS
```

**Part E —— `testing.py`**：

```
parse_num_bytes 6 例全对（1G/64M/512k/1048576/2 GiB/1tb）、4 个非法输入被拒
bench(2048² @, no flush) avg/min/max = 51.4/50.9/52.0 us
bench(flush_l2)          avg/min/max = 302.3/50.9/4814.6 us   ← 见下注
bench_kineto("gemm") = 47.7 us；EP_USE_NVIDIA_TOOLS=1 → 旁路返回 (1,1)
suppress_stdout_stderr 进入换 stdout / 退出恢复
→ Part E: 5/5 PASS
```

> **注**：`flush_l2` 的 avg 被一个 4814 us 的离群点拉高——本机被多个自驱任务共享，偶发抢占；
> 滤波器本身的 min 仍是 50.9 us。这与第 30 篇在共享 GPU 上观察到的抖动一致，**不是 `bench` 的缺陷**。

**结论**：`refs.py` 的 dispatch/combine、`gate.py` 的精确不均衡生成器、`testing.py` 的计时/工具
在 H100 上**真实可跑、可独立对拍**；它们是前面所有 DeepEP kernel 篇目的「语义底座」。唯一被本机
环境挡住的仍是 **`import deep_ep` 的 `check_nccl_so()`**（第 0 篇「阻塞」），本篇用注入合成包绕过，
从而在不依赖 DeepEP C++/NCCL 设备 API 的前提下验证了纯 host 语义层。

---

## 8. 边界与下一步

* 本篇只覆盖**架构无关的参考/生成/工具层**；真正的 dispatch/combine kernel（`impls/ep/*.cuh`）
  仍需 §「阻塞」里的 NCCL ≥ 2.32.3 环境才能在 H100 上端到端实测。
* `gate.py` 的 **approximate 模式**（`ratio` 非 1 且 `precise=False`）本就是随机近似，DeepEP
  测试也不校验其 ratio；本篇只在 precise 模式做硬校验，符合上游 `test_gate.py` 的约定。
* 可继续补的纯 host 语义层：`deep_ep/utils/refs.py` 的 **FP8 分支**（`sf` 的 per-128 传递）
  与 `combine` 的 **FP8 输出**变体（若上游新增）；以及 `deep_ep/utils/envs.py` 里
  `get_theoretical_num_sms` 之外的带宽/域解析（第 64 篇已碰过一部分）。

**参考链接**：DeepEP 仓库 <https://github.com/deepseek-ai/DeepEP>（本机核对 commit `93eb6eb`）；
参考文件 `deep_ep/utils/{refs,gate,math,testing}.py` 与 `tests/utils/test_gate.py`。
