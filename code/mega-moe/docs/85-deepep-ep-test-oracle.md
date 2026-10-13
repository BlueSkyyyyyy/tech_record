# 85 · DeepEP 的 EP dispatch / combine 端到端验收测试即正确性规范

> 目标源码：`~/github/DeepEP`（commit `93eb6eb`，v2.5.0）`tests/ep/test_ep.py`（616 行）；
> 依赖 `deep_ep/buffers/ep.py`（`EPHandle` 字段契约）、`deep_ep/utils/{refs,gate,math,testing}.py`。
> 本篇把这份旗舰验收测试当**「EP dispatch / combine 算什么对」的可执行规范**来拆。
> **端到端 buffer 路径依赖 DeepEP 的 C++/NCCL 设备 API（`ncclDevComm`/`ncclGin`），本机 NCCL 2.29.7
> 与编译期不匹配、运行还需 ≥2.31 → 端到端仅分析**；**测试侧的纯 Python / 张量逻辑架构无关，
> H100(sm90) 真机原样抽取运行、独立复刻对拍（17 项全 PASS，见 §10）**。

## 1. 这份测试在专题里的位置

专题拆过的测试栈：DeepGEMM 侧有计时/判对基座（第 40 篇）、操作数生成器（第 41 篇）、逐位断言
（第 43 篇），以及四个 mega 算子（Mega MoE 第 51、Mega Gate 第 57、MQA logits 第 60、Mega mHC 第 67）
与稠密 GEMM / einsum / SF layout / prenorm / 运行时 / 局部域 / legacy 七个顶层规格文件
（第 74/76/78/79/80/81/82 篇）。DeepEP 侧则拆过参考实现 `refs.py`／流量生成器 `gate.py`／基座
`testing.py`（第 65 篇）与 `EPHandle` host 编排（第 83 篇），但**把 dispatch/combine 端到端“算什么对”
定义清楚的那份测试 `tests/ep/test_ep.py` 本身从没被单独拆过**。它是第 7/11/48/51/52/61/63/71/83 篇里
反复引用的「逐位一致」「expand 不变量」「两套 psum」判据的出处，也是「dispatch/combine 的
SM/QP、overlap、带宽数字」的运行时来源。

一份测试如何同时是「正确性规范」与「性能口径」？它用**一次 launch 比两套实现**：
被测的是 `deep_ep.EPBuffer.dispatch/combine`（单 kernel，见第 7/71 篇），参照物是 P55 拆过的纯 PyTorch
`refs.dispatch`/`refs.combine`，判据是 **`torch.equal` 逐位**（不是 `calc_diff` 近似）——因为
「通信」只是一次数据搬移 + 确定性归约，任何数值差都是 bug。

## 2. 顶层结构

```
tests/ep/test_ep.py
├── imports (:1-18)        deep_ep, deep_ep.utils.{math,gate,envs,refs,testing}
├── enumerate_ep_modes ():22-30    6 维模式矩阵（生成器）
├── launch ():33-43              按 (previous_event, defer_epilogue, async) 调 buffer 方法
├── fold_expanded ():46-58       expand 折叠不变量
├── test_dispatch_combine ():61-523   主线：构造路由 → 五种 dispatch + 两种 combine → 逐位判据 + 性能
├── test_loop ():527-567         起 buffer → test_dispatch_combine → 压力测试（多 seed）
└── __main__ ():570-616          CLI（模型/场景/测试三类参数）→ mp.spawn(test_loop)
```

入口由 `torch.multiprocessing.spawn`(:616) 起 `num_processes`（默认 8）个进程，`test_loop`(:528)
在 `init_dist` 建组后构造 `deep_ep.EPBuffer`(:531-540)，把 `test_dispatch_combine` 跑一遍，可选
`do_pressure_test`(:554) 用不同 seed、每个 seed 可重建 buffer 再跑。

## 3. 模式枚举矩阵 `enumerate_ep_modes`（:22-30）

六层嵌套 `for`，逐维枚举「一次 dispatch/combine 会遇到的执行形态」：

| 维度 | 取值 | 触发的机制（篇目） |
|---|---|---|
| `expert_alignment` | `128, 1` | 每本地专家接收数是否对齐到 128（§6 的 psum 对齐） |
| `use_fp8_dispatch` | `1`（或 `1,0`） | 激活走 per-token FP8（`(x, sf)` 对）还是 bf16 |
| `num_bias` | `0, 1, 2` | combine 的 bias 个数（0/1/2；2 是 `(scaleup, scaleout)` 对） |
| `with_previous_event` | `0, 1` | 是否传 `previous_event=buffer.capture()`（通信流排序） |
| `async_with_compute_stream` | `1`（或 `0,1`） | 通信是否与计算流异步（第 75 篇 `EventOverlap`） |
| `allocate_on_comm_stream` | 受 `with_previous_event` 约束 | 中间张量是否记在通信流账上 |

`--fp8-dispatch-only` 把第二维钉成 1、`--defer-epilogue` 把第五维钉成 1；第六维在
`with_previous_event=1` 时只有一个取值（前面已有事件，分配只能在通信流）。**因此组合数是
`2 [align] × fp8 × 3 [bias] × Σ_{prev∈{0,1}} (async × alloc(prev))`**：`prev=1` 时 alloc=1、
`prev=0` 时 alloc=2，故对 `prev` 求和为 `async·(1+2)=3·async`。实测（§10 Part B）：
`(fp8,defer)=(0,0)→72`、`(0,1)→36`、`(1,0)→36`、`(1,1)→18`，与独立展开公式逐例吻合。

## 4. 五种 dispatch + 两种 combine

主线在同一个循环里对每套 `x`/`topk_idx`/`topk_weights` 连发**多次** dispatch、combine：

**① 普通 dispatch**（:146-155）。`dispatch_args` 含 `expert_alignment`、`num_sms`/`num_qps`
（`:67-68`，为 0 时用第 63 篇的 `get_theoretical_num_sms/num_qps` 现算）、以及三枚与流/事件有关的开关。

**② expand dispatch**（:159-162）。`dispatch_args | dict(do_expand=True, use_tma_aligned_col_major_sf=True)`。
expand 把每个 token 按 top-k 复制到**每个专家的 slot**（`[num_recv_tokens, topk, hidden]` 语义），
`use_tma_aligned_col_major_sf` 让 SF 走 TMA 对齐的列主序。

**③ cached dispatch**（:165-172）。只传 `x` 与 `handle=handle`（上一轮句柄），**复用整套布局计算**——这正是第 83 篇
讲的「handle 反哺 dispatch」。测试随后断言 `recv_x`、`recv_topk_idx`、`dst_buffer_slot_idx`、
两套 psum 与 `num_recv_tokens_per_expert_list` 都与非 cached 版**逐位一致**（:442-450）。

**④ cached expand + zero padding**（:175-179）。`do_expand=True, do_zero_padding=True, handle=expanded_handle`；
零填充把每个本地专家块内 `[真实计数, align(真实计数, alignment))` 的 padding 行显式清零，
测试逐专家验证（§6）。

**⑤ FP8 激活**。`use_fp8_dispatch` 时 `x = per_token_cast_to_fp8(x)`（:97），接收侧用
`per_token_cast_back`（:103-107）还原 bf16 再做逐位比较。

**combine 两种**：普通 combine（:211-219，`x=input_for_combine`）与 **reduced combine**（:221-233，
`x=input_for_expand_combine`、`handle=expanded_handle`）。`--allow-multiple-reduction=False` 时
reduced combine 不传 `topk_weights`（:229-231 注释：expand 模式要 `allow_multiple_reduction=True`
才支持 topk_weights）。归约配方 `(reduce_in_local, reduce_in_scaleup)`（:116-126）由
`allow_multiple_reduction` 与 `allow_hybrid_mode ∧ num_scaleout_ranks>1` 决定，与第 65 篇
`refs.combine` 的三配置（hybrid `(T,T)` / non-hybrid `(T,F)` / disabled `(F,F)`）一一对应。

## 5. 正确性判据：逐位 + refs oracle

参照物在 `if not args.skip_check` 分支里构造（:110-142）：`ref_dispatch` 给 `ref_recv_x/topk_idx/
topk_weights/src_token_idx/num_recv_tokens_per_rank`，`ref_combine` 用两套 recipe 给
`ref_combined_y`/`ref_reduced_combined_y`（:131-142）。

真正比较前，先坐实**「按 src 全局索引升序」**这条坐标约定（`refs.dispatch` docstring 声明、
P55 已验证）：

```
src_token_global_idx = handle.recv_src_metadata[:num_recv_tokens, 0]          # :188
sorted_src = torch.sort(src_token_global_idx).values                          # :190
assert torch.equal(ref_recv_src_token_idx, sorted_src)                        # :191
```

kernel 的接收行序由对称缓冲里的 `atomicAdd` 分配（第 52 篇），本身**不保证有序**；测试把 kernel 的
元数据**排序后**再和 ref（按构造就有序）比，这一步排除了「行序不确定」这一自由度，只留下数据本身。
随后对每个来源 rank 切一段（:483-507）：用 `sorted_metadata.indices` 把 kernel 输出重排进 ref 的坐标，
再 `torch.equal(ref_t, t)` 逐位比 —— `topk_weights` 还额外把 `ref_mask = ref_recv_topk_idx < 0`
的行屏蔽（:500-506），因为越界专家（不属于本 rank）的权重不参与 combine。

combine 的判据（:509-518）同样逐位：

```
assert torch.equal(combined_x, ref_combined_y)                    # :510
assert torch.equal(reduced_combined_x, ref_reduced_combined_y)    # :512
assert torch.equal(combined_topk_weights, topk_weights)           # :514
```

`combined_topk_weights == topk_weights` 是一条**结构性**断言：combine 只做加权重排、不改权重本身。

## 6. 三条不变量

### 6.1 expand 折叠不变量 `fold_expanded`（:46-58）

expand 布局把同一 token 复制到它的 k 个专家槽——**于是同一 token 的所有有效槽内容必须相同**。
`fold_expanded`（:47-56）正是把这一点当成可执行断言：`gathered = expanded[indices]`、
`first_valid_idx = valid_mask.to(int).argmax(dim=1)` 取每 token 第一个有效槽、
`folded = gathered[arange, first_valid_idx]`，然后
`assert ((gathered == folded.unsqueeze(1)) | ~valid_mask).all()`。对 `(x, sf)` 元组递归（:48-49）。
它同时完成两件事：验证 expand 的重复不变量、把 expand 布局折叠回「每 token 一行」供 combine 用（:438-439）。

### 6.2 handle 的两套 psum 语义（:452-476）

`EPHandle`（`ep.py:27-105`）里两种前缀和**语义不同**（第 83 篇已点明，这里给出测试的判据）：

- **非 expand**：`psum_num_recv_tokens_per_expert` 是**每专家对齐后计数**的含端点前缀和，故
  `psum[i+1]-psum[i] == align(真实计数_i, expert_alignment)`，且
  `num_recv_tokens_per_expert_list[i] == align(真实计数_i, alignment)`（:460、:468）。
- **expand**：`psum[i] == align(psum[i-1], alignment) + 真实计数_i`（未对齐），故
  **`psum[i+1] - align(psum[i], alignment) == 真实计数_i`**（:466-469）。

测试在 :461-469 同时用两套 psum 反推真实计数并与 `ref_recv_topk_idx == i` 的计数对齐，等价于
**两条恒等式的一次性验证**。§10 Part D 用合成计数向量独立复刻这两条恒等式（含它们共享的
「按对齐累加的起点序列」`align(expand_psum[i]) == 非expand_psum[i]`）。

scaleup 前缀和（:471-476）则是**去重**语义：`psum_num_recv_tokens_per_scaleup_rank[i+1] -
psum[i] == Σ_{r ≡ i (mod num_scaleup_ranks)} ref_num_recv_tokens_per_rank[r]`（一个 token 即使多个
top-k 专家落在同一 rank 也只计一次），末元素=收到 token 总数。

### 6.3 零填充与确定性双跑

零填充（:431-435）：对每个本地专家，`[start, end) = [psum[i], align(psum[i], alignment))` 的 padding
行，`recv_x`/`recv_topk_weights` 都必须是 0。`--deterministic` 时（:394-411）dispatch 连做两次，
逐位比较 `recv_x`/`recv_topk_idx`/`recv_topk_weights`/`recv_src_metadata[:, :1]`，验证第 52 篇的
确定性排序把物理行序规范回源 token 索引序（`expert_alignment==1` 时才比 expand 的 topk_weights，
因为 padding 行权重未初始化，:410-411）。

## 7. `EPHandle` 的字段契约

测试直接消费 handle 的这些字段，等价于一纸对外契约（docstring 在 `ep.py:27-60`）：

- `handle.topk_idx is topk_idx`（:371-373）：handle **持有原路由张量本身**（不是拷贝），且
  `cached_handle.topk_idx.data_ptr()` 与之相同——这既是「handle 冻结路由」的保证，也是第 83 篇
  `topk_idx` **版本守卫**的意义所在（前向期间原地改 `topk_idx` 会让 handle 语义崩坏）。
- `handle.dst_buffer_slot_idx`、`psum_num_recv_tokens_per_scaleup_rank`、
  `num_recv_tokens_per_expert_list`（:448-450）：cached 与非 cached 必须逐位一致。
- `handle.psum_num_recv_tokens_per_expert`：expand 模式下的 PSUM 布局，也是 `~/github/DeepGEMM/
  tests/test_mega_moe.py:307-333` 直接喂给 grouped GEMM 的 PSUM 布局（第 83 篇）。
- `recv_src_metadata[:, :2]` 是 `(src_token_global_idx, ...)`；expand 模式下 `[:, 2:]` 是
  各 top-k 槽在 expand 缓冲区里的行号（:390-392，`>=0` 为有效）——`input_for_expand_combine` 的散射
  就靠它（:206-208）。
- `cumulative_local_expert_recv_stats`（:413-416）：一个 int32 `[num_local_experts]` 输出计数器，
  dispatch 时按本地专家累计接收数；测试断言它等于 `(ref_recv_topk_idx == i).sum()`（:458-459）。

## 8. 带宽账目 `get_combine_bytes`（:298-343）

性能口径同样是「可执行规范」——它给每种 combine 配方定义「要过多少字节」。核心 helper
`get_unique_and_valid_dst_count`（:302-315）统计**去重后的有效目的数**：先把不同行错开
（`dst_idx + arange(T)[:,None]*(max_num+1)`，使行间不碰撞），再把「忽略项」（`-1`，以及
`ignore_local_traffic` 时区间 `[ignored_l, ignored_r)` 内）统一折成同一个哨兵值 `dst_idx[0][0]`
并 `torch.unique` 计数。于是三种配方的 combine 字节为（:317-343）：

- **无 multi-reduction**（:317-332）：非 expand 用 `topk_idx // num_experts_per_rank`（按 rank 去重）、
  expand 用每来源 rank 段的未屏蔽计数；scaleout 用 `topk_idx // num_experts_per_scaleout_rank`。
- **有 multi-reduction**（:333-340）：combine 的 token 数与 dispatch **完全相同**——scaleout 用
  `num_scaleout_send_tokens`、scaleup 用 `num_scaleup_recv_tokens`、reduction 读用
  `topk_idx // num_experts_per_scaleout_rank`（hybrid）或 `// num_experts_per_rank`（非 hybrid）。
- 单 scaleout（`num_scaleout_ranks==1`，:341-342）时 scaleout 字节视为 0。

这些计数再乘 `num_bytes_per_combine_token = count_bytes(recv_x_bf16, recv_topk_weights) / 行数`
（:294）就是 SO/SU/归约三类带宽的分子；dispatch 侧同理用 `num_bytes_per_dispatch_token`（:260）。
`bench_kineto` 按 kernel 名 `('dispatch_impl','dispatch_copy_epilogue_impl')` /
`('combine_impl','combine_reduce_epilogue_impl')` 拆出两段时长（:263-265、:346-348），与第 61/48 篇
的两个 epilogue kernel 一一对应。

## 9. 生命周期与 CLI

`test_loop`（:527-567）：每轮 `init_dist` → `construct_buffer`（把 CLI 参数映射到 `EPBuffer` 构造，
含 `deterministic`/`allow_hybrid_mode`/`allow_multiple_reduction`/`prefer_overlap_with_compute`/
`num_allocated_qps`/`explicitly_destroy=True`/两个 timeout）→ `test_dispatch_combine` → 循环
`do_pressure_test`（默认 `int(1e9)` 即不跑），每 seed 可 `reuse_buffer` 或 `buffer.destroy()` 重建。
末尾 `buffer.destroy()` + `dist.destroy_process_group()`。CLI（:570-608）分三类参数：资源
（`num_processes`/`num_sms`/`num_qps`/`num_allocated_qps`/两个 timeout）、模型
（`num_tokens=4096`/`hidden=7168`/`num_topk=6`/`num_experts=256`）、场景
（`do_cpu_sync`/`defer_epilogue`/`allow_hybrid_mode`/`allow_multiple_reduction`/
`prefer_overlap_with_compute`/`deterministic`）与测试（`skip_check`/`skip_perf_test`/
`masked_ratio`/`unbalanced_ratio`/`precise_unbalanced_ratio`/`fp8_dispatch_only`/…）。
`--masked-ratio` 会随机把 `topk_idx` 置 `-1` 并把对应权重置 0（:80-83）——这正是 §5/§6 里
`ref_mask`、`-1` 处理的输入端。

## 10. H100(sm90) 真机实测

实验：`harness/exp_deepep_ep_test.{py,sh}` → `harness/exp_deepep_ep_test.out.txt`。
做法：`sys.modules` 注入合成 `deep_ep`/`deep_ep.utils` 包绕过 `deep_ep/__init__.py:55 check_nccl_so()`，
按**文件路径** exec 真实 `refs.py`/`gate.py`/`math.py`（打印 sha256），并用 `ast` 从
`tests/ep/test_ep.py` 抽取顶层函数 `enumerate_ep_modes`/`launch`/`fold_expanded` **原样运行**。
环境：NVIDIA H100 80GB HBM3 / sm90 / driver 570.211.01，torch 2.12，NCCL 2.29.7，DeepEP `93eb6eb`，
`tests/ep/test_ep.py` sha256 `7346d98a17bce91e…`。

结果（17 项全 PASS）：

| Part | 内容 | 结果 |
|---|---|---|
| A | 真实源码加载/抽取（sha256） | 抽取到 3 个顶层函数 |
| B | 模式矩阵 vs 独立展开 | 72/36/36/18 全对（4/4） |
| C | `fold_expanded` 不变量 + 违反被 assert | 折叠正确、tuple 递归、反例被抓（3/3） |
| D | 两套 psum 恒等式 + scaleup 去重 | 4/4 |
| E | `get_unique_and_valid_dst_count` 去重计数 | 2/2 |
| F | 8×H100 真机 `refs.dispatch`「按 src 升序」 | 4/4 |

关键数字：

- Part B 首组元组 `(128, 1, 0, 0, 0, 0)`，与枚举顺序（`expert_alignment` 最外、`allocate_on_comm_stream`
  最内）一致。
- Part D（`num_local_experts=7, alignment=3`）四条恒等式成立；scaleup 例末元素=总数 251。
  `align(expand_psum[i],3) == 非expand_psum[i]` 逐步成立——两种 psum 共享同一「按对齐累加的起点序列」。
- Part E：`get_unique_and_valid_dst_count` 与独立 oracle 在「行内去重 + 忽略 `-1`」和
  「忽略区间 `[3,6)`」两例都相等（18==18、13==13）；这也**坐实了被忽略项被折成一个哨兵值只计一次**
  这一容易看漏的实现细节。
- Part F（8×H100 NCCL all_to_all，`num_tokens=512, hidden=256, E=32, k=4`）：本 rank 收到 1753 行、
  每来源 rank 计数 `[207,225,206,232,210,222,230,221]`；`recv_src` 本身已升序、等于其排序结果，
  且接收的专家索引都落在本 rank 的本地专家范围内或 `-1`——正是测试 `:191` 断言的根。

## 11. 「为什么跑不了」与结论

`tests/ep/test_ep.py` 的**端到端路径**需要一个 `deep_ep.EPBuffer`，其 dispatch/combine 走
DeepEP 的 C++/NCCL 设备 API（`ncclDevComm`/`ncclGin`，第 71 篇的 hybrid 亦然）；本机
`import deep_ep` 在 `deep_ep/__init__.py:55 check_nccl_so()` 处直接断言失败（loaded vs linked NCCL
不一致），且 `ncclGin` 需 NCCL ≥2.31（本机 2.29.7）。因此**端到端 buffer 测试与它跑出的
TFLOPS/GB/s 数字为「NCCL 设备 API 专属，仅分析」**（与 ROADMAP「阻塞」一致）。

但这份测试的**规范内核**——模式矩阵、expand 折叠不变量、两套 psum 恒等式、带宽账目的去重计数、
以及「按 src 升序」的 refs oracle——全部是纯 Python/张量逻辑，H100 上原样抽取运行、独立复刻对拍
通过。它补上了专题一直在 DeepEP 侧缺的一块：**「EP dispatch/combine 算什么对」的可执行定义**，
也把前面各篇零散引用的判据收敛到一份文件。
