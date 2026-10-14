# Mega MoE dispatch 的「round-robin min-peeling」认领协议：无锁唯一认领、与它到底在交错什么

> 本文对应路线图 **P184-1**，是「DeepSeek Mega MoE 前向 / 反向算子实现解析」的第 194 篇。
>
> 第 11 篇（P3-4）把 dispatch 段的 pull 主循环逐段拆过，其中一句话带过了核心映射：
> **`round-robin min-peeling` 把「本 rank 的第 `token_idx_in_expert` 个待拉 token」映射到
> 「某个源 rank 的哪个 `token_idx_in_rank`」**（`impls/sm100_fp8_fp4_mega_moe.cuh:480-528`）。
> 第 120 篇（P110-1）拆了**另一层**（GEMM 任务调度器的分域 + 工作窃取），第 180 篇（P170-1）
> 把「端到端不可复现的真凶」落到 DeepEP **训练侧** dispatch 的 `atomicAdd` 槽位分配上。
>
> 但这段 min-peeling 本身**从没被单独验证过**——它是不是一次**无锁唯一认领**（每个远端槽位
> 恰好被拉一次）、在**偏斜/零计数**下还成不成立、它到底在「交错」什么？本篇补上。
>
> **可跑性**：这段算法是**纯整数 + warp intrinsics**（`__reduce_add_sync`/`__reduce_min_sync`/
> `__ballot_sync`/`__fns`），**不碰 UMMA/TMA/tcgen05/NVLink**，故虽在 SM100 主 kernel 里，
> 却与架构无关——在 H100(sm90a) 上**直接编译运行真实设备代码**。主 kernel 本体 SM100 专属、仅分析。
>
> 原始输出：`harness/exp_mega_moe_dispatch_claim.out.txt`
> （`src/mega_moe_dispatch_claim.cu` + `harness/exp_mega_moe_dispatch_claim.{sh,py}`）。

参考坐标：DeepGEMM `057ca59`（Public release 26/09/30，v2.8.1）；宿主 CUDA 13.2 / H100 80GB HBM3。
目标卡代号敏感：全文用「目标卡」指代。

---

## 1. 这段算法在哪、为什么与架构无关

`sm100_fp8_fp4_mega_moe.cuh` 的 dispatch 段（`:430-689`）里，每个 dispatch warp 做的是：

```
kNumGlobalWarps = kNumSMs * kNumDispatchWarps                     // :448
for (token_idx = sm_idx*kNumDispatchWarps + warp_idx; ;          // :449  grid-stride
                 token_idx += kNumGlobalWarps) {
    ... 推进到 token_idx 所在专家 current_expert_idx ...
    从对称缓冲读该专家每个源 rank 的到达数 stored_rank_count[]  // :471-478（TODO: not coalesced）
    <=== round-robin min-peeling：把 token_idx_in_expert 映射到 (源 rank, token_idx_in_rank) ===>  // :480-528
    读源 topk 索引 src_token_topk_idx = *get_src_token_topk_idx_ptr(expert, rank, token_idx_in_rank) // :531
    src_token_idx = src_token_topk_idx / kNumTopk                    // :533
    src_topk_idx  = src_token_topk_idx % kNumTopk                    // :534
    ... 远端 pull token / weight / SF（:552-599）...
}
```

`kNumDispatchThreads=128`（`heuristics/mega_moe.hpp:220`）⇒ `kNumDispatchWarps=4`；H100 上
`kNumSMs=132` ⇒ `kNumGlobalWarps=528`。`kNumRanksPerLane = ceil_div(kNumRanks, 32)`（`:439`）：
每个 lane 持有 rank 集合 `{i*32 + lane}` 的计数（`j >= kNumRanks` 补 0，`:472-477`）。

这段**唯一的「并行」**就是 warp 内 32 lane 把「所有 rank 的计数」汇总成两个标量
（活跃 rank 数、最小剩余数），其余全是**标量整数**。没有内存序、没有跨 SM/跨 rank 通信、
没有张量核——所以把它从 SM100 kernel 里原样搬到 H100 上跑，测的是**真实语义**。

## 2. 语义解码：它一次算出一整条置换

把 `:480-528` 读成「每轮 peel」：

- 令 `remaining[r]` = 源 rank r 还没被消费的 token 数，`offset` = 已消费的「轮数×每轮长度」。
- 一轮：`active` = 所有 `remaining>0` 的 rank，`length = min_{r∈active} remaining[r]`（`:497-502`，
  用 `__reduce_min_sync` 跨 lane 取 min、`__reduce_add_sync` 数活跃 lane），
  `num_round_tokens = length * |active|`（`:505`）。
- 若目标 `slot_idx < num_round_tokens`：命中本轮的 `slot_idx_in_round = slot_idx % |active|`，
  **轮内按活跃 rank 升序 round-robin**（`:507-518`：`__ballot_sync` 出活跃 lane 掩码，
  `__fns` 找第 `slot_idx_in_round` 个活跃 rank），行号 `token_idx_in_rank = offset + slot_idx/|active|`。
- 否则进入下一轮：`slot_idx -= num_round_tokens; offset += length; remaining[r] -= min(remaining[r],length)`（`:522-527`）。

等价地：**它一次性枚举出一条 token 顺序**——第 k 轮把 `|active_k|` 个活跃 rank 各出
`length_k` 个 token，轮内交替排布：

```
token 顺序 = 轮 #0: r_a0,r_a1,… ; r_a0,r_a1,… (length_0 次)
           ⊕ 轮 #1: ...
```

对 `counts=[10,6,0,3,8,1,0,4]`（rank 2/6 空），min-peeling 的**实际 rank 序列**实测为（真机）：

```
0 1 3 4 5 7 0 1 3 4 7 0 1 3 4 7 0 1 4 7 0 1 4 0 1 4 0 4 0 4 0 0
轮次: #0(len=1,active=6) #1(len=2,active=5) #2(len=1,active=4) #3(len=2,active=3) #4(len=2,active=2) #5(len=2,active=1)
```

看得很清楚：**零计数的 rank（2、6）被自动跳过**（`active` 只收 `remaining>0`），每轮
`length=min` 先被「最穷」的活跃 rank 拉低，然后该 rank 出完就被剔出 `active`——这就是
「min-peeling」这个名字的来历（每轮削掉最小值那层）。这就是 dispatch pull 的**源坐标生成器**。

## 3. H100 真机：忠实复刻 + 双射/唯一认领

`src/mega_moe_dispatch_claim.cu` 把 `:439-528` 的 while 循环**逐行**搬进一个
`template<uint32_t kNumRanks>` 的 `claim_kernel`（保留 `__reduce_add/min_sync`、`__ballot_sync`、
`__fns`、`kNumRanksPerLane`），并用一个**独立的 host「聚合模型」oracle**（不模拟 lane/ballot，
只按 §2 的语义解析每个 token 的落点）逐位对拍。三个量：

1. **逐位**：kernel 输出 `(rank, token_idx_in_rank)` == oracle（对 total 个 token 全枚举）。
2. **双射**：枚举出的 `(rank,t)` 恰好是 `{(r,t): 0≤t<counts[r]}` 的一次排列 ⇒
   **每个远端槽位被唯一认领一次**（无重复 `dup`、无越界 `oob`、无遗漏 `uncovered`）。
   这就是「无锁」的依据：不需要任何原子/锁，仅凭每个 rank 的**计数**就得到唯一归属。
3. **频率**：经验 rank 频率精确等于 `counts`。

真机结果（H100 sm90a，`claim_kernel<<<528,32>>>`）：

| 用例 | total | kernel vs oracle | 双射 | 频率 |
|---|---|---|---|---|
| R=8 `[10,6,0,3,8,1,0,4]`（含零+偏斜） | 32 | mism=0 | dup=oob=unc=0 | 精确 |
| R=8 均匀 `[7]*8` | 56 | mism=0 | 0 | 精确 |
| R=8 单 rank `[100,0,…]` | 100 | mism=0 | 0 | 精确 |
| R=8 全 1 | 8 | mism=0 | 0 | 精确 |
| R=4 / R=16 / R=32 | 20/66/128 | mism=0 | 0 | 精确 |
| **R=64**（`kNumRanksPerLane=2`） | 258 | mism=0 | 0 | 精确 |
| **R=72**（`kNumRanksPerLane=3`，`kNumMaxRanks`） | 216 | mism=0 | 0 | 精确 |

再加 **200 组随机 sweep**（`R∈{8,16,32,64,72}`，一半均匀一半偏斜、随机插入零计数）：
**全部逐位+双射通过（bad=0）**。另有一个**完全独立**的 host/Python 复刻
（`harness/exp_mega_moe_dispatch_claim.py`，**不模拟 lane/ballot**，改用「逐轮构造整条置换」的
另一条代码路径），显式用例 + **500 组随机 sweep** 同样全过。

⇒ **min-peeling 是结构性正确的无锁唯一认领**，且对 `kNumRanksPerLane≥2`（跨 lane 分组、
rank 升序由 `i*32 + __fns` 保证）也成立。

## 4. 它到底在「交错」什么：并发窗口视角

一个自然的猜想是「round-robin = 让每个 SM 的 pull 摊到更多源 rank，避免 NVLink 串行」。
怎么量才对？**不是**「每个 warp 处理哪些 token」——因为 grid-stride（`:449`）已经把 token
本身摊得很散。物理相关的是**并发窗口**：第 k 次迭代时，528 个 dispatch warp 集体处理的是
**连续的 528 个 token** `[k·528, (k+1)·528)`；这批 pull 会**同时**打到源 rank 的对称缓冲上。

拿一个生产量级、含零的偏斜分布 `counts=[4096,3072,0,1024,5120,2048,256,1280]`
（每专家每 rank 的到达数，total=16896）对拍 min-peeling vs **朴素「按 counts 前缀连续切段」**
（token 0..counts[0]-1 全归 rank 0，……——最直白的实现）：

| 视角 | 指标 | min-peeling | 朴素连续切段 |
|---|---|---|---|
| 每 warp | 平均覆盖源 rank 数 | 5.5 | 6.5 |
| 每 warp | 平均最长同 rank 连跑 | 10.6 | 9.7 |
| **并发窗口** | **每窗口平均覆盖源 rank 数** | **4.5** | **1.2** |
| **并发窗口** | **平均最长同 rank 连跑** | **33.0** | **497.0** |
| **并发窗口** | **单 rank 窗口数** | **1 / 32** | **27 / 32** |

读法：

- **每 warp 视角几乎看不出差别**（5.5 vs 6.5）——grid-stride 把 token 打散后，任何映射在每个
  warp 上都会覆盖约一半的源 rank。这解释了为什么「每 warp 交错」是个**会误导**的指标。
- **并发窗口视角差别巨大**：朴素连续切段下 **27/32 个并发窗口是「单 rank」**（最长连跑 497），
  即整批 528 个 pull 全打到**同一个**远端 rank 上——**同一时刻只用一个源 rank 的链路出口**，
  其余 7 条 NVLink 空转；min-peeling 把每个窗口摊到 **4.5 个源 rank**（单 rank 窗口只剩 1/32），
  **让 8 条源链路并行饱和**。
- 这与第 191/192 篇的「每 rank NVLink 出口 ~8–16 SM 封顶」是同一件事的两面：**只要同一瞬间
  的 pull 不集中到一个源 rank，多条链路就能并行**；min-peeling 的 round-robin 正是这个保证。

## 5. 与「确定性」的关系：前向是确定性的，训练侧不是

值得对照第 180 篇（P170-1）：DeepEP **训练侧** dispatch 用 `atomicAdd` 抢槽位
（`dispatch.cuh:341`），槽位顺序**随机**，使 wgrad（收缩轴=token）逐位不可复现，必须靠
`deterministic_sort` 规范回序。

而 **Mega MoE 前向主 kernel 的这段 min-peeling 本身就是确定性的**：给定每个 rank 的到达数
（对称缓冲里由分布式前缀和算出、每轮固定），token→(源 rank, 行号) 的映射**唯一且与执行时序无关**。
实测 200+500 组用例的逐步逐位一致即为此。⇒ **前向 dispatch 的源顺序是结构性确定的，
前向不存在训练侧那种「靠 atomicAdd 决定 token 顺序」的不确定性**；第 180 篇的复现性风险
在**训练侧 batched dispatch**，不在这个融合前向里。这是「前向 fusion kernel 与训练 dispatch
是两套编排」的一个具体落点。

## 6. 边界与「仅分析」

- **仅分析（SM100 专属）**：min-peeling 之后的**远端 pull**（`:552-576` 的 `tma_load_1d` 跨 rank、
  `sym_buffer.map`、UTCCP SF 转置 `:584-599`）与 combine 段（`:1254-1520`）都要 UMMA/TMA/NVLink
  对称窗口，本机不跑。本篇只测它**前面的纯整数映射**。
- **真实设备代码**：`claim_kernel` 逐行复刻 `:439-528`，用真实 `__reduce_*_sync`/`__ballot_sync`/
  `__fns`，H100(sm90a) 真机运行（`PASS=629/FAIL=0`；Python 侧 `PASS=508/FAIL=0`）。
- **并发窗口**用 `kNumGlobalWarps=528`（H100 的 `kNumSMs=132 × 4`）；目标卡 SM 数不同则窗口宽度
  不同，但「min-peeling vs 连续切段」的定性结论（4.5 vs 1.2 个源 rank/窗口）不随 SM 数改变。
- 本实验量的是**映射结构**（谁从哪拉），不是**实际 NVLink 吞吐**——后者见第 191/192 篇。

## 7. 复现

```
bash code/mega-moe/harness/exp_mega_moe_dispatch_claim.sh \
  > code/mega-moe/harness/exp_mega_moe_dispatch_claim.out.txt
```

- `src/mega_moe_dispatch_claim.cu`：真实 `claim_kernel<kNumRanks>`（逐行复刻 `:439-528`）+
  独立 host oracle + 双射/频率检查 + 200 组随机 sweep + 并发窗口交错统计。`nvcc -arch=sm_90a`。
- `harness/exp_mega_moe_dispatch_claim.py`：**独立** Python 复刻（逐轮建整条置换 + 逐 token
  while oracle 两条路径互相印证）+ 500 组 sweep + 同样的交错统计。
