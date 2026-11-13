# （一）UniEP 的 Dispatch + FC1 GroupGEMM：把等待粒度压到一个 expert

**把「等整层 dispatch 完成」换成「等这个 expert 的 token 齐」，代价是一块 SMEM 要留给通信做中转**。

读完你会知道：
- 为什么 FC1 不必等整轮 all-to-all；
- 8 个 warp 怎么在同一批 SM 上各干各的；
- 这个粒度在什么形状下反而更慢。

副本 [`uniep/kernels/cutedsl_dispatch_group_gemm.py`](../uniep/kernels/cutedsl_dispatch_group_gemm.py)（上游 1314 行 → 注释副本 1879 行，其中 1222 行代码，Blackwell sm100 版）。行号会漂，用「搜 `xxx`」定位。

## 1. 等整层，GEMM 就空转一整轮

MoE 一层前向：

```text
gate → dispatch（按 expert 发到 owner rank）→ GroupGEMM(FC1) → SwiGLU → GroupGEMM(FC2) → combine
```

传统实现里 dispatch 和 GroupGEMM 是两个独立 kernel，中间是硬边界：**必须等 dispatch 全部完成，GEMM 才能启动**。哪怕第一个 expert 的数据早就到了。

GEMM 开第一块 tile 只要一个 expert 的 A 行（一块 256 对齐的 slab）；整层却要等 group_count 个 expert 全部落齐——中间全是空转。

## 2. 融成一个 kernel，等谁由 signal 说了算

UniEP 把 dispatch + FC1 融成一个 persistent kernel：一部分 warp 拉远端 token 行进本地 A buffer，另一部分对已经齐了的 expert 直接开算，两者之间只靠「每个 expert 一个 signal」沟通。

【复用】dispatch 的落点就是 GEMM 的输入 buffer，SMEM ring 只是让 TMA 合法的中转。

signal 单向：dispatch 只置 1，GEMM 只读，每次 launch 前 host 清零。

8 个 warp / 256 线程 / 2 个 warp group，角色硬编码（搜 `dispatch_warp_id_base`）：

```text
对端 A 行 --G2S--> SMEM ring --S2G--> 本地 A[slab] --signal[e]--> GEMM tile

warp 0-3  epilogue  TMEM→RMEM→SMEM→全局 C       资源：epi stage
warp 4    dispatch 生产  G2S：对端 buffer→ring    资源：ring 空位 mbarrier
warp 5    dispatch 消费  S2G：ring→本地 A；发信号  资源：per-expert 计数器
warp 6    MMA  tcgen05 UMMA 主循环              资源：A/B stage、TMEM
warp 7    TMA  A/B：GMEM→SMEM；换 group 时等 signal  资源：A/B stage
```

拆两个 warp 是因为 G2S 与 S2G 方向相反，中间必须有环：**TMA 不支持 GMEM→GMEM 直搬**。拆开后一个发 G2S，一个等上一行，通信内部也有流水。环深 `dispatch_num_stages`（1–5）是在途行数上限，也是 dispatch 领先 GEMM 的上限。

## 3. 一个 expert 一个 signal，等一次就够

【粒度】等待单位是一个 expert，不是整轮 dispatch，也不是单个 token——前者太粗，后者元数据爆炸。

```text
         expert 0        expert 1        expert 2
disp  ████████████████░░░░████████████░░░░████████
                 ↓signal0         ↓signal1
gemm           ░░░████████████████████████████████
               ↑等 signal0（此时 dispatch 已经在搬 expert 1）
```

`░` 是等待，`█` 是干活。GEMM 只在开头等一次；之后每进一个新 expert，等待已被上一段计算盖住。

三个动作：dispatch 两个 warp 搬完一个 expert → 最后一个 CTA 置 signal；TMA warp 在新 group 的第一个 tile 前 acquire-load 自旋；拿到 signal 才 producer_acquire stage。

## 4. 三段代码：谁写、谁等、从哪扣

### dispatch 侧：谁有资格宣布齐了

看 `expert_signal_counters`。

```python
old = atomic_add(expert_signal_counters + expert_idx, 1, sem="release", scope="gpu")
if old + 1 == num_blocks:           # 我是最后一个完成该 expert 分片的 CTA
    store(expert_signals + expert_idx, 1, sem="release", scope="gpu")
```

所以呢：每个 CTA 只搬自己的分片，凑满 num_blocks 才算齐——它是 grid 的 CTA 总数，因为所有 CTA 都会遍历所有 expert。`release` 保证 A 行先可见；最后一个 CTA 写，N 次压成一次。

### GEMM 侧：等在哪，等几次

看 `last_waited_group`；wait 在 `producer_acquire` 之前。

```python
if cur_group_idx != last_waited_group:
    with elect_one():
        v = load(expert_signals + cur_group_idx, sem="acquire", scope="gpu")
        while v == 0: v = load(..., sem="acquire", scope="gpu")
    last_waited_group = cur_group_idx
```

所以呢：【位置】等待在 GEMM 生产端（TMA warp），消费端（MMA）只见 pipeline，不被阻塞。

1. 只让一个 lane 自旋（`elect_one`），其余在 `sync_warp` 等（32 个 lane 打同一个地址会打爆 L2）；且必须放在 `producer_acquire` 之前，反过来那级 stage 会一直占着空转。
2. 只等「换 group」那一次：M-contiguous 保证 expert 单调，每个 expert 只等一次，等时前一个 expert 的 MMA 还在跑。

### SMEM：先给通信留，再给计算分

看 `_dispatch_smem_bytes`、`_compute_stages`。

```text
SMEM 总容量
  − reserved（1 KB）
  − dispatch ring buffer（dispatch_num_stages × K × 2 B）
  − dispatch 的 mbarrier / 指针表
  = 留给 GEMM 的容量 → 决定 A/B stage 数、epilogue stage 数
```

所以呢：【代价】通信和计算抢同一块 SMEM：ring 深一档，A/B pipeline 就可能少一档，MMA 更容易被 TMA 饿死。

`expert_alignment` 抬到 cluster tile 的 M（256）：slab 按它补齐，`problem_sizes_m` 是真实 token 数，混用会行偏移错位。

## 5. expert 太小或 SMEM 太紧就不划算

- **expert 太小**：dispatch 本来就比 GEMM 快，`░` 消失，per-expert signal 换不来收益，只剩一次 atomic。
- **token 太散或倾斜**：粒度是整个 expert，一个 CTA 慢，所有想算它的 cluster 都得等；倾斜到极致时问题在带宽，不在 kernel。
- **SMEM 不够**：ring 常驻占一块，A/B stage 变浅，省下的等待抵不过饿死 MMA。

两个前提，缺一个就静默出错：

- 调度器必须 M-contiguous。否则 `last_waited_group` 会漏等，读到还没落地的 A——结果错但不报错。
- 这是 intranode 版本（对端指针直接 TMA）。跨机看同目录 `*_inter.py`，走 NCCL GIN。

读代码的坑：`rank = -1` 表示 top-k 槽被丢弃，所以到处是 `if (src_rank + 1) != 0` 和 `safe_*`。

## 6. 如果只记三句

- overlap 的收益来自「更早开算」，不是「搬得更快」。
- 【可见性】release 写 / acquire 读 + `scope="gpu"`；intranode 够用，跨机要走 `_inter` 变体。
- 下一篇看 combine 侧：同一条链路，方向反过来。
