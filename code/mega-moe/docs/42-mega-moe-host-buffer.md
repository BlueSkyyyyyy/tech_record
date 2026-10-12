# Mega MoE 的 host 编排层与对称缓冲区布局

> 前面拆 Mega MoE 时（第 02/03/09/11/20 篇）反复出现 `SymmBuffer`、`x_sf`、`l1_acts_sf`、
> `num_ring_tokens`、`locality domain` 这些东西，但没有一篇把它们**从 Python 到裸字节**串起来：
> 用户调 `deep_gemm.fp8_fp4_mega_moe(...)` 时，那一大坨「对称显存」到底是谁算出来的、每个张量
> 落在哪个偏移、为什么 L1/L2 的 SF 是转置的、ring 容量为什么是那个数。
>
> 这一篇补上 **主 kernel 之前的一整层 host 编排**：Python `SymmBuffer` 生命周期 →
> `csrc/apis/mega_moe.hpp` 的 `get_symm_buffer_size_for_mega_moe`（ring 容量公式 + 12 个 view 的
> 形状/stride）→ `layout/mega_moe.cuh` 的 `Workspace`/`Data`/`Buffer`/`MegaMoEBuffer` 链式字节分配
> → `scheduler/mega_moe.cuh` 的 `get_num_max_live_pool_blocks`。
>
> **关键点**：主 kernel 是 **SM100 专属**（`mega_moe.hpp:256/367` 只在 `arch_major==10` 派发），
> 但**它之前的这整层编排全是纯 host 计算、架构无关**，DeepGEMM 还通过 pybind 暴露了
> `deep_gemm._C.get_symm_buffer_size_for_mega_moe` / `get_token_alignment_for_mega_moe` /
> `get_block_m_for_mega_moe`。因此本机 **H100（sm90）上可以真机对拍这层编排的每一个字节**。
>
> 参考版本：`~/github/DeepGEMM` 公开版 commit `057ca59`（v2.8.1），本机 `kernel_lab`（H100 sm90）。
> 实验：`harness/exp_mega_moe_host_buffer.py`（driver `harness/exp_mega_moe_host_buffer.sh`），
> 输出 `harness/exp_mega_moe_host_buffer.out.txt`，7 组配置 **num_bytes 与全部 view 逐字段 ALL PASS**。

---

## 1. 一次调用的完整链路

```
deep_gemm.fp8_fp4_mega_moe(y, l1_w, l2_w, sym_buffer, ...)      # deep_gemm/mega/__init__.py:168
   ├── SymmBuffer(group, E, Tmax, topk, H, I, mma_type)          # :18  分配/复用对称显存 + 切 view
   │      ├── align(Tmax, get_token_alignment_for_mega_moe())    # :28  Tmax 对齐到 1920
   │      ├── _C.get_symm_buffer_size_for_mega_moe(...)          # :43  ← C++ 算字节数 + 返回切片闭包
   │      └── slice_input_buffers(buffer) → 12 个 torch view     # :76
   └── _C.fp8_fp4_mega_moe(y, ..., buffer, ptrs, rank, ...)       # :179 ← C++ 入口

csrc/apis/mega_moe.hpp
   get_token_alignment_for_mega_moe()      # :17  = kLCMCandidateBlockM = 1920
   get_symm_buffer_size_for_mega_moe(...)  # :32  ① ring 容量 ② MegaMoEBuffer ③ slice 闭包
   fp8_fp4_mega_moe(...)                   # :155 权重/缓冲区检查 → arch 派发 → (debug) 清空
   bf16_mega_moe(...)                      # :285

layout/mega_moe.cuh
   kCandidateBlockM[]                      # :13
   get_num_max_pool_tokens / get_num_sf_ring_tokens / get_num_max_shared_sf_tokens  # :20/:31/:36
   MegaMoESignals / Workspace / Data / Buffer / MegaMoEBuffer                       # :54/:93/:256/:287/:345

scheduler/mega_moe.cuh
   get_num_l1_warmup_waves(...)            # :17  warmup 波数（避免 L1→L2 死锁）
   get_num_max_live_pool_blocks(...)       # :48  ring 里同时存活的 pool block 上界
```

`deep_gemm/__init__.py:85/89-95` 把 `get_block_m_for_mega_moe`、`SymmBuffer`、
`transform_weights_for_mega_moe`、`fp8_fp4_mega_moe`、`bf16_mega_moe` 导出到顶层。

---

## 2. Python 侧：`SymmBuffer` 生命周期（`deep_gemm/mega/__init__.py`）

`SymmBuffer` 只做三件事：**对齐、分配/复用、切 view**。

**① 对齐 token 数**（`:28`）：

```python
num_max_tokens_per_rank = align(num_max_tokens_per_rank, _C.get_token_alignment_for_mega_moe())
```

`get_token_alignment_for_mega_moe()`（`mega_moe.hpp:17-19`）返回 `layout::kLCMCandidateBlockM = 1920`
（`layout/mega_moe.cuh:17`）——即所有候选 BLOCK_M `{8,16,32,64,96,128,192,240}` 的最小公倍数。
**为什么必须对齐到 1920**：ring 容量、SF ring、shared SF 容量都在 host 侧预先算死，kernel 侧按
**任意**候选 BLOCK_M 切块；只有 Tmax 是候选 BLOCK_M 的公倍数，`num_max_routed_tokens` 才能被
每种 BLOCK_M 整除，pool block 数不出现零头。实测：`Tmax=512` 会被抬到 1920（见 §7 的 C 段）。

**② 分配或复用**（`:50-68`）：

```python
if base is None:
    allocator = torch if group.size() == 1 else symm_mem
    self.buffer = allocator.empty(num_bytes, dtype=torch.int8, device='cuda')
    self.handle = (...) if group.size() == 1 else symm_mem.rendezvous(self.buffer, group=group)
    self.buffer.zero_(); self.group.barrier(); torch.cuda.synchronize()
else:
    assert num_bytes <= base.buffer.nbytes, (...)
    self.buffer = base.buffer; self.handle = base.handle
```

- `group.size() == 1` 时退化成**普通 `torch.empty`**（不需要 symmetric memory）——这也是本机能
  真实构造 `SymmBuffer` 的原因。
- `group.size() > 1` 走 `torch.distributed._symmetric_memory`（NVLink/IB 的对称窗口，第 03 篇）。
- 复用路径（传 `base`）**不重新分配**，只校验新配置所需字节 ≤ 已有 buffer 字节。实测：小配置复用
  OK 且 `buffer` 指针相同；大配置被 `:64-66` 断言拒绝（打印所需 / 现有字节数）。
- 分配后 `zero_()` + barrier + `cuda.synchronize()`：因为 kernel 依赖信号区（`grid_sync_count`、
  `combine_ready_grid_idx` …）从零开始，且要保证所有 rank 都清零完。

**③ 切 12 个 view**（`:70-76`）：`self.x, self.x_sf, self.topk_idx, self.topk_weights,
shared_l1_acts, shared_l1_acts_sf, shared_l2_acts, shared_l2_acts_sf, l1_acts, l1_acts_sf,
l2_acts, l2_acts_sf`。这些全部来自 C++ 闭包 `slice_input_buffers`。

**④ 权重预变换**（同一文件 `:112-164`，第 02 篇 §3.4 已拆过，此处只定位）：

- `_interleave_weights(t, gran=8)`（`:112-126`）：L1 的 gate/up 输出维按 8 粒度交错成
  `[g0..7, u0..7, g8..15, u8..15, …]`，配合 L1 epilogue 的 in-place SwiGLU（第 10 篇）。
- `_transpose_sf_for_utccp(sf)`（`:129-143`）：把 SF 的 128 维按 `4×32 → 32×4` 转置，迎合 UTCCP。
- `transform_weights_for_mega_moe`（`:146-164`）：tuple 权重（FP8/FP4）→ L1 交错权重 + L1/L2 转 SF；
  非 tuple（BF16）→ 只交错 L1、L2 不动。

---

## 3. C++ host 入口：ring 容量（`csrc/apis/mega_moe.hpp:32-92`）

`get_symm_buffer_size_for_mega_moe` 返回 `(num_bytes, slice闭包)`。`num_bytes` 由两块决定：
**Workspace + 各 Buffer 的总字节**，而其中唯一需要「算」的是 **ring 容量**。

```cpp
// :48-50
num_experts_per_rank = num_experts / num_ranks;
num_active_topk      = min(num_topk, num_experts_per_rank);
num_max_routed_tokens = num_max_tokens_per_rank * num_ranks * num_active_topk;

// :56-63 在所有候选 BLOCK_M 上取上界
for (block_m : kCandidateBlockM) {
    num_pool_blocks      = ceil_div(num_max_routed_tokens, block_m) + num_experts_per_rank;
    num_live_pool_blocks = get_num_max_live_pool_blocks(num_pool_blocks, num_sms, hidden, intermediate_hidden);
    num_ring_tokens      = max(num_ring_tokens, num_live_pool_blocks * block_m);
}
num_ring_tokens = align(num_ring_tokens, kLCMCandidateBlockM);   // :63
```

**为什么要遍历所有候选 BLOCK_M 取值**：host 侧此刻还不知道 kernel 会选哪个 BLOCK_M
（`get_block_config_for_mega_moe`，`heuristics/mega_moe.hpp:88`，依赖运行时的实际 `num_tokens`）
——第 03/05 篇已拆。buffer 必须在**任何** BLOCK_M 下都够用，所以取所有候选里的最大
`num_live_pool_blocks * block_m`，再对齐到 LCM 1920。

`+ num_experts_per_rank`（`:58`）是**每专家一个 pool block 的对齐 padding**（一个专家的 token
即使只有 1 行也要占一个 block），细粒度在 `get_num_sf_ring_tokens` 里被 `align(block_m,128)` 放大。

**SF ring**（`:70-77`）：只有当 `with_sf`（FP8/FP4，即 `MmaKind::MXFP8FP4`）才算，同样在所有
候选 BLOCK_M 上取 max：

```cpp
num_sf_ring_tokens = max over block_m of get_num_sf_ring_tokens(num_ring_tokens, block_m);
```

`get_num_sf_ring_tokens(n, bm) = (n/bm) * align(bm,128)`（`layout/mega_moe.cuh:31-34`）：all 专家
共用一段连续 SF 区，按「pool block 数 × SF_BLOCK_M」记，其中每个 block 的 SF 行数按 128 向上
对齐。实测（R=8,Tmax=4096,H=7168,I=2048）：`block_m=8` 时 `align(8,128)=128`，最浪费，得
1,935,360；`block_m=128` 时最省，得 120,960。取 max → **1,935,360**（见 §7 D 段）。

**SF 缓冲的断言**（`:88-92`）：`hidden%128==0`、`intermediate_hidden%128==0`、
`shared_intermediate_hidden%128==0`、`num_sf_ring_tokens%4==0`。

**`num_sms`**（`:47`）：来自 `get_num_sms_for_mega_moe()`（`heuristics/mega_moe.hpp:67-71`），
它断言 `runtime->get_num_sms() == jit->device.get_num_sms()`——Mega MoE 因为要按 locality domain
分区，**必须用整块 GPU 的全部 SM**。本机 H100 = 132 SM。

---

## 4. `MegaMoEBuffer`：一整块 buffer 的字节布局（`layout/mega_moe.cuh:345-457`）

### 4.1 结构与链式分配

`MegaMoEBuffer` 是一组 `Buffer`，每个 `Buffer` 持有 `(Data, num_ranks, num_max_tokens_per_rank, base)`，
其中 `Data` 只记「单行多少字节 + 是否要 TMA 对齐」（`:256-285`）：

```cpp
get_num_bytes_per_rank() = num_max_tokens_per_rank * data_layout.num_bytes;   // :306-308
get_num_bytes()          = get_num_bytes_per_rank() * num_ranks;             // :311-313
get_end_ptr()            = base + get_num_bytes();                            // :321-323
```

构造函数（`:368-450`）从 `workspace.get_end_ptr()` 开始，**逐个拿上一块的 `get_end_ptr()` 当下一块
的 base**——一条纯加法链，没有空洞（除少数 `align`）。各 `Data` 的「单行字节」在 `:387-398` 定义。

### 4.2 Workspace 不是零字节

`MegaMoEBuffer` 第一块是 `Workspace`（`:346`），它不是空的，而是内核的**信号区 + 元数据区**：

- `MegaMoESignals`（`:54-90`）：grid/NVLink 同步计数、`l1/l2_task_count`（按 domain）、
  combine readiness（`combine_ready_grid_idx`/`peer_grid_idx`，每 rank 8 字节）、每专家收发计数、
  ring 信号（`l1_full_count`/`l1_empty_count`/`l2_full_mask`/`l2_empty_count`）。
  注意 `kNumMaxRingBlocks = 1<<20`、`kNumMaxSharedL2Blocks = 1<<15` 是**编译期固定上界**
  （`:58-59`），所以这坨信号本身就有 ~21 MB（实测 `sizeof(MegaMoESignals) = 21,140,864` 字节，
  且 `:91` 的 static_assert 要求 `combine_ready_grid_idx` 恰好落在偏移 128）。
- Workspace 末尾追加两段可变长元数据：`num_experts * num_max_tokens_per_rank * 4`（源 token-topk 表）
  和 `num_max_pool_tokens * sizeof(TokenSrcMetadata)=12`（combine push 源索引），最后 `align 16`
  （`:128-144`）。

`num_max_pool_tokens`（`:20-28`）本身也是一个上界：`align(num_max_recv_tokens * active_topk
+ num_experts_per_rank * (kMaxCandidateBlockM - 1), 1920)`——`(240-1)` 是对「最坏 BLOCK_M=240 时每
专家 padding」的估计。

### 4.3 12 个 view 的形状 / stride / dtype

`slice_input_buffers`（`mega_moe.hpp:96-151`）为每个 Buffer 造一个 `torch::from_blob` view。
下表是实测（R=8,E=256,Tmax=2048,topk=8,H=7168,I=2048,fp8xfp4）——**逐字段与复刻一致**：

| view | offset(B) | shape | stride | dtype |
|---|---:|---|---|---|
| `x` | 24,919,936 | (2048, 7168) | (7168, 1) | fp8 |
| `x_sf` | 39,600,000 | (2048, 56) | (56, 1) | int32 |
| `topk_idx` | 40,058,752 | (2048, 8) | (8, 1) | int64 |
| `topk_weights` | 40,189,824 | (2048, 8) | (8, 1) | fp32 |
| `l1_acts` | 40,255,360 | (63360, 7168) | (7168, 1) | fp8 |
| `l1_acts_sf` | 494,419,840 | (1013760, 56) | **(1, 1013760)** | int32 |
| `l2_acts` | 721,755,520 | (63360, 2048) | (2048, 1) | fp8 |
| `l2_acts_sf` | 851,516,800 | (1013760, 16) | **(1, 1013760)** | int32 |

几个**非平凡点**：

1. **`x` / `shared_l1_acts` 是同一个对象**（`:114`，`auto shared_l1_acts = x;`）：共享专家的 L1
   输入直接复用输入 token buffer。实测两者 `data_ptr` 相同。

2. **`x_sf` 是 K-major（行连续），而 `l1_acts_sf`/`l2_acts_sf` 是 M-major（stride `(1, n)`）**
   （注释 `:95`）。用户写 `x_sf` 时按行（K 维）连续；而 kernel 内部写的 L1/L2 激活 SF 按列
   （token 维）连续。这正是第 10 篇 UTCCP 要的布局：SF 沿 token 维连续，UTCCP 一次搬一个 SF 块。
   复刻里对未显式给 stride 的 view 用连续 stride 推导，对 M-major 的两个 SF 显式给 `(1, n)`。

3. **SF 每行字节数 `hidden/32`**（`Data(hidden/32)`，`:393`）而 view 的 shape 是
   `(Tmax, hidden/128)` int32——`hidden/128` 个 int32 = `hidden/32` 字节，**两种口径一致**。
   这些 SF 缓冲 `require_tma_alignment=false`（`:269` 允许 `num_bytes%16!=0`），所以
   `hidden/128` 不必是 4 的倍数也行（实测 `intermediate_hidden/128=11`、`/128=2` 都出现）。

4. **combine buffer 的“外层维”是 `num_topk(+shared)`**（`:447-449`）：`Data(hidden*2)` bf16，
   `num_ranks = num_topk + (shared?1:0)`。这就是 EP 的 **K 倍放大**在显存里的体现——每个 token 的
   combine 结果按 top-k（+shared）各留一份（第 01 篇 §2）。

5. **`l1_topk_weights_buffer` 每行只有 4 字节**（`:398`，`Data(sizeof(float))`），却有
   `num_ring_tokens` 行——ring 里每个 token 存一个 fp32 路由权重。

6. **总字节**（`:452-456`）：`combine_token_buffer.get_end_ptr() - workspace.signals`，即所有块
   首尾相接后的跨度。实测 R=8/Tmax=2048/FP8 配置为 **1,151,278,464 B ≈ 1.07 GiB**；BF16 同形状
   会更大（token 2 字节/元素）——与第 20 篇的 `symm buffer BF16 vs FP8 = 1.265×` 一致。

---

## 5. ring 容量 helper 的推导（`scheduler/mega_moe.cuh:48-80`）

`get_num_max_live_pool_blocks(num_total_m_blocks, num_sms, hidden, intermediate_hidden)` 给出
「ring 里同时存活的 pool block 数上界」。它是**纯 host 的保守闭式估计**，不依赖 GPU：

```
num_clusters       = num_sms / 2                       // 2-CTA cluster
num_l1_n_clusters  = intermediate_hidden*2 / (2*128)   // L1 输出 N 方向 cluster 数
num_l2_n_clusters  = hidden / (2*128)                  // L2 输出 N 方向 cluster 数
num_l1_clusters    = num_total_m_blocks * num_l1_n_clusters
num_l1_waves       = ceil(num_l1_clusters / num_clusters)
```

核心是 `num_live_blocks_after_warmup`：所有 CTA 先跑一段 **L1 warmup 波**（`:63-67`），
warmup 期间已经启动的 M block 数就是「初始存活数」；warmup 波数由
`get_num_l1_warmup_waves`（`:17-45`）给出——**这是为了不给 L1→L2 造成死锁**：L2 任务要等它对应的
L1 N block 全部 issue 才能开始（`l2_full_mask`，`:140-141`），所以调度必须先成批发 L1。

再叠加两项保守余量（`:69-78`）：

```
frontier_growth = ceil(num_total_m_blocks * (num_l2_n_clusters - num_l1_n_clusters) / num_l2_n_clusters)  // if L2>L1
wave_margin     = ceil(num_clusters / min(num_l1_n_clusters, num_l2_n_clusters))
return min(num_total_m_blocks, num_live_blocks_after_warmup + frontier_growth + wave_margin)
```

- `frontier_growth`：当 L2 的 N cluster 比 L1 多（`hidden > 2*intermediate_hidden` 时），L1 在
  M 方向推进更快，存活 block 数会线性增长，故按比例补。
- `wave_margin`：补一波全局波，覆盖部分波 / 流水滞后。
- **`min(..., num_total_m_blocks)`**：不可能比总 block 多。

实测这一项在 `num_pool_blocks` 上几乎是线性的（见 §7 D 段）：`pool_blocks 32800→14072 live`，
且 `live*BM` 随 BLOCK_M 单调上升到 240 时的 119,280——所以 ring 大小对 BLOCK_M 敏感，
必须取 max。注释 `:141` 也点明：**L1→L2 位图的相位翻转依赖 ring 容量**，容量算错会直接串相位。

---

## 6. `fp8_fp4_mega_moe` / `bf16_mega_moe` 入口（`mega_moe.hpp:155-390`）

两个 C++ 入口做同样的事，差别只在权重类型与是否有 SF：

1. **recipe 契约**（`:176-177`）：FP8/FP4 只接受 `(rm,rn,rk)==(1,1,32)`，即 per-token/per-32 block
   scale（对应第 25/27 篇的 `per_token_cast_to_fp8` + gran32 packed UE8M0）。
2. **权重布局检查**（`:188-204`）：`check_weights_layout_3d` 要求 L1 `[E/R, 2I, H]`、L2 `[E/R, H, I]`；
   `check_sf_layout` 要求 SF 是 UE8M0、MN-major、TMA 对齐、`kGranMN=1,kGranK=32`。BF16 分支
   （`:312-316`）只检查权重是 BF16。
3. **shared expert**（`:206-228` / `:322-340`）：可选，`num_shared_experts = shared_intermediate_hidden
   / intermediate_hidden`；SF 侧 `shared_l1_weights_sf` 无 group 维（`std::nullopt`）。
4. **缓冲区校验**（`:237-248` / `:349-359`）：再次调用 `get_symm_buffer_size_for_mega_moe` 算所需字节，
   断言 `sym_buffer.nbytes() >= num_required_bytes`，然后 `slice(sym_buffer)` 拿回 12 个 view。
5. **arch 派发**（`:256-277` / `:367-384`）：**只有 `arch_major==10`** 才进
   `sm100_fp8_fp4_mega_moe` / `sm100_bf16_mega_moe`，否则 `DG_HOST_UNREACHABLE`。
   **这就是「H100 跑不了端到端」的唯一闸门**——第 1~5 步纯 host 逻辑在 H100 上畅通无阻。
6. **debug 清零**（`:279-282` / `:386-389`）：环境变量 `DG_COMM_KERNEL_DEBUG` 打开时，每次调用后
   把整个 symm buffer 清零（调用者需要每次重新拷贝输入）。

`register_apis`（`:392-398`）把这 5 个函数绑成 pybind：`get_token_alignment_for_mega_moe`、
`get_block_m_for_mega_moe`、`get_symm_buffer_size_for_mega_moe`、`fp8_fp4_mega_moe`、`bf16_mega_moe`。

---

## 7. H100 实测

环境：H100 80GB HBM3（sm90）、132 SM、`arch_major=9`、`deep_gemm._C` 可用（v2.8.1 / commit
`057ca59`）。脚本用 Python **逐字段复刻** `MegaMoESignals` 结构体（C 布局规则）+
`Workspace`/`Data`/`Buffer`/`MegaMoEBuffer` 的链式 offset 分配 + ring 容量公式，再与真实 binding
返回的 `num_bytes` 和每个 view 的 `(offset, shape, stride, dtype)` 逐项对拍。

**A/B) 7 组配置全部 `num_bytes` MATCH，全部 view `OK`**：

| R | E | Tmax | topk | H | I | mma_type | shared | ring_tokens | num_bytes | 判定 |
|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---|
| 8 | 256 | 2048 | 8 | 7168 | 2048 | fp8xfp4 | 0 | 63360 | 1,151,278,464 | MATCH |
| 8 | 256 | 4096 | 8 | 7168 | 2048 | fp8xfp4 | 1 | 120960 | 2,287,634,816 | MATCH |
| 1 | 256 | 4096 | 8 | 7168 | 2048 | fp8xfp8 | 0 | 44160 | 1,137,541,504 | MATCH |
| 4 | 128 | 8192 | 6 | 4096 | 1408 | fp8xfp4 | 2 | 69120 | 1,155,066,752 | MATCH |
| 16 | 512 | 1024 | 4 | 2048 | 512 | bf16xbf16 | 0 | 49920 | 300,947,328 | MATCH |
| 64 | 1024 | 512 | 8 | 3072 | 1536 | bf16xbf16 | 1 | 5760 | 112,627,072 | MATCH |
| 2 | 64 | 32 | 2 | 1024 | 256 | fp8xfp4 | 0 | 9600 | 39,900,288 | MATCH |

复刻里 `sizeof(MegaMoESignals)=21,140,864`（含 `:91` 的 offset==128 static_assert 复核）。
`num_bytes` 完全吻合 ⇒ **整个链式分配 + ring 公式（含 `get_num_max_live_pool_blocks`）都被独立验证**。

**C) 真实 `SymmBuffer` 类**（gloo world_size=1 → torch allocator 路径）：
`Tmax=512` 被 `:28` 抬到 **1920**（`x` shape `(1920, 2048)`）；`l1_acts_sf` stride `(1, 153600)`
（M-major）；`shared_l1_acts is x == True`；写入 `x[0,0]=1.0 / x[1,3]=2.0` 原位读回正确；
小配置用 `base` 复用 OK 且 buffer 指针相同；大配置复用在 `:64-66` 被拒绝
（`requires 224,384,384 bytes, but ... only 93,824,384 bytes`）。

**D) ring 容量阶梯**（R=8,Tmax=4096,H=7168,I=2048, num_sms=132, routed_tokens=262144）：

```
BLOCK_M  pool_blocks  live_blocks   live*BM
      8        32800        14072     112576
     16        16416         7050     112800
     32         8224         3539     113248
     64         4128         1784     114176
     96         2763         1199     115104
    128         2080          906     115968
    192         1398          614     117888
    240         1125          497     119280   ← max
→ max(live*BM)=119280, align 1920 → num_ring_tokens=120960
sf_ring per BLOCK_M: 8→1935360 ... 128→120960 ... 240→129024
→ num_sf_ring_tokens = max = 1935360
```

坐实了：`live_blocks` 随 BLOCK_M 增大而降，但 `live*BM` 反升（更少 block × 更大块），
**所以必须在所有候选上取 max**；SF ring 的 max 落在最细的 `BLOCK_M=8`（`align(8,128)` 浪费最大）。

**总判定：ALL PASS**（7 组 × 全 view 逐字段）。

---

## 8. 小结

- Mega MoE 的 **host 编排层**（`SymmBuffer` → `get_symm_buffer_size_for_mega_moe` →
  `MegaMoEBuffer` → scheduler ring helper）是一整条**架构无关、纯 host** 的字节计算链，
  H100 上可逐字段对拍（本文 7 组 ALL PASS）。
- 三个「为什么」：
  1. **Tmax 对齐 1920**：让任意候选 BLOCK_M 都能整除、ring/pool 无零头（`:28`）。
  2. **ring 在所有候选 BLOCK_M 取 max**：host 此刻不知道 kernel 会选哪个 BLOCK_M
     （基运行时 token 数），buffer 必须对任意选择都够用（`:56-63`）。
  3. **`x_sf` K-major vs `l1/l2_acts_sf` M-major**：前者是用户输入、后者是 kernel 内 UTCCP 要的
     沿 token 维连续布局（`:95`，第 10 篇）。
- **唯一挡住 H100 的是 `:256/367` 的 `arch_major==10` 派发**——前面所有 host 逻辑本机都能跑。
  这也是为什么后续要在 H100 上继续做的、与 Mega MoE 相关的分组 GEMM / 通信实验（第 04/12/13/14 篇）
  都能成立。

### 真实链接 / 源码坐标

- DeepGEMM：https://github.com/deepseek-ai/DeepGEMM （本地 `~/github/DeepGEMM`，核对 commit `057ca59`）
- 引入 Mega MoE 的 release：[PR #304](https://github.com/deepseek-ai/DeepGEMM/pull/304)（2026-04-17）、
  [PR #316](https://github.com/deepseek-ai/DeepGEMM/pull/316)（2026-04-24，含基准）、
  [PR #462](https://github.com/deepseek-ai/DeepGEMM/pull/462)（2026-09-30，locality-aware）
- 关键文件：`csrc/apis/mega_moe.hpp`、`deep_gemm/include/deep_gemm/layout/mega_moe.cuh`、
  `deep_gemm/include/deep_gemm/scheduler/mega_moe.cuh`、
  `csrc/jit_kernels/heuristics/mega_moe.hpp`、`deep_gemm/mega/__init__.py`
- 相关篇章：第 02 篇（架构与权重布局）、第 03 篇（`SymmBuffer`/对称显存）、第 05 篇（BLOCK_M 启发式）、
  第 09 篇(第 34 篇 locality domain)、第 20 篇（BF16 变体）
