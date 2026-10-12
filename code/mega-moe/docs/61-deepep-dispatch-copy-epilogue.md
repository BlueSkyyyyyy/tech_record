# DeepEP dispatch 的拷贝 epilogue：TMA 搬运、专家区对齐分配与链式列表

> 专题第 61 篇。前面第 7 篇（`docs/07-deepep-dispatch-combine.md`）讲了 DeepEP dispatch/combine 的
> 数据流与接口，第 48 篇（`docs/58-deepep-combine-reduce.md`）拆了 **combine 的归约 epilogue**
> （`combine_reduce_epilogue`：专家算完后把结果加回源 token）。本篇补上**对称的另一半**——
> **dispatch 的拷贝 epilogue**（`dispatch_copy_epilogue_impl`）：主 dispatch kernel 把 token 拉进
> 本 rank 的对称缓冲后，由一个**独立的第二 kernel** 把缓冲里的 token 落盘成下游 GEMM 需要的
> 目的张量（`recv_x`/`recv_sf`/`recv_topk_idx`/`recv_topk_weights`/`recv_src_metadata`），并做
> **专家区分组对齐分配**、**零填充**与**链式列表**。
>
> 关键观察：这段 kernel 整段**不触碰 NCCL**（没有 `ncclDevComm`/`ncclGin`/`ncclTeam`），只依赖
> `TokenLayout`/`BufferLayout`/`EPWorkspaceLayout` 三个纯结构体 + 第 59 篇讲过的 `ptx` 原语 +
> PDL。因此它既能被完整精读，又能在 H100（sm90）上**直接 include 真实头文件、跑真实设备代码**。
> 主 dispatch kernel 本身（`impls/ep/dispatch.cuh`）用 `NCCLGin`，本机 NCCL 版本受限（见「阻塞」），
> 但本篇的 epilogue 与之完全解耦，不依赖那个环境。

## 1. 它在流水里的位置：为什么拆成两个 kernel

DeepEP 的 dispatch 在 **scaleup（NVLink 域）/ scaleout（RDMA 域）** 两层拓扑上做 all-to-all，把每个
token 送到其 top-k 命中的专家所在 rank。主 kernel（`impls/ep/dispatch.cuh`）用 `comm::NCCLGin` 把数据搬进**本 rank 的对称 recv buffer**（`buffer`），但它**不负责**把 buffer
里的 token 展开/重排成下游友好的张量。这一步交给 `dispatch_copy_epilogue_impl`：

- **主 dispatch kernel**：跨 rank 搬运 + 写计数信号（`psum_num_recv_tokens_per_scaleup_rank`、
  `num_unaligned_recv_tokens_per_expert`、`psum_num_recv_tokens_per_expert`）。
- **拷贝 epilogue**（本篇）：在通信与计算 **overlap** 的语义下，从 buffer 把 token 一槽槽搬到最终输出，
  顺带做专家区分组对齐、零填充、链式列表维护。

二者靠 **PDL（Programmatic Dependent Launch）** 建立顺序，而不是 stream 顺序：
`dispatch_copy_epilogue.cuh:61` 的 `cudaGridDependencySynchronize()`（即 `griddepcontrol.wait`）会阻塞
到主 dispatch kernel「触发依赖」之后。注释 `:59-60` 明确警告：**PDL 下不得用 `__ldg` 读前序 kernel 写的
数据**（要显式同步后才可见）。这与第 48 篇 combine 归约 epilogue 的 `:57-59` 是同一套约定（第 52 篇
`docs/52-kernel-launch-pdl.md` 拆过发射侧的 PDL 属性）。

host 侧发射在 `csrc/kernels/ep/dispatch.hpp:168-226`：

- `:188-190` 用 `TokenLayout` 算单槽字节数，`num_warps = min(num_smem_bytes / token_bytes, 32)`——**用共享内存
  反推 warp 数**，每 warp 一个 token 槽（TMA 流水）。
- `:193-204` 通过 DeepJIT 现编一个只 include 本 `.cuh` 并强制实例化一个模板的微源码，14 个模板参数见下。
- `:207-214` 发射：`grid = num_sms`、`block = num_warps*32`、`enable_pdl = true`。

### 模板参数（`dispatch_copy_epilogue.cuh:12-22`）

```
kDoExpand, kCachedMode, kDoZeroPadding,
kNumSMs, kNumChannels, kNumWarps,
kNumScaleoutRanks, kNumScaleupRanks,
kNumHiddenBytes, kNumSFPacks, kNumMaxTokensPerRank,
kNumExperts, kNumTopk, kExpertAlignment,
kNumRanks = kNumScaleoutRanks * kNumScaleupRanks,
kNumThreads = kNumWarps * 32,
kNumMaxTokensPerChannel = ceil_div(kNumMaxTokensPerRank, kNumChannels),
kDoCreateLinkedList = (kNumScaleoutRanks > 1 and not kCachedMode)
```

四个布尔开关决定了它一次要处理**五种输出形态**（见 §4）。`kNumChannels` 只在 hybrid（scaleout>1）
时非 1，来自 `csrc/buffers/ep.hpp:437-456` 的 smem 预算。

## 2. 三种结构体：从「裸字节」到「token 槽」

### 2.1 `TokenLayout`（`layout/ep/token.cuh:10-82`）

一个 token 槽在缓冲区里被切成四段，全部按 `kNumTMAAlignmentBytes = 32` 对齐：

```
hidden (num_hidden_bytes)                    // :51-53
sf     (num_sf_bytes)                        // :55-57  packed block-scale
metadata:  topk_idx   (num_topk × 4B)        // :63-65
           topk_weights (num_topk × 4B)      // :67-69
           src_token_global_idx (4B)         // :71-73
           linked_list_idx (num_topk × 4B)   // :75-77  仅 with_metadata
mbarrier (8B, 仅 kWithMBarrier 视图)          // :79-81
```

`num_metadata_bytes = num_topk*(4+4) + (with_metadata ? (1+num_topk)*4 : 0)`（`:25-26`）。
`get_num_bytes<kWithMBarrier>()`（`:32-41`）把 `hidden/sf/metadata` 三段各自 32B 对齐后求和、再整体
对齐到 `kNumRDMAAlignmentBytes = 64`，最后（smem 视图）追加一个对齐到 32B 的 mbarrier。于是同一个
`TokenLayout` 有两个字节数：**gmem/对称缓冲用 `<false>`、smem TMA 槽用 `<true>`**。

本实验实测（`harness/exp_deepep_dispatch_copy_epilogue.out.txt`，hidden=128B / sf=16B / topk=4 /
with_metadata=1）：偏移 `hidden=0, sf=128, metadata=160, topk_idx=160, weights=176, src_global=192,
linked_list=196`，`<false>=256`、`<true>=288`（差一个 32B 的 mbarrier）。Python oracle 逐字段复刻一致。

### 2.2 `BufferLayout`（`layout/ep/token.cuh:84-144`）

把若干 rank、每个 rank 若干 token 摆成**规则网格**：

- `get_rank_buffer(r)`（`:122-126`）= `base + r * (num_max_tokens_per_rank * token_bytes)`；
- `get_token_buffer(i)`（`:139-143`）= `rank_base + i * token_bytes`；
- `get_channel_buffer<kNumTokensPerChannel>`（`:128-136`）= 按 channel 切一维视图，且**特意保留
  `num_max_tokens_per_rank` 作为 stride**（注释 `:133-134` 说明为什么不用除法）。

kernel 里用两个视图（`:48-50`）：

```cpp
tma_buffer      = BufferLayout<true >(token_layout, kNumWarps, 1, smem)  // smem：每 warp 一槽
scaleup_buffer  = BufferLayout<false>(token_layout, kNumScaleupRanks,
                                      kNumScaleoutRanks * kNumMaxTokensPerRank, buffer)  // 对称缓冲
```

即 **TMA 从 `scaleup_buffer`（gmem）载入到 `tma_buffer`（smem）**，处理后再 TMA 存出到最终输出。

### 2.3 `EPWorkspaceLayout`（`layout/ep/workspace.cuh:9-112`）

`EPSignals`（`:9-23`）是每个 rank 的对称 workspace，含屏障、每 rank/专家计数、以及**链式列表尾部指针**
`channel_scaleup_tail[kNumMaxChannels][kNumMaxRanks]`（`:20`）。本实验实测 `sizeof(EPSignals)` = 15,831,168 B
（`kNumMaxRanks=1024`、`kNumMaxChannels=1280` 撑起来的稀疏大表）。

## 3. 主循环逐段精读（`dispatch_copy_epilogue.cuh:70-209`）

循环把全局 token 序号 `i` 按 `kNumWarps*kNumSMs` 的步长分配给所有 warp（`:71`）。

**① 归属 rank 的确定（`:73-82`）**：token 在缓冲里按**来源 scaleup rank** 分段，段界由
`psum_num_recv_tokens_per_scaleup_rank`（累积）给出。代码用一次 `ptx::exchange` 就把「当前段结束」
广播给 warp：`stored_lane_idx = current_rank_idx % 32`，只有对应 lane 读 `psum[...]`，再 `exchange`
展开（`:76-80`）。这样一个 warp 每跨过一个 rank 段只读一次全局内存。

**② TMA 载入（`:84-95`）**：先 `tma_store_wait()`（`:85`）等**上一轮 TMA 存出**读完 smem，再让
elect-one lane 发 `tma_load_1d` 把整槽（data+sf+metadata）搬进 smem，`mbarrier_arrive_and_set_tx`
登记字节数（`:91-93`）。这里的「等存出→再载入」是复用同一 smem 槽的关键次序。

**③ 专家号校验与重映射（`:97-111`）**：
- 先**单独**读 top-k 专家号（`:99-101`）以容忍 TMA 延迟；
- `in_range = expert_start_idx <= e < expert_end_idx`（`:105`，`expert_start/end` 由本地 `rank_idx` 与
  `kNumExpertsPerRank` 给出，`:41-43`）；
- `dst_expert_idx = in_range ? e - expert_start : -1`——**映射成本地专家号**；
- `ptx::deduplicate` 断言同一 token 的有效槽专家号互异（`:108`）；
- 非 expand 直接写 `recv_topk_idx[i*topk + l]`（`:109-110`）。

**④ 目的行 `dst_tensor_idx` 的三条分支（`:113-124`）**——这是本 kernel 最核心的语义分叉：

| 模式 | dst_tensor_idx | 含义 |
| --- | --- | --- |
| 非 expand | `i`（elect-one） | 输出第 i 行就是第 i 个收到的 token（`:116-117`） |
| expand + cached | `recv_src_metadata[i*stride + 2 + l]` | 前向已缓存好目的槽，直接读（`:118-120`） |
| expand + 非 cached | `atomicAdd(psum_num_recv_tokens_per_expert + e, 1)` | **运行时按专家区原子分配**（`:121-122`） |

**⑤ 链式列表（`:131-136`）**：hybrid（scaleout>1）非 cached 时，`channel_linked_list[
linked_list_idx[master]] = i`——把当前 token 序号写进「上一跳指定的槽」，形成一条 per-channel 的链。
`master = ptx::get_master_lane_idx(ptx::gather(in_range))`（`:106`）取最高有效 lane。

**⑥ TMA 存出 hidden（`:138-144`）**：expand 只在 `dst>=0` 时存、非 expand 由 elect-one 存；每次
`tma_store_commit()`。目的地址 `recv_x + dst*kNumHiddenBytes`。

**⑦ 存 SF（`:146-178`）**：把 smem 里的 `sf_pack_t` 读进寄存器（`:154-159`），然后对每个有效 lane 把
一整列 SF 写到 `recv_sf + dst*token_stride`、逐元素步进 `hidden_stride`（`:169-175`）。**两种 stride 对应
host 的两种布局**：`recv_sf_token_stride=num_sf_packs, hidden_stride=1`（行主序）或
`token_stride=1, hidden_stride=align(num_allocated_tokens, 4)`（TMA 对齐的列主序，供下游 GEMM，
`csrc/buffers/ep.hpp:688-698`）。

**⑧ 存权重（`:180-187`）**：expand 写 `recv_topk_weights[dst]`（每 token 一个）、非 expand 写
`recv_topk_weights[i*topk + l]`（每 token topk 个）。

**⑨ 写 `recv_src_metadata`（`:189-208`）**：非 cached 时写源 token id（`+0`）与「combine 用的槽」
（`+1`：非 hybrid `current_rank_idx*topk + master`，hybrid `(i-current_rank_start)*topk + master`，
`:196-200`）；expand 再写每槽的 `dst_tensor_idx`（`+2+l`，`:205-206`）——这条元数据让后续 combine 知道
该去哪个目的行取数。

## 4. 五种输出形态与零填充

### 4.1 非 expand vs expand

- **非 expand**（`do_expand=false`）：dispatch 的输出与原始 token 一一对应，`recv_x` 有 `num_recv_tokens`
  行；`recv_topk_idx`/`recv_topk_weights` 是 `[T, topk]`。下游需按 token 再 gather 到专家。
- **expand**（`do_expand=true`）：每个 (token, 命中专家) 对**复制**成一行，输出按专家分组，
  `recv_x` 有 `num_expanded_tokens` 行；目的槽由**原子计数器**分配（非 cached）或从 metadata 读
  （cached）。这是「专家连续布局」——下游 grouped GEMM 可直接把每个专家的连续行段喂进去。
  host 侧 worst-case 上界：`num_expanded = N_ranks * T_max * min(topk, E_local) + (align-1)*E_local`，
  再对齐（`csrc/buffers/ep.hpp:666-668`）。

### 4.2 零填充（`:232-325`）

expand 且 `kExpertAlignment>1` 时，每个专家的行段要对齐到 `kExpertAlignment`，尾部空槽必须清零，
否则下游 GEMM 会读到脏数据。算法：

1. 先 `tma_store_wait()` 等最后一批存出（`:235`），再 `ptx::st_bulk<kNumHiddenBytes>` 把 smem 的 hidden
   区整体清零 + `tma_store_fence()`（`:239-240`）。
2. 每个 warp 读若干专家的**实际 token 数** `num_unaligned_recv_tokens_per_expert`（`:244-251`）。
3. 对每个「专家 wave」算 per-lane padding 数 `align - count%align`（`:266-267`），用
   `reduce_add`/`warp_inclusive_sum`/`warp_exclusive_sum` 在 warp 内定位
   `dst_tensor_idx = base_e + count_e + local_pad`（`:269-286`，`__ffs(__ballot(...))` 取**最低**满足
   lane）。
4. 对每个 padding 行：TMA 存出清零后的 hidden、把权重和 SF 也清零（`:292-319`）。

本实验用 4 专家、`counts=[4,3,3,4]`、`alignment=4` 触发 2 个 padding 行（专家 1→行 7、专家 2→行 11），
真实设备代码实测：**token 行 hidden 正确、padding 行 hidden/SF/权重全零**，`psum` 计数器从
`base` 被 atomicAdd 到 `base+count`。Python oracle 独立复刻了这段 warp 算术（含 `__ffs` 取最低 lane
这一关键点，第一版误用最高 lane 会得错解）。

### 4.3 cached 模式（`:113-124, 189-208`）

cached dispatch（`deep_ep/buffers/ep.py:161` 提到的场景）下，前向已把每个 token 的目的槽写进
`recv_src_metadata`，kernel 直接读 `[i*stride + 2 + l]` 当 `dst_tensor_idx`、**不再 atomicAdd**，
也不再重写 metadata（`:193` 的 `if constexpr (not kCachedMode)`）。这保证同一输入**逐位可复现**
（无原子顺序不确定性）。实测：输出严格落到 metadata 指定的行，且 `psum` 计数器保持不变。

### 4.4 链式列表（`:211-230`）

hybrid 非 cached 时，主循环把 token 序号串成链；循环结束后把**每条 (channel, scaleup-rank) 链的尾**
写 `-1`（哨兵）并把 `channel_scaleup_tail` 清零（`:216-229`），供 combine 阶段复用同一 workspace。
实测（scaleout=2、T=1 使写序确定）：`channel_linked_list[linked_list_idx[master]] = i`、
尾部槽 `-1`、`channel_scaleup_tail` 归零。

## 5. H100 实测（真实设备代码）

`harness/exp_deepep_dispatch_copy_epilogue.sh` 用 `nvcc -arch=sm_90a` **直接 include 真实
`deep_ep/impls/ep/dispatch_copy_epilogue.cuh`**，构造真实 `BufferLayout<false>` token 缓冲喂进去，
与独立 host oracle 对拍。**21 PASS / 0 FAIL**（`harness/exp_deepep_dispatch_copy_epilogue.out.txt`）：

- **Part 0 布局**：`TokenLayout` 各字段偏移与 `<false>/<true>` 字节数、`BufferLayout` 每 rank/总量、
  `EPWorkspaceLayout.num_experts_per_rank` 全对。
- **Part 1 非 expand**：`recv_x` 行 = 源 hidden（TMA 载入→存出）、SF 两种 stride 正确、
  `recv_topk_idx` 本地专家号、`recv_topk_weights`、`recv_src_metadata[0]/[1]` 全对；
  `metadata[2+l]` 24/24 未被触碰（非 expand 不写）。
- **Part 2 expand + 零填充**：每专家对齐区内的 token 集合正确（原子分配）、计数器到 `base+count`、
  token 行 hidden 正确、padding 行 hidden/SF/权重全零、`metadata[2+l]` 指向专家区内行。
- **Part 3 cached**：输出严格按 metadata 行、`psum` 不变（不 atomicAdd）。
- **Part 4 链式列表**：主循环写槽、尾部哨兵、workspace 归零。
- **Python oracle**：独立复刻 `TokenLayout` 偏移、`BufferLayout`、expand 对齐分配与零填充 dst 公式，
  ALL PASS。

## 6. 小结

- `dispatch_copy_epilogue_impl` 是 DeepEP dispatch 的**第二段 kernel**，把对称缓冲里的 token 落盘成
  下游张量；与主 dispatch 靠 **PDL** 而非 stream 顺序衔接，故读前序数据必须显式
  `cudaGridDependencySynchronize()`。
- 一个 kernel 覆盖**非 expand / expand(+cached) / 零填充 / 链式列表**五种形态，由四个模板布尔开关选择；
  expand 的目的槽要么原子分配、要么从缓存 metadata 读——后者保证逐位可复现。
- 输出张量的 stride 有两种（行主序 vs TMA 对齐列主序 SF），由 host 决定；`TokenLayout` 的三段对齐 +
  mbarrier 追加是「gmem 视图 vs smem 视图」差一个字节数的根源。
- 与第 48 篇 combine 归约 epilogue 合起来，**EP 的输入展开与输出归约两侧**都拆清了；两段都不碰 NCCL
  设备 API，因此本机可跑真实设备代码。
