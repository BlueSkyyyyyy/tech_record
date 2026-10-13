# 共享 TMA 拷贝层与环形流水线（`common/tma_copy.cuh` + `common/ring_pipeline.cuh`）

> 本专题（Mega MoE 前向 / 反向算子实现解析）的第 77 篇。第 44 篇补齐了 DeepGEMM 的**设备端公共
> 地板** `common/{math,utils,types,packing,compile,exception,cute_tie}.cuh`，并在结尾列了「可再补」
> 的清单；第 56 篇补了每一行 `cp.async.bulk.tensor` 前面那 128 字节 `CUtensorMap` 的**来源**
> （`runtime_utils.hpp` 的 `make_tma_*`）；第 46 篇补了 GEMM 入口的**主序契约**。
>
> 但夹在这两者中间的那一层——**「描述符造好之后，谁来发 TMA、发几次、搬到 smem 的哪里」**——
> 一直没被单独讲过：它就是 `deep_gemm/include/deep_gemm/common/tma_copy.cuh`（106 行）。它被
> **16 个 impl 文件** include（`sm100_fp8_fp4_mega_moe.cuh` / `sm100_bf16_mega_moe.cuh` /
> `sm90_bf16_gemm.cuh` / `sm90_fp8_gemm_1d1d.cuh` / `sm100_mqa_logits.cuh` / `sm100_mega_mhc.cuh`
> …），前面每一篇讲 TMA 时贴的 `tma::copy<...>` 调用点都源于此。
>
> 与之成对的还有 `common/ring_pipeline.cuh`（40 行）：一个把「环形 stage 索引 + 相位翻转 + 返回旧
> 状态」封装成 5 行的 `RingPipeline<kNumStages>::advance`。它把第 29 篇（`sm90_bf16_gemm.cuh`）里
> 手写的 `advance_pipeline` lambda 标准化了，MQA logits / 稀疏 MQA logits / Mega mHC 全都换用了它。
>
> 两个文件都**架构无关到设备函数级**：H100(sm90) 上可直接 include 真实头文件跑真实设备代码。
> 只有 `copy_nd` 里 `__CUDA_ARCH__ >= 1000` 的 **2-SM TMA** 分支是 SM100 专属（仅分析）。
> 本文所有行号均已对照 DeepGEMM commit `057ca59`（Public release 26/09/30）核对过。实测见
> `code/mega-moe/harness/exp_tma_copy_ring.{sh,out.txt}` + `src/tma_copy_ring.cu`。
> 目标卡代号敏感：全文一律用「目标卡」指代。

---

## 1. 这只箱子为什么存在：一个 `tma::copy` 要回答四个问题

在任意一个 SM90/SM100 kernel 里，搬一块 A/B/SF 进 smem 需要同时定下四件事：

1. **描述符**：`CUtensorMap`（第 56 篇），其中 inner box 的宽度由 **swizzle 模式**决定；
2. **坐标**：TMA 指令只认「inner 坐标 + 若干 outer 坐标」，而主序（K-major / MN-major）会交换谁当
   inner（第 46 篇）；
3. **smem 落点**：一块 `BLOCK_OUTER × BLOCK_INNER` 的 box 在带 swizzle 时，**不能被一条 TMA 全搬
   进去**——硬件 box 的 inner 宽度被 swizzle 原子（32/64/128 字节）卡死，所以要**沿 inner 方向切成
   若干原子、逐条发**；
4. **多播 / 2-SM**：cluster 里两个 CTA 可能共享同一块数据，这时不是各自发一条，而是「leader 发一条
   `multicast`，硬件把数据扇出并给每个 CTA 的 mbarrier 记 tx」。

`tma_copy.cuh` 就是把这四个问题收进**一个 `copy<BLOCK_INNER, BLOCK_OUTER, kSwizzleMode, dtype_t,
kIs3DTMA>` 模板**（`:96-104`）。调用点因此短到一行（`sm90_bf16_gemm.cuh:189-199`）。

```cpp
// deep_gemm/include/deep_gemm/common/tma_copy.cuh:96-104
template <uint32_t BLOCK_INNER, uint32_t BLOCK_OUTER, uint32_t kSwizzleMode,
          typename dtype_t, bool kIs3DTMA = false, ...>
CUTLASS_DEVICE void
copy(void const* desc_ptr, cutlass::arch::ClusterTransactionBarrier* barrier_ptr,
     dtype_t* smem_ptr, const uint32_t& inner_idx, const uint32_t& outer_idx,
     const uint32_t& num_tma_multicast = 1, const uint32_t& batch_idx = 0) {
    if constexpr (kIs3DTMA)
        copy_nd<..., 3, ...>(desc_ptr, barrier_ptr, smem_ptr, num_tma_multicast, inner_idx, outer_idx, batch_idx);
    else
        copy_nd<..., 2, ...>(desc_ptr, barrier_ptr, smem_ptr, num_tma_multicast, inner_idx, outer_idx);
}
```

`kIs3DTMA` 只服务**批处理 GEMM**（`GemmType::Batched`）：此时 A/B 的描述符是 3D 的（batch 走最外层），
调用点用第三个坐标 `batch_idx` 取「第几号专家矩阵」（`sm90_bf16_gemm.cuh:186-190`：
`const uint32_t batch_idx = (kIsBatchedMM ? scheduler.current_group_idx : 0);`）。

---

## 2. `tma_copy.cuh` 逐段

### 2.1 swizzle 原子的 inner 宽度：`get_inner_block_atom_size`（`:12-15`）

```cpp
template <uint32_t BLOCK_INNER, uint32_t kSwizzleMode, uint32_t kPackFactor, typename dtype_t>
constexpr uint32_t get_inner_block_atom_size() {
    return kSwizzleMode == 0 ? BLOCK_INNER / kPackFactor : kSwizzleMode / sizeof(dtype_t);
}
```

- **swizzle == 0**：没有原子概念，一条 TMA 就把整个 `BLOCK_INNER`（逻辑元素）搬进 smem。除以
  `kPackFactor` 是因为 **FP4 在 smem 里一字节塞两个逻辑元素**（`packing.cuh:9-10`
  `get_smem_pack_factor`：`float_e2m1_t → 2`，其余 → 1）——所以「一个 TMA lane 覆盖的**存储字节**数」
  是 `BLOCK_INNER / kPackFactor`。
- **swizzle != 0**：一个 swizzle 原子横跨 `kSwizzleMode` 个**字节**（32/64/128），换成 **storage 元素**
  要除以 `sizeof(dtype_t)`。注意这里 `sizeof(float_e2m1_t) == 1`（被打包类型是 1 字节存储），所以
  FP4 sw128 的原子是 **128 storage 元素 = 256 逻辑元素**。

这一点直接决定了 tensormap 的 box 宽度：`runtime_utils.hpp` 造描述符时把 inner box 设成
`swizzle==0 ? BLOCK_INNER : swizzle/elem`（第 56 篇已从 `make_tma_*` 侧确认），于是 `copy_nd` 发
`BLOCK_INNER / 原子` 条，正好拼成完整一块。

### 2.2 维度 → cute 载入原语的映射：`LoadOps`（`:17-37`）

```cpp
template <uint32_t kNumDimensions> struct LoadOps;
template <> struct LoadOps<2> {
    using Load          = cute::SM90_TMA_LOAD_2D;
    using Load2SM       = cute::SM100_TMA_2SM_LOAD_2D;
    using LoadMulticast = cute::SM90_TMA_LOAD_MULTICAST_2D;
};
// 同样地 3D / 4D / 5D（:23-37）
```

三个原语对应三种「谁收数据」：

| 别名 | cute 原语 | 收方 | 架构 |
|---|---|---|---|
| `Load` | `SM90_TMA_LOAD_{2..5}D` | 本 CTA | sm90+ |
| `LoadMulticast` | `SM90_TMA_LOAD_MULTICAST_{2..5}D` | cluster 内被 mask 选中的 CTA | sm90+ |
| `Load2SM` | `SM100_TMA_2SM_LOAD_{2..5}D` | 2-SM 搭档（UMMA 2-SM） | **sm100 专属** |

支持 2~5 维，是给不同描述符形态用的：普通 GEMM 的 A/B 是 **2D**；批处理 GEMM 是 **3D**（多一维
batch）；而 Mega MoE 的权重是 **4D** ——`sm100_fp8_fp4_mega_moe.cuh:793` 的
`tma::copy_nd<BLOCK_K, LOAD_BLOCK_N, kSwizzleBMode, weight_dtype_t, 4>` 取的坐标是
`(k_idx, half_n_idx, half_idx, expert_idx)`：`[k, n半区, 上下半, 专家号]`，一把描述符对上
「路由专家 + 共享专家」的全部权重（第 34 篇）。

### 2.3 核心：`copy_nd`（`:39-77`）

```cpp
template <uint32_t BLOCK_INNER, uint32_t BLOCK_OUTER, uint32_t kSwizzleMode,
          typename dtype_t, uint32_t kNumDimensions, uint64_t kCacheHint = ..., typename... coord_ts>
CUTLASS_DEVICE void
copy_nd(void const* desc_ptr, cutlass::arch::ClusterTransactionBarrier* barrier_ptr,
        dtype_t* smem_ptr, const uint32_t& num_tma_multicast,
        const uint32_t& inner_idx, const coord_ts&... outer_coords) {
    DG_STATIC_ASSERT(sizeof...(coord_ts) + 1 == kNumDimensions, "Invalid coordinates");          // :46
    DG_STATIC_ASSERT(uint64(...EVICT_NORMAL_sm90) == uint64(...EVICT_NORMAL_sm100), "...");       // :47-48
    constexpr uint32_t kPackFactor = get_smem_pack_factor<dtype_t>();                             // :49
    DG_STATIC_ASSERT(kPackFactor == 1 or sizeof(dtype_t) == 1, "Packing expects a 1-byte storage type"); // :51
    constexpr uint32_t BLOCK_INNER_ATOM_STORAGE = get_inner_block_atom_size<...>();               // :52
    constexpr uint32_t BLOCK_INNER_ATOM_LOGICAL = BLOCK_INNER_ATOM_STORAGE * kPackFactor;         // :53
    DG_STATIC_ASSERT(BLOCK_INNER % BLOCK_INNER_ATOM_LOGICAL == 0, "...");                          // :54
    ...
}
```

三条**编译期断言**各自在防一类真实错误：

- `:46`：坐标个数必须与描述符维度一致——防止有人给 2D 描述符传 3 个坐标（批处理 GEMM 忘记
  `kIs3DTMA`）；
- `:47-48`：`EVICT_NORMAL` 这个 cache hint 常量在 sm90 与 sm100 头文件里必须数值相等，否则同一
  份默认参数在两个架构上语义不同；
- `:51`：只有「打包因子 1」或「存储宽度 1 字节」的类型才能进原子换算——因为 `BLOCK_INNER_ATOM_STORAGE`
  是**字节/元素混算**，`:53` 再乘回逻辑因子。FP4 恰好两项都满足（`sizeof(float_e2m1_t)==1`、pack=2）。

发指令的主体（`:56-77`）分三支：

```cpp
if (num_tma_multicast == 1) {                       // :56 单 CTA：自己搬
    #pragma unroll
    for (uint32_t i = 0; i < BLOCK_INNER / BLOCK_INNER_ATOM_LOGICAL; ++ i)
        Ops::Load::copy(desc_ptr, (uint64_t*)barrier_ptr, kCacheHint,
                        smem_ptr + i * BLOCK_OUTER * BLOCK_INNER_ATOM_STORAGE,   // :60 逐原子挪 smem
                        inner_idx + i * BLOCK_INNER_ATOM_LOGICAL, outer_coords...); // :61 逐原子挪坐标
} else {
    #if __CUDA_ARCH__ >= 1000                        // :64-71 2-SM：每条信号只发给 leader
        Ops::Load2SM::copy(...);
    #elif __CUDA_ARCH__ >= 900                       // :72-77 多播：只有 rank0 真发
        if (cute::block_rank_in_cluster() == 0)
            Ops::LoadMulticast::copy(desc_ptr, (uint64_t*)barrier_ptr,
                                     (1 << num_tma_multicast) - 1, kCacheHint, ...);
    #endif
}
```

三个关键「为什么」：

1. **原子循环的两条步进**（`:60-61`）：smem 偏移 `i * BLOCK_OUTER * ATOM_STORAGE`（每个原子块占
   `OUTER × 原子` 个 storage 元素），坐标偏移 `i * ATOM_LOGICAL`（逻辑元素）。**smem 用 storage、
   gmem 用 logical** 正是 `:51` 那条断言守护的混算约定。
2. **多播 mask = `(1 << num_tma_multicast) - 1`**（`:75`）：约定 cluster 内前 `num_tma_multicast` 个
   rank 都收。所以多播只对「连续的低位 rank 组」生效，调用方必须保证 cluster 形状与之匹配。
3. **多播只由 rank0 发**（`:73`）：`LoadMulticast` 硬件把数据 + tx 计数扇出到每个目标 CTA **同
   smem 偏移**的 mbarrier，因此**每个 CTA 仍要各自 `arrive_and_expect_tx` 自己的 barrier**——这就是
   `sm90_bf16_gemm.cuh:200` 每个 CTA（仅其 elected TMA 线程）都调一次
   `full_barrier.arrive_and_expect_tx(...)`、即便 rank>0 根本没发指令的原因。

SM100 的 2-SM 分支为什么**只给 leader CTA 发信号**？因为 2-SM UMMA 把两块 CTA 当一个更大的 MMA，
leader 持有期间只需要自己的 barrier 被唤醒（注释 `:65` 写死）。这条路径本机 H100 跑不了，**仅分析**。

---

## 3. `ring_pipeline.cuh`：把 5 行流水线状态机标准化

多 stage 流水线的 stage 索引本质是一个模 `kNumStages` 的计数器 + 一位相位。第 29 篇
`sm90_bf16_gemm.cuh:138-145` 里它是手写 lambda：

```cpp
uint32_t stage_idx = 0, phase = 0;
auto advance_pipeline = [&](uint32_t& k_block_idx) {
    ++ k_block_idx;
    stage_idx = stage_idx == kNumStages - 1 ? 0 : stage_idx + 1;
    phase ^= stage_idx == 0;
};
```

`RingPipeline` 把它封装成有状态对象 + 一个返回**旧状态**的 `advance`（`:13-38`）：

```cpp
template <uint32_t kNumStages>
struct RingPipeline {
    DG_STATIC_ASSERT(kNumStages > 0, "...");
    uint32_t stage_idx = 0;
    uint32_t phase = 0;
    // Callers advance by at most one full ring, so the phase toggles at most once.
    CUTLASS_DEVICE cute::tuple<uint32_t, uint32_t> advance(const uint32_t step = 1) {   // :21
        const uint32_t current_stage_idx = stage_idx;
        const uint32_t current_phase = phase;
        const uint32_t next_stage_idx = stage_idx + step;
        if constexpr ((kNumStages & (kNumStages - 1)) == 0) {        // :26 2 的幂
            stage_idx = next_stage_idx % kNumStages;
            phase ^= next_stage_idx / kNumStages;
        } else {                                                     // :29 非 2 的幂
            stage_idx = next_stage_idx;
            if (stage_idx >= kNumStages) { stage_idx -= kNumStages; phase ^= 1u; }
        }
        return cute::tuple<uint32_t, uint32_t>(current_stage_idx, current_phase);  // :36 返回旧状态
    }
};
```

四个设计点：

1. **返回旧 (stage, phase)**（`:36`）：调用方拿到的是「本轮到手的 buffer」的坐标，紧接着用它去
   `wait` 对应的 full barrier、再从该 stage 取数据；等到下一轮 `advance` 才换格。用
   `CUTE_TIE_DECL(q_pipeline.advance(), q_stage_idx, q_phase)`（`sm100_mqa_logits.cuh:227`）结构化
   绑定接住——这是 `common/cute_tie.cuh`（第 44 篇）存在的理由。
2. **2 的幂走除法/取模**（`:26-28`）：注释 `:25` 说明「除以 2 的幂会被编译器降成位运算」，所以
   这条路径是免费的；而 **`phase ^= next/kNumStages` 一次异或就等价于「每绕一圈翻一次相位」**，
   把「step 跨多圈」也顺手处理了（前提是 `step ≤ kNumStages`）。
3. **非 2 的幂走减法**（`:29-35`）：只能减一次、`phase ^= 1`——因为 `advance` 的契约是
   **「单次步进不超过一整圈」**（`:20` 注释）。跨整圈时相位最多翻一次，正好对应「一个 mbarrier
   的相位只有 0/1 两态」这个硬件事实。这不是偷懒，而是 mbarrier 语义决定的。
4. **支持 `step > 1`**：`advance(math_warpgroup_idx)`（`sm100_mqa_logits.cuh:439`）、
   `a_pipeline.advance(kPostWorkerIdx)`（`sm100_mega_mhc.cuh:415`）——当多个消费者 warpgroup
   错开取同一环形缓冲时，一步跨多个 stage 比自己循环 `advance()` 更直接。

---

## 4. H100(sm90) 真机实验

`harness/exp_tma_copy_ring.sh` 用 `nvcc -arch=sm_90a --expt-relaxed-constexpr` **直接 include 真实
`tma_copy.cuh` / `ring_pipeline.cuh`**（`--expt-relaxed-constexpr` 是必需的：`get_inner_block_atom_size`
是 `constexpr __host__`，但要从设备函数 `copy_nd` 里调用；DeepJIT 也带这个 flag）。跑出 **27 PASS /
0 FAIL**。以下为原始输出摘录。

### 4.1 原子尺寸与打包因子（host constexpr）

```
bf16: sw0->256 sw32->16 sw64->32 sw128->64 (logical)
fp8 : sw0->256 sw128->128 | fp4: sw0->128 sw128->128 storage, pack=2
```

对照独立公式 `swizzle==0 ? BLOCK_INNER/pack : swizzle_bytes/sizeof(dtype)`：bf16 的 sw128 原子
= `128/2 = 64` 逻辑元素，fp8 的 sw128 原子 = `128/1 = 128`，fp4 的 sw0 原子 = `256/2 = 128`
（逻辑），而 **fp4 的 sw128 原子 = 128 storage 元素**（= 256 逻辑，因为一字节两元素）。10 条断言全过。

### 4.2 `RingPipeline::advance` 状态机（设备端）

对 `N ∈ {1,2,3,4,5,8}`（含非 2 的幂 3/5）× `step ∈ {1,2,3,N}`，每条跑 `5N` 次，与 oracle
「第 i 次返回 `(i·step mod N, (i·step/N) & 1)`」逐项比较：

```
N=1  step=1 :: 5 calls mismatch=0      N=3  step=1 :: 15 calls mismatch=0
N=4  step=1 :: 20 calls mismatch=0     N=2  step=1 :: 10 calls mismatch=0
N=8  step=3 :: 40 calls mismatch=0     N=8  step=8 :: 40 calls mismatch=0   # 恰好一整圈
```

9 组共 245 次调用 **0 不符**——坐实「返回旧状态」与「非 2 的幂分支」都正确。

### 4.3 真实 `tma::copy`（2D 单原子 / 多原子 atom 循环 / 3D 批次）

源张量位模式 = `row*inner + col`，逐字节对比「预测物理布局」（swizzle XOR 键取自绝对 smem 地址
bits[7:10)，见第 56 篇 §Part 2）：

```
2d_sw0_1atom      box=64x8  atom=64 atoms=1  mismatch=0/1024
2d_sw128_1atom    box=64x8  atom=64 atoms=1  mismatch=0/1024
2d_sw128_4atom    box=64x8  atom=64 atoms=4  mismatch=0/4096   # BLOCK_INNER=256 -> 4 条 TMA 拼一块
2d_sw64_2atom     box=32x8  atom=32 atoms=2  mismatch=0/1024   # BLOCK_INNER=64  -> 2 条
3d_sw128_batch2   box=64x8  atom=64 atoms=1  mismatch=0/1024   # kIs3DTMA，取 batch=2
```

`2d_sw128_4atom` 正面证明 §2.3 的原子循环：**一条 `tma::copy<256,...>` 展开成 4 条
`cp.async.bulk.tensor.2d`**，坐标 `inner_idx + i*64`、smem `+ i*8*64` 元素交替推进，拼出来的
4096 字节与「一个 256 宽的 box」完全一致；`3d_sw128_batch2` 证明 `kIs3DTMA` 的批次坐标生效。

### 4.4 真实 `tma::copy` 多播（cluster=2）

`__cluster_dims__(2,1,1)` 起一个 2-CTA cluster，只有 rank0 真发 `LoadMulticast`（`copy_nd` 的
rank 判断），**两个 CTA 各自 init 自己的 `ClusterTransactionBarrier` 并 `arrive_and_expect_tx`**：

```
rank=0 mismatch=0/1024
rank=1 mismatch=0/1024
```

两个 CTA 收到的字节都正确——正面验证 §2.3 第 3 点「硬件按同 smem 偏移给每个 CTA 的 barrier 记 tx」。
这是 H100 上能跑通的 sm90 多播路径；`sm100` 的 `Load2SM` 分支（`copy_nd:64-71`）本机不可跑，仅分析。

---

## 5. 边界与「仅分析」清单

- **SM100 专属（仅分析）**：`copy_nd` 的 `__CUDA_ARCH__ >= 1000` 分支——`SM100_TMA_2SM_LOAD_{2..5}D`
  与「2-CTA 只给 leader 信号」。Mega MoE 的 A/SF 载入正是走这条（`sm100_fp8_fp4_mega_moe.cuh:743-745`
  的 `tma::copy<BLOCK_K, LOAD_BLOCK_M, kSwizzleAMode, a_dtype_t>` 传 `num_tma_multicast=2`）。
- **缓存的局限**：实验里 smem 基址 `__align__(1024)`，故 swizzle XOR 键的基准行位为 0；真实
  kernel 里 smem 各 stage 的绝对地址不一定是 1024 的整数倍，swizzle 键会随绝对地址整体平移（第 56
  篇已量化过这个陷阱）。
- **FP4 TMA 载入未实跑**：`get_inner_block_atom_size` 的 FP4 数值只做了 constexpr 对拍，没有真的发
  `16U4` 描述符的 TMA（需要 `CU_TENSOR_MAP_DATA_TYPE_16U4_*` + 打包布局，且实际由 DeepJIT 在
  目标卡上生成）；此处只证明「字节/元素混算」的编译期公式。

---

## 6. 小结

- `common/tma_copy.cuh` 是**「描述符已造好之后」的唯一 TMA 发射层**：用 `get_inner_block_atom_size`
  把 `BLOCK_INNER` 切成 swizzle 原子，用 `LoadOps<维度>` 选 `Load/LoadMulticast/Load2SM`，用三条
  静态断言守住坐标数 / cache hint / 打包混算，最后把「单 CTA 自己搬 / rank0 多播 / sm100 2-SM」三支
  收进一个模板——这就是前面 16 个 impl 里所有 `tma::copy<...>` 的全部实现。
- `common/ring_pipeline.cuh` 把「环形 stage + 相位翻转 + 返回旧状态」标准化，用 2 的幂位运算优化与非
  2 的幂减法两种分支，支撑 `step ≤ kNumStages` 的任意步进。
- 两者都**架构无关到设备函数级**：H100 真机直接 include 真实头文件跑真实设备代码 **27 PASS / 0
  FAIL**（含 atom 循环展开成多条 TMA、3D 批次坐标、sm90 多播两 CTA 同步收 tx）；唯一例外是
  `copy_nd` 的 sm100 2-SM 分支。
