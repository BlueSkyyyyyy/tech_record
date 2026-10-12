# DeepGEMM einsum `bmk,bnk->mn`：把「batch 维」并进收缩轴的批归约 GEMM（H100 实测）

> 本文是「DeepSeek Mega MoE 前向 / 反向算子实现解析」专题的 P8 增量（第 17 篇）。
> 前面第 10~14 篇把 MoE 反向的 **grouped GEMM**（`k_grouped` / `m_grouped`）讲完了：
> wgrad 的不等长轴在**收缩维 K（token）**，dgrad 的不等长轴在 **M（token）**。
> 但训练里还有一类「反向」是 grouped GEMM 覆盖不到的——**沿一个 batch 维做归约的权重梯度**：
> 当同一个权重被 `b` 个批次（rank / 头 / chunk）分别消费时，权重梯度天然是
> `D = Σ_b A_bᵀ B_b`。DeepGEMM 把它做成一个硬编码 einsum：`bmk,bnk->mn`。
>
> 本篇拆解这个**批归约 GEMM**：它的数学、host 契约、SM90 split-K kernel 与 `atomicAdd` 归约、
> SM100 变体（TMA reduce-add），并在 H100 上给出数值/等价性/非确定性/性能的**实测数字**。
>
> 源码坐标：`~/github/DeepGEMM`，公开版 commit `057ca59`（v2.1.1.post3）。
> SM100 分支为**仅分析**。本机 H100 sm90。

## 1. 为什么需要它：grouped GEMM 之外的第「三」种 wgrad

MoE 训练的反向有三类收缩形态，前两类已被本专题覆盖：

| 形态 | 数学 | 不等长轴 | DeepGEMM 入口 | 编号 |
|---|---|---|---|---|
| dgrad | `dX = dG·Wᵀ` | M（token） | `m_grouped_*` | 第 11 篇 |
| wgrad（分组） | `dW_e = x_eᵀ·dG_e` | K（token，逐专家） | `k_grouped_*` | 第 10 篇 |
| **批归约 wgrad** | **`D = Σ_b A_bᵀ B_b`** | **b（batch/rank）** | **`einsum('bmk,bnk->mn')`** | **本篇** |

第三类的典型场景：

1. **EP / DP 下的权重梯度归约**。每个 rank 只持有部分 token，算出**部分**权重梯度 `P_r`；
   全局权重梯度是 `Σ_r P_r`。`bmk,bnk->mn` 把这个「逐 rank 的 GEMM + 跨 rank 求和」
   融进一次 launch，中间量 `[b,m,n]` 不用落显存。
2. **Attention / MLA 的按头共享权重**。Q 投影或吸收矩阵按 `b`（batch/序列 chunk）分组、
   `h`（头）共享时，权重梯度同样是沿 `b` 归约。
3. **任意「同一权重被多批数据消费」的层**（例如 test 里 `b` 从 4 到 8192 扫描）。

数学上它就是「把 batch 维展平进收缩轴」的普通 GEMM：

```
D[m,n] = Σ_b Σ_k A[b,m,k]·B[b,n,k]
       = A.view(m, b·k) @ B.view(n, b·k)ᵀ        （收缩轴 = flatten(b,k)，k 内层）
```

但**实现上不能用普通 GEMM 替代**：因为 `m,n` 通常很小（一个权重 tile），而 `b·k` 极大，
普通 GEMM 要么先把 `[b,m,n]` 逐个算出来再求和（中间量爆炸），要么把 `b` 维展平后物化
`[m, b·k]` / `[n, b·k]`（显存爆炸）。第 8.3 节实测这两条朴素路线分别慢 **7~18×**、
显存多 **8×**（且展平要物化 GB 级副本）。

## 2. DeepGEMM 的 einsum 家族总览

`csrc/apis/einsum.hpp` 注册了两个函数：`einsum`（BF16）与 `fp8_einsum`（FP8 输入、走 BMM）。
它们**不支持任意表达式**（`einsum.hpp:163-165` 的 TODO），而是硬编码了几个 Einstein sum：

| expr | 语义 | 输出 | arch | 实现 |
|---|---|---|---|---|
| `bmk,bnk->mn` | `Σ_{b,k} A[b,m,k]B[b,n,k]` | FP32 / BF16 | 9 / 10 | `sm90/sm100_bmk_bnk_mn` |
| `bhr,hdr->bhd` | `Σ_r A[b,h,r]B[h,d,r]` | BF16 | 9 / 10 / cuBLASLt | `sm90/sm100_bf16_gemm` |
| `bhd,hdr->bhr` | `Σ_d A[b,h,d]B[h,d,r]` | BF16 | 9 / 10 / cuBLASLt | 同上 |
| `bhd,bhr->hdr` | `Σ_b A[b,h,d]B[b,h,r]` | FP32（累积） | cuBLASLt | `cublaslt_bhd_bhr_hdr` |
| `fp8 bhr,hdr->bhd` | FP8 版 MLA 投影 | BF16 / FP8 | 9 / 10 | `sm90/sm100_fp8_bmm` |
| `fp8 bhd,hdr->bhr` | FP8 版 | BF16 | 10 | `sm100_fp8_bmm` |
| `fp8 bhd,bhr->hdr` | FP8 版跨 batch wgrad | FP32 | 10 | `sm100_fp8_bmm` |

注意到 `bhd,bhr->hdr` 已经是「沿 `b` 归约的 FP32 权重梯度」，但它**只走 cuBLASLt**
（`einsum.hpp:147`，无自研 kernel）；而 `bmk,bnk->mn` 才是 DeepGEMM 自研的批归约 GEMM。
本篇聚焦后者。dispatch 在 `einsum.hpp:166-178`：

```
if      (expr == "bmk,bnk->mn") bmk_bnk_mn(a, b, d, c);       // :166-168
else if (expr == "bhr,hdr->bhd") bhr_hdr_bhd(...);            // :169-170
else if (expr == "bhd,hdr->bhr") bhd_hdr_bhr(...);            // :171-172
else if (expr == "bhd,bhr->hdr") bhd_bhr_hdr(...);            // :173-175
else DG_HOST_UNREACHABLE(...)                                 // :176-177
```

## 3. host 端契约（`einsum.hpp:22-58`）

`bmk_bnk_mn(a, b, d, c)` 的分支逻辑：

- **FP32 累积模式**（`:25-26`）：`d` 是 fp32 时，强制 `c` 与 `d` **同一张量**（
  `c->data_ptr()==d.data_ptr()`）——因为 kernel 是 `atomicAdd` 到 `D`，`C` 就是 `D` 的初值。
  实测传不同张量被拒（`einsum.hpp:26`）。
- **BF16 输出模式**（`:28-38`）：`d` 是 bf16 时不允许 `c`；内部开一个 fp32 workspace、
  `cudaMemsetAsync` 清零、跑 fp32 累积、再 `d.copy_(workspace)`（隐式 fp32→bf16）。
  实测 bf16 传 `c` 被拒（`einsum.hpp:29`）。
- 连续性（`:41-43`）与形状（`:45-47`）：`a/b/d` 必须连续；`a:[s,m,k]`、`b:[s,n,k]`，
  断言 `s==s_ and k==k_`。
- **arch 派发**（`:49-57`）：`arch_major==9 → sm90_bmn_bnk_mn_gemm`，
  `==10 → sm100_bmn_bnk_mn_gemm`。这是本机 H100 可跑的分叉点。

注意：`m,n,k` 全部取自 `a/b`，`d` **只校验 dtype、不校验形状**。实测把 `d` 开成 `(128,128)`
而 `(m,n)=(64,64)` 时，kernel 只写左上 `64×64`，`d[64:,:]` 保持零——
调用方需自己对好 `d` 的 `(m,n)`（`harness/exp_einsum_bmk_bnk.py` §F 记录）。

## 4. 调度与 split-K（`sm90_bmk_bnk_mn.hpp`）

SM90 host 端（`csrc/jit_kernels/impls/sm90_bmk_bnk_mn.hpp`）：

```
block_m=128, block_n=128, block_k=64                              // :19-21
num_tma_threads=128, num_math_threads=256                        // :22-23
DG_HOST_ASSERT(k % 64 == 0);                                     // :24
DG_HOST_ASSERT(m % 64 == 0 and n % 64 == 0);                     // :25
num_mn_blocks = ceil_div(m,128) * ceil_div(n,128)                // :33
num_sk_blocks = s * (k / 64)      // 收缩轴 = flatten(b,k)        // :34
split_factor  = ceil_div(num_sk_blocks, max(num_sms/num_mn_blocks, 1))  // :35
```

- **收缩轴切块**（`:34-35`）：把 `s*(k/64)` 个 K 块按「每个 MN tile 分到约
  `num_sms/num_mn_blocks` 个 block」的原则打包，形成 split-K factor。
- **流水级数**（`:38-53`）：从 4 级起，按 `(A+B) × stages + barrier` 装进 SM90 的
  `smem_capacity`（232448B）为止，`block_k=64` → 每级 `128×64×2×2 = 32KB`，一般降到 1~2 级。
- **TMA 描述符**（`:64-65`）：`a` 视为 2D `[s*m, k]`、`b` 视为 `[s*n, k]`，
  即「把 batch 维拼进行维」，`swizzle=128`。
- **网格**（`:93`）：`grid = num_mn_blocks × ceil_div(num_sk_blocks, split_factor)`，
  块内 `num_tma_threads + num_math_threads = 384` 线程。

## 5. SM90 kernel 逐段精读（`sm90_bmk_bnk_mn.cuh`）

### 5.1 warp 角色与预取（`:30-81`）

- 类型：`WGMMA = BF16MMASelector<BLOCK_N>`（`:31`），`BLOCK_M % WGMMA::M == 0`（`:33`）。
- 静态断言把配置钉死：`BLOCK_M==128`、`kNumTMAThreads==128`、`kNumMathThreads==256`（`:42-44`）。
- warp 0 的一个 lane 预取 A/B 的 TMA 描述符（`:47-50`），`__syncwarp()` 后继续。
- smem 布局：A 在前 `kNumStages` 段、B 紧随（`:56-61`），随后是 full/empty barrier 数组（`:64-66`）。
  warp 1 初始化：`full_barriers[i]->init(1)`（TMA 到达 1 次）、`empty_barriers[i]->init(kNumMathThreads)`
  （**每个 math 线程各到达一次**，`:69-78`），`fence_barrier_init()` 使其在 async proxy 可见。

### 5.2 寄存器再平衡（`:84-85`）

`kNumTMARegisters=40`、`kNumMathRegisters=232`：TMA warp-group 让出寄存器给 math warp-group
（`:99-101` dealloc / `:127` alloc）。这是 Hopper warp-specialization 的标准两档分配。

### 5.3 block 索引与 PDL（`:88-97`）

```
mn_block_idx = blockIdx.x % num_mn_blocks       // 展平的 (m,n) tile
sk_block_idx = blockIdx.x / num_mn_blocks       // split-K 分片
num_total_stages = min(kSplitFactor,
                       shape_s*(SHAPE_K/BLOCK_K) - sk_block_idx*kSplitFactor)  // :94
```

`num_total_stages` 处理**尾分片**：最后一个 sk 分片可能不够 `kSplitFactor` 个 K
块，要按剩余的取（避免越界读）。随后：

```
cudaGridDependencySynchronize();   // :97  PDL：等待前驱 kernel 完成
```

这是 **Programmatic Dependent Launch（PDL）** 的消费者侧等待点。host 端 `jit->launch`
在 `enable_pdl` 时加 `CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION`
（`deep_jit/backend/cuda/kernel.hpp:148-152`），由 `deep_gemm.set_pdl(bool)` 开关
（`csrc/apis/config.hpp:27-32`）。有了 PDL，本 kernel 的**前导阶段**（预取描述符、初始化
barrier、寄存器再平衡）可与前驱 kernel 的**尾部**重叠；真正碰数据前在 `:97` 等到依赖满足。
本 kernel 在 `:97` 很早就同步，所以重叠窗口只有「launch 到 `:97` 之间」的一小段——第 8.4 节
实测 PDL on/off 差异在噪声级（这与 mega-kernel 里大段 prologue 的做法不同）。

### 5.4 TMA 生产者（`:99-124`）

只有一个 warp（`kNumMathThreads/32 = 8`）的一个 lane 发 TMA：

```
for s in 0..num_total_stages:
    empty_barriers[stage_idx]->wait((s/kNumStages + 1) & 1)     // :110 消费者已释放
    sk_idx = (sk_block_idx*kSplitFactor + s) * BLOCK_K          // :113 收缩轴坐标
    k_idx  = sk_idx % SHAPE_K                                   // :114  k 内层
    s_idx  = sk_idx / SHAPE_K                                   // :115  batch 外层
    tma::copy<BLOCK_K,BLOCK_M>(a, k_idx, m_block_idx*BLOCK_M + s_idx*SHAPE_M)  // :118
    tma::copy<BLOCK_K,BLOCK_N>(b, k_idx, n_block_idx*BLOCK_N + s_idx*SHAPE_N)  // :120
    full_barrier.arrive_and_expect_tx(SMEM_A + SMEM_B)          // :122
```

关键点：**A/B 的 TMA 行坐标 = `tile + s_idx*SHAPE`**——把「第 `b` 个 batch 的 m/n tile」
通过 `s_idx*SHAPE_M` 折算成 2D `[s*m, k]` 里的行偏移。所以 host 只造了 2D 描述符，
kernel 用一次乘加换出 batch 维度。相位约定与第 6 篇 mbarrier 一致：
消费者（math）用 `(s/kNumStages)&1`，生产者用 `(s/kNumStages+1)&1`（buffer 初空）。

### 5.5 WGMMA 消费者（`:125-158`）

256 个 math 线程（2 个 warpgroup）各自持有 `accum[WGMMA::kNumAccum]` 寄存器累加器（`:131`）。
每个 stage：`full_barrier.wait((s/kNumStages)&1)`（`:137`）→ 发 `BLOCK_K/WGMMA::K` 条
`WGMMA::wgmma`（`make_smem_desc` 的 A 描述符带 `math_wg_idx*WGMMA::M` 行偏移，`:146`）→
`commit_batch/wait<0>`（`:150-154`）→ `empty_barriers[stage_idx]->arrive()`（`:157`）。

注意：**每个 block 只算自己分片的部分和**，跨分片的归约不在这里做——留到 epilogue 的
`atomicAdd`（这是本 kernel 与普通 GEMM 最本质的区别）。

### 5.6 epilogue：`atomicAdd(float2)` 直接归约到 D（`:160-174`）

```
row = m_block_idx*BLOCK_M + warp_idx*16 + lane_idx/4            // :160
col = n_block_idx*BLOCK_N + (lane_idx%4)*2                      // :161
for i in 0..WGMMA::kNumAccum/4:
    atomicAdd(&d[(row  )*SHAPE_N + col + i*8], {accum[i*4+0], accum[i*4+1]})  // :167
    atomicAdd(&d[(row+8)*SHAPE_N + col + i*8], {accum[i*4+2], accum[i*4+3]})  // :171
```

- **直接 `atomicAdd` 到全局 `D`**：不同 `sk_block` 的部分和以原子加合并。
  `D` 必须是 fp32（bf16 走 workspace，见 §3）。
- 这也是**非确定性**的来源：原子加的顺序不固定。第 8.2 节实测 20 次运行有
  ~5 万个元素 bit 级不同（幅度 ~4e-7，数值可忽略，但**逐位复现不保证**）。
- 边界保护：`col+i*8 >= SHAPE_N` 提前 break（`:164`），`row/row+8 < SHAPE_M` 才写（`:166/170`）。

## 6. SM100 变体对比（**仅分析**，`sm100_bmk_bnk_mn.cuh`）

SM100 版（arch≥1000，`:25`）语义相同，但实现三处不同：

1. **TMA reduce-add 取代 atomicAdd**（`:255`）：
   `cute::SM90_TMA_REDUCE_ADD_2D::copy(&tensor_map_d, smem_cd, n_idx, m_idx)`——
   硬件把 D 当作 reduce 目标，省掉逐元素 `atomicAdd` 的指令开销与 L2 往返。
   host 侧因此多造一个 D 的 TMA 描述符（`sm100_bmk_bnk_mn.hpp:69`）并选 CD swizzle（`:28`）。
2. **tcgen05 UMMA + TMEM**（`:131-183`）：warp 1 发
   `SM100_MMA_F16BF16_SS::fma`（`:174`，首条 `s>0 or k>0` 决定是否累加），
   累加器在 TMEM；warp 2 负责 `TMEM::Allocator1Sm().allocate`（`:90-92`）。
   epilogue 用 `SM100_TMEM_LOAD_32dp32b4x` 读回 TMEM 再 STSM 到 smem（`:245-248`）。
3. **单 warpgroup 架构**（`kNumThreads=128`，`:19-24`）：warp 0 TMA load、warp 1 MMA、warp 2
   分配 TMEM，其余 warp 参与 epilogue 的 STSM/TMA store。仍是 384 线程？不——SM100 用
   `num_threads=128`（`sm100_bmk_bnk_mn.hpp:22`），因为 tcgen05 的 MMA 只需单 warp 发起。

其余一致：同样的 split-K 展平（`:103`）、同样的 PDL 等待点（`:106`）、同样的
`num_total_stages` 尾分片处理（`:103`）、同样的 `m,n,k` 契约（`sm100_bmk_bnk_mn.hpp:23-24`）。

## 7. H100 实测

实验 `harness/exp_einsum_bmk_bnk.py`（driver `.sh`，输出 `.out.txt`）在本机
**H100 80GB HBM3 / sm90 / 132 SM** 上真跑。以下均为原始数字摘录。

### 7.1 语义正确性（§A）

`ref = torch.bmm(A,Bᵀ).sum(0)`（fp64 思维下的 fp32 参考）：

| s | m | n | k | FP32 reldiff | BF16 reldiff |
|---|---|---|---|---|---|
| 129 | 128 | 384 | 128 | 3.9e-7 | 3.5e-3 |
| 4096 | 384 | 128 | 384 | 3.9e-5 | 2.9e-3 |
| 8192 | 256 | 256 | 256 | 7.7e-5 | 2.6e-3 |

FP32 误差随 `s` 增长（收缩长度越大、fp32 累加顺序偏差越大），8192×256×256×256 也不到 1e-4；
BF16 输出受 bf16 舍入限制，稳定在 ~3e-3。**语义与定义完全一致**。

### 7.2 三种等价视角

- **展平 GEMM**（§B）：`D == A.permute(1,0,2).reshape(m,s*k) @ B.permute(1,0,2).reshape(n,s*k)ᵀ`
  的 reldiff 1.0e-6~2.1e-4（差异仅来自累加顺序）——**坐实「batch 并入收缩轴」恒等式**。
- **rank 归约模型**（§C）：`s=8` 时
  `reldiff(einsum, Σ_r (A_r@B_rᵀ)) = 2.8e-7`——**坐实「部分 wgrad 再求和」语义**。
  朴素 `[s,m,n]` fp32 中间量 1.6MB vs 融合输出 0.197MB，**省 8×**（`s` 越大越省）。
- **dtype 双路**（§A）：FP32 路径（`c==d`）与 BF16 路径（内部 fp32 workspace）都验证。

### 7.3 非确定性（§D）

`s=8192,m=n=k=256` → `(num_mn,num_sk,split,grid)=(4, 32768, 993, 132)`（split=993！）。
同一输入跑 20 次，与首次结果比：

```
bit 级不同元素数 min/median/max = 51473/52113/52751  (总 65536)
最大相对偏差 = 4.0e-7
```

即 **65536 个输出元素里有 ~5.2 万个出现了 bit 级抖动**，因为 993 个 split 分片的
`atomicAdd` 顺序随机。幅度 4e-7 对训练无害，但**要求逐位可复现的场合不能用它**
（对照：普通 `torch.sum` 两次运行 bit 差异恒为 0）。SM100 的 TMA reduce-add 同样
不保证确定性。

### 7.4 性能（§E）

`FLOPs = 2·s·m·n·k`，对比 torch 朴素（`bmm+sum`，会物化 `[s,m,n]`）与「展平后的稠密
GEMM」（预先把 fp32 展平副本算好、不计入计时）：

| s | m | n | k | einsum | TFLOPS | GB/s | torch 朴素 | 倍数 | 展平 GEMM |
|---|---|---|---|---|---|---|---|---|---|
| 129 | 128 | 384 | 128 | 10.7us | 152 | 1599 | 121.8us | 11.4× | 22.1us |
| 4096 | 256 | 256 | 256 | 761us | 181 | 1411 | 6944us | 9.1× | 2271us |
| 8192 | 384 | 128 | 384 | 2548us | 121 | 1264 | 18366us | 7.2× | 5726us |
| 8192 | 256 | 256 | 256 | 1662us | 165 | 1292 | 13825us | 8.3× | 4722us |
| 4096 | 512 | 512 | 128 | 922us | 298 | 1166 | 11829us | 12.8× | 2181us |
| 4096 | 1024 | 1024 | 64 | 1750us | 314 | 616 | 31612us | 18.1× | 3005us |

观察：

- einsum **比 torch 朴素快 7~18×**，且**不物化 `[s,m,n]`**（`s=8192,m=n=256` 时朴素
  中间量 = 8192·256·256·4B = **2.1GB**，融合输出仅 256KB）。
- **`m·n` 越大越快**：`m=n=1024` 达 314 TFLOPS，而 `m=384,n=128` 只有 121——
  因为 `num_mn_blocks` 决定并行度与原子归约的竞争度。`m=n=256` 时 `num_mn=4`、
  split 到 993 个分片全往 4 个 tile 上 `atomicAdd`，竞争很重。
- 带宽 616~1599 GB/s，说明它**既非纯计算受限也非纯访存受限**：算术强度约
  `m·n/(m+n)`（256 时 ~128），远低于 H100 bf16 roofline 拐点 ~295，但小 `m·n` 时
  原子归约竞争拉低了有效吞吐。**这类算子适合大 `m·n`（大权重 tile）+ 极大 `b`**。

### 7.5 参数契约（§F）

| 非法输入 | 结果 |
|---|---|
| `k%64 != 0`（k=96） | 拒（`sm90_bmk_bnk_mn.hpp:24`） |
| `m%64 != 0`（m=96） | 拒（`sm90_bmk_bnk_mn.hpp:25`） |
| `n%64 != 0`（n=96） | 拒（`sm90_bmk_bnk_mn.hpp:25`） |
| fp32 `c != d` | 拒（`einsum.hpp:26`） |
| bf16 带 `c` | 拒（`einsum.hpp:29`） |
| `d` 形状偏大 | **不校验**，只写左上 `(m,n)`（`m,n` 取自 a/b） |

### 7.6 PDL（§G）

`set_pdl(False)` vs `set_pdl(True)`：reldiff 都是 3.485e-05（语义不变），
耗时 818.5us vs 756.6us——差异在噪声级。原因见 §5.3：本 kernel 在 `:97` 很早就
`cudaGridDependencySynchronize()`，PDL 重叠窗口太小。PDL 真正发光的地方是 prologue
很长的 mega-kernel（如第 7 篇的 MMA/epilogue 流水），不是这种「开局即等」的小 kernel。

## 8. 与前向/反向主题的联系与结论

- **它是 MoE wgrad 的「跨 rank 求和的融合版」**。第 10 篇的 `k_grouped_*` 处理的是
  「同一 rank 内、逐专家的不等长 token 收缩」；`bmk,bnk->mn` 处理的是「同一权重、
  跨 batch/rank 的求和」。两者拼起来，才覆盖 MoE 训练里权重梯度的完整归约树：
  `专家内不等长收缩(k_grouped) → 跨 rank/chunk 归约(bmk,bnk->mn)`。
- **它的设计取舍是「用小 `m,n` 换大 `b`」**：省掉 `[b,m,n]` 中间量与一次归约 pass，
  代价是 fp32 原子归约的**非确定性**与小 `m·n` 时的**竞争**。
- **SM100 把它从「软件原子加」升级成「硬件 TMA reduce-add」**：同样的数学、
  同样的 split-K 调度，但 D 的归约交给 TMA（`:255`），这是 Hopper→Blackwell
  在「归约类 GEMM」上最直接的一处硬化。
- **边界**：本机没有 SM100，SM100 分支仅源码分析；`bmk,bnk->mn` 的 fp32 结果不可
  逐位复现；`fp8_einsum` 的 `bhd,bhr->hdr`（FP8 跨 batch wgrad）在 sm90 不可用
  （`einsum.hpp:287` 要求 arch 10），留待后续（需 SM100）。

## 附录：实验复现

```
bash code/mega-moe/harness/exp_einsum_bmk_bnk.sh > exp_einsum_bmk_bnk.out.txt
```

环境：H100 80GB HBM3 / sm90 / 132 SM，CUDA 13.2，PyTorch 2.12，DeepGEMM 公开版
`057ca59`（editable 安装）。实验脚本同步放在 `harness/`，输出见 `*.out.txt`。
