# Mega mHC 反向在目标卡上的算子接线推演：per-token 融合 worker 与「省掉的中间张量」

> 本专题第 130 篇把 **Mega mHC（HyperConnection）变换的反向数学**补齐了（残差多流混合、
> Sinkhorn 迭代 VJP、`g` 耦合项、RMSNorm，并证明手写 VJP ≡ autograd）；第 131 篇换**工程视角**
> 把这条反向拆成**六段**（K0/G0/K1/G1a/G1b/K2）做了融合/roofline 账，发现整条反向 **AI≈6.5**、
> **访存受限**，并在结尾点了一个**目标卡级**的接线问题：
>
> > 「三段元素级按前向 `sm100_mega_mhc.cuh` 的 per-token worker 对偶融进 epilogue
> > （同款 `shfl_xor` 做 Sinkhorn，把 G0 的 TMEM 累加器直接喂 K1 省一次 `u0` 往返）。」
>
> 本篇就把这句话**坐实成可核对的源码接线 + 一个 H100 上真跑的 per-token 融合 worker**。
> 主 kernel（`sm100_mega_mhc.cuh`）是 **SM100（目标卡）专属、且只有前向**，反向在 DeepGEMM 里
> **不存在**——所以「接线」部分是**源码精读 + 推演，仅分析**；但「融合省掉哪些全局中间张量、
> 省多少、真机收益几何」是**架构无关**的，本篇给真实数字。
>
> 参考版本：`~/github/DeepGEMM` 公开版 commit `057ca59`（v2.8.1）；本机 `kernel_lab`（H100 sm90）。
> 实测：`harness/exp_mhc_backward_worker.{sh,py,out.txt}` + `src/mhc_backward_worker.cu`。
> 目标卡代号敏感：全文一律用「目标卡」指代。

---

## 1. 前向 mHC 的机器：反向接线的**事实底座**

反向的每一段都要落回前向的某个部件，所以先把前向三件东西的真实行号钉住：
**残差混合（Post worker）**、**系数投影（TF32 UMMA）**、**系数 epilogue（Sinkhorn/RMSNorm）**。

### 1.1 残差多流混合：`Post` worker 一个 warp 管一段 hidden（`impls:482-508`）

前向每 token 先算 `new_residual = post ⊙ x + comb @ residual`（`impls:488` 的 `x*post`、
`:494` 的 `comb@residual` 用 `__ffma2_rn` 累进），随后**立刻 round 成 bf16**
（`:497`，注释 `:496` 明说「this BF16 round is the shared numerical boundary for all consumers」）。
这一 round 是第 130 篇让反向用 straight-through 的那条边界。混合结果既写 smem
（`:499-500`）又当场累进 `hc_norm_sqr_sums`（`:501`，即 `q = Σ A²` 的**融合点**），
shifted 时再累一条 `x1`（`:502-506`、`:520-523`）。

### 1.2 系数投影：一条 **TF32 TS UMMA**（`impls:235-238 / 273-275`）

`z = A @ fnᵀ` 用 `SM100_MMA_TF32_TS<tfloat32, tfloat32, float, BLOCK_M, BLOCK_N>`（`impls:235-236`），
A（混合后的残差）在 **TMEM**、B（`fn`，全 batch 共享的投影权重）在 **SMEM**：
`umma_t::fma(tmem_base + a_stage_idx*…, b_desc, tmem_base + kAccumTmemStartColumn + …)`（`:273-275`）。
A 由 Post worker 通过 `SM100_TMEM_STORE_16dp256b4x::copy` 写进 TMEM（`:512-515`）。
之所以走 **TF32 而不是 BF16**，是因为系数投影要求高于 bf16 的尾数（第 5 篇结论）。

### 1.3 系数 epilogue：Sinkhorn 与 RMSNorm 全在 **warp 内** 做（`epilogue:17-101`）

`run_mix_task`（`epilogue:17`）把「split 归约 → 仿射/sigmoid → Sinkhorn」全压进一个 warp：

- split 归约：`hc_norm_sqr_sum`（`:27-33` 的 `warp_reduce_sum`）、`gemm_partials`（`:36-42`），
  得到 `hc_rms_scale = rsqrtf(q/(hc*H)+eps)`（`:43`）；
- 仿射 + sigmoid：`affine_value = hc_output*scale[seg]+bias`（`:53`），`sigmoid = fast_rcp(1+exp(-z))`
  （`:55`），pre 项 `+hc_pre_eps` / post 项 `*hc_post_scale`（`:56`）；
- **Sinkhorn**（`:63-86`）：`warp_reduce_max` 求行 max（`:66`）→ `exp`（`:67`）→ `warp_reduce_sum`
  求行和（`:68`）→ `+sinkhorn_eps`（`:69`）→ **列和用两次 `__shfl_xor_sync`**（`:71-76`，
  先合并相邻行、再合并两组行对）→ 迭代（`:80-83`）；
- 输出 `normalized_comb`（`:88`）写 `new_comb_res_mix` 等（`:91-96`）。

RMSNorm 则在 `run_normal_worker`（`epilogue:141`）里与 Mix+Pre **融成 per-token worker**：
`rms_scale = rsqrtf(Σsq*kInvHidden+eps)*scale`（`:216-220`），配 `normalize_bf16_pair`
（`:112-113`）直接产出 bf16/FP8。**这三段（Sinkhorn / 归一化 / RMSNorm）是反向里要逆放的部分。**

---

## 2. 反向六段 → 目标卡接线的**对偶表**

第 131 篇的六段（K0 重算混合 → G0 `u0=A@fnᵀ` → K1 系数反向 → G1a `dz=du0@fn` /
G1b `dfn=du0ᵀ@A` → K2 混合反向）与上面前向机器的对偶关系：

| 反向段 | 形状角色 | 前向对偶部件 | 目标卡原语（仅分析） |
|---|---|---|---|
| K0 `A = post⊙x + comb@R` | 元素级 | Post worker `impls:482-508` | 同一 warp-pair 模式，`__ffma2_rn` |
| G0 `u0 = A @ fnᵀ` | dgrad（m=T,n=Nout,k=D） | TF32 UMMA `impls:273-275` | **同一条 `SM100_MMA_TF32_TS`**，A 在 TMEM、B=`fn` 在 SMEM |
| K1 系数反向 | 元素级 + Sinkhorn VJP | `epilogue:17-101` | per-token worker + 两次 `shfl_xor` 逆放 |
| G1a `dz = du0 @ fn` | dgrad（m=T,n=D,k=Nout） | 无（前向无此向） | `SM100_MMA_TF32_TS` 或 BF16，A=`du0` 在 TMEM |
| G1b `dfn = du0ᵀ @ A` | wgrad（m=Nout,n=D,k=T） | 无 | 标准 UMMA（**跨 token 归约**，必须落全局累加） |
| K2 `dA = dAbase+dz → dx/dR/dpost/dcomb` | 元素级 | Post worker 反向 | 同款 warp-pair |

**两条可直接复用的接线**（这是第 131 篇那句话的精确含义）：

1. **G0 直接复用前向的同一条 TF32 TS UMMA**：前向已经在算 `A @ fnᵀ`（`impls:273-275`），
   反向 G0 是同一个算子——只是把 `fn` 转置的角色换成「B 描述符换个主序」。**A 本来就在 TMEM**，
   若把反向也接成「K0 的 Post worker 把 A 写 TMEM → UMMA 出 `u0`」，就**省掉 `A` 与 `u0` 各一次
   全局往返**。
2. **K1 的 Sinkhorn VJP 复用同一套 `shfl_xor` 列和/行和**（`epilogue:63-86`）：前向的
   `get_col_sum` 就是两次 XOR（`:73-74`），反向的每步「除以和」要同时改分子与分母两路
   （第 130 篇推的 `dX=dY/(S+ε)−Σ(dY·X)/(S+ε)²`）——**同一 warp 布局，逆序放**。

---

## 3. 融合的**边界**：什么必须落全局

不是所有中间量都能留在片上。判据只有两条：

- **必须跨 token 归约** → 落全局。反面例子：G1b `dfn = Σ_t du0[t,k]·A[t,d]`
  （wgrad，收缩维是 token），一个 block 看不到别的 token，**只能落全局累加**；
- **必须喂给独立 GEMM** → 落全局。反面例子：`du0[T,Nout]` 要喂 G1a（dgrad）与 G1b（wgrad），
  所以 `du0` 仍写全局。

**可以留在片上的**：`dAbase[T,D]`（K1 产、K2 用的元素级量）与 **G1a 的 `dz[T,D]`**
（`du0@fn` 的 dgrad，若跟着 K1 一起融进同一个 block，就不必单独落全局再读回）。
这正是第 131 篇那句「**省一次 `u0` 往返**」在同一侧的推广：**把 K1、G1a、K2 融成一个
per-token worker，省掉 `dAbase` 与 `dz` 两张 `[T,D]` 全局中间张量**。

---

## 4. H100 实测：per-token 融合 worker（`src/mhc_backward_worker.cu`）

我在 `src/mhc_backward_worker.cu` 里实现了这个融合 worker：**一个 block 处理一个 token**，
线程 = hidden 列（H≤64），K1 的 24 维系数反向由 thread 0 完成、`dz` 在 block 内用
`Σ_k du0[k]·fn[k·D+j·H+h]` 现算，K2 的混合反向紧接其后——**全程不写 `dAbase`、不写 `dz`**。
六段参考路径与 host double 参考全部沿用第 131 篇的 `src/mhc_backward_fused.cu`，保证对拍一致。

### 4.1 正确性：融合 worker ≡ 六段 ≡ host double

fp32 device vs host 单片 double 参考，6 组形状（normal/shifted），最坏 relerr：

| 形状 | 六段（P121） | **融合 worker（P122）** |
|---|---|---|
| T=64 H=8 it=3 | 1.99e-7 | 1.99e-7 |
| T=128 H=32 it=4 | 2.85e-7 | 2.85e-7 |
| T=256 H=64 it=2 | 5.92e-7 | 5.28e-7 |
| T=256 H=64 it=5 | 4.90e-7 | 5.28e-7 |

两者**同量级、同 worst 张量族**（dfn/dscales/dshift），坐实融合**没改数值语义**。

### 4.2 计时：H=64 上 ~5% 墙钟收益（被串行 Sinkhorn 压制）

`T=65536 H=64`（元素级 K1+G1a+K2，不含 G0/G1b 两个真实 GEMM）：

| 路径 | 时间 |
|---|---|
| 分段 K1 + dz + K2（4 个 kernel + reduce） | 6000.3 us |
| **融合 per-token worker**（1 个 kernel + reduce） | **5702.8 us** |

只快 ~5%。**为什么不是省下那么多字节的倍数**：这个形状 `D=hc*H=256` 太小，
K1 里**串行的 Sinkhorn**（thread 0 跑 `2*iters` 轮小矩阵迭代、其余线程空等）就是
`3814 us`，占分段路径的 **~64%**（见 `out.txt` 的 `K1 coeff 3813.5us @56 GB/s`、
`K1 -sinkhorn 1394.8us` 的对照）。**融合省的是访存，不是 Sinkhorn**。

### 4.3 真实形状的账：省掉 ~7.5 GB 的中间张量往返

融合省掉的全局往返 = `dAbase[T,D]` 与 `dz[T,D]` 各「写 + 读」= `4·T·D·4` B。逐形状：

| T | H | D=hc·H | 省掉 | 按 H100 实测带宽投影 |
|---|---|---|---|---|
| 2048 | 512 | 2048 | 0.067 GB | 73.8 us |
| 8192 | 2048 | 8192 | 1.074 GB | 1181 us |
| **16384** | **7168** | **28672** | **7.516 GB** | **8271 us** |

（带宽用 `h100_sm90` 上 1 GiB D2D copy 实测 **908.8 GB/s**；本机为共享环境，非峰值，
故投影是**保守下界**。）真实形状下省掉 **7.52 GB** —— 与第 131 篇「整条反向访存下界
3648.9 us」相对照，这是**纯属分段引入、与 GEMM 无关**的额外流量。

---

## 5. 目标卡接线推演（**仅分析**）

把 §2 的对偶表落成一个**单 mega-kernel 的反向**（目标卡 `sm100_mega_mhc.cuh` 的镜像），
需要三处接线：

1. **G0/G1a 复用 TF32 TS UMMA**（`impls:235-238 / 273-275`）：`u0` 的累加器直接在 TMEM，
   K1 worker 用 `SM100_TMEM_LOAD_*`（前向 `impls:602-607` 就是这套）读回——**省 `u0` 往返**；
2. **K1 的 Sinkhorn VJP 用前向同款 `get_col_sum`**（`epilogue:71-76`）：把前向的
   `sinkhorn_normalize`（`:63-86`）换成「逆序放 + 分母项」，warp 布局不变；
3. **K2 复用 Post worker 的反向**（`impls:482-508` 的对偶）：`x/R/post/comb` 已在片上，
   `dpost/dcomb` 用 `block/warp` 归约、`dR` 每 h 局部。

**必须留在全局的**：G1b 的 `dfn`（wgrad，跨 token 收缩）、以及 `du0`（喂 G1a/G1b）。
**边界由此闭合**：§4 的融合 worker 恰好只覆盖「K1+G1a+K2」，与这条判据一致。

**为什么本机只能给到 H=64 的微基准**：融合 worker 一个 block 一 token、每线程一列 hidden，
`MAXH=64`。真实形状 `H=7168` 下需要「一个 block 跨多段 hidden / 一个 warp-pair 管一段」
（正是前向 Post worker 的做法 `impls:482-508`），那部分要落到目标卡的 TMEM/mbarrier 接线，
本机 H100（sm90）**不跑**——故标 **SM100 专属，仅分析**。

---

## 6. 结论与关系

- 第 130 篇给了 mHC 反向的**数学**、第 131 篇给了**roofline 账**，本篇补上**接线**：
  反向的每一段都能落回前向一个**真实存在的部件**（Post worker / TF32 TS UMMA / Sinkhorn warp），
  并给出「什么必须落全局」的**两条判据**。
- **可验证的那半**（融合省掉的中间张量）在 H100 上真跑：融合 worker 与六段/host double
  **同量级**（最坏 5.28e-7），H=64 墙钟快 ~5%（Sinkhorn-bound），真实形状省 **7.52 GB**。
- **仅分析的那半**（目标卡单 kernel 接线）把第 131 篇的一句 TODO 展开成三处可核对的接线点，
  行号均已在 commit `057ca59` 上核对。

> 与第 4/127 篇（MoE router / Mega Gate 反向）同构：三条门控反向都共享「硬 top-k 对梯度透明、
> 归一化/归一化族 Jacobian 跨槽耦合」这套结构；mHC 只是把「槽」换成了 **Sinkhorn 的行/列**。
