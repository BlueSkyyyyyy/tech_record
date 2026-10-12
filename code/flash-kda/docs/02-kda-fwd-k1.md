# KDA 前向 K1（prepare）逐阶段精读

> 本文是「Kimi Delta Attention 前向 / 反向算子实现解析」的第 2 篇，对应路线图 **P2-1**。
> 逐阶段拆解 FlashKDA 前向的第一个 kernel **K1 = `_flash_kda_fwd_prepare`**（`csrc/smxx/fwd_kernel1.cuh`）：
> TMA 载入 → QK L2 归一化 → 门激活 + cumsum → decay（构造 $k^{\text{dec}}/q^{\text{dec}}/k^{\text{inv}}/k^{\text{rest}}$
> 与 $E^{\text{tot}}$）→ 构造 $L$ / $M_{qk}$ → 块内求逆 → 写 workspace。
> 公式记号沿用第 1 篇 `01-kda-theory.md` §3；行号均为 `~/github/FlashKDA` 下**核对过**的真实行号。

## 0. K1 的职责、形状与 launch 配置

K1 是**单次 TMA load、无 pipeline** 的短 kernel：一个 CTA 处理一个 `(head, tile)`，把
$K=16$ 个 token 的块预处理成 K2 递推所需的中间件，全部写进 workspace。它**不做递推、不读状态**，
所以同一块内的 token 互相独立，天然 token 并行。

| 项 | 值 | 出处 |
|---|---|---|
| kernel 入口 | `_flash_kda_fwd_prepare` | `fwd_kernel1.cuh:120` |
| grid | `dim3(total_tiles, H)` | `fwd_launch.cu:169` |
| block | `256` 线程，`__launch_bounds__(NumThreads, 8)` | `fwd_launch.cu:149`、`fwd_kernel1.cuh:120` |
| 动态 smem | `sizeof(SharedStorageK1<K1L>)`，用 `cudaFuncSetAttribute` 放开 | `fwd_launch.cu:151–162` |
| 输入 | `q,k,g`（bf16 $[H,T,D]$）、`beta`（bf16 1D，host 已转置 `[H*T]`）、`dt_bias`（fp32 $[H,D]$）、`A_log`（fp32 $[H]$） | `flash_kda.cpp:130–132`、`fwd_launch.cu:50–96` |
| 输出（workspace） | `k_decayed,q_decayed,k_restored`（bf16 $[H{\cdot}tiles,16,128]$）、`g_total`（fp32 $[H{\cdot}tiles,128]$）、`INV,Mqk`（bf16 $[H{\cdot}tiles,16,16]$） | `fwd_launch.cu:64–84` |
| 模板参数 | `<D, CHUNK, NumThreads, IsVarlen>`；host 侧 `D=128, CHUNK=16` | `fwd_kernel1.cuh:115–118`、`flash_kda.cpp:10–11` |

`IsVarlen` 是编译期参数，决定 tile 定位走整除还是二分（见 §1）；host 侧据 `cu_seqlens` 是否存在
实例化两套（`flash_kda.cpp:184–213`）。

> **为什么 K1 只有一个 TMA load、没有 pipeline？** 它每个 CTA 只干一块活，没有跨 tile 的复用，
> 流水线的 stage 收益为零；与之相对 K2 一个 CTA 串完整个序列，必须靠 3-stage 载入流水掩盖访存
> （第 3 篇讲）。把 K1 做成短 kernel 也让它的 token 并行度能塞满 SM。

## 1. 阶段 ①：tile 定位（`fwd_kernel1.cuh:168–199`）

每个 CTA 先算出自己负责的 `(seq_idx, local_t)`：

- **非 varlen**（`fwd_kernel1.cuh:187–195`）：每个序列长度都等于 `T_seq = T_total/N`，
  `tiles_per_seq = ceil(T_seq/16)`，直接除。此时 `blocks = N * tiles_per_seq` 是**精确**的
  （`flash_kda.cpp:180`），没有废 CTA。
- **varlen**（`fwd_kernel1.cuh:175–186`）：长度不等，host 无法预知每个序列几块，于是用
  `total_tiles = ceil(T_total/16) + N` 作**上界**（`flash_kda.cpp:178`），多出的 CTA 在
  `fwd_kernel1.cuh:199` 的 `if (local_t >= t_tiles_this_seq) return;` 提前退出。
  `global_tile_idx → seq_idx` 用预生成的 `tile_prefix`（每序列累计 tile 数）做 **$O(\log N)$ 二分**
  （`fwd_kernel1.cuh:176–186`），而不是每个 CTA 线性扫 `cu_seqlens`。
  `tile_prefix` 由独立的小 kernel `_flash_kda_build_tile_prefix`（`fwd_kernel1.cuh:89–104`）在
  K1 之前跑一次写好（`fwd_launch.cu:164–167`），只有 32 个线程、单 CTA。

**为什么要把上界单独处理？** varlen 下 `B=1`、`cu_seqlens` 给的是「拼接后」的序列边界，
host 侧不跑一遍前缀和就无法把 `global_tile_idx` 映射回序列；把前缀和放进 GPU 小 kernel，
避免了 host-device 同步。

## 2. 阶段 ②：单次 TMA 载入（`fwd_kernel1.cuh:200–263`）

只有 `threadIdx.x == 0` 发起 TMA（`fwd_kernel1.cuh:202`），用 cluster transaction barrier 同步：

1. `tma_load_barrier.init(1)` + `fence_barrier_init()`（`fwd_kernel1.cuh:204–205`），
   再 `arrive_and_expect_tx(kTmaTransactionBytes)`（`fwd_kernel1.cuh:206`）声明本次要等多少字节。
2. 依次发起 5 个 copy：`q`、`k`、`beta`、`g_bf16`、`dt_bias`（`fwd_kernel1.cuh:231–254`）。
3. 所有线程在 `fwd_kernel1.cuh:260–263`：`__syncthreads()` → `tma_load_barrier.wait(0)` →
   `fence_view_async_shared()` → `__syncthreads()`。前一个 sync 保证 barrier 的 init 对全部线程可见，
   后一个保证 TMA 数据对后续 generic 访问可见。

`kTmaTransactionBytes`（`fwd_kernel1.cuh:158–161`）=
`cosize(QKLayout)*(3*2) + 32*2 + D*4`，对应 `q + k + g_bf16`（3 个 $16{\times}128$ bf16）、
`beta`（32 bf16）、`dt_bias`（128 fp32）。

**两个容易忽略的细节：**

- **地址偏移复用**：`q` 与 `k` 的 gmem 布局相同，于是只算一次 `qk_off`
  （`fwd_kernel1.cuh:216`，`g_q.layout()(head_idx, bos+local_t*16, 0)`），k 直接复用
  （`fwd_kernel1.cuh:220`）。`g_bf16` 也用同一 `qk_off`（`fwd_kernel1.cuh:241`）。
- **beta 的 8 元素对齐**（`fwd_kernel1.cuh:222–225`）：beta 是 `[H*T]` 的 1D 数组，1D TMA 的
  最小传输粒度是 8 个 bf16，所以起点按 `beta_linear & ~7` 下取整；真正第 0 个 token 的偏移记在
  `beta_smem_offset = (head*T_total + bos + local_t*16) & 7`（`fwd_kernel1.cuh:345`），
  后面取 beta 时加上它（`fwd_kernel1.cuh:502`）。**为什么要转置成 `[H,T]`？** 若保持 `[T,H]`，
  同一 head 的 beta 在行方向 stride=H，无法满足 1D TMA 的对齐/连续要求；转置后 `[H,T]` 连续，
  正好一次 1D copy 取到一块 token。

## 3. 阶段 ③：QK L2 归一化（`fwd_kernel1.cuh:265–303`）

对每个 token 的 $q,k$（各 128 维）做 $\ell_2$ 归一化，映射到 **16 线程/行、每线程 8 元素**
（`fwd_kernel1.cuh:268–271`：`ELEMS_PER_THREAD=8`、`THREADS_PER_ROW=16`）：

1. 每线程把自己的 8 个 bf16 转 fp32，边转边累加平方和 `q_sq/k_sq`（`fwd_kernel1.cuh:279–287`）。
2. 5 步 `__shfl_xor_sync` 做 warp 内**树形归约**（`fwd_kernel1.cuh:289–293`，`delta` 从 8 到 1）。
   注意一行 128 元素跨 16 个 lane，16 个 lane 恰好在半个 warp 内，`xor` 8/4/2/1 正好聚合一行、
   不串行。**为什么用树形而不是顺序累加？** 归一化只是 scale，顺序无关紧要，但树形让所有 lane
   拿到同一结果、且无 bank 冲突。
3. `rsqrtf(x + 1e-6f)`（`fwd_kernel1.cuh:295–296`）防零向量，再逐元素乘回、写回 bf16
   （`fwd_kernel1.cuh:298–302`）。

> 与 `torch_ref.py` 的 `l2_normalize_kernel_match`（`01-kda-theory.md` §2 提到的）保持同一归约
> 顺序，是为了 host 参照实现能和 kernel 的浮点结果对齐——归一化顺序不同会让 $q,k$ 末位有差，
> 进而在对拍时制造假误差。

## 4. 阶段 ④：门激活 + cumsum + 尾部置零（`fwd_kernel1.cuh:306–338`）

这一段用**列并行、不做跨行同步**来天然实现前缀和（`fwd_kernel1.cuh:309–338`）：

- **线程 0–127**（一路一个通道 `col = compute_tid`）：对 16 行循环，取该行该通道的
  raw gate bf16，加 `dt_bias`，乘 `a_log_exp`，再乘 `gate_scale` 后过 sigmoid
  （`fwd_kernel1.cuh:321–323`）：
  $$
  g_{\text{row,col}} = \underbrace{(\text{lower\_bound}\cdot\log_2 e)}_{\text{gate\_scale}}
  \;\sigma\big(e^{A\_log}\cdot(\text{raw\_g}+\text{dt\_bias})\big).
  $$
  `gate_scale = lower_bound * 1.4426950408889634`（$\log_2 e$）在 host 算好传进来
  （`flash_kda.cpp:128`），于是 $g$ 直接是 **base-2 log**，后面 `ex2` 免换底。
  sigmoid 用 `sigmoid_tanh_approx_f32`（`utils.cuh:50–53`，`0.5*tanh(x/2)+0.5`，PTX `tanh.approx.f32`），
  比 `__expf`+除法快得多，精度对门控足够。
  循环内 `sum += g_val; g_smem[row*D+col] = sum;`（`fwd_kernel1.cuh:327–328`）——**因为每个通道
  只由一个线程负责，行之间的累加天然串行**，无需任何同步就在 smem 里写出累积前缀 $g^{\text{cum}}$。
  末行的 `sum` 就是整块累积 $g^{\text{total}}$，写进 `g_total[col]`（`fwd_kernel1.cuh:330`）。
  越界行（`row >= actual_len`）门取 0（`fwd_kernel1.cuh:324–326`），避免污染前缀和。
- **线程 128–255**：把 k 的**尾部越界行**清零（`fwd_kernel1.cuh:331–337`），
  这样尾部 token 对 $L/M_{qk}$ 的贡献为 0，等价于块内填充。

**为什么 `actual_len` 在这里算？** `actual_len = min(16, seq_len - local_t*16)`（`fwd_kernel1.cuh:310`）
只有尾块 < 16，非尾块恒为 16。把 padding 处理放在 K1（一次性），K2 的递推就能对整块无分支地跑。

## 5. 阶段 ⑤：$E^{\text{tot}}$ 与 decay_apply（`fwd_kernel1.cuh:353–476`）

先把 `g_total` 原地取 $2^{g^{\text{total}}} = E^{\text{tot}}$（`fwd_kernel1.cuh:354–357`，
`ex2_approx_ftz_f32`，`utils.cuh:38–42`），然后构造四个衰减变体
（对照 `01-kda-theory.md` §3 ②，`fwd_kernel1.cuh:449–469`）：

| 输出 | 公式 | 代码 |
|---|---|---|
| `q_decayed` | $q_i\,2^{g^{\text{cum}}_i}\cdot\text{scale}$ | `fwd_kernel1.cuh:454` |
| `k_decayed` | $k_i\,2^{g^{\text{cum}}_i}$ | `fwd_kernel1.cuh:455` |
| `k_inv` | $k_i\,2^{-g^{\text{cum}}_i}$ | `fwd_kernel1.cuh:466–467` |
| `k_restored` | $k_i\,2^{-g^{\text{cum}}_i}\,E^{\text{tot}}$ | `fwd_kernel1.cuh:468` |

**这里最关键的是 smem union 复用与「先入寄存器再覆写」的时序**：

- Phase A 的 `{q, k, g}` 与 Phase B 的 `{k_decayed, q_decayed, k_inv, L, INV, Mqk}` 共用同一块
  smem（`fwd_kernel1.cuh:57–71` 的 `union`，注释说省 ~14KB）。因为 decay 要**同时读 q/k/g、
  写 q_decayed/k_decayed/k_inv**，而目标地址就是源地址，所以代码先把所有源数据搬进**寄存器**
  （`fwd_kernel1.cuh:384–421`），`__syncthreads()`（`fwd_kernel1.cuh:425`），再整块覆写
  （`fwd_kernel1.cuh:427–473`）。这就是「先读后写、中间一次 sync」的由来。
- 线程映射（`fwd_kernel1.cuh:365–368`）：`g = lane/4`（0–7）、`t = lane%4`（0–3）；
  8 个 warps 的 `(warp_id+g)%8` 恰好覆盖 8 行，`t` 覆盖 8 列里的 2 列（`thr2_2d`，
  `fwd_kernel1.cuh:370–401`），于是 32 lane 摊满一个 $8\times64$ 子块、每线程 2 个元素。
  $16\times128$ 块共 4 个子块（`N_M=2, N_N=2`，`fwd_kernel1.cuh:375–377`），循环 4 次。
- `g_bf16` 与 `k_restored` 也做成 union（`fwd_kernel1.cuh:75–78`）：`g_bf16` 在阶段 ④ 读完后死掉，
  随后 decay 才写 `k_restored`。`dt_bias` 与 `g_total` 同理（`fwd_kernel1.cuh:79–82`）。

> 精度上，`q/k` 的衰减乘子都在 **bf16** 里算（`exp_cumsum` 先 cast 到 bf16 再乘，
> `fwd_kernel1.cuh:453–455`），只有门/前缀和保持 fp32。这是精度取舍：衰减乘子相对大小
> 由 fp32 gate 决定，乘进 bf16 的 q/k 与其本来的 bf16 精度同量级。

## 6. 阶段 ⑥：$L$ 与 $M_{qk}$ 两个 GEMM（`fwd_kernel1.cuh:478–487`）

```
L_fp32 = k_decayed @ k_invᵀ      (fp32 累加)
Mqk    = q_decayed @ k_invᵀ      (bf16 累加)
```

- warp 0（`compute_tid < 32`）调 `mma_m16n16_bf16bf16fp32_1warp`（`utils.cuh:167–192`）算
  $L=k^{\text{dec}}\,{k^{\text{inv}}}^\top$，**fp32 累加器**——因为 $L$ 马上要求逆，保留 fp32
  精度。
- warp 1（`32 ≤ compute_tid < 64`）调 `mma_m16n16_bf16bf16bf16_1warp`（`utils.cuh:145–165`）算
  $M_{qk}=q^{\text{dec}}\,{k^{\text{inv}}}^\top$，**bf16 累加并直接存 bf16**——它只参与输出投影，
  精度要求低。
- 两个都基于 `SM80_16x8x16_F32BF16BF16F32_TN`（`utils.cuh:153/175`），即 **TN**：A 行主、
  B 列主，所以 `cooperative_gemm` 算的是 $A B^\top$。这正是我们要的「$i,j$ 点积」形式：
  $L_{ij}=\langle k^{\text{dec}}_i, k^{\text{inv}}_j\rangle$。
- 只有 2 个 warp 在算，其余 6 个 warp 空转——这是短 kernel，无所谓。

**为什么 $L$ 的对角/上三角之后要补？** $\langle k^{\text{dec}}_i,k^{\text{inv}}_i\rangle
= k_i^2\,2^{g^{\text{cum}}_i-g^{\text{cum}}_i}=k_i^2$，不为零；但分块公式里 $L$ 必须是
**严格下三角**（对角线归 1，体现在 $I+L$ 的 $I$）。所以下一步要 mask。

## 7. 阶段 ⑦：tril + beta 掩码（`fwd_kernel1.cuh:489–508`）

$16\times16=256$ 个元素，正好 256 线程一人一个（`fwd_kernel1.cuh:494–498` 用列分块
`col_block_size=8` 把 `(i,j)` 解出来）：

- $i\le j$：`L_fp32(i,j)=0`（`fwd_kernel1.cuh:499–500`）——严格下三角，对角清零。
- $i>j$：`L_fp32(i,j) *= sigmoid_tanh_approx_f32(beta_i)`（`fwd_kernel1.cuh:501–503`）——
  把写入门 $\beta_i=\sigma(\text{beta\_logit}_i)$ 乘进去。注意用 `beta_smem_offset` 定位到对齐
  载入里的正确 token（`fwd_kernel1.cuh:502`）。
- $i<j$：`Mqk(i,j)=0`（`fwd_kernel1.cuh:504–506`）——$M_{qk}$ 抹上三角，保留**下三角含对角**。

同时把 $L$ 的 **bf16 版本当作合并矩阵 $M$ 的载体**：`M_bf16` 直接复用已死的 `k_inv` buffer
（`fwd_kernel1.cuh:490–491`），交给求逆例程内部按块填 $C$（见下）。

## 8. 阶段 ⑧：块内求逆 `inv_fwd_subst_fused_1warp`（`fwd_kernel1.cuh:510–511`，`utils.cuh:213–369`）

要求 $P=(I+L)^{-1}$，$I+L$ 是 $16\times16$ **单位下三角**（$L$ 严格下三角）。直接求逆不如
按 $8\times8$ 分块做 Schur 补：

$$
X = I+L = \begin{bmatrix} A & 0 \\ C & B\end{bmatrix},\quad
X^{-1} = \begin{bmatrix} A^{-1} & 0 \\ -B^{-1} C A^{-1} & B^{-1}\end{bmatrix},
$$

其中 $A,B$ 是 $8\times8$ 单位下三角，$C=L[8{:}16,0{:}8]$。例程分三步：

**(a) 两个 $8\times8$ 对角块的 fp32 前代求逆**（`utils.cuh:229–252`）。
lane $i$（`i = tid & 7`）拥有 block 内第 $i$ 行，`inv[p]` 初始化为 $L$ 的严格下三角 + 对角 1
（`utils.cuh:238–242`）。然后 $s=0\ldots6$：`row_scale = -inv[s]`，对 $p<s$，
`pivot = __shfl_sync(..., inv[p], (tid & ~7) | s)` 把已定型的第 $s$ 行从它的 owner lane 广播出来，
`if (i > s) inv[p] = fmaf(row_scale, pivot, inv[p])`，最后 `inv[s]=row_scale`
（`utils.cuh:243–252`）。这就是**逐行消元的 forward substitution**：处理完第 $s$ 行后，
第 $i>s$ 行被消成 $A^{-1}$ 的第 $i$ 行。8 个 lane 一组，`row0 = tid & 8` 决定做 block A
（rows 0–7）还是 B（rows 8–15）；groups 0/2 做 A、1/3 做 B，冗余拷贝让所有 shuffle 收敛
（`utils.cuh:232–242` 注释）。

**(b) 组装 $P$ 与 $M$**（`utils.cuh:254–277`）：group 0 把 $A^{-1}$ 写进 `INV_bf16` 左上，
group 1 把 $B^{-1}$ 写进右下，group 2/3 把 $M=[0\,0;C\,0]$ 写进 `M_bf16`（$C$ 由 fp32 $L$ 现 cast 到 bf16）。
所以 $P=[A^{-1}0;0B^{-1}]$、$M=[00;C0]$，量化为 bf16 只发生在 HMMA 输入。

**(c) 两次 bf16 HMMA + 一次元素加**（`utils.cuh:298–368`）：
$$
dc = P\,M \;(\text{=}[0\,0;B^{-1}C\,0]),\qquad
o = \text{bf16}(-dc)\,P \;(\text{=}[0\,0;-B^{-1}CA^{-1}\,0]),\qquad
INV = P + \text{bf16}(o).
$$
用 `mma_16x16`（两个 `SM80_16x8x16` 沿 N 拼，`utils.cuh:311–315`），
`transpose_u32x4` 用 `MOVM_T`（`utils.cuh:304–309`）把 A 片段直接当 B 操作数，省掉 smem 往返。
最后 `__hadd2` 把两块不相交的 bf16 结果相加（`utils.cuh:343–355`），STSM 存回 `INV`（`utils.cuh:357–368`）。

**为什么不用 Neumann 级数？** 旧实现 $(I-L)(I+L^2)(I+L^4)(I+L^8)$ 在高相关 key（$|L|\to1$）时
$L^k$ 中间量可达 $10^3$、fp16 里灾难性相消；前代 + Schur 补从不构造 $L^k$，对近共线 key 也稳
（`utils.cuh:194–212` 注释）。这也是 K1 里最烧脑、也最值得写进文档的一段。
块内求逆的代数正确性有独立自测：`harness/check_inv_identity.py`（见 §10）。

## 9. 阶段 ⑨：写 workspace（`fwd_kernel1.cuh:513–585`）

`fence_view_async_shared()` + `__syncthreads()`（`fwd_kernel1.cuh:513–514`）保证求逆结果对
async proxy 可见后，`threadIdx.x == 0` 连发 **6 个 TMA store**（`fwd_kernel1.cuh:515–583`）：

| 字段 | 布局 | 行号 |
|---|---|---|
| `k_decayed` | $[H{\cdot}tiles,16,128]$ bf16 | `fwd_kernel1.cuh:517–527` |
| `q_decayed` | 同上 | `fwd_kernel1.cuh:528–538` |
| `k_restored` | 同上 | `fwd_kernel1.cuh:539–549` |
| `g_total` | $[H{\cdot}tiles,128]$ fp32 | `fwd_kernel1.cuh:550–560` |
| `INV` | $[H{\cdot}tiles,16,16]$ bf16 | `fwd_kernel1.cuh:561–571` |
| `Mqk` | 同上 | `fwd_kernel1.cuh:572–582` |

每发一个就 `tma_store_arrive()`，最后 `tma_store_wait<0>()`（`fwd_kernel1.cuh:584`）等全部落地。
workspace 是 **SoA**（六个独立大数组，不是结构体数组），指针切分在 `fwd_launch.cu:64–71`，
尺寸在 `flash_kda.cpp:20`（`3*16*128*2 + 128*4 + 2*16*16*2`）与 `utils.cuh:64–77` 的
`WorkspaceSizes` 一一对应。**为什么 SoA**？每个字段能独立配 TMA descriptor（`fwd_launch.cu:98–103`）
且都能 128B 对齐（`flash_kda.cpp:16–18` 的 `static_assert`）。

`ws_idx = head_idx * total_tiles + global_tile_idx`（`fwd_kernel1.cuh:516`）——注意索引顺序是
**head 在外、tile 在内**，与 K2 一个 CTA 串一个 head 的所有 tile 的顺序一致，K2 顺序读即连续。

## 10. 行号速查 + 自测

| 阶段 | 函数/位置 | 行号 |
|---|---|---|
| 布局定义 | `K1Layouts` | `fwd_kernel1.cuh:5–42` |
| smem union | `SharedStorageK1` | `fwd_kernel1.cuh:44–84`（union `57–71`） |
| tile 定位 | `_flash_kda_fwd_prepare` | `fwd_kernel1.cuh:168–199` |
| TMA 载入 | thread0 + barrier | `fwd_kernel1.cuh:200–263` |
| QK norm | 16 线程/行 + shfl | `fwd_kernel1.cuh:265–303` |
| gate+cumsum | 列并行前缀和 | `fwd_kernel1.cuh:306–338` |
| decay_apply | 寄存器暂存 + union 覆写 | `fwd_kernel1.cuh:353–476` |
| $L/M_{qk}$ | 2 个 1-warp MMA | `fwd_kernel1.cuh:478–487` |
| tril+beta | 逐元素 mask | `fwd_kernel1.cuh:489–508` |
| 求逆 | `inv_fwd_subst_fused_1warp` | `fwd_kernel1.cuh:510–511`；`utils.cuh:213–369` |
| 写 workspace | 6×TMA store | `fwd_kernel1.cuh:513–585` |

**自测**：`harness/check_inv_identity.py` 用纯 PyTorch 复现 §8 的「8×8 前代 + Schur 补」块内求逆，
与直接解 $(I+L)^{-1}$（fp64 `torch.linalg.inv`）对比。实测（`harness/check_inv_identity.out.txt`）：
纯公式在 **fp64** 下 max abs $\le 2.2\times10^{-16}$（机器精度，代数严格正确），**fp32** 下
$\le 1.2\times10^{-7}$；把 HMMA 输入 $P/M$ 与中间量按 kernel 一样量化到 bf16 后，
max abs $\approx 2.5\times10^{-3}\!\sim\!5.9\times10^{-3}$——与 bf16 输入本身的量化误差同量级。
对近共线 key（$|L|$ 接近 1）的构造，fp64 仍 $\le5.6\times10^{-17}$，bf16 量化 $\approx1.7\times10^{-3}$，
没有 Neumann 级数那种放大。这验证了求逆例程的代数（`utils.cuh:213–369`）确实产出块内注意力矩阵的逆。

> 优化细节（smem union 尺寸、`__launch_bounds__`、`ex2.approx`/`tanh.approx` 的选择理由、
> 前代求逆的精度取舍）留到 **P2-2**；workspace 逐字段字节数与 host 对齐留到 **P2-3**（本文已给出
> 对应行号）。K2 递推见第 3 篇。

## 11. 下一步

- **P2-2** 关键优化逐条量化（union 省多少 smem、近似指令误差）。
- **P2-3** workspace 布局逐字段与 `get_workspace_size` 对齐。
- **P3** K2 精读（warp specialization + 八个 Phase）。
