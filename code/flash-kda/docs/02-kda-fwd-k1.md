# KDA 前向 K1（prepare）逐阶段精读

> 本文是 KDA 专题的第 2 篇：第 1 篇给了公式地图，这一篇把它落到代码。
> FlashKDA 前向的第一个 kernel **K1 = `_flash_kda_fwd_prepare`**（`csrc/smxx/fwd_kernel1.cuh`）用九道工序，
> 把一个 16-token 的块预处理成六样中间件写进 workspace；它不做递推、不读状态，所以天然 token 并行。

## 0. 先记住一个画面

**K1 是一间只有一张操作台的备菜间。**

它一次只接一张订单——一个 16-token 的块。生料（`q`、`k`、`g`、`beta`）整托盘搬上台，
切好、腌好、分装进六个保鲜盒（workspace），这间备菜间就下班了：
它不关心半成品被谁用。

台面很窄，只有 21.2 KB smem（§11.1），所以同一块台面要反复复用：
切完 `q/k/g`，台面立刻被改成 `k_decayed / q_decayed / k_inv`（§5）。
而"擦台面"在代码里不是一个动作，是一次 `__syncthreads()`：旧数据先搬进寄存器，sync，再整块覆写。

> **类比的边界**：备菜间擦台面是物理动作，擦没擦干净看结果就知道；K1 的 smem 复用是**地址别名**，
> 一旦"读旧值"和"写新值"重叠就是静默算错，编译器不会替你检查，全靠"先入寄存器 + 一次 sync"兜住。
> 再者，九道工序不是九个工人并行，而是**同一批 256 个线程**在换角色：阶段④里 0–127 号线程算门、
> 128–255 号线程清 `k` 的尾行。最后一处：求逆（§8）在类比里没有对应动作——它是在解一个
> 小型线性方程组，也是 K1 里最烧脑的一段。

读完这一篇，你应该能回答：

1. 九道工序里，哪几步真的在"算"，哪几步只是搬数据、整布局？
2. 为什么 smem 必须做 union，为什么 union 中间一定要有一次 `__syncthreads()`？
3. workspace 每 tile 的 13824 B 是怎么算出来的，为什么选 SoA 而不是 AoS？

### 0.1 K1 的职责、形状与 launch 配置

K1 是**单次 TMA load、无 pipeline** 的短 kernel：一个 CTA 处理一个 `(head, tile)`，
把 $C=16$ 个 token 的块预处理成 K2 递推所需的中间件，全部写进 workspace。
它**不做递推、不读状态**，所以同一块内的 token 互相独立，天然 token 并行。

| 项 | 值 | 出处 |
|---|---|---|
| kernel 入口 | `_flash_kda_fwd_prepare` | `fwd_kernel1.cuh:120` |
| grid | `dim3(total_tiles, H)` | `fwd_launch.cu:169` |
| block | `256` 线程，`__launch_bounds__(NumThreads, 8)` | `fwd_launch.cu:149`、`fwd_kernel1.cuh:120` |
| 动态 smem | `sizeof(SharedStorageK1<K1L>)`，用 `cudaFuncSetAttribute` 放开 | `fwd_launch.cu:151–162` |
| 输入 | `q,k,g`（bf16 $[H,T,D]$）、`beta`（bf16 1D，host 已转置 `[H*T]`）、`dt_bias`（fp32 $[H,D]$）、`A_log`（fp32 $[H]$） | `flash_kda.cpp:130–132`、`fwd_launch.cu:50–96` |
| 输出（workspace） | `k_decayed,q_decayed,k_restored`（bf16 $[H{\cdot}tiles,16,128]$）、`g_total`（fp32 $[H{\cdot}tiles,128]$）、`INV,Mqk`（bf16 $[H{\cdot}tiles,16,16]$） | `fwd_launch.cu:64–84` |
| 模板参数 | `<D, CHUNK, NumThreads, IsVarlen>`；host 侧 `D=128, CHUNK=16` | `fwd_kernel1.cuh:115–118`、`flash_kda.cpp:10–11` |

`IsVarlen` 是编译期参数，决定 tile 定位走整除还是二分（见 §1）；
host 侧据 `cu_seqlens` 是否存在实例化两套（`flash_kda.cpp:184–213`）。

> **为什么 K1 只有一个 TMA load、没有 pipeline？** 它每个 CTA 只干一块活，没有跨 tile 的复用，
> 流水线的 stage 收益为零；与之相对，K2 一个 CTA 串完整个序列，必须靠 3-stage 载入流水掩盖访存
> （第 3 篇讲）。把 K1 做成短 kernel，也让它的 token 并行度能塞满 SM。

### 0.2 九道工序总览

先看全景：

| 阶段 | 一句话 | 关键代码位置 | 输出 |
|---|---|---|---|
| ① tile 定位 | 算出我是哪个序列的第几块 | `fwd_kernel1.cuh:168–199` | `(seq_idx, local_t)` |
| ② TMA 载入 | 一次把 5 样输入搬进 smem | `fwd_kernel1.cuh:200–263` | smem 里的 `q,k,g_bf16,beta,dt_bias` |
| ③ QK L2 归一化 | 每行 128 维除以自己的模长 | `fwd_kernel1.cuh:265–303` | 归一化后的 `q,k` |
| ④ 门激活 + cumsum | 造 128 维 log 门，顺手做块内前缀和 | `fwd_kernel1.cuh:306–338` | `g`（$g^{\text{cum}}$，fp32）+ `g_total` |
| ⑤ decay_apply | 算出四种 q/k 变形与 $E^{\text{tot}}$ | `fwd_kernel1.cuh:353–476` | `q_decayed,k_decayed,k_inv,k_restored` |
| ⑥ 两个 GEMM | 块内注意力矩阵 $L$、$M_{qk}$ | `fwd_kernel1.cuh:478–487` | `L`（fp32）、`Mqk`（bf16） |
| ⑦ tril + $\beta$ | 清上三角，把 $\beta$ 乘进 $L$ | `fwd_kernel1.cuh:489–508` | 严格下三角的 `L`、下三角的 `Mqk`、`M_bf16` |
| ⑧ 块内求逆 | $P=(I+L)^{-1}$，Schur 补 | `fwd_kernel1.cuh:510–511`、`utils.cuh:213–369` | `INV`（bf16） |
| ⑨ 写 workspace | 6 个 TMA store 交给 K2 | `fwd_kernel1.cuh:513–585` | 6 个 SoA 数组 |

数据流的形状大致是这样，最后一步是"倒着流回显存"：

```
q,k,g,beta ──TMA──▶ ① 定位 ─▶ ② 载入 ─▶ ③ 归一化 ─▶ ④ 门+cumsum ─▶ ⑤ 四种变形
                                                                        │
       workspace ◀── ⑨ 写回 ◀── ⑧ 求逆 ◀── ⑦ tril+β ◀── ⑥ 两个 GEMM ◀──┘
```

读图时留意三个同步点：② 等齐、④ 之后、⑤ 覆写之前。

## 1. 阶段 ①：tile 定位（`fwd_kernel1.cuh:168–199`）

**每个 CTA 上来先回答"我是谁"：哪个序列、序列里的第几块。**

**为什么需要这一步。**
kernel 里只有两个 blockIdx，却要还原"这是哪个序列的第几块"：非 varlen 一次除法，
varlen 就必须查表。

**代码怎么做。**
非 varlen（`fwd_kernel1.cuh:187–195`）：每个序列长度都等于 `T_seq = T_total/N`，
`tiles_per_seq = ceil(T_seq/16)`，直接除。此时 `blocks = N * tiles_per_seq` 是**精确**的
（`flash_kda.cpp:180`），没有废 CTA。

varlen（`fwd_kernel1.cuh:175–186`）：长度不等，host 无法预知每个序列几块，于是用
`total_tiles = ceil(T_total/16) + N` 作**上界**（`flash_kda.cpp:178`），多出的 CTA 在
`fwd_kernel1.cuh:199` 的 `if (local_t >= t_tiles_this_seq) return;` 提前退出。
`global_tile_idx → seq_idx` 用预生成的 `tile_prefix`（每序列累计 tile 数）做 **$O(\log N)$ 二分**
（`fwd_kernel1.cuh:176–186`），而不是每个 CTA 线性扫 `cu_seqlens`。

`tile_prefix` 由独立的小 kernel `_flash_kda_build_tile_prefix`（`fwd_kernel1.cuh:89–104`）在
K1 之前跑一次写好（`fwd_launch.cu:164–167`），只有 32 个线程、单 CTA。

**读的时候注意。**
二分的比较是 `tile_prefix[mid] <= global_tile_idx`，所以 `lo` 落在"最后一个累计数不大于我"的序列上，
边界序列不会错位。上界带来的废 CTA 靠 early return 丢掉，**不是**靠 grid 精确——
§12.3 会再遇到同一个套路：分配用上界、launch 有时用精确值。

所以这一步只产出两个整数，却决定了后面所有 TMA 偏移。

## 2. 阶段 ②：单次 TMA 载入（`fwd_kernel1.cuh:200–263`）

**五样输入、一条 barrier、一次等齐——K1 没有流水线，所以载入阶段可以极简。**

**为什么需要这一步。**
后面每一步都在 smem 上做，所以先把 `q`、`k`、`g`、`beta`、`dt_bias` 一次搬齐。
用 TMA 而不是逐线程 `ld.global`，省指令，也让拷贝走 async proxy、不占寄存器。

**代码怎么做。**
只有 `threadIdx.x == 0` 发起 TMA（`fwd_kernel1.cuh:202`），用 cluster transaction barrier 同步：

1. `tma_load_barrier.init(1)` + `fence_barrier_init()`（`fwd_kernel1.cuh:204–205`），
   再 `arrive_and_expect_tx(kTmaTransactionBytes)`（`fwd_kernel1.cuh:206`）声明本次要等多少字节。
2. 依次发起 5 个 copy：`q`、`k`、`beta`、`g_bf16`、`dt_bias`（`fwd_kernel1.cuh:231–254`）。
3. 所有线程在 `fwd_kernel1.cuh:260–263`：`__syncthreads()` → `tma_load_barrier.wait(0)` →
   `fence_view_async_shared()` → `__syncthreads()`。

`kTmaTransactionBytes`（`fwd_kernel1.cuh:158–161`）=
`cosize(QKLayout)*(3*2) + 32*2 + D*4`，对应 `q + k + g_bf16`（3 个 $16{\times}128$ bf16）、
`beta`（32 bf16）、`dt_bias`（128 fp32）。

**读的时候注意。** 两个细节最容易看漏：

- **地址偏移复用**：`q` 与 `k` 的 gmem 布局相同，于是只算一次 `qk_off`
  （`fwd_kernel1.cuh:216`，`g_q.layout()(head_idx, bos+local_t*16, 0)`），`k` 直接复用
  （`fwd_kernel1.cuh:220`）。`g_bf16` 也用同一 `qk_off`（`fwd_kernel1.cuh:241`）。
- **beta 的 8 元素对齐**（`fwd_kernel1.cuh:222–225`）：

  ```cpp
  int beta_linear = head_idx * T_total + (int(bos) + local_t * CHUNK);
  int beta_aligned = beta_linear & ~7;
  auto beta_off = g_beta.layout()(beta_aligned);
  ```

  `beta` 是 `[H*T]` 的 1D 数组，1D TMA 的最小传输粒度是 8 个 bf16，所以起点按 `beta_linear & ~7`
  下取整；真正第 0 个 token 的偏移记在
  `beta_smem_offset = (head*T_total + bos + local_t*16) & 7`（`fwd_kernel1.cuh:345`），
  后面取 beta 时加上它（`fwd_kernel1.cuh:502`）。

  **为什么要转置成 `[H,T]`？** 若保持 `[T,H]`，同一 head 的 beta 在行方向 stride=H，
  无法满足 1D TMA 的对齐/连续要求；转置后 `[H,T]` 连续，正好一次 1D copy 取到一块 token。

三段式等待里，第一个 `__syncthreads()` 保证 barrier 的 init 对全部线程可见，
最后一个保证 TMA 数据对后续 generic 访问可见；少一个就会偶发读到未初始化数据。

## 3. 阶段 ③：QK L2 归一化（`fwd_kernel1.cuh:265–303`）

**把每个 token 的 `q`、`k` 各归一化成单位向量——delta rule 的秩一投影要求方向干净。**

**为什么需要这一步。**
块内递推里 $k_tk_t^\top$ 是"往 key 方向写"的投影，写得重不重由 $\beta$ 管；
$k$ 的模长若也参与，$\beta$ 的语义就被污染。所以先归一化，把"方向"和"强度"拆开。

**代码怎么做。**
映射是 **16 线程/行、每线程 8 元素**（`fwd_kernel1.cuh:268–271`：`ELEMS_PER_THREAD=8`、
`THREADS_PER_ROW=16`）：

1. 每线程把自己的 8 个 bf16 转 fp32，边转边累加平方和 `q_sq/k_sq`（`fwd_kernel1.cuh:279–287`）。
2. 4 步 `__shfl_xor_sync` 做 warp 内**树形归约**（`fwd_kernel1.cuh:289–293`，`delta` 取 8、4、2、1）。
3. `rsqrtf(x + 1e-6f)`（`fwd_kernel1.cuh:295–296`）防零向量，再逐元素乘回、写回 bf16
   （`fwd_kernel1.cuh:298–302`）。

**读的时候注意。**
一行 128 元素跨 16 个 lane（半个 warp），`xor` 8/4/2/1 正好聚合一整行、不跨行。
用树形而不是顺序累加，好处是所有 lane 同时拿到同一结果，且无 bank 冲突。

> 与 `torch_ref.py` 的 `l2_normalize_kernel_match`（`01-kda-theory.md` §2 提到的）保持同一归约
> 顺序，是为了 host 参照实现能和 kernel 的浮点结果对齐——归一化顺序不同会让 $q,k$ 末位有差，
> 进而在对拍时制造假误差。

所以这一步看着像"顺手 normalize"，其实是一次**为了对拍而刻意定序**的归约。

## 4. 阶段 ④：门激活 + cumsum + 尾部置零（`fwd_kernel1.cuh:306–338`）

**这一段是整个 K1 里最省的一次并行：列并行，天然就是前缀和。**

**为什么需要这一步。**
KDA 的门是**逐通道**的：每个 token、每个通道一个 $\log_2$ 域的门值，而块内要的又是它的累积量
$g^{\text{cum}}_i=\sum_{j\le i}g_j$。一个 $16\times128$ 的前缀和，如果按行做，行与行之间就得同步；
按列做，每个通道只有一条串行链，**一次同步都不需要**。

**代码怎么做。**
这一段（`fwd_kernel1.cuh:309–338`）用**列并行、不做跨行同步**来天然实现前缀和。
线程 0–127 一路一个通道（`col = compute_tid`），对 16 行循环，取该行该通道的 raw gate bf16，
加 `dt_bias`，乘 `a_log_exp`，再乘 `gate_scale` 后过 sigmoid（`fwd_kernel1.cuh:321–323`）：

$$
g_{\text{row,col}} = \underbrace{(\text{lower\_bound}\cdot\log_2 e)}_{\text{gate\_scale}}
\;\sigma\big(e^{A\_log}\cdot(\text{raw\_g}+\text{dt\_bias})\big).
$$

循环里 `sum += g_val; g_smem[row*D+col] = sum;`（`fwd_kernel1.cuh:327–328`）——
**因为每个通道只由一个线程负责，行之间的累加天然串行**，无需任何同步就在 smem 里写出 $g^{\text{cum}}$。
末行的 `sum` 就是整块累积 $g^{\text{total}}$，写进 `g_total[col]`（`fwd_kernel1.cuh:330`）。

线程 128–255 干另一件事：把 `k` 的**尾部越界行**清零（`fwd_kernel1.cuh:331–337`），
这样尾部 token 对 $L/M_{qk}$ 的贡献为 0，等价于块内填充。

**读的时候注意。**

- `gate_scale = lower_bound * 1.4426950408889634`（$\log_2 e$）在 host 算好传进来
  （`flash_kda.cpp:128`），于是 $g$ 直接是 **base-2 log**，后面 `ex2` 免换底。
- sigmoid 用 `sigmoid_tanh_approx_f32`（`utils.cuh:50–53`，`0.5*tanh(x/2)+0.5`，PTX `tanh.approx.f32`），
  比 `__expf`+除法快得多，精度对门控足够（误差见 §11.3）。
- 越界行（`row >= actual_len`）门取 0（`fwd_kernel1.cuh:324–326`），避免污染前缀和。
- `actual_len = min(16, seq_len - local_t*16)`（`fwd_kernel1.cuh:310`）只有尾块 < 16，非尾块恒为 16。
  把 padding 放在 K1 一次性处理掉，K2 的递推就能对整块无分支地跑。

所以这一段同时干完了三件事：算门、做前缀和、清 padding，却只用了一次 `__syncthreads()`。

## 5. 阶段 ⑤：$E^{\text{tot}}$ 与 decay_apply（`fwd_kernel1.cuh:353–476`）

**把"衰减"一次算成四种形态，是 K1 最像流水线的一段，也是 union 复用最刺激的一段。**

**为什么需要这一步。**
块内衰减有四种用法：算块内注意力时 key 要往前衰减（$k^{\text{dec}}$），
query 要同向乘一次并带 scale（$q^{\text{dec}}$），还原到块首要往回补（$k^{\text{inv}}$），
状态更新要的是整块衰减后的 key（$k^{\text{rest}}$）。它们都是 $k$ 乘一个对角矩阵，
逐元素乘就能出，但要**同时**产出四个，还得复用同一块 smem。

先把 `g_total` 原地取 $2^{g^{\text{total}}} = E^{\text{tot}}$（`fwd_kernel1.cuh:354–357`，
`ex2_approx_ftz_f32`，`utils.cuh:38–42`），然后构造四个衰减变体
（对照 `01-kda-theory.md` §3 ②，`fwd_kernel1.cuh:449–469`）：

| 输出 | 公式 | 代码 |
|---|---|---|
| `q_decayed` | $q_i\,2^{g^{\text{cum}}_i}\cdot\text{scale}$ | `fwd_kernel1.cuh:454` |
| `k_decayed` | $k_i\,2^{g^{\text{cum}}_i}$ | `fwd_kernel1.cuh:455` |
| `k_inv` | $k_i\,2^{-g^{\text{cum}}_i}$ | `fwd_kernel1.cuh:466–467` |
| `k_restored` | $k_i\,2^{-g^{\text{cum}}_i}\,E^{\text{tot}}$ | `fwd_kernel1.cuh:468` |

**代码怎么做：一次"先读后写、中间一次 sync"。**
Phase A 的 `{q, k, g}` 与 Phase B 的 `{k_decayed, q_decayed, k_inv, L, INV, Mqk}` 共用同一块
smem（`fwd_kernel1.cuh:57–71` 的 `union`，注释说省 ~14KB）。
因为 decay 要**同时读 q/k/g、写 q_decayed/k_decayed/k_inv**，而目标地址就是源地址，
所以代码先把所有源数据搬进**寄存器**（`fwd_kernel1.cuh:384–421`），`__syncthreads()`
（`fwd_kernel1.cuh:425`），再整块覆写（`fwd_kernel1.cuh:427–473`）。

线程映射（`fwd_kernel1.cuh:365–368`）：`g = lane/4`、`t = lane%4`。
`g` 同时决定列组（`col_base = n_blk + g*8`）和行号（`row = m_blk + (warp_id+g)%8`），
8 个 warp 合起来正好摊满一个 $8\times64$ 子块（`thr2_2d`，`fwd_kernel1.cuh:370–401`），每线程 2 个元素。
$16\times128$ 块共 4 个子块（`N_M=2, N_N=2`，`fwd_kernel1.cuh:375–377`），循环 4 次。

**读的时候注意。**

- 那 4 个 `reg_*` 数组（`float reg_g[N_TILES][2]` 等）是**正确性**要求，不是优化花活：
  它们在覆写前把旧数据整体挪走，`__syncthreads()` 之后才敢写同一块地址。
- `g_bf16` 与 `k_restored` 也做成 union（`fwd_kernel1.cuh:75–78`）：`g_bf16` 在阶段 ④ 读完后死掉，
  随后 decay 才写 `k_restored`。`dt_bias` 与 `g_total` 同理（`fwd_kernel1.cuh:79–82`）。
- 精度上，`q/k` 的衰减乘子都在 **bf16** 里算（`exp_cumsum` 先 cast 到 bf16 再乘，
  `fwd_kernel1.cuh:453–455`），只有门/前缀和保持 fp32。这是精度取舍：衰减乘子相对大小
  由 fp32 gate 决定，乘进 bf16 的 q/k 与其本来的 bf16 精度同量级。

  ```cpp
  BF16 exp_cumsum = BF16(ex2_approx_ftz_f32(g));
  r_qd(0, v) = q * exp_cumsum * BF16(scale);
  r_kd(0, v) = k * exp_cumsum;
  ```

  也就是说：**门是 fp32 的，衰减乘子是 bf16 的**，别把两者混成一件事。

所以这一步的产出是三个大数组加 $E^{\text{tot}}$——K2 要的"衰减记忆"都在这里。

## 6. 阶段 ⑥：$L$ 与 $M_{qk}$ 两个 GEMM（`fwd_kernel1.cuh:478–487`）

**两个 $16\times16$ 的块内注意力矩阵，正好一个 warp 一个，各用各的累加精度。**

**为什么需要这一步。**
$L_{ij}=\langle k^{\text{dec}}_i,k^{\text{inv}}_j\rangle$ 是"第 $i$ 个 token 对第 $j$ 个的 delta 修正系数"，
$M_{qk,ij}=\langle q^{\text{dec}}_i,k^{\text{inv}}_j\rangle$ 是块内"query 看 key"的分数。
两者形式完全一样（都是 $A B^\top$），只差 A 用哪个矩阵，所以映射到同一个 MMA 模板。

**代码怎么做。**

```
L_fp32 = k_decayed @ k_invᵀ      (fp32 累加)
Mqk    = q_decayed @ k_invᵀ      (bf16 累加)
```

- warp 0（`compute_tid < 32`）调 `mma_m16n16_bf16bf16fp32_1warp`（`utils.cuh:167–192`）算
  $L=k^{\text{dec}}\,{k^{\text{inv}}}^\top$，**fp32 累加器**——因为 $L$ 马上要求逆，保留 fp32 精度。
- warp 1（`32 ≤ compute_tid < 64`）调 `mma_m16n16_bf16bf16bf16_1warp`（`utils.cuh:145–165`）算
  $M_{qk}=q^{\text{dec}}\,{k^{\text{inv}}}^\top$，**bf16 累加并直接存 bf16**——它只参与输出投影，
  精度要求低。
- 两个都基于 `SM80_16x8x16_F32BF16BF16F32_TN`（`utils.cuh:153/175`），即 **TN**：A 行主、
  B 列主，所以 `cooperative_gemm` 算的是 $A B^\top$，正好是我们想要的"$i,j$ 点积"形式。

**读的时候注意。**
只有 2 个 warp 在算，其余 6 个 warp 空转——短 kernel，无所谓：并行度来自 8 个 CTA 同驻一个 SM
（§11.2），不来自单 CTA 内的 8 个 warp。

还有一处必须记住：$L$ 的对角并不为零。
$\langle k^{\text{dec}}_i,k^{\text{inv}}_i\rangle = k_i\cdot k_i\,2^{g^{\text{cum}}_i-g^{\text{cum}}_i}=k_i^2$，
L2 归一化后就是 1 左右。但分块公式里 $L$ 必须是**严格下三角**（对角线归 1，体现在 $I+L$ 的 $I$），
所以下一步必须 mask。

## 7. 阶段 ⑦：tril + beta 掩码（`fwd_kernel1.cuh:489–508`）

**一次 256 线程的逐元素整理：把 $L$ 削成严格下三角、把 $\beta$ 乘进去、把 $M_{qk}$ 削成下三角。**

**为什么需要这一步。**
三件事都是"形状"问题：$L$ 多算了上三角和对角（因为 GEMM 不知道因果性）；
$\beta_i$ 这个写入门还没进 $L$；$M_{qk}$ 的上三角在因果注意力里永远是 0。

**代码怎么做。**
$16\times16=256$ 个元素，正好 256 线程一人一个（`fwd_kernel1.cuh:494–498` 用列分块
`col_block_size=8` 把 `(i,j)` 解出来）：

- $i\le j$：`L_fp32(i,j)=0`（`fwd_kernel1.cuh:499–500`）——严格下三角，对角清零。
- $i>j$：`L_fp32(i,j) *= sigmoid_tanh_approx_f32(beta_i)`（`fwd_kernel1.cuh:501–503`）——
  把写入门 $\beta_i=\sigma(\text{beta\_logit}_i)$ 乘进去。注意用 `beta_smem_offset` 定位到对齐
  载入里的正确 token（`fwd_kernel1.cuh:502`）。
- $i<j$：`Mqk(i,j)=0`（`fwd_kernel1.cuh:504–506`）——$M_{qk}$ 抹上三角，保留**下三角含对角**。

同时把 $L$ 的 **bf16 版本当作合并矩阵 $M$ 的载体**：`M_bf16` 直接复用已死的 `k_inv` buffer
（`fwd_kernel1.cuh:490–491`），交给求逆例程内部按块填 $C$（见 §8）。

**读的时候注意。**

- $\beta$ 是**行**方向的（第 $i$ 个 token 的写入门），所以是 `beta_i` 乘第 $i$ 行，不是列。
- $L$ 用 `i <= j` 清（含对角），$M_{qk}$ 用 `i < j` 清（不含对角）——差一个等号，正好对应
  "$I+L$ 的单位对角" 和 "$M_{qk}$ 的对角自注意力"。
- `M_bf16` 与 `k_inv` 是同一块地址：写 $M$ 时 $k^{\text{inv}}$ 已经没人要了（两个 GEMM 都算完了），
  这就是"复用死 buffer"的典型手法，前提是你确定它真的死了。

所以这一步没有浮点运算，却决定了求逆的输入长什么样。

## 8. 阶段 ⑧：块内求逆 `inv_fwd_subst_fused_1warp`（`fwd_kernel1.cuh:510–511`，`utils.cuh:213–369`）

**K1 最烧脑的一段：求 $P=(I+L)^{-1}$，实现是"两个 $8\times8$ 前代 + 一次 $16\times16$ 的 bf16 合并"。**

**为什么需要这一步。**
块内 16 步递推是 $S_i=S_{i-1}(I-\beta_ik_ik_i^\top)+\cdots$ 的连乘，展开后是一个下三角线性系统，
解它就等于求 $(I+L)^{-1}$。$I+L$ 是 $16\times16$ **单位下三角**（$L$ 严格下三角），
直接求逆不如按 $8\times8$ 分块做 Schur 补：

$$
X = I+L = \begin{bmatrix} A & 0 \\ C & B\end{bmatrix},\quad
X^{-1} = \begin{bmatrix} A^{-1} & 0 \\ -B^{-1} C A^{-1} & B^{-1}\end{bmatrix},
$$

其中 $A,B$ 是 $8\times8$ 单位下三角，$C=L[8{:}16,0{:}8]$。

**代码怎么做。例程分三步：**

**(a) 两个 $8\times8$ 对角块的 fp32 前代求逆**（`utils.cuh:229–252`）。
lane $i$（`i = tid & 7`）拥有 block 内第 $i$ 行，`inv[p]` 初始化为 $L$ 的严格下三角 + 对角 1
（`utils.cuh:238–242`）。然后 $s=0\ldots6$（`utils.cuh:243–252`）：

```cpp
for (int s = 0; s < 7; ++s) {
    const float row_scale = -inv[s];
    for (int p = 0; p < s; ++p) {
        const float pivot = __shfl_sync(0xFFFFFFFFu, inv[p], (tid & ~7) | s);
        if (i > s) inv[p] = fmaf(row_scale, pivot, inv[p]);
    }
    if (i > s) inv[s] = row_scale;
}
```

这就是**逐行消元的 forward substitution**：处理完第 $s$ 行后，第 $i>s$ 行被消成 $A^{-1}$ 的第 $i$ 行。
8 个 lane 一组，`row0 = tid & 8` 决定做 block A（rows 0–7）还是 B（rows 8–15）；
groups 0/2 做 A、1/3 做 B，冗余拷贝让所有 shuffle 收敛（`utils.cuh:232–242` 注释）。

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
最后 `__hadd2` 把两块不相交的 bf16 结果相加（`utils.cuh:343–355`），STSM 存回 `INV`
（`utils.cuh:357–368`）。

**读的时候注意。**

- **为什么不用 Neumann 级数？** 旧实现 $(I-L)(I+L^2)(I+L^4)(I+L^8)$ 在高相关 key（$|L|\to1$）时
  $L^k$ 中间量可达 $10^3$、fp16 里灾难性相消；前代 + Schur 补从不构造 $L^k$，对近共线 key 也稳
  （`utils.cuh:194–212` 注释）。实测量化见 §11.4。
- **为什么一个 warp 就够？** 16 个 lane 各持一行、冗余成 4 组，每个 shuffle 都在单 warp 内收敛。
- 求逆结果按 bf16 落进 `INV`，精度从 $10^{-8}$ 退到 $\sim10^{-3}$——这个取舍在 §11.4 有账。

块内求逆的代数正确性有独立自测：`harness/check_inv_identity.py`（见 §10）。
所以这一步是"数学最难、代码最短"的一站：一行调用，背后是整个 Schur 补。

## 9. 阶段 ⑨：写 workspace（`fwd_kernel1.cuh:513–585`）

**六次 TMA store，把半成品放进 K2 的货架，然后 K1 就可以退场了。**

**为什么需要这一步。**
产出必须落到 gmem，K2 才能按 head 顺序读。用 TMA store 而不是逐线程写，省指令、不占寄存器，
代价是要显式 fence 和 wait。

**代码怎么做。**
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
`WorkspaceSizes` 一一对应。

**读的时候注意。**

- **为什么 SoA**？每个字段能独立配 TMA descriptor（`fwd_launch.cu:98–103`）
  且都能 128B 对齐（`flash_kda.cpp:16–18` 的 `static_assert`）。展开见 §12.2。
- 索引顺序是 `ws_idx = head_idx * total_tiles + global_tile_idx`（`fwd_kernel1.cuh:516`）——
  **head 在外、tile 在内**，与 K2 一个 CTA 串一个 head 的所有 tile 的顺序一致，K2 顺序读即连续。
- `tma_store_wait<0>()` 在 `if (threadIdx.x == 0)` 外面，全 CTA 都要等，别以为只有 store 线程需要。

所以这一步之后，K1 的痕迹只剩六个数组，K2 只认这六个数组。

## 10. 行号速查 + 自测

速查表，读代码时对着看：

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

**自测**：`harness/check_inv_identity.py` 用纯 PyTorch 复现 §8 的"8×8 前代 + Schur 补"块内求逆，
与直接解 $(I+L)^{-1}$（fp64 `torch.linalg.inv`）对比。实测（`harness/check_inv_identity.out.txt`）：
纯公式在 **fp64** 下 max abs $\le 2.2\times10^{-16}$（机器精度，代数严格正确），**fp32** 下
$\le 1.2\times10^{-7}$；把 HMMA 输入 $P/M$ 与中间量按 kernel 一样量化到 bf16 后，
max abs $\approx 2.5\times10^{-3}\!\sim\!5.9\times10^{-3}$——与 bf16 输入本身的量化误差同量级。
对近共线 key（$|L|$ 接近 1）的构造，fp64 仍 $\le5.6\times10^{-17}$，bf16 量化 $\approx1.7\times10^{-3}$，
没有 Neumann 级数那种放大。这验证了求逆例程的代数（`utils.cuh:213–369`）确实产出块内注意力矩阵的逆。

> workspace 逐字段字节数与 host 对齐见 **§12**。K2 递推见第 3 篇。

## 11. 关键优化逐条量化

本节把 §5–§8 提到的几处"为什么这么写"全部换成**本机实测数字**。探针：
`harness/smem_probe.cu`（编译期 `sizeof`/`cosize`）、`harness/approx_probe.cu`（近似指令误差）、
`harness/check_inv_precision.py`（求逆精度路径对比）、以及对已编译 `.so` 的
`cuobjdump -res-usage`（真实寄存器/占用）。原始输出见同目录 `*.out.txt`。

### 11.1 smem union 到底省了多少

用 `smem_probe.cu` 打印 `K1Layouts<128,16>` 各布局的 `cosize`（`fwd_kernel1.cuh:5–42`）：

| 缓冲 | 布局 | cosize | 字节 |
|---|---|---|---|
| `q`,`k` | `QKLayout` bf16 $16{\times}128$ | 2048 | 4096 各 |
| `g` | `GLayout` fp32 $16{\times}128$ | 2048 | 8192 |
| `k_decayed`,`q_decayed`,`k_inv` | `MMALayout` bf16 $16{\times}128$ | 2048 | 4096 各 |
| `L` | `LF32Layout` fp32 $16{\times}16$ | 256 | 1024 |
| `INV`,`Mqk` | `LMLayout` bf16 $16{\times}16$ | 256 | 512 各 |
| `beta` | `BetaSmemLayout` bf16 32 | 32 | 64 |
| `dt_bias`,`g_total` | `GTotalLayout` fp32 128 | 128 | 512 各 |

- **Phase A（`q,k,g`）** = `2·4096 + 8192` = **16384 B**；
- **Phase B（`k_dec,q_dec,k_inv,L,INV,Mqk`）** = `3·4096 + 1024 + 2·512` = **14336 B**；
- 两者 union 后取 **max = 16384 B**，即省 **14336 B ≈ 14 KB**——正是 `fwd_kernel1.cuh:56`
  注释里的"~14KB"。
- 还有两个小 union：`g_bf16`/`k_restored` 各 4096 B，省 4096；`dt_bias`/`g_total` 各 512 B，省 512。
- 若完全不做 union，总和 = `16384 + 14336 + 4096 + 64 + 4096 + 512 + 512` = **39936 B**；
  实测 `sizeof(SharedStorageK1<128,16>)` = **21248 B**（含 128B 对齐填充与 barrier）。
  即 **union 共省 18688 B ≈ 18.25 KB**，smem 单块从 39 KB 压到 **21.2 KB**。

  > 记账口径：上面这一行逐项相加其实是 **40000 B**（`16384+14336+4096+64+4096+512+512`），
  > 原文的 39936 / 18688 都按**不含 `beta` 64 B** 的口径记；实测 `sizeof=21248 B` 与探针一致。
  > 若按 40000 B 记，省下的是 **18752 B ≈ 18.3 KB**。两种口径差的就是那 64 B，不影响结论。

省这 18 KB 直接决定了 §11.2 的占用率：不做 union 时单块 39 KB，SM 的 228 KB 只放得下 5 块；
做了之后 `8 × 21248 = 169984 B = 166 KB`，才够 8 块。

### 11.2 `__launch_bounds__(256,8)`：占满 SM

`fwd_kernel1.cuh:120` 写死 `__launch_bounds__(NumThreads, 8)`（`NumThreads=256`），
第二个参数是**每 SM 最少 8 个 CTA**。在 H100（sm90）上逐项核对（`fwd_launch.cu:149–172`）：

| 约束 | 上限（sm90 每 SM） | 8 CTA 需求 | 满足 |
|---|---|---|---|
| 线程 | 2048 | `8 × 256 = 2048` | 恰好占满 |
| smem | 228 KB | `8 × 21248 B = 166 KB` | ✓ |
| 寄存器 | 65536 | `65536 / 2048 = 32`/线程 | 编译器按此封顶 |

实测（`cuobjdump -res-usage` 对已编译 `.so`，`harness/k1_res_usage.out.txt`）K1 的两个
varlen 变体均为 **`REG:32 STACK:0 LOCAL:0`**——正好压在 32 寄存器上限、**零 spill**。这说明
`__launch_bounds__` 的寄存器约束对 K1 不痛：K1 只有 2 个 warp 做 MMA（§6），其余 6 个 warp
主要搬数据/求逆，寄存器压力本就不大；用 ILP 换满占用率（8 CTA/SM = 100%）是对的——K1 是
**访存延迟受限的短 kernel**（单次 TMA load、无 pipeline，§0），高占用才能在等 TMA 时切到别的
CTA。（对比 K2 是长 kernel、warp specialization，寄存器达 74–80，第 3 篇讲。）

### 11.3 `ex2.approx` / `tanh.approx` 的误差：可以忽略

`harness/approx_probe.cu` 在 sm90 上对 2×10⁶ 个点与双精度参照对比
（`harness/approx_probe.out.txt`）：

| 指令（出处） | 测试范围 | max abs | max rel |
|---|---|---|---|
| `ex2_approx_ftz_f32`（`utils.cuh:38`） | $x\in[-30,0]$ | $1.14\times10^{-7}$ | $1.42\times10^{-7}$ |
| `tanh.approx.f32`（`utils.cuh:44`） | $x\in[-30,30]$ | $7.63\times10^{-6}$ | $9.88\times10^{-6}$ |
| `sigmoid_tanh_approx_f32`（`utils.cuh:50`） | $x\in[-10,10]$ | $3.85\times10^{-6}$ | $8.77\times10^{-3}$ |

`sigmoid` 的 max rel 大是因为小 $x$ 处 $\sigma\to0$、分母极小，看 **abs** 更公平：全区间
$\le 3.85\times10^{-6}$。真正要关心的是**门控衰减乘子的端到端误差**
$\text{decay}=2^{\,\text{gate\_scale}\cdot\sigma(x)}$（kernel 路径用 `ex2_approx(gate_scale * sig_approx(x))`，
`fwd_kernel1.cuh:321–323,453`）：

| `lower_bound`（`gate_scale = lb·log₂e`） | max abs | max rel |
|---|---|---|
| −1.0 | $3.54\times10^{-6}$ | $3.85\times10^{-6}$ |
| −2.0 | $6.53\times10^{-6}$ | $7.71\times10^{-6}$ |
| −5.0 | $1.31\times10^{-5}$ | $1.93\times10^{-5}$ |

即衰减乘子的相对误差 $\lesssim2\times10^{-5}$，比 bf16 表示 $q/k$ 本身的舍入（~$4\times10^{-3}$）
**小两个数量级**，所以这两处近似是"免费"的。注意 `ex2` 是 **base-2**：`gate_scale` 在 host
侧已经乘过 $\log_2 e$（`flash_kda.cpp:128`），kernel 里一条 `ex2.approx` 就够，省掉换底乘法。

### 11.4 前代求逆的精度取舍：为什么值得

`harness/check_inv_precision.py` 对同一 seed $L$ 比较四条路径（golden = fp64 `inv(I+L)`）：

- **A** 纯公式：fp32 前代 + fp32 Schur 合并（无量化上界）；
- **B** **kernel 实际路径**：seed $L$、8×8 前代全程 fp32，仅 HMMA 输入 $P/M$ 量化 bf16（`utils.cuh:254–296`）；
- **C** 反面：把 seed 也量化 bf16 再求逆；
- **D** 旧实现：fp16/bf16 **Neumann 级数** $(I-L)(I+L^2)(I+L^4)(I+L^8)$。

| 场景 | A fp32 | B kernel | C bf16 seed | D Neumann fp16 | D Neumann bf16 |
|---|---|---|---|---|---|
| 随机 $\lvert L\rvert\sim0.3$ | $7\times10^{-8}$ | $3.6\!\sim\!5.9\times10^{-3}$ | 同 B | $5.6\!\sim\!7.0\times10^{-4}$ | $3.1\!\sim\!4.2\times10^{-3}$ |
| 完全共线 $c=0.7$ | $2.5\times10^{-8}$ | $7.8\times10^{-4}$ | $7.8\times10^{-4}$ | $1.7\times10^{-2}$ | $1.6\times10^{-1}$ |
| 完全共线 $c=0.9$ | $3.0\times10^{-8}$ | $1.6\times10^{-3}$ | $1.6\times10^{-3}$ | $\mathbf{5.8\times10^{-1}}$ | $\mathbf{2.8}$ |

三个结论：

1. **纯公式（A）在 fp32 下就到 $10^{-8}$**——Schur 补 + 前代的代数严格正确（§8、`check_inv_identity`）。
2. **误差来源是"HMMA 输入量化 bf16"，不是 seed**（B≈C，且都 ~$10^{-3}\sim10^{-2}$，与 bf16 的
   $\varepsilon\approx7.8\times10^{-3}$ 同量级）。把 seed 保 fp32 的价值在"完全共线 c=0.9"之外的
   强相关场景才显现（C 略大于 B）；这里 A/B 的差距说明真正买精度的是**把 seed 留在 fp32 走前代**，
   而不是把合并也做成 fp32。
3. **Neumann 在强共线时崩溃**（D 在 $c=0.9$ 时 fp16 误差 $0.58$、bf16 误差 $2.83$，而 $\lvert X^{-1}\rvert_{\max}=1$）：
   $L^k$ 的元素是 $\binom{14}{7}c^7\approx1.7\times10^3$，在 16-bit 里相减灾难性相消。
   前代 + Schur **从不构造 $L^k$**，所以对近共线 key 也稳——这正是 `utils.cuh:194–212` 注释里
   "Neumann powers reach ~1e3 there"的实测确认。

代价（取舍）：为了用 bf16 HMMA 做合并，$P/M$ 被量化 bf16，使结果误差从 $10^{-8}$ 退到
$\sim10^{-3}$；但比块内求逆的**用途**（$k^{\text{inv}}$ 只做 bf16 衰减变体、$M_{qk}$ 只做输出投影，
§5/§6）所需要的精度已经足够，且避免了 Neumann 的失稳。

## 12. workspace 逐字段字节账与 SoA 指针切分

§9 讲了 K1"往哪些数组写"，这一节把**每个数组占多少字节、基址落在哪、总分配怎么算**逐字对上，
并用实跑验证。三方口径必须一致：host 的 `get_workspace_size`（`flash_kda.cpp:5–26`）、
C++ 常量 `WorkspaceSizes`（`utils.cuh:63–77`）、host 指针切分（`fwd_launch.cu:61–71`）。

### 12.1 每 tile 字节预算

workspace 是 **SoA**（六个独立大数组），每个 tile 把六个字段**顺序**排一遍
（`utils.cuh:70–76` 的 `kKDecayed…kMqk` / `kPerTile`）：

| 字段 | 元素类型 | 每 tile 形状 | 字节 | 出处 |
|---|---|---|---|---|
| `k_decayed` | bf16 | $[16,128]$ | 4096 | `utils.cuh:70` |
| `q_decayed` | bf16 | $[16,128]$ | 4096 | `utils.cuh:71` |
| `k_restored` | bf16 | $[16,128]$ | 4096 | `utils.cuh:72` |
| `g_total` | fp32 | $[128]$ | 512 | `utils.cuh:73` |
| `INV` | bf16 | $[16,16]$ | 512 | `utils.cuh:74` |
| `Mqk` | bf16 | $[16,16]$ | 512 | `utils.cuh:75` |
| **`kPerTile`** | | | **13824** | `utils.cuh:76` |

host 用等价的一行硬编码这个和（`flash_kda.cpp:20`）：
`3*CHUNK*D*2 + D*4 + 2*CHUNK*CHUNK*2 = 3·4096 + 512 + 2·512 = 13824`。
三个 `static_assert`（`flash_kda.cpp:16–18`）保证每个字段都是 **128 B 的整数倍**——
这一点直接决定了 §12.2 的"六个基址天然 128 B 对齐、字段之间零填充"。

**为什么是 13824 而不是把 K1 smem 里那 6 块照搬？** K1 的 smem Phase B（§11.1）是
`3·4096 + 1024(fp32 L) + 2·512 = 14336`，比 13824 **多 512 B**：因为 smem 里 $L$ 是 fp32
（1024 B，求逆要用），而 workspace 只存求逆**之后**的 bf16 `INV`（512 B），$L$ 本身不必落盘。
换句话说 workspace 只保留 K2 递推真正要读的东西：三个衰减变体 + 块累积门 + 块内逆 + $M_{qk}$。

### 12.2 SoA 基址切分与对齐

host 侧 `n_ht = H * total_tiles`（`fwd_launch.cu:62`），六个基址是**前缀和偏移**
（`fwd_launch.cu:64–71`）：

| 指针 | 偏移（字节） | 代码 |
|---|---|---|
| `ws_kd` | `ws + 0` | `fwd_launch.cu:64` |
| `ws_qd` | `ws + n_ht·4096` | `fwd_launch.cu:65` |
| `ws_kr` | `ws + n_ht·8192` | `fwd_launch.cu:66` |
| `ws_gt` | `ws + n_ht·12288` | `fwd_launch.cu:67` |
| `ws_inv` | `ws + n_ht·12800` | `fwd_launch.cu:68` |
| `ws_mqk` | `ws + n_ht·13312` | `fwd_launch.cu:69` |
| `ws_tile_prefix` | `ws + n_ht·13824` | `fwd_launch.cu:71` |

因为 `n_ht·(每个字段大小)` 里字段大小都是 128 的倍数，只要 `ws` 基址按 128 B 对齐
（`torch.empty(uint8)` 给 256 B 对齐），**六个基址全部 128 B 对齐**，且一个 tile 内部无填充
（13824 也是 128 的倍数）。这正是六个字段能各自配一个 TMA descriptor（`fwd_launch.cu:98–103`）的前提。

**为什么用 SoA 不用 AoS？** 这六个字段在 K2 里的访问模式、想要的 smem swizzle
都不同：`k/q_decayed/k_restored` 走 $16{\times}128$ 的 `TMAVOLayout`（`MMALayout` K-major），
`g_total` 是 1D `GTotalLayout`，`INV/Mqk` 走 $16{\times}16$ 的 `TMALMLayout`
（`fwd_kernel1.cuh:98–103` 的 store descriptor 各自独立）。若做成 AoS（结构体数组），
每个字段的 stride 被一个结构体尺寸钉死，凑不出各自最优的 TMA 布局，也难保证字段级 128 B 对齐。
SoA 让"布局、对齐、descriptor"三者解耦，代价只是 host 要多算几个前缀和偏移。

**索引顺序**：`ws_idx = head_idx * total_tiles + global_tile_idx`（`fwd_kernel1.cuh:516`）——
**head 在外、tile 在内**。K2 的 grid 是 `dim3(N, H)`（`fwd_launch.cu:203`），一个 CTA 负责
一个 `(seq, head)` 并**顺序**跑完该 head 的所有 tile；head-major 让这个 CTA 读 workspace 时
`ws_idx` 逐块 +1、地址连续，正好是 TMA/缓存的顺读。若反过来 tile-major，同一 CTA 每步要跨
`H·kPerTile` 步进，白白浪费预取。

### 12.3 总分配 = 上界，launch = 精确/上界

`get_workspace_size(T_total,H,N)`（`flash_kda.cpp:5–26`）**永远按上界**算：

$$
\text{tiles}_{\text{alloc}} = \left\lceil \tfrac{T_{\text{total}}}{16}\right\rceil + N,\qquad
\text{bytes} = H\cdot\text{tiles}_{\text{alloc}}\cdot 13824 + \left\lceil\tfrac{4(N+1)}{128}\right\rceil\!\cdot\!128 .
$$

尾部的 `tile_prefix_bytes = ceil(4(N+1)/128)*128`（`flash_kda.cpp:23`）是 varlen 才用到的
每序列 tile 累计数（$N+1$ 个 int32，向上取整到 128 B）。

**关键：分配用上界、launch 有时用精确值，两者可以差。** `fwd` 里（`flash_kda.cpp:176–181`）：

- 非 varlen：`total_tiles = N·ceil(T_seq/16)`（精确，`flash_kda.cpp:180`）；
- varlen：`total_tiles = ceil(T_total/16) + N`（上界，`flash_kda.cpp:178`）。

非 varlen 的精确值不超过上界：因 $T_{\text{total}}=N\cdot T_{\text{seq}}$，

$$
N\left\lceil \tfrac{T_{\text{seq}}}{16}\right\rceil \;\le\; N\left(\tfrac{T_{\text{seq}}}{16}+1\right)
= \tfrac{T_{\text{total}}}{16}+N \;\le\; \left\lceil\tfrac{T_{\text{total}}}{16}\right\rceil + N .
$$

所以按上界分配一定够。但注意 §12.2 里 `n_ht`、以及 `ws_tile_prefix` 的绝对偏移，用的是
**launch 的** `total_tiles`（`fwd_launch.cu:62,71` 传的是 `total_tiles` 参数）。非 varlen 时它比
分配上界小，于是 `tile_prefix` 落在分配区**内部偏前**的位置——因为非 varlen 根本不写、不读
`tile_prefix`（只有 varlen 的 `_flash_kda_build_tile_prefix` 写它，`fwd_launch.cu:164–167`），
这段"错位"无害。

**这是一个容易踩的坑**：`ws_tile_prefix` 的偏移不能按分配上界去猜，
必须按真正 launch 的 `total_tiles`。下面 §12.4 的实跑就专门验证了 varlen 下三者对齐。

### 12.4 实跑验证

`harness/check_workspace_layout.py`（原始输出 `harness/check_workspace_layout.out.txt`）
在 `kernel_lab` 里对**已编译**扩展做三项核对：

1. **A 字段预算**：六个字段全部 128 B 对齐，`kPerTile=13824` 与 `flash_kda.cpp:20` 的
   `3*16*128*2+128*4+2*16*16*2` 相等。
2. **B `get_workspace_size` vs 闭式**：9 组 `(T,H,N)`（含 varlen `N=2/4/7` 与不整除 `T=1000/12345/101/77`）
   全部 `OK`。例：`T=512,H=4,N=1 → tiles=33, ws=4·33·13824+128 = 1,824,896 B`（≈1.74 MB）。
3. **C 实跑 varlen**：`lens=[30,16,5,50]`（$T_{\text{total}}=101$、$N=4$），把 workspace 用 `0xAA`
   填满后跑 `fwd`，在预测偏移 `H·launch_tiles·13824 = 2·11·13824 = 304128` 读 int32：
   得到 `[0, 2, 3, 4, 8]`，与每序列 tile 数 `[ceil(30/16),1,1,ceil(50/16)]=[2,1,1,4]`
   的累计前缀**完全一致**；其后 108 B 仍是 `0xAA`（kernel 没有越界写），前 304128 B 里有
   259423 B 被写过（六个数组确实落盘）。

这三项把"host 公式 ↔ C++ 常量 ↔ kernel 实际写入位置"钉死在同一套字节账上。

## 13. 实验与出处索引

正文不写任务编号，对账集中在这里：

| 编号 | 产出 | 对应小节 | 证据文件 |
|---|---|---|---|
| P2-1 | 九道工序逐段精读 + 行号速查 | §1–§10 | `harness/check_inv_identity.out.txt` |
| P2-2 | 关键优化逐条量化 | §11 | `harness/smem_probe.out.txt`、`harness/approx_probe.out.txt`、`harness/k1_res_usage.out.txt`、`harness/check_inv_precision.out.txt` |
| P2-3 | workspace 逐字段字节账 | §12 | `harness/check_workspace_layout.out.txt` |

探针与结论的对应：`harness/smem_probe.cu` 测编译期 `cosize`/`sizeof`（§11.1），
`cuobjdump -res-usage` 测真实寄存器与占用（§11.2），`harness/approx_probe.cu` 测近似误差（§11.3），
`harness/check_inv_precision.py` 比四条求逆路径（§11.4），`harness/check_inv_identity.py`
验求逆代数（§10），`harness/check_workspace_layout.py` 对字节账（§12.4）。

所有行号均在 `~/github/FlashKDA` 下按当前源码核对；第 1 篇 §3 的公式记号本文直接沿用。

## 14. 三句话小结

- K1 是**一间只有一张操作台的备菜间**：一个 CTA 一个 16-token 块，九道工序全在 21.2 KB smem 上，
  产出六个 workspace 数组就退场，不读状态、不做递推。
- 真正"算"的只有三处——归一化、两个 $16\times16$ GEMM、$8\times8$ 前代 + Schur 补的块内求逆；
  其余六步都在搬数据、整布局、清 padding，但每一步的**顺序**都关系到正确性。
- 省下来的账都能对上：smem union 省 18 KB 换来 8 CTA/SM 满占用（§11.1–§11.2），
  `ex2/tanh` 近似把门控误差压到 $2\times10^{-5}$（§11.3），workspace 13824 B/tile 的 SoA
  让六个字段各自 128 B 对齐（§12）。

下一篇进入 K2：同一个块，从"预处理"变成"递推"，warp specialization 与 TMA 流水都在那里。
