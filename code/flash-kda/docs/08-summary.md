# KDA 前向 / 反向算子实现解析：一页纸汇总

> 本文是「Kimi Delta Attention（KDA）前向 / 反向算子实现解析」的收尾篇（路线图 **P8-1**）。
> 把前七篇（数学 `01`、K1 `02`、K2 `03`、数值 `04`、反向推导 `05`、反向 kernel `06`、
> 性能 `07`）压成一张可独立阅读的速查表：**公式 → 结构 → 误差 → 性能 → 反向要点**。
> 所有数字都来自本机实测，出处见各篇与 `code/flash-kda/harness/*.out.txt`。

## 1. 一句话

KDA 是「**逐通道衰减** $\alpha$ + **写入门** $\beta$」的 delta rule；FlashKDA 把它的前向拆成
**K1（token 并行 prepare）+ K2（head 并行 recurrence）** 两段，块内是 16×16 的小矩阵代数，
瓶颈不是 TensorCore 而是 **$D\times D$ 状态与 workspace 的访存**；反向（FlashKDA 未开源）可
**复用 K1 落盘的 workspace**，块内并行、块间只传状态伴随。

## 2. 数学（记号与两套等价形式）

单头、状态 $S\in\mathbb{R}^{K\times V}$（$K=V=D=128$），$\beta_t\in(0,1)$ 已 sigmoid，
$g_t\le0$ 是 **log2** 衰减、$\alpha_t=2^{g_t}$ 逐 key 通道，输出 $o_t=\text{scale}\cdot q_tS_t$。

**逐 token gated delta rule**（golden）：

$$
p_t=(\alpha_t\odot k_t)S_{t-1},\quad
\delta_t=\beta_t(v_t-p_t),\quad
S_t=\operatorname{diag}(\alpha_t)S_{t-1}+k_t\delta_t^\top .
$$

**CHUNK=16 分块**（$g^{\text{cum}}_i=\sum_{j\le i}g_j$，$E_i=2^{g^{\text{cum}}_i}$，$E^{\text{tot}}=E_{C-1}$）：

$$
\begin{aligned}
k^{\text{dec}}_i=k_i\odot E_i,\ &\ q^{\text{dec}}_i=q_i\odot E_i\cdot\text{scale},\quad
k^{\text{inv}}_i=k_i\odot E_i^{-1},\quad k^{\text{rest}}_i=k^{\text{inv}}_i\odot E^{\text{tot}},\\
L=\operatorname{tril}(k^{\text{dec}}{k^{\text{inv}}}^\top,-1)\odot\beta,\ &\quad
M_{qk}=\operatorname{tril}(q^{\text{dec}}{k^{\text{inv}}}^\top),\quad P=(I+L)^{-1},\\
\tilde v=(v-k^{\text{dec}}S_{\text{in}})\odot\beta,\ &\quad U=P\tilde v,\quad
o=q^{\text{dec}}S_{\text{in}}+M_{qk}U,\\
S_{\text{out}}&={k^{\text{rest}}}^\top U+E^{\text{tot}}\odot_{\text{row}}S_{\text{in}} .
\end{aligned}
$$

两套形式在 fp64 下 **逐元素完全相等**（`src/kda_ref.py` 互拍，`max_abs=0`）；fp32 下差
$\sim10^{-9}\!\sim\!10^{-7}$，只是计算顺序的舍入。**KDA vs GLA/DeltaNet**：DeltaNet 有 $\beta$
无 $\alpha$，GLA 有 $\alpha$（每 head 标量）无 $\beta$，KDA 两者兼有且 $\alpha$ 逐通道。

> 布局坑：FlashKDA 的 `initial_state`/`final_state` 是 **$S^\top$（$[N,H,V,K]$）**，与参考的
> $[K,V]$ 互为转置；方阵不触发形状检查，错布局只表现为「初态处误差大、有衰减时又收敛」。

## 3. 前向结构（FlashKDA，`CHUNK=16`）

| | **K1 prepare** | **K2 recurrence** |
|---|---|---|
| grid | $N\cdot H\cdot\text{num\_chunks}$（token 并行） | $N\cdot H$（head 并行，沿 T 串行） |
| 线程 / 角色 | 256 线程、单次 TMA load、无 pipeline | 192 = 4 MMA + 1 LOAD + 1 STORE（warp-specialized） |
| 流水线 | 无（短 kernel，高占用） | 输入 `PipelineTmaAsync<3>`、输出 `PipelineAsync<2>` |
| 干什么 | 门激活/归一化/衰减 → $L,M_{qk},P$ → 写 workspace | 八相 delta 递推 + 输出 + 状态更新 |
| smem | `SharedStorageK1`=21248 B；union 复用省 14 KB | `SharedStorageK2`=98432 B（union 后，含状态缓冲） |
| 寄存器 | `REG:32`（正好压上限、零 spill） | `REG:74–80`（零 spill） |
| 占用 | 8 CTA/SM（占满） | **2 CTA/SM**（smem 钉死，理论 occ 18.75%） |

- **K1 关键段**（真实行号）：tile 定位 `fwd_kernel1.cuh:168–199` → TMA 载入 `200–263` →
  L2 norm `265–303` → 门激活+cumsum `306–338` → decay_apply `353–476` → $L/M_{qk}$ `478–487`
  → tril+beta `489–508` → 求逆 `510–511`（`utils.cuh:213–369`）→ 写 workspace `513–585`。
  求逆用「两个 8×8 fp32 前代 + 一次 bf16-HMMA 合并」，比 Neumann 级数在 $|L|\to1$ 时更稳。
- **K2 八相**（`fwd_kernel2.cuh`）：dual GEMM `533–568` → cast+载 v/INV/β `570–587` →
  $u=(v-u)\beta$、$u=Pu$（`MOVM_T` 把 C 片段当 B 操作数）`589–623` → $M_{qk}U+o$ `625–650`
  → 写 out `652–657` → 状态更新 `s\!\mathrel{+}\!=\ E^{\text{tot}}\odot s+k^{\text{rest}\top}U`
  `659–731`。
- **为什么拆两段**：单 kernel 会被 K2 的低并行度（$B\cdot H$）拖住，拆开至少快 15%；
  K1 把 token 并行的重活一次算完落盘，K2 专心递推。
- **workspace**：每 tile 六字段 = 3×4096 + 512 + 2×512 = **13824 B**（`utils.cuh:70–76`），
  SoA 排布且各 128B 对齐；分配恒取上界 `ceil(T/16)+N`，varlen launch 用精确 `N·ceil(T_seq/16)`。

## 4. 数值（全部 vs fp64 golden，除非注明）

| 量 | 误差 | 说明 |
|---|---|---|
| 分块 vs 逐 token（fp64） | **max_abs = 0** | 实数域严格等价 |
| 前向 $o$（FlashKDA bf16） | max_rel **4.0e-3 ~ 7.6e-3** | 峰值相对误差，随 T 不漂移 |
| 前向终态 $S$（FlashKDA bf16） | max_rel **5.0e-3 ~ 1.4e-2** | $g=0$ 无衰减时最大 |
| 三方：`flash↔fla` 的 $o$ | max_rel **3.65e-3** < 各自 vs golden（6.22e-3） | **共模抵消**（共享同一份 bf16 输入量化） |
| 反向六项 vs autograd（fp64） | **max_abs = 0** | 显式 adjoint 与链式法则逐位相等 |
| 反向 chunked vs 逐 token（fp64） | max_rel **~1e-15**（含 $D=128,T=512$） | 13 组 case 全机器精度 |
| FLA `chunk_kda` 反向（bf16） | 六项 max_rel **3e-3 ~ 8.2e-3** | 与前向误差同阶，无算法级偏差 |

- **误差归因**：bf16 输入舍入（~3.9e-3）是主项；bf16 状态存储边际 ~1e-4（有衰减）~2e-3（$g=0$）；
  门控 `ex2.approx`/`tanh.approx` 端到端 ≤2e-5（可忽略）。算法（分块/求逆/`MOVM_T`）无可测额外损失。

## 5. 性能（H100 80GB HBM3，bf16，$D=128$）

口径：`wall` = 一个 event 对包 50 次取均值（含 host 派发）；`dev` = CUPTI 纯 kernel 时间。

| 规模 | FlashKDA wall/dev | FLA wall/dev | 结论 |
|---|---|---|---|
| T=128 H=4 | 36.6 / 25.1 µs | 779.7 / 57.8 µs | wall **21×**（赢在省 host 税） |
| T=2048 H=4 | 208.7 / 204.8 | 880.2 / 122.9 | wall 4.2×，dev 0.60（FLA h-kernel 沿 T 并行） |
| B8 T2048 H8 | 337.4 / 333.9 | 886.7 / 741.3 | **wall 2.6×、dev 2.2×**（真实 GPU 收益） |
| T=8192 H=64 | 1261.9 / 1261.2 | 2327.9 / 2288.9 | wall 1.84×、dev 1.81× |

- **FLA 每次调用 ~0.8 ms host 税**（~10 个独立 kernel launch），中小 shape 上 FLA 是 host-bound；
  FlashKDA 单 host 调用、`wall≈dev`。
- **K2 沿 T 串行**：$B=1,H=4$ 时 dev/T ≈ **100 ns/token** 恒定；固定 T 扫 $H=1\to16$ dev 只
  106→126 µs（per-CTA 串行延迟受限，非吞吐受限）。
- **ncu bound**：K1/K2 **都不是 TensorCore bound**（K1 ≤1.6%、K2 22–24%）。放大后两者都撞
  **L2 带宽**（K1 SOL L2 88.2%、K2 填满 grid 时 L2 91.6%），硬成本是 13824 B/tile workspace 的搬运。
  K1 第二瓶颈是 **barrier stall**（27 cyc/issue，来自 union 复用的同步圈）+ 2.7-way smem bank
  conflict；K2 被 smem 钉在 **2 CTA/SM**，设计点 H=64 < 132 SM 时 achieved occ 仅 9.4%。
  **占用率不是 K2 的可优化余量**（`docs/07` §8）：削 pipeline stage 不减 smem（被 64KB fp32 缓冲
  钉死），强抬到 3 CTA/SM 反而更慢，因为 L2 已 94% SOL。

## 6. 反向要点

**推导（`05`）**：从 $o_t=q_tS_t$ 逐 token 求伴随，状态伴随先合成
$G=\bar S_t^{\text{rec}}+\text{scale}\,q_t^\top\bar o_t$，再按状态更新/输出/pred/衰减链拆到
$dq,dk,dv,d\beta,dg,dh_0$；衰减链 $\bar g=\bar\alpha\odot\ln2\cdot\alpha$，门控链到
`g_logits/A_log/dt_bias`。全程 **token 维串行**，块内可并行。

**分块化（`05` §7）**：折叠成「**块内全 GEMM、块间只传 $D\times D$ 状态伴随**」。求逆伴随
**复用前向 $P$**：$\bar L=-P^\top\bar P\,P^\top$（只多两次矩阵乘，不必再求逆）；跨块状态伴随
$\bar S_{\text{out}}^{(c-1)}={q^{\text{dec}}}^\top\bar o+E^{\text{tot}}\odot\bar S_{\text{out}}^{(c)}-{k^{\text{dec}}}^\top\bar w$。

**kernel（`06`）**：FlashKDA 无开源 CUDA 反向，以 FLA Triton 的 5 个 kernel 为参照——
`dAv`(输出支路) / `dhu`(跨块状态伴随) / `wy_dqkg`(逐块 dq/dk/dv/dβ/dg) /
`intra`(块内 $A$ 反向) / 反向 `cumsum` + `kda_gate_bwd`(衰减/门控链)，与分块反向**逐式对应**；
记号映射实测钉死：`Akk`=$P$、`kg`=$k^{\text{rest}}$、`qg`=$q^{\text{dec}}$、`h`=$S_{\text{in}}$。

**目标卡 CUDA 反向设计**：复用 K1 workspace 六字段（$k^{\text{dec}},q^{\text{dec}},k^{\text{rest}},
E^{\text{tot}},P,M_{qk}$），只缺 `beta`/门 logits → 补轻量 **B0 门控重算**，再落三个 GEMM kernel：
**B3 intra**（token 并行，复用 K1 模板）/ **B2 dqkg**（per-tile，吃 B1 的 $\bar S$）/
**B1 state-adjoint**（head 并行，**直接镜像前向 K2 的八个 Phase**，只是逆序扫）。
**B1/B2/B3 均已做成最小可运行 Triton kernel**并位级核验：B1 与 FLA `dhu` 逐行同构、实测 **1e-5**
（tf32 底噪），状态伴随 **bf16 存储可行**（受控消融 ~2.5e-3）；B2 `wy_dqkg` 与 FLA **逐位相同**
（六项梯度 5.8e-4~2.1e-3 = tf32 底噪）；B3 `intra` 的 `dq/dk` 与 FLA **逐位相同**、`db/dg` ~1e-7
（六项 3.0e-4~8.8e-4 = tf32 底噪），三者含非整除 T。
**且这条链已被端到端驱动在 FlashKDA 自己的 K1 workspace 上**（`06` §6.10 P6-8）：按 head-major
SoA 偏移切六字段 + B0，重建全部反向操作数，六项梯度对 autograd 最坏 **6.4e-3**（含非整除 T、$g=0$、
$B=2$）——「复用 K1 workspace」的设计前提**端到端成立**，不再是逐字段点对点核验。

## 7. 踩坑速查

1. **状态布局**：FlashKDA $S^\top$（`[N,H,V,K]`）vs 参考 $[K,V]$；错布局只在初态处暴露。
2. **FLA `transpose_state_layout=True`** 会把传入的 `initial_state` 也当 V-first 重解释。
3. **$\ln2$ 换元**：FLA 用自然对数 $g$、参考用 log2，$\partial L/\partial g_{\log2}=\partial L/\partial g_{\text{fla}}\cdot\ln2$
   （误写成除 $\ln2$ 时 `dg` 的 max_rel 稳定停在 0.52）。
4. **$S_{\text{in}}$ 伴随不可整体乘 $E^{\text{tot}}$**：只有状态更新支路带 $E^{\text{tot}}$，独立累加。
5. **求逆伴随转置顺序**：$\bar L=-P^\top\bar P\,P^\top$（$P$ 非对称时顺序敏感）。
6. **计时**：背靠背 event 对的「单次中位数」会读到真值一半，必须用一个 event 对包 N 次取均值。
7. **workspace 偏移**：`ws_tile_prefix` 按 **launch** 的 total_tiles 算，勿按分配上界猜。
8. **`dhu` 初始 `dS`** 来自终态梯度 `dht`（非初态 `h0`，后者只进 B2 的 `detot`）。

## 8. 交付物索引

| 类别 | 产物 |
|---|---|
| 分析文档 | `docs/01`–`08`（本篇） |
| 参考实现 | `src/kda_ref.py`（前向逐 token + 分块）、`src/kda_bwd_ref.py`（逐 token + chunked 反向） |
| 对拍/基准 | `harness/check_*.py`、`bench_fwd.py`、`probe_ncu.sh`（原始输出 `*.out.txt`） |
| 博客 | `content/posts/flash-kda-00-index` … `flash-kda-08-summary` |

**待验证 / 后续方向**：~~K1「global load 4/32 字节/sector」具体落在哪条 load~~（**已解决**，`docs/07` §7
P7-4：唯一标量 load = `fwd_kernel1.cuh:258` 的 `A_log_ptr[head_idx]`，是 warp 内地址一致的**广播**
标量读，`Sectors Global == Ideal`、`Excessive=0`，`4/32` 为启发式误报，DRAM 占比 <2%）；~~K2 的 L2 流量中
workspace 读 vs $o$/state 写占比~~（**已解决**，`docs/07` §6 P7-3：K2 读 ≈ 75% workspace + 22.5% `v`，
写 ≈ `o` 为主，DRAM 读与闭式预测吻合 1.000–1.010）；~~B1 状态伴随 kernel 的「非整除 T」边界~~（**已解决**，
`docs/06` §6.5 P6-3：尾块 `E_tot` 按最后一个有效 token 取，非整除 T 下 Triton↔FLA `max_rel` 1e-8~4e-5，
可整除回归逐位不变；「K 的 64 分段」经查是 FLA 同 program 内展开、非正确性约束）；~~K2 突破 2 CTA/SM 后
L2 墙是否后移~~（**已解决**，`docs/07` §8 P7-5：不会后移——`sizeof` 被**无条件预留的 65536 B fp32 状态缓冲**
钉死为 98,432 B，只减 pipeline stage 不动 smem；真进 3 CTA/SM 的配置必须砍 pipeline 深度，实测更慢
`+14%~+40%`，且 `lts` 已 94% SOL、K2 是 L2 带宽 bound，占用率无可用余量）；~~FLA host 税的组成~~
（**已解决**，`docs/07` §9 P7-6：FLA 前向 host CPU 固定 ~0.82–0.86 ms、与 T 无关，拆开 =
**12 次 Triton kernel 的 Python launch 包装**（`l2norm_fwd` 单次 77 µs，而裸 launch 仅 2.4 µs）
\+ **~190 µs 后端 `dispatch` 派发** \+ ~75 µs 真 driver launch；GPU 计算仅 ~82 µs、裸分配 0.07 µs；
FlashKDA 3 个 launch、host ~34 µs 且 `wall≈dev`）。**后续方向**：反向四个 kernel 里 B0（门控
重算，`docs/06` §6.8 P6-6）、B1（状态伴随，`docs/06` §6）、B2（`wy_dqkg`，`docs/06` §6.6 P6-4）、
B3（`intra`，`docs/06` §6.7 P6-5）**均已做成最小可运行 Triton kernel** 并三方/位级核验（B0 还额外与
FlashKDA K1 当场写的 workspace `kGTotal` 对到 1.6e-5~8.1e-5），四条支路全覆盖；且 P6-8（`06` §6.10）
已把整条链**架在 FlashKDA 自己的 K1 workspace 上端到端跑通**（六项梯度对 autograd 最坏 6.4e-3）。
剩余的是把最小 kernel 工程化为 CUTLASS/TMA + warp-specialization 的目标卡 CUDA（复用 K1 workspace、
`CHUNK=16`），属更大工程，留待有明确需求时再做。

**反向性能（`docs/07` §10 P7-7 + §11 P7-8 更正）**：把上面那条 workspace 驱动反向链和
`torch.autograd` 穿 `chunk_kda` 在统一口径（`wall`+CUPTI `dev`）下比，**只计反向**。四组 shape 实测
ours **wall 0.40–0.62×、dev 0.61–0.80×** FLA 反向（端到端前向+反向 dev 0.5–0.8×），根因是 **FLA
反向里重算了前向状态**（`...fwd_kernel_h` + `recompute_w_u`，修正后占其 dev 的 **17.8%**，$T512H4$
21.9%），而我们复用 K1 workspace 从不重算——把「复用 K1 workspace」的设计收益从正确性（P6-8）兑现
成性能。注：P7-8 复核发现 P7-7 的表是在**未隔离 GPU**（GPU 0 被另一容器打满）时测的，两侧绝对值都
被抬高约 2×，且关键两行（ours B1 @B2T1024H8 539.8→**214.3**、FLA `fwd_h` 578→**83.6**）不可复现；
比值结论不变，但「重算占 46.2%」应改为 **17.8%**（`docs/07` §11）。**B1 `dhu` 仍是 ours 反向链最大
单 kernel（占 dev 44–63%），但与 FLA 同类 kernel 的差距被更正为 1.6–1.9×（不是 3.9×）**，性质是
「CHUNK=16 粒度（比 64 多 1.3–1.6×）+ 一个常数因子」；**最便宜的杠杆是让 B1 吃 workspace 原生
bf16（不 `.float()`）→ B1 快 2.2–2.5×，B2T1024H8 已快过 FLA dhu**；其次才是按更大 chunk 重组操作数、
最后才是镜像 K2 的 warp specialization。`dAv/B2/B3` 已领先或接近。反向 host 税非主要矛盾
（ours 数的百 µs、FLA 500–730 µs/次，大 shape 全被 GPU 时间隐藏）。
