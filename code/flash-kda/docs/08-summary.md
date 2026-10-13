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
`CHUNK=16`）；**该工程化已由 P7-13…P7-73 完成并收口**——B1 做成 K2 形状手写 CUDA（TMA 输入 + TMA store +
`fence.proxy.async`），B2/B3 做成 K2 形状 CUDA 六段生产核（逐位无损、但整核非净赢），最终部署取
`cf`（`dAv→CUDA-B1→融合 Triton B2/B3→反向 cumsum`），见文末「K2 形状 CUDA B2 反向终局」段。**B1 的 CHUNK=16 粒度损失**已由 P7-10 通过
「从 16-chunk workspace 免重算恢复 64-chunk 操作数」消掉（B1@64 快 2.1–2.3×，但 torch 原型恢复 ~ms 是
硬伤）；**P7-11**（`docs/07` §14）把该恢复融成**单个 Triton pass**（对角块复用落盘 $P16$、只新算 6 个
非对角 $A_{ij}$ + 4×4 块前代合并 + $w64$），kernel-only 降到 **41–953 µs（ieee）/ 16–492 µs（tf32）**、
为 torch 原型的 **0.11–0.25×**，B1@64 端到端仍落 bf16 底噪——恢复不再是相对 B1 的硬伤。
**P7-13**（`docs/07` §16）用 `ncu` 剖析 B1 的硬件 bound：它**不是** tensor/带宽/占用率 bound（tensor
4.5–14.9%、DRAM ≤28.6%、achieved occ 一律 ~6.2%＝小 grid 症状），而是**延迟/并行度受限**——同 BT=64 下
ours 比 FLA `dhu` 慢 2.14×，差在 **`long_scoreboard`（≈47–55% issue 间隔在等全局加载）+ 单一 `[128,BV]`
依赖链 + `num_stages=1` 无流水**，FLA 用「K 拆两条 64 链 + `num_stages∈{2,3,4}` 流水」补上。**P7-14**
（`docs/07` §17）随后在 bf16 Triton 原型上把这两条分开量：单靠 **`tl.range` 软件流水**就把 B1 从 **1.96×**
拉到 **1.00×（`B2T1024H8`）/ 0.85×（`B8T2048H8`）FLA**、`long_scoreboard` 5.27→**0.96**（≈FLA 1.06），
而**拆 K 两条链单独负收益（慢 13%）、与流水叠加更慢**——故目标卡 CUDA B1 的优先级修正为 **只做「镜像 K2 的
TMA 输入流水」**，拆累加链降级为不推荐。**P7-15**（`docs/07` §18）又用**手写纯 CUDA** B1 补了「显式预取后
是否还需拆链」的边界：该原型与 FLA `dhu` **逐位相同**（含非整除 T），但裸 wmma 版**根本没到 load-bound**
（每 CTA 2 warp、~108KB smem、3 GEMM/chunk ~1.9 TFLOP/s），`cp.async` 预取只值 **2–6%**、大 shape 反被占用率
损失反噬（1.40–1.45×）——**「镜像 K2 流水」不是可单独加上的开关，只在 kernel 已是 K2 形状时兑现**；裸 wmma
版要先把结构做到 K2。**P7-16**（`docs/07` §19）再用 `ncu` 剖析同一支 CUDA B1，把 §18.4 的定性归因量化并
**修正一处**：瓶颈**不是全局加载**（`cp.async` 已把 `long_sb` 压到 0.66；换成同步写则暴涨到 3.96、慢 64%）、
**不是同步**（`barrier` 仅 0.02，§18.4 猜的「占用率/发射/同步」要改成「mma 依赖 + smem 取数」）、**也不是
占用率**（与 FLA 同为 6.25% theo / 6.1% achieved 却慢 4.9×），而是 **`wait`（mma 固定延迟，1.80 拍≈36% 发射
间隔）+ `short_sb`（smem 操作数取数）+ 76% shared-load bank conflict**（FLA ldmatrix 仅 0.8%）。故目标卡
CUDA B1 的第一优先级是**换掉 wmma-from-smem（ldmatrix/TMA 整块、提 mma 密度）**，`cp.async`/TMA 预取只是
「不退化」保底。**P7-17**（`docs/07` §20）再用 smem padding（行 stride +8 bf16）的受控实验**修正了这条排序**：
把 bank conflict 从 76% 打到 19%（数值逐位不变）端到端**只快 10–23%**，大 shape BT=64 因 smem +29% 掉占用率
反慢 1.21×；ncu 显示 **`wait` 纹丝不动**——故 bank conflict 是 **~1.2× 分项、不是量级项**，量级差来自**并行度/
mma 密度**。目标卡 CUDA B1 真实优先级＝**(1) 补并行度/warp-specialization、(2) 零成本 XOR swizzle/`ldmatrix`
消冲突（勿用 padding）、(3) 预取保底**。**P7-18**（`docs/07` §21）在动「K2 形状重写」前用最便宜的受控实验
检验「并行度/mma 密度是量级项」里有多少**免费**：同一份 CUDA 原型加 warp 内 mma ILP（循环换序暴露独立
累加链，**逐位不变**，`ilp2` 快 6–12%）与 CTA 形状（`BV=16/64` 反慢）；ncu 显示加 16 条独立链后 `wait` 仅
1.82→1.76、tensor 3.5→3.9%（FLA 14%）、`Eligible Warps/Scheduler` 仍 **0.16**（每调度器只挂 ~1 active
warp，warp 内 ILP 填不满）。**结论**：免费杠杆（conflict ~1.2× + ILP ~1.1×）都只是分项，量级项的落点是
**提高每 warp 的有效 mma 密度**（`ldmatrix`/TMA + 流水，2.3%→14%），不是加 warp 数或加指令级并行。
**P7-19**（`docs/07` §22）**直接把这一步兑现**：手写 CUDA B1 从「wmma-from-smem」改成 **K2 形状**（运行态
$S$ **常驻寄存器**、mma 换裸 `mma.sync.m16n8k16` + `ldmatrix`、操作数走 XOR 填充无冲突 smem），比旧 CUDA
基线快 **2.3–2.7×**、`xFLA` 从 ~8× 收到 **2.1–3.75×**（BT=16 部署粒度 81.8 vs 旧 214.4 µs）；ncu 显示
bank-conflict **827392→0**、tensor **2.3→5.8%**、`wait` 1.79→1.54。**P7-20**（`docs/07` §23）在 K2
形状上**补完 TMA + warp-specialization**（专职 producer warp 发 `cp.async.bulk.tensor`、128 B swizzle、
per-stage `mbarrier`、无 `__syncthreads`），正确性**逐位保持**，achieved 占用 6.32→7.75%、tensor
8.85→**10.73%**（FLA 12.12%）、`short_sb`/`barrier` 降，`BT=64` 只快 **~15–20%**（同形状 `xreg64=0.81–0.85×`）、
`xFLA` 收到 **1.73×**。**P7-21**（`docs/07` §24）**钉死并修掉这块残差的第一部分**：§23.1「省一次 TMA」
 让 TMA 版在 mma 循环里**逐 chunk 直读 global `E^tot`**（reg 版是 `cp.async` 预取），正是 §23 相对 §22
 的 `long_sb` 抬升（2.25→3.52）来源；用一条 512 B 的 `cp.async.bulk.tensor.3d`（`-DB1T_ETMA`）搬回 smem
 后，`long_sb` 降 **0.6–0.8**、`wait` 降 0.16–0.36、tensor 逼近 FLA，**算术逐位不变**，kernel-only
 **0.77–0.98×**、**全表最好 `xFLA=1.63`**（B8T2048H8 BT=64 **NS=1**：218.5→**195.0 µs** vs FLA 119.4）。
 **结论**：TMA+warp-spec 是**真实但次线性**的一步（约 +15–20%）；**「加深 pipeline」与「占用率」两条
 路线都被实验排除**（`E^tot` 修好后最优回到 `NS=1` 的 2 CTA/SM；FLA achieved 占用 6.15% 比 ours 7.7%
 还低却快 1.7×）；**剩余 1.63× 已定位到每 warp 发射效率**（`wait` 1.24 vs FLA 0.96、`short_sb` 0.90 vs
 0.50）＝手写 mbarrier 最小实现的结构上限。**剩余工程化**＝换 CUTLASS `PipelineTmaAsync` 或直接接受
 Triton 版（§14/§17 已到 1.0×/0.85×），不再在 stage 深度/占用率/ILP/padding 上找。

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

**P7-22（`docs/07` §25）**：把 §17 的 **Triton 软件流水**从孤立原型**落进真实组装链**（§12 的
链此前写死 `num_stages=1`）——只把 B1 的反向 chunk 循环从 `range(NT-1,-1,-1)` 换成前向
`tl.range(0,NT,num_stages=NS)`（算术一字未动），`NS=3` 下 B1 从 122→**61.9 µs**（`B2T1024H8`）、
387→**319**（`B8T2048H8`），且 `dq/dk/dv/dg/dh0` 与 `NS=1` **逐位相同**（`dbeta` ≤1.5e-8 =
`atomic_add` 序）。这是「**直接接受 Triton 版**」那条路线的链级落地：链级瓶颈明确转向 B2/B3
（§15 已证整链跟到 64 净亏）。方法学：`docs/07` §25.5 指出**孤立小 grid kernel 的 CUPTI 归因比链内
背靠背高 ~1.3×**（固定启停/排空被算进 device-active 窗口），此后 kernel-only 微观基准须以
**链内归因**为链级口径。~~B1 在所有 shape 反超 FLA `dhu`（2.2×）、整链 dev/FLA 0.48–0.54×~~——
**P7-34（`docs/07` §37）更正**：该结论的分母是 **fp32** FLA `dhu`，而我方 B1 是 **bf16**，属
**跨 dtype**；matched bf16 下 FLA `dhu`（BT=64）只有 **13.9 / 26.7 / 126.0 µs**，故 B1 实为
**慢 ~2.3–2.5×**、整链 `dev/FLA` 实为 **0.86–0.95×**（与 §22–24 CUDA 原型的 1.7–3.8× 自洽，
差距来自**部署的 16 块粒度**；孤立 BT=64 时 ours≈FLA，见 §17.2）。诚实定位＝**与 FLA 反向同档
（~0.9×）且省掉 FLA 的前向状态重算**；工程重点（B1 粒度 + store、B2/B3 融合）不变。

**P7-28（`docs/07` §31）**：消掉 B3 `intra` 的**链外 elementwise 税**——组装链此前让 B3 在**全新置零**
的 `dq/dk/dg` 上跑、再在 kernel 外物化 3 张 `[B,T,H,K]` 的求和（`check_bwd_b1_pipe_e2e.py:162-166`），
白读零缓冲 + 又读写一遍 B3 产出 + 3 个 memset。FLA 的 `chunk_kda_bwd_intra` 本就要求传入的
`dq/dk/dg` 是**累加器**（`chunk_intra.py:554-560/657-667`：读旧值、加块内贡献、store 到独立缓冲）。
改成 B3 **就地累积进 B2 缓冲**（grid 的 `i_k`×子块轴划分每个 `(token,k)`、无跨 program 竞争；`db`
仍 `atomic_add`），**kernel 算术逐行不动**：`dq/dg` **逐位相同**、`dk/db ~1e-7`（fp32 加法次序），
落链正确性 vs FLA 逐值同 §25/§26；链 `dev` **0.84–0.88×**、`wall` **0.76–0.86×**，per-kernel 归因
显示 elementwise 从 **316→49 µs**（`B8T2048H8`）、每次少 8 个 elementwise/memset kernel。**约束**：
B3 契约改为消费 B2 缓冲（其后 B2 输出不可再用）。下一步＝§27.7 剩下的 kernel 内对角平铺 L1 局部性优化。

**P7-29（`docs/07` §32）**：检验 §27.7 遗留的「B3 减少对角平铺重复读 / L1 局部性」，给**空结论**——
两处冗余读（`k/g` 双读被 CSE、`dAqk/dAkk` 每 `i_k` 重读被 L2 吸收）都不进 DRAM、**DRAM 字节逐位不变**、
`reuse` 恒 1.00×、`BK=64` 只值 ~5%。真杠杆是**把 `k` 分块从 grid 轴搬进 program 内并 `tl.range` 软件
流水**（`_intra_kernel_loop`）：孤立大 shape **0.79–0.81×**、`long_scoreboard 16.7→4.0`、DRAM 不变；
落链 `dev` **0.97×**、六项梯度逐值不变。定位：B3 kernel 级优化收敛（已彻底 L2/DRAM bound）。

**P7-30（`docs/07` §33）**：兑现「**B2/B3 融合**」——部署粒度 `NC=1` 下 B3 只剩**对角** Parts B/D
（两个 `tl.dot`），折进 B2 的 `(NT,B*H)` program（B2 结束时已有完整 `dAkk`），把 intra 贡献加到
`dq/dk/dg` **寄存器**上再存一次、`dA` 不落盘。`dq/dv2` **逐位相同**、`dk/db/dg ≤2.2e-7`（fp32 加法序，
含非整除 T）；落链六项梯度 vs FLA **逐值同 §25/§26**，链 `dev` **0.83–0.93×**、`wall` **0.90–0.92×**、
DRAM 字节 **−23%**、L2 **−17%**（`B8T2048H8`；`ncu` Duration 1193→1050 µs，仍 1 block/SM）。**唯一使能
条件＝`BK=K`+`BV=V`+`nw=8`**（留 `[BT,K]` 累加器只能 `BK=K`，靠 `BV=V` 把溢出从 222/60 压到 **42**），
是**结构前提**而非可选旋钮。至此 Triton 侧反向三元组（B1 流水 §25、B2/B3 融合 §33）工程化收敛。

**P7-31（`docs/07` §34）**：量化「K2 形状内 B1→B2 融合」的**收益上界**（不建融合 kernel）。受控消融把
`dh` 落盘往返拆两半——**B1 去掉 `dh` store 快 0.28–0.67×**（`dv2/dh0` 逐位不变），`ncu` 定位为**逐标量
`st.global` 的 `lg_throttle`（0.67→0.06）+ L2 流量放大 ~3×**、**不是 DRAM 带宽**；**B2 不读 `dh` 只值 ~5%**。
故**往返上界 ≈ B1+B2 的 ~10%，且 ~2/3 在 B1 侧且属 store 效率（非融合专属）**。**修正 §30.5 的下一步**：
目标卡 CUDA 反向优先把 **B1 的 `dh`/`dv2` 落盘改成 TMA / 专职 STORE warp**（镜像前向 K2 的 STORE warp），
而非 B1→B2 融合——融合相对「高效 `dh` store」的净增量只剩 ~5%（同 §26/§29 对 `h/dh` 的判定）。

**P7-32（`docs/07` §35）/ P7-33（`docs/07` §36）**：兑现上条的**待验证**——把 B1 的 `dh`（§35，`-DB1T_TMADH`）
与 `dv2`（§36，`-DB1T_TMADV`）两条逐标量 `st.global` **都换成 TMA store**（per-warp 未 padding tile + 双缓冲 +
一块一个 bulk group，OOB 尾行由 TMA 按 `T` 自动裁）。两条均与 scalar 路径 **逐位相同**（含非整除 `T`）；
`dh` TMA 恢复「删 store」上界 58–84%，`dh`+`dv2` 全 TMA 后 `tma2/base` = **0.39–0.71**、对「两条全删」上界
恢复 **62–83%（均值 75%）**；`ncu` 把 `lg_throttle` **0.66/0.90→0.04/0.07**（坐实「逐标量 store 打满队列」）、
DRAM write 基本不变（**不是带宽**）、L2 字节 **−52%**、tensor 抬到 **15.5/16.9%**。至此 B1 的 store 侧收敛，
下一块瓶颈只剩**每 warp 发射效率**（`wait` 1.40），即 §24.5 的手写 mbarrier 结构上限（需换 CUTLASS
`PipelineTmaAsync`，属更大工程）。**坑**：TMA 张量图 `globalStrides` 按 dim 从 1 起算，写反 T/H stride 时
`H==1` 全对、仅 `H>1` 时 `dv2` 错。

**P7-35（`docs/07` §38）**：把「K2 形状 CUDA B1」的累计收益**收口成 matched-bf16 的 `xFLA`**（§22–§24 报过、
§35/§36 的 store 轮只报相对收益），并**发现手写 pipeline 的 run-to-run 竞态**。三变体
`base`/`tma`/`tma2`（=ETMA/+TMADH/+TMADV）在 `BT=32/64 NS=1/2`（含非整除 `T=250`）`tma2` vs `base`
**逐位相同**。**`xFLA`（ours/FLA bf16，同 `BT=64`）**：大 shape `B8T2048H8` **1.15（NS=1）/0.83（NS=2）**
（§24 标量 store 时 1.63 → TMA store 兑现「大 shape 追平 FLA」）；小 shape `2.0–2.7×`（残差＝每 warp
发射效率）。**BT=16 无法在探针测**（FLA 前向只支持 chunk 32/64），部署口径仍归 §37.3。**踩坑（新）**：
run-to-run 确定性扫描暴露 `base` kernel 在 `NSTAGE≥2` 大 shape 上不可复现（`BT=16 NS=2` 3.64e-2、
`BT=32 NS=3` 4.52e-3），`NSTAGE=1` 全 `BT` 稳定 → §35/§36 的「逐位相同」属**观测而非保证**。

**P7-36（`docs/07` §39）**：**定位并修掉**上条竞态。根因不是 `full`/`empty` parity，而是手写 TMA 流水
复用的 stage 缓冲缺一条 **`fence.proxy.async`**：`full`/`empty` mbarrier 只在 **generic proxy** 内定序，
consumer 用普通 `ld` 读 `st->e`、producer 用 `cp.async.bulk.tensor`（**async proxy**）回填同一缓冲，WAR
未定序 → 间歇读到被覆写的字节。9 变体 run-to-run 扫描（`B8T2048H8 NR=40`）：`base` racy（`16/2`=39/40、
`32/3`=19/40）、`noetma`/`dropread`=0、`proxy`/`pro`/`con`（`fence.proxy.async`，两侧任一侧单独加都够）
**全 0/40**、`esync`/`evol` 修不了、`sync` 能修但折回串行。错误形状＝**一整个 compute warp 的 16 列
V-slice 跨连续若干 chunk**（WAR 指纹）。`proxy/base`=**0.96–1.02（免费）**，`proxy` `NS=2/3` vs `base`
`NS=1` `dh` **逐位相同**。**回授撤回**：多 stage 安全不需要 CUTLASS `PipelineTmaAsync`，`-DB1T_EPROXY`
即修复；手写「TMA 复用缓冲」的握手必须显式 `fence.proxy.async`（`__threadfence_block`/`volatile` 会漏）。

**P7-37（`docs/07` §40）**：关闭上条的 **store 侧待验证**——`tma2` 输出侧也有对称的一对 generic↔async
（generic store 写 `dh`/`dv2` tile → `cp.async.bulk.tensor` 读出去，**RAW**）。加 `-DB1T_SPROXY`（写完 tile、
`__syncwarp()` 后、lane 0 发 store 前一条 `fence.proxy.async`，全 lane 执行）。`NR=40` 确定性扫描：
`scalar` 三次大配置全 racy（39/40），而 **`tma2` 全族（含无 fence 的 `tma2`）全 0/40**——输出侧 RAW/WAR
**在本探针观测不到**（但「观测 clean≠有保证 clean」，部署仍取两侧 fence）。`tma2_full` vs `scalar_p` 16 格
`dh/dv2/dh0` **逐位相同（含非整除 T=250）**、vs bf16 FLA 1 ulp；`full/tma2`=**0.997–1.081（噪声内，免费）**；
`tma2_full` `NS=2/3` vs `NS=1` 逐位相同。**部署配置**＝`-DB1T_ETMA -DB1T_TMADH -DB1T_TMADV -DB1T_EPROXY
-DB1T_SPROXY`（`NSTAGE` 可 >1）；「输入/输出两个方向的 TMA 复用缓冲都需 `fence.proxy.async`」收口。

**P7-38（`docs/07` §41）**：关闭上条的**唯一边界（更深 pipeline / 不同 store 时序）**——给探针加
`NSTAGE=4/5/6` 与输出缓冲深度宏 `-DB1T_SDEPTH`（输出 tile 缓冲 1–4 深、`i % SDEPTH` 复用）。`NR=80`
确定性扫描覆盖 `BT`×`NS`(≤6)×`SDEPTH`(≤4)×4 变体：标量对照稳定复现输入侧竞态（`BT16 NS2/3/4`=79/80），
而 **`tma2` 全族（含无 fence 的）在全部 60 个组合上仍 0/80**——**加深 pipeline / 推极端 store 时序仍
观测不到**两侧 `fence.proxy.async`（扫描有功率）。细节：`scalar` 的 racy 概率随 `NS` **反降**（强时序
相关，clean 不可外推）。**新正面数据**：TMA store + `BT=16/32` 上深 pipeline 有正收益（`B8T2048H8`
BT16 `NS=1→4` 1.69×、BT32 `NS=1→3` 1.63×），**`NSTAGE` 建议 ≥4（修正 §24 的「回到 NS=1」**，那是
BT=64/标量 store 的结论）；`SDEPTH` 非旋钮。数值深配置与 `NS=1`/`scalar_p` **逐位相同**、vs bf16
FLA 仍 1 ulp。

**P7-39（`docs/07` §42）**：第一次把 K2 形状的**手写 CUDA B1 落进完整反向链**（此前全是孤立 kernel
vs kernel；组装链 B1 一直是 Triton）。给 `b1_tma_probe.cu` 补 `dht`（终态伴随）入口
（`b1t_launch_dht`，语义对齐 Triton 的 `HAS_DHT`）。**drop-in 核验**：同 workspace 操作数、`dht≠0`
下 CUDA-B1 vs Triton-B1 `dh/dv2/dh0` `max_abs` 1.4e-5~3.7e-4（=1 bf16 ulp）。**链级正确性**：
CUDA-B1 链（`BT=16`、`NSTAGE=4`）vs FLA 六项最坏 `dq 3.42e-3…dh0 9.37e-3`，与 Triton-B1 链同阶
（`dq`/`dh0` 逐位相同）。**链级 wall**：CUDA-B1 链 0.94–1.03× Triton-B1（`B8T2048H8` 快 6%），
不拖慢整链；~~`T250H3` 的 1.65× 回归＝launch 每次重编码张量图的 host 税（**待验证**）~~——
**P7-40（`docs/07` §43）已撤回**：加 `-DB1T_MAPCACHE`（编码一次复用）后 cache 只省 0.15–0.75µs、
空闲卡 `T250H3` 实为 **1.049×**，1.65× 不可复现（§42 的 Triton 基线抽到低离群）。**单核**：
部署 `BT=16` 下小 shape CUDA 1.05–1.63×、大 shape **0.53×**（1.9× 快）——「手写 CUDA 是否比
Triton 快」**依赖 shape**。至此目标卡 CUDA 反向**第一次端到端可跑**（B1=CUDA、其余 Triton）。

**P7-40…P7-45（`docs/07` §43–§48）——收尾段**：P7-40 撤回 §42 的「张量图重编码＝host 税」（1.65× 不可
复现、纯编码 3.9µs 被 launch 隐藏）；P7-41 把 B1 手写 mbarrier 换 CUTLASS `PipelineTmaAsync`＝**空杠杆**
（`cut2/hnd 0.985–1.003`、ncu 逐指标重合），手写已到结构上限；P7-42 把 CUDA B1 与融合 B2/B3 **合成一条链**
（可加、`cf/cu 0.67–0.71×`、`dev/FLA 0.31–0.37×`），B1 按 grid 切换、B2/B3 一律融合；P7-43 给「融合 B2/B3
→ K2 形状 CUDA」**go/no-go**：延迟受限、同字节地板核 509µs vs 融合核 988µs＝**1.94× 头寸**、Triton 够不着；
P7-44 把「放弃 `tl.dot` 手写 SIMT」读法**证伪**（慢同 op Triton 核 3–5×、L1/TEX bound），端口必须保留
`mma.sync`；**P7-45** 用受控**字节消融**（同核只把选定输入 `tl.load`→`tl.zeros`）把 §47 的「头寸在字节还是
发射」问清：`no_hdh` −1074MB→**0.56×**、`bf16_hdh`（h/dh 砍半）**Duration 持平（wash）**、`no_qkg` −201MB→
0.73×、`no_all`（大操作数不读）**170µs=0.16×**；ncu：`full` DRAM 55.9% SOL、tensor 6.21%、issue 0.14/cyc、
1 block/SM。**结论**：头寸＝**大操作数（h/dh）常驻寄存器→spill/发射饥饿**，非纯字节、非占用率；`h/dh` bf16
非独立杠杆；K2 形状端口首位动作＝**大操作数出寄存器保活、只留 warp 分片小累加器**。

**P7-46…P7-71（`docs/07` §49–§74）——K2 形状 CUDA B2 反向的「可行 → 贵 → 收口」全过程**：这一整段把
§47/§48 定位出的「融合 B2/B3 有 ~1.9× 延迟头寸、Triton 够不着」真正做成手写 CUDA，并诚实收口为「**结构
闭合、逐位无损，但整核仍比融合 Triton 慢 ~1.7×**」。链条：

- **§49（P7-46）首个正面**：保留 `mma.sync.m16n8k8.tf32` + `cp.async` 流式操作数 + 寄存器只留每 warp `[16,16]`
  小累加器的 K2 形状 B2 GEMM 核，`B8T2048H8` **CUDA 517µs vs identical-op Triton 554µs = 0.93×**（本专题
  **第一个跑赢同 op Triton 的反向 CUDA 核**，离 §47 地板 442µs 1.17×）；regs **226→70**、occ 12.4→24.6%、
  DRAM 79.9→**84.6% SOL**。**§47 的「放弃 `tl.dot`、手写 SIMT fp32」读法在同轮被证伪**（慢同 op Triton 3–5×、
  L1/TEX 89.6% bound），故端口**必须保留 tensor core**——SIMT vs tensor 判据是「**算术强度 op/word**」。
- **§50–§54（P7-47…51）把核做成 K2 形状的正确形态**：bank conflict **8.81M 的 ~97% 来自 `dgk` 列归约**
  （`-DB2M_DGKCF` 归零，但部署 shape 不兑现）；L2 **+35%** 拆成 **`cp.async .BYPASS` 读路径（+0.64GB）
  \+ 逐标量 32-bit store（+0.13GB）**，**TMA 输入**把读放大整条消掉且**净赢 ~5%**（`tma_vecst` 比 identical-op
  Triton 快 **1.12×**、贴 DRAM 地板 1.11×）；`TMA+BVK=64` 只能走 **NAT 个 128B swizzle atom** 拆分
  （`SWIZZLE_NONE` 2.5–3× 不可用），`atom64/base32` **0.853–0.974×**。**向量化 store 一律做**（免费逐位、小 shape 10%+）。
- **§55–§60（P7-52…57）六段非 GEMM 全搬进 CUDA**：`dv2`(tf32)/`dkgb`(tf32)/`dApre`(fp32 **5.31e-7**，但贵
  1.49–1.84×，收缩维是全 `V`/`K` 的标量 FMA)/`dA` sandwich(近免费 1.02–1.10×，**并更正 §57 的 `dw` 符号**)/
  `db`(fp32 1.3–1.8e-7，+2~13%)/门控链(`dq` 逐位、`dg` fp32 序 2.2e-7，**最贵 +30~38%**)。
  **至此 B2 六项 `dq/dk/dv2/db/dg/dA` 全在 CUDA**。
- **§61（P7-58）把门控「`q/k/g` 合流进 smem」证伪**：非 TMA 版常驻三张 `[BT,KD]` tile 把 smem 顶过占用率
  阈值（2→1 block/SM）→ 反慢 **1.11–1.51×**；TMA 版 global-load sector 确 **−34%** 但本核 **DRAM/L2 双 bound**、
  `Duration +8.7%`——**散读被 L2 吸收、不进端到端**（ncu 计数当成本前须做受控对拍）。
- **§62–§64（P7-59…61）六段并成单生产 launch**：`-DB2M_PROD`（一次 launch 摊掉两份重复 GEMM 核，**−47~53%**）、
  `-DB2M_HDW`（同 CTA `dw` 交接，`dw` 从 ABI 消失；隔离段快 7–9% 但生产核 **wash**——原往返被流水藏住，
  交接反成串行尾巴）、B3 对角 intra 折进 K2 形状（`b3_intra_probe.cu`，比 Triton MMA B3 快 **1.3–1.45×**）。
  **K2 形状 CUDA 化在结构上闭合**（中间量要么本就核内、要么可核内）。
- **§65–§68（P7-62…65）折进 vs 多核的诚实账**：`-DB2M_INTRA`（B3 intra 折进单 launch）**逐位无损但大 shape
  反慢 1.04–1.06×**、小 shape 0.88–0.89×（独立 B3 的工作量在 executor 层被更好流水，折进成关键路径尾部串行尾巴）；
  生产核落完整链**正确**（`dv/dh0` 与最优链 `cf` 逐位）但**整核比融合 Triton B2/B3 慢 1.27–1.73×**；
  逐段归因（`-DB2M_LEVEL=k`）六段非 GEMM 共 **651.6µs**、**OUT+DA+GATE 占 83%**——是**交付物 IO + 标量数学**，
  不是可再融合掉的中间量（重新融回寄存器上界仅 ~40µs）。
- **§69–§74（P7-66…71）研究形态内部收尾**：`mma.sync` 化 DA（段内边际 185→**83µs = 0.45×**，整核仅
  0.93–0.95×）；TMA×DAMMA（`mma_tma/mma` 0.978–1.006，**TMA 的生产代价原本就是「标量 DA 的 +4.85M bank
  conflict」**，随 DA 实现切换）；DAMMA 进链（六项逐位/≤8e-6、链级快 1–2.5%）；`dA/dApre` 的 `st2` store
  （0.93–0.96×，**相邻列扇区已最小、向量化只砍指令数**）；`-DB2M_VECST` 落整链并与 TMA/DAMMA 三向交叉
  （**四对孪生六项全逐位 `max_abs=0`**、链级 0.968–1.003×）。**方法学**：同叫「标量 store」先看线程→地址映射
  （是否连续）再看 bound 类型；「段内砍 2×」与「整核砍 2×」是两回事（要乘段占比）。

**最终部署建议（`docs/07` §45/§67/§74）**：目标卡 CUDA 反向默认 ＝
**`dAv`(Triton) → **CUDA-B1**（K2 形状、TMA store、`NSTAGE≥4`、`BT=16`）→ **融合 Triton B2/B3** → **反向 cumsum**
（代号 `cf`）：
整链反向 `dev/FLA` **0.31–0.37×**（FLA 反向还额外重算了前向状态）、两优化**可加**（`cf/cu` 0.67–0.71×）、
对全 Triton 反向链 `cf/fu` 大 shape **0.91×**（小 shape 因 B1 依 shape 交叉 1.05–1.16×，故 **B1 按 grid 切换**）。
**CUDA B2 生产核**（六段单 launch）**正确、逐位可核验，但在部署链上不是净赢（~1.6–1.7×），不作为默认**——
它每一段都「CUDA 赢 Triton」，六段叠进同一个 1 block/SM、tensor 3%、发射受限的核后，反被 Triton「中间量留寄存器
\+ 融进一张 `tl.dot` 图」打败。**核心判据**：段级赢**不能外推**到整核，凡「把独立核合起来」必须以**整链 `dev`** 收口。

**P7-72 / P7-73（`docs/07` §75/§76）——研究形态收尾与「bf16 原生入口」**：§75（P7-72）关闭 `BVK=128`（数值逐位
但无收益：NS 被迫=1 + 无 V-tile 并行）与 `atom64` 的 L2 +0.04 GB 归因（**NAT×NS 相关的 atom 边界 L2 重触**，
srcunit/DRAM/全局读三者恒定）。§76（P7-73）兑现「**bf16 原生入口**」：B2 GEMM 核的五个 streamed 操作数改
bf16 staging（mma 保持 `m16n8k8.tf32`，因 **bf16 ⊂ tf32** 故与真 bf16 mma 逐位），与 fp32 核在 `bf16.float()`
上**逐位相同**（`.float()` 往返是 drop-in）、大 shape kernel **0.73×**、**DRAM 读精确减半**（1.28 GB→637 MB）。
即 §67.4「`.float()` 往返是部署税」里 **B2 侧那部分可用 bf16 入口消掉**。（仅覆盖非 TMA 基础 GEMM 核，未落链。）
