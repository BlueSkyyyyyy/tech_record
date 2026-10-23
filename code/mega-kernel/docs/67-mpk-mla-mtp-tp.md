# MPK MLA-MTP decode 的 TP 三兄弟走读：per-rank 头/KV/分裂分区、Oa/La 部分量 ABI 与 flat-grid 适配（P153-1）

> 定位：本文补掉「当前进度/下一步」**明留的下一步候选**之一——
>
> > 或 `mla_mtp_decode_tp2/4/8_sm100.cuh` 三兄弟的**内部未逐行处**。
>
> `docs/34`（P120-1）已把 MLA-MTP 家族的**基版**（`mla_mtp_decode_sm100.cuh`，748 行，QK→softmax→PV→
> online-rescale→归一化 + reduce）逐段走完，并且对 TP 变体给了一段**差异摘要**（`docs/34` §3.4）。但那只是
> 「骨架一致、差异集中在定义常量」的**目录级概括**：三兄弟本身（tp2 796 / tp4 787 / tp8 775 行）的
> **TP 专属主循环与分区语义从未逐行打开**。本文补这一段。
>
> 快照 `mirage-project/mirage` 分支 `mpk`，commit `f9eb70c`（本地 `~/mirage_mpk_build`，与
> `docs/06`–`docs/66` 同快照）。**纯源码走读（资料分析），无新实验**——三份身体是 SM100（`sm_100a`）
> 专属，在目标卡 H100/sm90 不可执行。所有行号为快照静态事实。

---

## 0. 三条 headline

**(A)（headline）「基版」不是单卡版，而是被参数化过的「TP1/父版」；真正的三种 TP 是三个独立文件，
且它们的**块索引契约与网格形状和基版不同**。** 基版 `mla_mtp_decode_sm100_task_impl` 的签名带运行期
`local_num_heads = mla_mtp::NUM_HEADS(128)` 与已分解好的 `gi/si/bi`（`:81-89`），`hpb = local_num_heads /
num_head_groups`（`:123`，注释 `:70-73` 明说「TP 模式下等于 num_heads/tp_size」）——**基版本身就能表达
任意 per-rank 头数**，只是靠**额外运行期参数**；而 tp2/4/8 是**另外三份文件**，`NUM_HEADS` 硬编码
`64/32/16`（tp2 `:40` / tp4 `:29` / tp8 `:26`），签名改成收**原始 `block_x/block_y`** 再在 kernel 内
`gi = block_x / sk`、`si = block_x % sk`（tp2 `:157-158`）、`bi = block_y`。前端把 TP 的网格从基版的
**3D `(num_splits, num_head_groups, max_req)`** 换成 **2D `(num_groups*num_splits*x_mul, max_req, 1)`**
（`persistent_kernel.py:1837-1840`）——**TP 路径把 `(group,split)` 压成扁平 x 维**，这正是 kernel 内要
自己除法拆包的原因。

**(B)（headline）tp4 与 tp8 各有一处「MPK 逼出来的形状适配」，基版没有。** (i) **tp4 的 V-split**：tp4 把
`D_V=512` 沿 V 维一分为二（`PV_CHUNKS = V_CHUNKS/2 = 4`，`:149`），两个 half 各做一个任务，并把
`v_half` **折进 `block_x` 的最低位**（`:141-143`，注释原话 *"Mirage MPK has only a flat task index, no
z-dim launch"*）——因为 MPK 只给一维扁平 task 索引，没有 z 维 launch 来表达「第三维 = v_half」。副产物是
**`La` 只由 `v_half==0` 写**（`:711-713`，两个 V half 共享同一份 LSE）。(ii) **tp8 的 `Q_LEN` 偶对齐**：
tp8 要求 `Q_LEN` pad 成偶数，故签名比 tp2/tp4 多一个 `int Q_LEN_real`（`:124`），**所有下界/因果判断
用 `Q_LEN_real`**（`:135-136/346-357`），而 grid 计算用 pad 后的 `Q_LEN`（`num_groups = Q_LEN/qpg` 无
`ceil`，`:146`）——注释解释「TP=8 下 `qpg=2`，`Q_LEN` 已 pad 成 `qpg` 倍数」。

**(C)（headline）三兄弟共用同一份「部分量 ABI」：`Oa`/`La` 是 per-(batch, group, split) 的未归一化
attention 输出与 log2-LSE，reduce 再做一次 log-sum-exp 归并；TP 只改 `NUM_HEADS`/`qpg`/`RD_DV`。**
decode 主核把 `Oa`（bf16，`[num_groups*sk, D_V, 128]` 视角、行 = 128 个 (query,head) 槽）与 `La`
（fp32，`[num_groups*sk, 128]`，`log2f(max(row_sum,1e-30))+row_max`，tp2 `:720`）写到全局；reduce
（tp2 `:732-796`）对 `sk` 个 split 做「`max` 递推 + `exp2` 预测权重」的 **online LSE 归并**，最后按
`inv_sum` 归一化写回 `O`。这套 ABI 在基版 `docs/34` §3.3 与三兄弟**逐位同构**（tp4/tp8 只在细节上加
`q_in_group>=qpg` 守卫、`La` 单写者门）；生产上由 `register_mla_mtp_decode_*` / `..._reduce_*` 生成，
`world_size` 2/4/8 选路（`builder.py:743-772`）。

---

## 1. 定位：本文与 `docs/34` 的关系

| 层 | `docs/34`（P120-1）已做 | 本文（P153-1）补 |
|---|---|---|
| 基版 `mla_mtp_decode_sm100.cuh` | QK/softmax/PV/epilogue/reduce **逐段** | 只补它「TP-aware 父版」的签名与 `hpb` 语义（§2） |
| TP 差异 | `docs/34` §3.4 **一段摘要**（NUM_HEADS/bar.sync/qpg/PV/RD_DV） | tp2/4/8 主体**逐行**：块索引契约、V-split、Q_LEN 对齐、reduce 索引（§3–§6） |
| 生产布线 | `docs/34` §4 起止行号 | `world_size` 选路 + 三份 `register` 的 `params`/`grid`/`qpg` 对照（§7） |
| ptx 命名空间 | `docs/31` §9 一句「各自重定义」 | 逐文件核对（§8） |

判据口径同 `docs/07`/`docs/29`–`docs/66`：**一个 device task 身体只贡献 C3**；C1/C2/C4/C5 由运行层拿
（本文 §9 重申）。

---

## 2. 分区模型：TP 到头/KV/分裂这一层到底切了什么

DeepSeek-V3 MLA 的 MTP 投机解码一次处理 `Q_LEN` 个 query（≤8，`q1to8` 区制）对一段 KV。整个 attention
被切成三层并行（对基版/TP 通用）：

- **head 分组（`gi`，`num_groups = ceil(Q_LEN/qpg)`）**：把 `Q_LEN` 个 query 按 `qpg` 个一组；
  `128/Q_LEN` 个 head 每 query 地被分到若干 block。注意 **`M=128` 的 TMEM 行槽映射为
  `Q_LEN_grp × (128/Q_LEN)` 个 (query,head) 对**（`docs/34` §3.2）。
- **KV split（`si`，`sk = num_splits = ceil(kv_len/128)`）**：KV 按 `TILE_S=128` 切块，再摊到 `sk`
  个 split；每个 split 只看 `[t0,t1)` 段 tile（tp2 `:163-170`）。
- **batch（`bi`）**：这里恒 `bi = block_y = request_id`（tp2 `:158-159`）。

**TP 的「切什么」只切 head**：per-rank 只持有 `NUM_HEADS = 128/TP` 个 head 的 Q/KV 切片，KV split 与
batch 维不变。因此：
- `hpb`（heads per block）在 TP 里恒 `= NUM_HEADS`（tp2 `:177`，一个 block 处理本 rank 的**全部**本地
  head）；基版则是运行期 `local_num_heads/num_head_groups`（`:123`）。
- `qpg`（queries per group）随 `NUM_HEADS` 变：`64→min(2,q_len)`、`32→min(4,q_len)`、`16→2`
  （`persistent_kernel.py:1829-1834`）——head 少则 query/block 多。
- Q 张量按本地 head 索引：`global_row = bi*Q_LEN*NUM_HEADS + actual_q_idx*NUM_HEADS`（tp2 `:299`），
  即 Q 的最后一维是**本地** `NUM_HEADS`。

**为什么基版与 TP 是不同文件而不是一份模板？** 基版已经参数化 `local_num_heads`，理论上 TP=2/4/8 可以
复用它。快照选择三份独立文件（文件头 `:3-13` 注明是对 ferret standalone 的**机械 port**：`__global__`→
`__device__ __noinline__`、`blockIdx`→参数、`__grid_constant__ CUtensorMap`→指针、删 `cudaTrigger...`
PDL、加线程守卫），三份各自硬编码 `NUM_HEADS`/`qpg`/PTX 命名空间——**这是「同一族实现 N 遍」在
MTP attention 上的又一肉身**（同 `docs/58` §2「`tcgen05` 封装三份」、`docs/52`/`docs/61` 的「P2 实现两遍」）。

---

## 3. 基版是「TP1 父版」：签名与 `hpb`（补 `docs/34` §3.2 的一处误读）

`docs/34` §3.2 把基版当作「单卡版」描述。更精确的读法是它**已经是 TP-aware 的父版**：

- 签名（`mla_mtp_decode_sm100.cuh:79-89`）：末位带默认参 `int local_num_heads = mla_mtp::NUM_HEADS`；
  入参是**已分解**的 `gi/si/bi`（`:86-88`），不是原始 `block_x`。
- `hpb = local_num_heads / num_head_groups`（`:123`，注释 `:70-73`「单卡 == NUM_HEADS；TP 模式 ==
  num_heads/tp_size；用于 Q 索引与边界检查以支持 per-rank 本地头数而无需 padded buffer」）。
- 生产上基版由 `register_mla_mtp_decode_sm100_task`（`task_register.cc:4405-4471`）发射，**不传**
  `local_num_heads`（走默认 128），`gi = task_metadata.request_id`、`si = task_metadata.kv_idx`、
  `bi = merge_task_offset`（`:4466-4468`）。
- reduce 侧基版同样是参数化的：`mla_mtp_reduce_sm100_task_impl<NUM_THREADS_=512>` 带
  `local_num_heads = mla_mtp::NUM_HEADS`（`:684`），并用 `hpb = local_num_heads/num_head_groups`（`:697`），
  MPK 实发 `<256>`（`register...` `:4473-4508`）。

**含义**：基版是「用运行期参数表达 TP」的路线，三兄弟是「用编译期常量 + 独立文件表达 TP」的路线；两条
路线在快照里**并存**，且各自有完整生产布线（基版走 `world_size∉{2,4,8}` 的 fallback，
`builder.py:767-772`）。这解释了一个表面上矛盾的现象：基版 `mla_mtp_decode_sm100.cuh` 明明「会 TP」，
却仍存在三份独立的 TP 文件。

---

## 4. tp2（`NUM_HEADS=64`）内部：扁平 x 维的拆包

`mla_mtp_tp2_main< SINGLE_TILE >`（`:131-730`）。与基版骨架相同（`docs/34` §3.2），**TP 专属的只有索引
与常量**：

- **块索引拆包**（`:157-159`）：`gi = block_x / sk`、`si = block_x % sk`、`bi = block_y`。这里 `sk`
  即 `num_splits`（split 数），而 `block_x = task_metadata.kv_idx`（`task_register.cc:5244`）——TP 前端把
  `(group, split)` 拍成扁平 x 维（`num_groups*num_splits`），故 kernel 内自己做整除/取余。
- **early-exit（三类，全部 uniform、不与 `bar.sync` 冲突）**：`tid>=TB(128)`（`:143-145`）、
  `Q_LEN>8`（dual-dispatch gate，`:146-151`）、`gi>=num_groups`（`:160-162`）；另加「空 split」时写
  `La=-1e30f` 后返回（`:170-176`）。
- **`M=128` 行 → (query,head)**：`hpb = NUM_HEADS(64)`（`:177`），`actual_qpg = min(qpg, Q_LEN-gi*qpg)`
  （`:178`）。
- **TMEM 分配用 warp0**（`:186-212`）：三兄弟统一 warp0（`docs/34` §3.5 / `docs/57` §4.1 的常驻复用约束），
  `tcgen05.alloc.cta_group::1 [addr], D_V(=512)`（`:205-209`）；同 warp 内先 `elect_sync` 做 5 组
  `mbar_init`（`:200-203`）。
- **`SINGLE_TILE` 模板**（`:131`）：`single_tile = (tps==1)`（`task_register.cc:5203`），为真时 `o_save`
  退化成 1 个寄存器（`:231`）且跳过跨 tile 的 `o_save` 重载（`:237-262`）——即「每个 split 只含 1 个 KV
  tile」时不需要 online-rescale 保存/恢复。
- **QK 阶段**：producer `wid0` 逐 `ki=0..8` 发 3D TMA 载 K tile（`:285-296`）+ 每 query 一条 TMA 载 Q 行
  （`:297-308`，坐标用 `bi*Q_LEN*NUM_HEADS + actual_q_idx*NUM_HEADS`）；consumer `wid1` 每 `ki` 对
  `k2=0..BK/MMA_K-1` 发 `tcgen05_mma`（`:319-327`）再 `tcgen05_commit(mma_bar)`。与基版逐段同构。
- **softmax / PV / epilogue**：与基版同构（`docs/34` §3.2）；epilogue 用 `valid_rows = actual_qpg*hpb`
  （`:680`）限定写出，`La[block_linear*128+tid] = log2f(max(row_sum,1e-30))+row_max`（`:720`），
  最后 `bar.sync 12,128` + warp0 `tcgen05.dealloc`（`:723-729`）。

**关键差异 vs 基版（一句）**：基版收 `(gi,si,bi)` 且 `hpb` 运行期算；tp2 收 `(block_x,block_y)` 自己拆包、
`hpb=NUM_HEADS` 编译期常量、`bar.sync 12,128` 替代 `__syncthreads()`（`docs/34` §3.5 已解释为什么）。

---

## 5. tp4（`NUM_HEADS=32`）内部：V-split 折进 `block_x` 低位

`mla_mtp_tp4_main`（`:121-...`）。相对 tp2 的**全部差异**（用 `diff tp2 tp4` 逐条核过）：

- **V-split 折进 `block_x`**（`:141-145`）：签名收 `block_x_packed`，`block_x = block_x_packed >> 1`、
  `v_half = block_x_packed & 1`，再 `gi=block_x/sk`、`si=block_x%sk`、`bi=block_y`。注释原话——
  *"V-split is folded into block_x: low bit = v_half, rest = original block_x. Mirage MPK has only a flat
  task index, no z-dim launch."* 前端对应地把 x 维乘 2（`grid_dim[0] = num_groups*num_splits*x_mul`，
  `x_mul=2`，`persistent_kernel.py:1836-1841`）。
- **`PV_CHUNKS = V_CHUNKS/2 = 4`**（`:149`）：每个 half 只处理 4 个 V chunk；所有 PV 相关的 `vc` 循环
  改成 `vc_base + vi`，`vc_base = v_half*PV_CHUNKS`（`:335-338/547/559-560/589/668-673`）——即
  **V 的半段分给两个任务并行**，两个 half 写**不相交的 TMEM 列段**（前端注释 `persistent_kernel.py:1914-1915`）。
- **`La` 单写者门**：只有 `v_half==0` 写 `La`（`:710-713`，注释 *"Only v_half==0 writes La (shared across
  V halves)"*）——因为两个 V half 的 `row_sum`/`row_max` 相同，LSE 是两者共享的标量。
- **`NUM_HEADS=32`（`:29`）**：Q 索引、`qpg=min(4,q_len)`（register `:5300`）。

**含义**：tp4 是「同一 attention 的 V 维再切半」——**head 切 4 路（TP=4）× V 切 2 路 = 8 路并行任务**，
但只有 V 的两半是 MPK 用扁平 x 维「挤」出来的、不是真正的 launch 维。这是 TP>2 时并行度不够、用 V-split
补的工程折中，也是「MPK 无 z 维 launch」这条运行层约束在 device 侧的**具体让步**（同族：`docs/32` §5.3
`kNumSMs=1` HACK、`docs/34` §3.5 TMEM-warp、`docs/37` §2 `blockIdx.x==worker id`）。

---

## 6. tp8（`NUM_HEADS=16`）内部：pad 到偶数 + `Q_LEN_real`

`mla_mtp_tp8_main`（`:118-...`）。相对 tp2 的差异：

- **签名多一个 `int Q_LEN_real`**（`:124`）——pass 的是「未 pad 的真实运行期 Q_LEN」；`Q_LEN` 参数是
  **pad 成偶数**的值（register 里 `q_len_padded_rt_ = q_len_real_rt_ + (q_len_real_rt_ & 1)`，
  `task_register.cc:5419`）。
- **所有判断用 `Q_LEN_real`**：dual-dispatch gate `Q_LEN_real>8`（`:135-136`）、因果掩码
  `causal_limit = kv_len - Q_LEN_real + actual_q_tid + 1`（`:346-357`）。
- **grid 用 pad 后 Q_LEN**：`num_groups = Q_LEN/qpg`（**无 `ceil`**，`:146`，注释 *"Q_LEN already padded
  to multiple of qpg"*）；reduce 同步（`:725-727`）。
- **无 V-split**（与 tp2 同，`V_CHUNKS=8`）。
- **reduce 多一条 `q_in_group>=qpg` 守卫**（`:744`）——因 `qpg=2` 固定、`Q_LEN` 已对齐，多余的行要被剔掉。
- **`NUM_HEADS=16`（`:26`）、`qpg=2`（register `:5403`）**。

**含义**：tp8（`NUM_HEADS=16`）下 head 太少，`Q_LEN` 与 `qpg` 的对齐关系变紧，故 pad `Q_LEN` 到偶数以
保证 `num_groups` 整除——与 tp2/tp4 的 `ceil` 写法形成对照，是「per-rank 头数越小、Q 分组越需要对齐」的
直接后果。数值语义不变（`Q_LEN_real` 只用于下界/掩码，不用来算偏移）。

---

## 7. reduce 三兄弟的索引契约（补充基版 `docs/34` §3.3）

三份 reduce（tp2 `:732-796` / tp4 `:726-...` / tp8 `:712-...`）与基版 reduce 的逻辑**逐位同构**（online
LSE 归并），差异只在：

| 量 | 基版 `mla_mtp_reduce_sm100` | TP 三兄弟 |
|---|---|---|
| 模板线程数 | `<256>`（MPK）/`<512>`（standalone） | 直接写死 `RD_TB=256`（无模板参） |
| `RD_DV`（每 block 处理多少 D_V 维） | `4`（常量 `:54`） | `2`（tp2 `:61`），前端 `rd_dv=2` |
| `hpb` | `local_num_heads/num_head_groups`（运行期） | `NUM_HEADS`（编译期常量） |
| `gi` | 入参（`= request_id`） | `block_y` |
| `bi` | 入参（`= merge_task_offset`） | `block_z`（tp2 `:750-752`） |
| `d` | `dv_base + lane`（`lane=tid/128`） | 同（tp2 `:756`） |
| `row → (q_in_group,h)` | `row/hpb`、`row%hpb`（亦有 global head 计算） | `row/NUM_HEADS`、`row%NUM_HEADS`（tp2 `:761-762`） |
| 输出索引 `O` | `(bi*Q_LEN+q)*H*D_V + h*D_V + d` | `(bi*Q_LEN+actual_q)*NUM_HEADS*D_V + h*D_V`，`O[o_base+d]`（tp2 `:791-796`） |

reduce 的核心计算（tp2 `:775-795`）：

```
maxVal = max(maxVal, localMax); corr0 = exp2(oldMax-maxVal); corr1 = exp2(localMax-maxVal);
sumVal = sumVal*corr0 + corr1; acc = acc*corr0 + oa_val*corr1;   // 遍历 sk 个 split
O[o_base+d] = bf16(acc * (1/sumVal));
```

即 **log2 域的 online softmax 归并**（主核写的是 `log2(sum)+max` 的 LSE，reduce 用 `exp2` 递推预测权重）。
TP 的 `(row,h)` 用 `NUM_HEADS`（本地头）索引，故不同 rank 的 reduce 各写自己那段 head 的 `O`——**TP 的
跨卡归并不在 device 侧做，而是每 rank 各自 reduce 本 rank 的 head**（`O` 的布局是
`[B, Q_LEN, local_heads, D_V]`）。

`Oa` 的布局（主核 `:686-687`、reduce `:783-785`）：`Oout = Oa + block_linear*D_V*128`、
`oa_ptr = Oa + (bi*num_groups*sk+gi*sk)*D_V*128 + d*128 + row`——即 **`Oa` 视角为
`[batch*num_groups*sk, D_V, 128]`**，第二维 D_V 在外、第三维 128 行在内；`La` 视角为
`[batch*num_groups*sk, 128]`。**tp4/tp8 共享同一 `Oa`/`La` buffer**（前端传同一 `mla_partial_o`/
`mla_partial_lse`，`builder.py:752-763`），V-split 的两个 half 写不同 `d` 段、`La` 单写者。

---

## 8. ptx 命名空间：三兄弟各复制一份

`docs/31` §9 记过「tp2/tp4/tp8 各自重定义一份 `ptx` 命名空间」。逐文件核实：三份文件各自定义了
`namespace ptx { elect_sync / mbar_init / mbar_wait / mbar_tx / desc_enc / make_desc / tcgen05_mma /
tcgen05_commit }`（tp2 `:65-128`），**与基版共享的 `kernel::sm100_ptx`（`sm100_ptx.cuh`）是两套独立
实现**（`docs/57` §2「blackwell 目录里有两套互不共享的 SM100 张量核原语」的又一处）。也就是说 MTP
attention 这一族**既不复用 `sm100_ptx.cuh`、也不互相复用**，而是基版 + 三 TP = **四份 PTX 助手**。

同一份 `make_desc`（tp2 `:97-101`）里硬编码 `SBO = 8*128`、`(1<<46)`、`(2<<61)`——只对「128B swizzle、
K 主序、fp16」这一种布局成立；4 份复制意味着**改一处 swizzle 契约要改四处**（同族于 `docs/58`
§5「TMA swizzle 的 SMEM 基址对齐」是「常驻里加一个 task 可能静默破坏邻居」的静态风险）。

---

## 9. 生产布线：`world_size` 选路 + 三份 register 对照

- **选路在前端/builder，靠 `world_size`**（`models/deepseek_v3/builder.py:743-772`，两处
  `:1389/1413` 同构）：`world_size==2 → tp2`、`==4 → tp4`、`==8 → tp8`、`else → 基版`。每条分支调用
  `mla_mtp_decode_tpN_layer` + `mla_mtp_decode_tpN_reduce_layer`，把 `q_nope_pe`/`contiguous_kv` 喂进去、
  `mla_partial_o`/`mla_partial_lse` 当中间 buffer、`attn_out` 当最终输出。
- **前端 layer**（`persistent_kernel.py`）：`mla_mtp_decode_tpN_layer`（`:1892/1910/1930`）统一走内部
  helper `_mla_mtp_decode_tp_layer`（`:1814-1863`）——它按 `num_heads` 选 `qpg`（64→min(2)/32→min(4)/
  16→2）、算 `num_groups = ceil(q_len/qpg)`、`num_splits = ceil(kv_len/128)`、`x_mul = 2 if has_v_split`，
  网格 `(num_groups*num_splits*x_mul, max_num_batched_requests, 1)`、`block_dim=(128,1,1)`；tp8 额外传
  `q_len_real`。reduce helper `_mla_mtp_reduce_tp_layer`（`:1865-1899`）网格
  `((512+rd_dv-1)//rd_dv, num_groups, max_req)`、`rd_dv=2`、`block_dim=(256,1,1)`。
- **`graph.cc` dispatch**（`:805-840`）：六个 op 名 → 六个 `register_*`，`task_config` 均
  `(2,2,...)`（decode）或 `(2,1,...)`（reduce）——**2 入 2 出**（Q 的 TMA 描述符 + KV 的 TMA 描述符 入、
  `Oa` + `La` 出），reduce 是 2 入 1 出。
- **三份 register 的 `params` 形状**（`task_register.cc`）：
  - tp2 `:5192-5250`：`params = [num_groups, q_len, kv_len, num_splits, group_id]`（5 个），
    `qpg=min(2,q_len)`，`kv_len_` 从 `paged_kv_indptr_buffer/kv_group_block_sizes/paged_kv_last_page_len`
    现场算，`q_len_rt_` 从 `qo_indptr_buffer` 算（dual-dispatch）；传 `block_x = task_metadata.kv_idx`、
    `block_y = request_id`（`:5244-5245`）。
  - tp4 `:5287-...`：参数同 tp2，`qpg=min(4,q_len)`；`block_x` 传的还是 `kv_idx`（**已含 v_half 低位**，
    由前端 grid x 维 ×2 保证）。
  - tp8 `:5384-...`：`params = [num_groups, q_len_padded, kv_len, num_splits, q_len_real, group_id]`
    （6 个），`qpg=2`，`q_len_padded_rt_ = q_len_real_rt_ + (q_len_real_rt_&1)`。
  - reduce 三份（`:5252/5349/5447`）：`params = [num_groups, q_len, num_splits, rd_dv]`，`bi_ =
    merge_task_offset`，模板 `<256>`。
- **`TaskType`**（`runtime_header.h:177-182`）：TP2={287,288}、TP4={289,290}、TP8={291,292}（基版
  269/270 在 `:156-157`）。
- **`ss` 软缩放**（三份 register 同）：`_sm = (1/sqrt(192))*_mscale²`，`_mscale = 0.1*log(40)+1`
  （DeepSeek-V3 YARN，`task_register.cc:5238-5242` 等）——**注意 `d_k=576` 是吸收后 latent dim，点积
  缩放维是 192**（`docs/34` §4 已记）。

---

## 10. 判据映射（C1–C5）

同 `docs/07`/`docs/29`–`docs/66`：**一个 device task 身体只贡献 C3**。

- **贡献 C3**：把「per-rank 头/KV/split 分区的 MLA-MTP 解码 + 部分量（`Oa`/`La`）+ log2-LSE reduce」编成
  可被 `register_task` 生成、可被常驻 worker 调度的 device 函数；TP 的 `NUM_HEADS`/`qpg`/`RD_DV` 都在
  编译期/前端钉死，device 身体不感知「自己是第几个 rank」。
- **不贡献 C1/C2**：单次 launch / 常驻由运行层拿（`docs/06`/`docs/22` §8）。
- **不贡献 C4/C5**：task 内 `mbarrier`/`bar.sync 12,128` 是 CTA 内同步；跨 rank 的 head 归并不在 device
  侧（每 rank 各写本 rank 那段 `O`），故**没有跨卡会合**——TP 的「协调」全在 host/builder 的选路里。
- **与常驻有关的 device 侧事实（本文再证三条，同 `docs/32` §5.3 / `docs/34` §3.5 / `docs/57` §4.1）**：
  ① V-split 折进 `block_x` 低位 = **MPK 无 z 维 launch** 的直接后果；② TMEM alloc/dealloc 统一 warp0；
  ③ 命名屏障 ID=12（CUTLASS 用户区）替代 `__syncthreads()`。

---

## 11. 局限（诚实标注）

- **纯源码走读，未构建 / 未运行**：SM100（`sm_100a`）在目标卡 H100/sm90 不可执行，未取任何 device 计数；
  「同构」「逐位」都是**结构性对照**（`diff` + 行号），非运行期验证。
- **`docs/34` 已覆盖基版骨架**：本文对 QK/softmax/PV/epilogue 只给「与基版同构 + 行号」，不重复逐段展开；
  真正的增量集中在§4–§7 的 TP 专属索引/分区/reduce。
- **未与 ferret standalone 原核对照**：三份文件头都注明是机械 port，本文只核 port 后的 MPK 形态，未拉
  ferret 源码逐行比对（同 `docs/34` §6）。
- **`Q_LEN_real`/V-split 的数值等价未验**：pad `Q_LEN` 与 V 切半的语义等价性只由源码结构推断（pad 值
  在因果掩码下 `effective_len=0`、V 两 half 写不相交 `d` 段），未构造实例。
- **单卡版 vs TP 的分区「谁切了什么」系静态推断**：`O`/`Oa` 布局的 per-rank 切片来自 `NUM_HEADS` 与
  `global_row` 写法，未加载真实 ckpt 校对 Q/KV tile 形状。
- **未走读的相邻层**：`mla_prefill_sm100`/`mla_prefill_tp8_sm100`/`mla_decode_sm100`/`mla_reduce_sm100`/
  `mla_sm100_2sm.cuh` 已由 `docs/35`/`docs/36` 覆盖（本文不重复）。

---

## 12. 参考坐标（可核对）

- 快照：`mirage-project/mirage` 分支 `mpk`，commit `f9eb70c`（本地 `~/mirage_mpk_build`，与
  `docs/06`–`docs/66` 同快照）。
- **基版（TP1 父版）**：`include/mirage/persistent_kernel/tasks/blackwell/mla_mtp_decode_sm100.cuh`——
  常量 `:36-62`（`NUM_HEADS=128` `:38`、`RD_DV=4`/`RD_TB=512`/`MAX_SK=32` `:54-56`、`MTP_SMEM_SIZE`
  `:60`）；主函数 `:79-671`（签名 `:79-89` 含 `local_num_heads`；`hpb` `:123`；TMEM warp1 alloc `:146-152`）；
  reduce `:676-746`（模板 `NUM_THREADS_=512` `:676`、`local_num_heads` `:684`、`hpb` `:697`）。
- **TP 三兄弟**（`include/mirage/persistent_kernel/tasks/blackwell/`）：
  - `mla_mtp_decode_tp2_sm100.cuh`（796）：`NUM_HEADS=64` `:40`、`RD_DV=2`/`RD_TB=256` `:61-62`、
    `ptx` 命名空间 `:65-128`、`mla_mtp_tp2_main< SINGLE_TILE >` `:131-730`（拆包 `:157-159`、
    guards `:143-176`、`hpb` `:177`、TMEM warp0 `:186-212`、`o_save` `:231-262`、QK `:281-336`、
    `valid_rows` `:680`、`La` `:720`、dealloc `:723-729`）、`mla_mtp_tp2_reduce` `:732-796`。
  - `mla_mtp_decode_tp4_sm100.cuh`（787）：`NUM_HEADS=32` `:29`、`mla_mtp_tp4_main` `:121-...`
    （`block_x_packed`/`v_half` `:141-145`、`PV_CHUNKS` `:149`、`vc_base` `:335-338/559-560/668-673`、
    `La` 单写者 `:710-713`）、`mla_mtp_tp4_reduce` `:726-...`。
  - `mla_mtp_decode_tp8_sm100.cuh`（775）：`NUM_HEADS=16` `:26`、`mla_mtp_tp8_main` `:118-...`
    （`Q_LEN_real` 签名 `:124`、gate `:135-136`、`num_groups=Q_LEN/qpg` `:146`、因果 `:346-357`）、
    `mla_mtp_tp8_reduce` `:712-...`（`q_in_group>=qpg` 守卫 `:744`）。
- **生产**：`src/kernel/graph.cc:805-840`（六 op dispatch）；`src/kernel/task_register.cc`
  （基版 `register_mla_mtp_decode_sm100_task:4405-4471` / `..._reduce:4473-4508`；TP2 `:5192/5252`、
  TP4 `:5287/5349`、TP8 `:5384/5447`）；`include/mirage/persistent_kernel/runtime_header.h:156-157,177-182`。
- **前端/builder**：`python/mirage/mpk/persistent_kernel.py`（基版 `mla_mtp_decode_layer:1749`/
  `mla_mtp_reduce_layer:1780`；TP helper `_mla_mtp_decode_tp_layer:1814`/`_mla_mtp_reduce_tp_layer:1865`；
  `mla_mtp_decode_tp2/4/8_layer:1892/1910/1930`）；`python/mirage/mpk/models/deepseek_v3/builder.py:743-772`
  （及 `:1389-1417`）。
- **同专题相关篇**：`docs/34`（本文补其三兄弟内部；基版骨架与三类常驻适配）、`docs/31` §9（TP standalone
  port 与 ptx namespace）、`docs/35`/`docs/36`（MLA 家族其余身体 / 2SM）、`docs/57` §4（TMEM alloc warp）、
  `docs/22` §8（C 判据总映射）。
- **论文快照**：`~/github/ai-infra-book/experiments/ch05/05-08/sources/mpk-v2.txt`——**未提及本层**
  （device task 内部实现质量，同 `docs/28`–`docs/66`）。
- **交叉专题**：DeepGEMM MoE/mega 家族见 `../mega-moe/`（本专题不重复）。
