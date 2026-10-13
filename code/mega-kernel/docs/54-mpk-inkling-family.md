# MPK Inkling task 家族走读：GQA 相对位置 bias attention、深度短卷积与 logsigmoid-softmax 路由（P140-1）

> 快照：`mirage-project/mirage` 分支 `mpk`，commit `f9eb70c`（与 `docs/06`–`docs/53` 同快照）。
> **纯源码走读（资料分析），本轮无新实验**——SM100（`sm_100f`）在目标卡 H100/sm90 不可执行，
> 与 `docs/35`–`docs/53` 同。
>
> 本文补掉 `docs/35` §9 明留的「**注意力之外的 SM100 task 家族**（`dflash_*`/`inkling_*`/…）未逐行」
> 这条尾巴里的第二支：把 **Inkling（Thinking Machines / thinkingmachines/Inkling）** 三个 device
> 身体（`inkling_attention_sm100` / `inkling_sconv_sm100` / `inkling_moe_router_sm100`）从 `docs/37`
> §5.3 的摘要推进到**逐行**，并核清 `TaskType`/`graph.cc`/`register`/前端/builder/单测六层接线。
> Inkling 是继 DFlash（`docs/53`）之后同一批「非 MLA 专用身体」的第二支，也是本专题首个把
> **相对位置 bias + 深度短卷积 + logsigmoid 路由**三件事各做成一个独立薄 task 的模型家族。

## 0. 三条 headline

**(A) Inkling attention 是「带每 (head, 相对距离) bias 的 GQA decode」，且 bias/tau 都折进 pre-softmax 分数**：
参考数学（`inkling_attention_sm100.cuh:22-31`）——单新 token 位置 `P = ctx_len`，缩放 `1/HEAD_DIM`
（q/k 已各自 per-head RMS-norm），`s_j = tau * (dot(q_h, k_j) * scaling + bias[h, P-j])`，
其中 `tau = 1 + alpha * ln(max((P+1)/n_floor, 1))` 是**全局层才有的 log-scaling**（乘到整个分数，`:102-106`），
`bias[h, d]` 由上游一个小 GEMM（`r[NQ,16] @ proj[16,extent]`，`docs/26` §3）每步预计算。相对位置
`d = P - j` 只用于 bias 查表与滑窗过滤（`:169-184`），**不写进 key**——这是它与 RoPE 路线的根本区别，
也是 `docs/26` §3 所称「无 RoPE 相对位置」的 device 落点。4 warp 在 **key 维**上 stripe
（`j = j_start + warp_idx; j += 4`，`:151-152`），每 warp 每 q head 一个在线 softmax、V 累加器按 lane
分布（`ELTS_PER_LANE = HEAD_DIM/32`，D=128 时 4），**末尾跨 warp 用 SMEM + NamedBarrier 6 合并**
（`:199-241`）——即它**没有** DFlash mma 路径「warp-local 无归约」的便宜，走的是最朴素的手写路线。

**(B)（headline）三个 task 全是「一算子一 CTA、grid 维在通道/头维上切」的薄身体，且 `sconv` 用 fp32 状态原地滑窗**：
- `inkling_attention`：`grid.x = G` 切 **KV heads**（每 task `NUM_Q_HEADS/G` 个 q、`NUM_KV_HEADS/G` 个 kv），
  `NUM_THREADS=128`（4 warp），bias 走 `imap` **dim 0** 切、q/kv/out 走 dim 1 切（`task_register.cc:402-403`）。
- `inkling_sconv`：`grid.x` 切 **hidden 通道**（`c = threadIdx.x; c += blockDim.x`，`:59`），
  深度可分离短卷积 k=4、fp32 算术、`conv_state [K-1, HIDDEN]` **原地更新**成最近 K-1 个 token
  （`:60-90`）；一行固定 `acc = x + w[3]*x + Σ w[i]*s[i]`（残差 + 最新 tap 合并，`:74`），最后写回 state。
- `inkling_moe_router`：**单 task（`grid (1,1,1)`）**、一线程一行（`:92`），sigmoid+bias top-6 选路由，
  权重 = `softmax(logsigmoid(sel ++ shared)) × route_scale(8) × global_scale`（`:125-146`），
  并在 Phase 2 用 `atomicAdd(active+258,1)` 把活跃专家 id **紧凑化**（`:150-166`）。
  三者都以「**把预处理/写回各拆成独立薄 task**」的方式接线（对照 `docs/40`/`docs/41` 的融合边界结论）。

**(C) Inkling 是本批里「接线与测试最完整」的一支，且它**复用**了 DFlash 的 `dflash_kv_store` task**：
与 `docs/53` §4 记的 DFlash 两条落差（builder 引的单测目录不存在、`DFlashBuilder` 零实例化）相反，
Inkling 有**真实存在的单测目录** `tests/runtime_python/blackwell/sm100_inkling/`（三个 `*_testmode.py`
+ `pytorch_reference.py`，共 458 行），且 `InklingBuilder` 经 `models/__init__.py` 的 `rglob("builder.py")`
自动导入而**进入 `_MODEL_BUILDERS` 注册表**（`model_registry.py:5-23`），可被 `get_builder("Inkling")`
取用（`mpk.py:472`）——即「device 家族 + builder + 单测」三件齐全。更值得记的是：`_build_attention`
写 KV cache 时**不调用 inkling 自己的任何 device task**，而是直接
`mpk.dflash_kv_store_layer(kv_in=k_normed, …)` / `(kv_in=v_conv, …)`（`builder.py:410-417`）——
**Inkling 复用了 DFlash 家族的 paged KV store task**（`docs/53` §1 的 `TASK_DFLASH_KV_STORE_SM100=298`），
「attention 只读、写回独立成 task」的边界因此跨模型家族共享，是 megakernel「薄 task 可跨模型复用」的
一个干净实例。

---

## 1. 家族定位与六层生产接线

三个 `TaskType`（`include/mirage/persistent_kernel/runtime_header.h:208-215`，家族块 `350–399`）：

| TaskType | op 名 | 入/出 | device 身体 | 运行 grid |
|---|---|---|---|---|
| `TASK_INKLING_SCONV_SM100 = 351` | `inkling_sconv` | 3 / 1 | `inkling_sconv_sm100.cuh`（94 行） | `G = hidden/128` |
| `TASK_INKLING_MOE_ROUTER_SM100 = 352` | `inkling_moe_router` | 3 / 3 | `inkling_moe_router_sm100.cuh`（168 行） | `(1,1,1)` |
| `TASK_INKLING_ATTENTION_SM100 = 353` | `inkling_attention` | 7 / 1 | `inkling_attention_sm100.cuh`（245 行） | `G = num_kv_heads` |

六层接线（均在快照可核）：

- **L1 登记（`TaskType`）**：`runtime_header.h:210/212/214`（`:208/:215` 为 `BEGIN/END` 占位）。
- **L3 编入**：`tasks/blackwell/task_header.cuh:34-36` 三头全 include（**SM100 专属**闭包内）。
- **L2 发射器映射**：`src/kernel/graph.cc:472-491` 把 op 名映射到 `register_*`，并写 `(n_in, n_out, TaskType, variant)`：
  `inkling_sconv (3,1)`、`inkling_moe_router (3,3)`、`inkling_attention (7,1)`（`graph.cc:474-476/484-486/489-491`）。
- **发射器实现**：`src/kernel/task_register.cc` 的
  `register_inkling_sconv_sm100_task:317-365`、
  `register_inkling_attention_sm100_task:367-428`、
  `register_inkling_moe_router_sm100_task:430-487`。
- **前端 layer**：`python/mirage/mpk/persistent_kernel.py` 的
  `inkling_attention_layer:1050-1097`、`inkling_sconv_layer:1099-1123`、`inkling_moe_router_layer:1125-1163`。
- **builder**：`python/mirage/mpk/models/inkling/builder.py`（705 行，`InklingBuilder`，
  `@register_model_builder("Inkling", "thinkingmachines/Inkling", "inkling")`，`:70-71`）。
- **单测**：`tests/runtime_python/blackwell/sm100_inkling/` 三个 `*_testmode.py` + `pytorch_reference.py`。

---

## 2. `inkling_attention_sm100.cuh` 逐行（245 行）

**契约与模板**（`:20-47` 头注释 + `:51-61` 模板参数）：模板 `<T, NUM_Q_HEADS, NUM_KV_HEADS, HEAD_DIM,
EXTENT, SLIDING_WINDOW, Q_STRIDE, KV_STRIDE, O_STRIDE, BIAS_STRIDE>`，另有运行期参数
`log_scaling_alpha`（0 = 不做 log-scaling）与 `log_scaling_n_floor`。入参 7 个
（q / ctx_k / ctx_v / blk_k / blk_v / bias / step）+ 1 输出（`:63-72`）。
`static_assert(NUM_Q_HEADS % NUM_KV_HEADS == 0)`（`:74`）与 `HEAD_DIM % 128 == 0 || HEAD_DIM == 128`（`:75`）。
注意 `BIAS_STRIDE` 是**完整 bias 行宽**（`task_register.cc:416` 传 `extent`），但设备侧 `d` 用它做列索引
（`:183`）。

**线程划分与守卫**（`:77-87`）：固定 `NUM_WARPS=4`、`NUM_THREADS=128`；
`if (threadIdx.x >= NUM_THREADS) return;`（`:85-87`）——**runtime 可能发更多线程，多出的不碰 smem 也不到
named barrier**（注释 `:83-84`），这是常驻 kernel「task 粒度 ≠ launch 粒度」的适配惯例（对照 `docs/34`/
`docs/37`）。NamedBarrier ID 固定 **6**（`:81-82`，与 `docs/37` §8 记的 6/7 张力同）。

**tau / 位置 / 滑窗**（`:101-113`）：`P = ctx_len`；`tau` 仅在 `log_scaling_alpha != 0` 时算
（`:103-106`）；`j_start` 在 `SLIDING_WINDOW > 0` 时取 `max(0, P - SW + 1)`（`:110-113`），
故滑窗与因果（`j <= P`）都由 `j_start` 与上界共同表达。

**跨 warp 合并的 SMEM 布局**（`:115-122`）：`extern __shared__ __align__(128) uint8_t smem_raw[]`
（`:118`），依次叠 `s_acc[NUM_WARPS][NUM_QO_PER_KV][HEAD_DIM]`（fp32）、`s_m`、`s_l`。
注释 `:116-117` 明确写：**「extern shared 符号同名者必须在整个 megakernel TU 内一致」**，并对齐
`mla_prefill_sm100.cuh:258` 的同名声明——这是 `docs/51`「布局类型即 SMEM 的 ABI」在**函数级常驻
SMEM 基址**上的一层额外硬约束（见 §6 已知坑）。

**主体**（`:124-242`）：对每个 kv head `g`：
1. 预载 q fragment（`q_frag[NUM_QO_PER_KV][ELTS_PER_LANE]`，lane 持每 head 的连续 4 维，`:127-136`）；
2. 初始化 per-head 在线 softmax 状态 `m_local/l_local/o_local`（`:138-149`）；
3. **key 维 stripe**：`for j = j_start + warp_idx; j <= P; j += NUM_WARPS`（`:152`）——
   `j < ctx_len` 取 `ctx_k/ctx_v`，否则取本步 `blk_k/blk_v`（`:153-161`），即「本步 key 不进 cache 就读」；
4. 每 head：`partial = Σ q_frag*k_frag`，再做 **`__shfl_xor` 蝶形 warp allreduce**（`off=16..1`，`:178-181`）；
5. bias 查表：`d = P - j`，`d < EXTENT` 才取 `bias[(g*NUM_QO_PER_KV+h)*BIAS_STRIDE + d]`，否则 0（`:182-184`）；
6. 分数 `s = tau*(partial*inv_scale + b)`（`:185`），标准在线 softmax 更新 `m/l/o`（`:186-196`）。

**跨 warp 合并**（`:199-241`）：每 warp 把 `m/l`（lane0 写）与 `o`（全 lane 写）落 smem，
`wg_barrier.arrive_and_wait()`（`:212`）后，**head 由 `warp_idx` 轮转归约**
（`for h = warp_idx; h < NUM_QO_PER_KV; h += NUM_WARPS`，`:215`），算全局 `m_star`、`l_sum`、
按 `scale_w = exp(s_m - m_star)` 重标定后写出 `out`（`:216-239`），再 `arrive_and_wait()` 复用 smem
（`:241`）。**注意**：与 DFlash mma 路径不同，这里**必须**有跨 warp 归约（key 被 4 warp 切开），
归约方式是最朴素的 SMEM 中转。

---

## 3. `inkling_sconv_sm100.cuh` 逐行（94 行）

**契约**（`:18-36`）：语义与 HF 参考一致（decode / short-extend），
`out[t,c] = x[t,c] + Σ_{i=0..K-1} w[c,i]*win[i]`，`win = [state…, x[<=t]]` 为因果滑窗，**全部 fp32**；
`weight [HIDDEN, K]`（`w[.,0]` 是最旧 tap）、`conv_state [K-1, HIDDEN]`（行 0 最旧，**原地更新**）。
模板 `<T, HIDDEN, K, SEQ, X_STRIDE, OUT_STRIDE, STATE_STRIDE>`（`:40-47`）。

**主体**（`:48-92`）：对 `c = threadIdx.x; c < HIDDEN; c += blockDim.x`（`:59`）：
- 读入 `s[K-1]` 历史（`:61-65`）与 `taps[K]`（`:66-70`）；
- 对每个 token `t`（decode 时 `SEQ=1`）：`xv = x[t*X_STRIDE+c]`，`acc = xv + taps[K-1]*xv`
  （**残差与最新 tap 合并成一行**，`:74`），再 `Σ taps[i]*s[i]`（`:76-78`）；
- 写 `out`（`:79`），**移动窗口**（`s[i]=s[i+1]`，末位补 `xv`，`:81-85`）；
- 循环后把最终窗口写回 `state`（`:87-90`）。

`register_inkling_sconv_sm100_task`（`task_register.cc:317-365`）以 `hidden/G` 为 `HIDDEN`
（`G = grid.x`，`:349-350`），并断言 `weight.dim[0]==hidden`、`state.dim[0]==kernel_size-1`（`:344-346`）。
builder 侧 `_conv_w` 把 checkpoint 的 `[C,1,K]` squeeze 成 `[C,K]` fp32（`builder.py:129-135`），
`conv_state` 清零（序列起点零填充，`builder.py:359-362`）。**Inkling 每层有 4 段 sconv**
（attn 内 k/v 各一、attn 输出一、mlp 输出一，见 `builder.py:353-365/445-448/594-606`）。

---

## 4. `inkling_moe_router_sm100.cuh` 逐行（168 行）

**契约**（`:18-41`）：参考 `transformers` 的 `modular_inkling.py`——
`logits = x @ W.T`（`[rows, 258]` = 256 routed + 2 shared）；`scores_for_choice = sigmoid(logits[:,:256]) + e_score_correction_bias`；
`topk_idx = top-6(scores_for_choice)`（**无 group limiting**，与 GLM 的 group-aware 不同，`docs/37` §2）；
`w = exp(logsigmoid(topk_logits) - logsumexp) * route_scale(8) * global_scale`。**关键 MPK 映射**：
共享专家折叠成 expert 256/257（gamma 与 down_proj 可交换、两条路径都在 down_proj 之后施权），
故本 kernel **对 258 个专家发 top-(K+S) 选择**，下游 `moe_w13/silu_mul/moe_w2/mul_sum_add` 管线原样复用
（`:29-33`）。输出 ABI 与 `topk_sigmoid_sm100` 相同（`:35-38`）。

**辅助函数**（`:45-50`）：`inkling_logsigmoid(x) = -softplus(-x)`，用 `nx>20` 分支避免 `exp` 上溢
（`:48`）。

**主体三阶段**：
- **Phase 0**（`:79-89`）：清空 `indices`（`[NUM_TOTAL, rows]` 全 0）与 `active`（-1），
  `active[NUM_TOTAL]=0` 为计数器；`__syncthreads()`。
- **Phase 1**（`:91-147`）：一线程一行。读 `l[e]` 后**立即把 `lrow[e]` 写 0**（`:96-99`，供 split-k gate
  linear 复用，`:40`/`:98` 注释）；`choice[e] = sigmoid(l[e]) + bias[e]`（`:102-106`）；
  **O(TOPK×NUM_ROUTED) 朴素选 top-6**（`:108-119`）；shared 槽固定为 `NUM_ROUTED+s`（`:120-123`）；
  再 `lp[k]=logsigmoid(l[sel[k]])`、`denom=Σexp(lp-max)`、`scale=route_scale*gscale/denom`（`:126-139`），
  写 `weights[row*K_OUT+k]`、`indices[sel[k]*rows+row]=k+1`、`active[sel[k]]=sel[k]`（`:141-146`）。
  最后一行是**良性写竞争**（所有写者都存 `e`，注释 `:145`）。
- **Phase 2**（`:150-166`）：**先把各线程认领的活跃 id 快照进寄存器** `owned[MAX_OWNED]`
  （`MAX_OWNED = (NUM_TOTAL+127)/128 + 1`，`:153-160`，注释 `:151-152` 说明「写入 `active[pos]` 不得
  覆盖尚未读到的标记」），再 `atomicAdd(active+NUM_TOTAL,1)` 拿位置紧凑化（`:162-165`）。

`register_inkling_moe_router_sm100_task`（`task_register.cc:430-487`）从 `params=[route_scale_bits, n_shared]`
解出 `route_scale`/`n_shared`，断言 `grid (1,1,1)`（`:470`）、`logits_stride >= num_total`（`:465`），
并按输出形状反推 `topk = weights.dim[1] - n_shared`（`:463`）。builder 侧把 `gate.weight [258,H]`
**pad 到 `[384,H]`**（`GATE_PADDED=384`，`builder.py:497-504`），使 gate linear 的 grid 48 每块仍有
≥8 行 bf16（16 B）——**router 读 stride 384、只读前 258 列**（注释 `builder.py:497-498`）。

---

## 5. builder 粒度与三处家族特异折叠（与 `docs/26` §3 交叉引用，此处只补新发现）

`docs/26` §3 已给 Inkling builder 的家族结构摘要；本文只补**device 视角下的三处新发现**：

1. **KV 写回复用 DFlash 的 task**（headline C，`builder.py:410-417`）：`_build_attention` 用
   `mpk.dflash_kv_store_layer` 写 `k_normed` 与 `v_conv`（即 sconv/norm 之后的 k/v），而不是任何
   `inkling_*` 写回 task。故 `TASK_INKLING_*` 只含「读 attention + sconv + router」，**写回路完全借 DFlash**
   （`docs/53`）——「attention 只读、写回独立」的边界跨家族共享。
2. **bias 表由一次普通 `linear_layer` 现算**（`builder.py:385-399`）：`r_buf[NQ,16]` 先 pad 到 `[NQ,64]`
   （`D_REL_PAD=64`，`:389-393`）再 `r @ proj_pad` 得 `bias[NQ, extent]` bf16——即 attention 头部注释说的
   「tiny GEMM upstream」，在 builder 里就是一次标准线性 task，**不是**融进 attention 身体。
3. **相对位置 bias 的 dtype 口径落差（只记不判）**：前端 `inkling_attention_layer` 声明
   `bias [num_q_heads, extent] **bf16**`（`persistent_kernel.py:1057`），device 也按 `T = bfloat16` 读
   （`inkling_attention_sm100.cuh:94,182-184`），但 `task_register.cc:422` 的行内注释写成 `bias (fp32)`
   ——**注释与两端实现不一致**，本文只记录，不构成 bug 判定（与 `docs/37` §8 对命名屏障 ID 的口径张力
   同处理）。

---

## 6. 单测层（与 DFlash 的关键差异）

`tests/runtime_python/blackwell/sm100_inkling/` **真实存在**（对照 `docs/53` §4 DFlash 引的单测目录不存在）：

- `test_inkling_attention_testmode.py`（103 行）：三 layer 一张图，`local`（GQA 4:1、sw=512、ctx 700
  触发滑窗）、`global`（GQA 8:1、无窗、extent 1024、alpha=0.1 且小 n_floor 使 tau>1、ctx 1200 触发
  `d>=extent` 零 bias 分支）、`edge`（ctx_len=0 只 attend 本步 blk）——三 case 覆盖滑窗、log-scaling、
  越界 bias、首 token 四条边；校验 `atol=rtol=2e-2`（`:80-96`）。
- `test_inkling_moe_router_testmode.py`（141 行）：对 `weights` 校验 `atol=rtol=1e-4`（`:95`），并另起
  Python 侧重算 `exp_indices`/`exp_active` 比对 `routing_indices`/`active_ids`（`:103-112`）。
- `test_inkling_sconv_testmode.py`（86 行）：同时校验 `out`（`atol=rtol=1e-2`）与**原地更新后的
  `state`**（`atol=rtol=1e-5`，`:73-74`），即把「conv_state 滑窗语义」也纳入回归。
- `pytorch_reference.py`（128 行）：`inkling_attention_ref` 等纯 PyTorch 参考。

单测直接调 `pk.inkling_attention_layer`/`…_sconv_layer`/`…_moe_router_layer`（test_mode），
与 `docs/53` 的 DFlash 单测同路线（**device 身体层**回归，非端到端 step）。

---

## 7. 判据映射

| 判据 | Inkling 家族是否贡献 |
|---|---|
| C1 单次 launch | 间接（三个 task 只是同一常驻图里的节点，launch 由 `PersistentKernel` 统一发） |
| C2 常驻 | 间接（同 C1） |
| C3 多算子范围 | **是**——本文给出 Inkling 段「哪些 task、融合边界在哪、粒度多大、哪些算子借了别家 task」的清单（§1 表 + §5 的跨家族复用） |
| C4 设备内协调 | 间接（跨 task/跨 step 依赖由 MPK event 承担，`docs/06`/`docs/09`） |
| C5 SM 级调度 | 否（device body 层不涉 worker/scheduler 划分） |

结论：**本文贡献 C3 的「Inkling 段的范围与融合边界」**，与 `docs/37` §5.3（摘要）、`docs/26` §3
（builder 视角）互补。特别地，Inkling 把「相对位置 bias（GEMM）、短卷积、路由」各做成**独立薄 task**，
且**写回借 DFlash task**——是「融合边界不是越融越好、薄 task 可跨模型复用」的又一实证
（对照 `docs/40`/`docs/41` 的退役融合 attention 与 `docs/49` 的融合算子层）。

---

## 8. 局限（诚实标注）

- **纯源码走读，未构建 / 未运行**：SM100（`sm_100f`）在目标卡 H100/sm90 不可执行，未取任何 device 计数；
  所有行号为快照 `f9eb70c` 静态事实，不构成对上游的 bug 判定（§5.3 的 dtype 口径落差也只记不判）。
- **数值等价未验证**：三个单测的 `atol/rtol` 口径只**声明**了 device 与 PyTorch 参考的容许误差，
  本文未跑；`inkling_logsigmoid` 的 `nx>20` 分支、Phase 2 的寄存器快照在竞争下的正确性都只按源码读。
- **Inkling 模型侧未核**：`builder.py` 的常量（`HIDDEN_SIZE=6144`/`NUM_LAYERS=66`/`GLOBAL_KV_HEADS=8`/
  `LOCAL_KV_HEADS=16`/`EXTENT 1024/512`/`D_REL=16`/`NUM_ROUTED_EXPERTS=256`）皆取自 docstring 所述
  `config.json text_config + modular_inkling.py`，本文未加载真实 ckpt，未逐层核权重名与 HF 的对应。
- **`world_size==1` / decode-only 未越过**：builder 首行即 `assert world_size == 1`（`:104`），docstring
  声明 v1 仅 decode（`max_num_batched_tokens==1`）、BF16；故多卡/预填充路径未在本文范围。
- **`extern __shared__` 同名一致的 TU 级约束未验证**：`:116-117` 的注释只声明「同名须一致」，
  本文未在 SM100 上实测该 megakernel TU 的 SMEM 合并行为（本机不可编译 SM100）。

---

## 9. 参考坐标（可核对）

- 快照：`mirage-project/mirage` 分支 `mpk`，commit `f9eb70c`（与 `docs/06`–`docs/53` 同快照）。
- **device 身体**：
  - `include/mirage/persistent_kernel/tasks/blackwell/inkling_attention_sm100.cuh`（245 行；
    头契约 `:20-47`；模板 `:51-61`；impl `:62-72`；常量/static_assert `:73-79`；NamedBarrier 6 与
    thread 守卫 `:81-87`；位置与 tau `:101-106`；滑窗 `:108-113`；SMEM 布局 `:115-122`；kv-head 主循环
    `:124-242`；q 预载 `:127-136`；在线状态 `:138-149`；key stripe `:151-161`；dot+蝶形归约 `:162-181`；
    bias+分数 `:182-185`；在线 softmax `:186-196`；跨 warp 合并 `:199-241`）。
  - `…/inkling_sconv_sm100.cuh`（94 行；契约 `:18-36`；模板 `:40-47`；主体 `:48-92`；窗口滑动 `:81-90`）。
  - `…/inkling_moe_router_sm100.cuh`（168 行；契约 `:18-41`；`inkling_logsigmoid:45-50`；模板 `:52-58`；
    Phase 0 `:79-89`；Phase 1 `:91-147`；top-k `:108-119`；weights `:125-146`；Phase 2 `:150-166`）。
- **共享原语**：`tasks/common/common_header.cuh`；`cutlass/arch/barrier.h`（`NamedBarrier`）；
  `tasks/ampere/smem_layout.cuh`（布局，`docs/51`，本文未用）。
- **登记 / 编入 / 布线**：`runtime_header.h:208-215`；`tasks/blackwell/task_header.cuh:34-36`；
  `src/kernel/graph.cc:472-491`；`src/kernel/task_register.cc:317-487`。
- **前端**：`python/mirage/mpk/persistent_kernel.py:1050-1163`（三 layer；attention 的 grid-split
  断言 `:1073-1082`）。
- **builder**：`python/mirage/mpk/models/inkling/builder.py`（docstring `:1-24`；常量 `:38-67`；
  `_conv_w:129-135`；`_is_local:137-140`；`_build_attention:290-456`；q/k/v/r proj `:312-350`；
  k/v sconv `:352-365`；q_norm/k_norm `:367-383`；bias proj `:385-399`；KV store 借 DFlash `:410-417`；
  `inkling_attention_layer` 调用 `:419-426`；attn sconv `:445-448`；`_build_dense_mlp:458-489`；
  `_build_moe_mlp:491-575`；gate pad `:497-510`；router `:526-531`；共享专家折叠 `:533-568`；
  `_build_mlp` 的 mlp sconv `:594-614`）。
- **注册表机制**：`python/mirage/mpk/model_registry.py:5-23`；`python/mirage/mpk/models/__init__.py`
  （`rglob("builder.py")` 自动导入）；取用处 `python/mirage/mpk/mpk.py:472`。
- **单测**：`tests/runtime_python/blackwell/sm100_inkling/`（`test_inkling_attention_testmode.py` 103 行、
  `test_inkling_moe_router_testmode.py` 141 行、`test_inkling_sconv_testmode.py` 86 行、
  `pytorch_reference.py` 128 行）。
- **本专题交叉引用**：`docs/37` §5.3（摘要）、`docs/26` §3（builder 视角）、`docs/53`（DFlash 家族，
  Inkling 复用其 `dflash_kv_store`）、`docs/40`/`docs/41`（退役 paged attention 与「拆写回」）、
  `docs/49`（融合算子层）、`docs/51` §B（SMEM 布局 ABI）、`docs/24`（`TaskType` 351/352/353）。
