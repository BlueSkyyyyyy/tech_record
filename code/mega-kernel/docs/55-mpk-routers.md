# MPK 路由 task 走读：GLM-4.x 标量路由与 DeepSeek group-aware sigmoid 路由（P141-1）

> 快照：`mirage-project/mirage` 分支 `mpk`，commit `f9eb70c`（与 `docs/06`–`docs/54` 同快照）。
> **纯源码走读（资料分析），本轮无新实验**——两条路由都是 SM100 task 家族，目标卡 H100/sm90 不能跑
> SM100 编译单元；但其中 **GLM 路由是 arch-agnostic 标量身体**（见 §0-A），本身可在 sm90 编译，
> 本文仍只做静态走读、未单独构建该 task，与 `docs/35`–`docs/54` 同。
>
> 本文补掉 `docs/37` §2（路由三兄弟只给摘要）与 `docs/35` §9 明留的「注意力之外的 SM100 task 家族
> 未逐行」这条尾巴里的**路由支**：把 **`topk_sigmoid_sm100`（DeepSeek-V3 group-aware）** 与
> **`glm_moe_router_sm100`（GLM-4.x n_group=1 标量）** 两条 device 身体从摘要推进到**逐行**，并核清
> `TaskType`/`graph.cc`/`task_register`/前端/builder/单测**六层接线**。第三条路由（Inkling
> logsigmoid-softmax）已由 **P140-1/`docs/54`** 补掉，本文与之互补、不重复。

## 0. 三条 headline

**(A) `glm_moe_router_sm100.cuh` 是全仓唯一「`_sm100` 后缀却被 Hopper task_header 共享」的 device 身体，
因为它是纯标量、arch-agnostic 的**：文件本身不含任何 TMA / UMMA / `mma.sync` / `cp.async` / SMEM 声明，
只有一个 `for (row = threadIdx.x; …)` 的逐行标量循环（`:86`）＋两段 `__syncthreads`（`:83`/`:135`）。
因此它同时被 **Blackwell** `tasks/blackwell/task_header.cuh:33` 与 **Hopper**
`tasks/hopper/task_header.cuh:28` include——后者还带一行显式注释：
`// arch-agnostic scalar kernel (GLM-4.x router), shared with the sm100 build`（`:27`）。
这在「`_sm100` = SM100 专属」的命名惯例里是个**反例**：`docs/45` 已指出 GLM-4.6 走 `cc≥90` 路径，
故它必须能被 Hopper 闭包编进去；后缀只表示「首次为 SM100 渠道写」，不表示硬件绑定。
（Hopper 头里另有 4 个 `*_sm100.cuh`：`argmax`/`attention`/`mul_sum_add`/`topk_softmax`，但带「共享」
注释的只有 GLM router 一支。）

**(B) 两条路由的「模型家族差异」全部折平在 device task 内部，下游 MoE 管线零改动**，但两条实现策略因
**幂约束与规模**完全分道：`topk_sigmoid` 是 **group-aware 向量化**（要求专家数、`VPT`、`BYTES_PER_LDG`
都是 2 的幂，`static_assert` `:121-126`；warp 内一行、每线程持 `VPT=8` 个专家、`__shfl_*_sync` 做归约）；
`glm_moe_router` 因 GLM 有 **160**（非 2 的幂，`:33-36` 注释自陈不能复用 sigmoid 版）退化成
**一线程一行**的标量实现。二者产物 ABI 相同（`weights[rows, TOPK+NSHARED]` /
`indices[NUM_TOTAL, rows]` / `active[NUM_TOTAL+1]`），且都把**共享专家折进专家张量**（GLM 折成
`NUM_ROUTED` 槽、权重恒 1.0，`:120-125`），故下游 `moe_w13 / silu_mul / moe_w2 / mul_sum_add`
原样复用——`TaskType` 只多两个号，MoE 管线零改动。

**(C)（headline）两条路由的六层接线全通且都有生产 builder 调用**，与 `docs/53` DFlash 的「影子实现」
形成对照：`glm_moe_router` 有真实单测目录 `tests/runtime_python/blackwell/sm100_glm4_moe/`
（4 文件，含**端到端 builder 测试** `test_glm4_moe_builder_testmode.py`：4 层 GLM-4.6、随机权重、
跑一次 decode step 与 **HF `Glm4MoeForCausalLM` 参考**逐位比 lm-head logits），`topk_sigmoid` 有
test-mode 测试 + 一套**遗留 standalone wrapper 测试**（`sm100_moe_sigmoid/`，含
`runtime_kernel_wrapper_sm100.cu` 与 `setup.py`）。生产生产者分别
`glm4_moe/builder.py:305`（GLM 主链）与 `deepseek_v3/builder.py:905/1501`（DeepSeek 主链 + MTP 旁路）。

---

## 1. 家族定位与六层接线

两个 `TaskType`（`include/mirage/persistent_kernel/runtime_header.h`）：

| TaskType | 所在块 | op 名 | 入/出 | device 身体 | 运行 grid |
|---|---|---|---|---|---|
| `TASK_MOE_TOPK_SIGMOID_SM100 = 280` | SM100 块（`160–299`） | `moe_topk_sigmoid_sm100` | 2 / 3 | `topk_sigmoid_sm100.cuh`（364 行） | `(1,1,1)`，`block 256` |
| `TASK_GLM_MOE_ROUTER_SM100 = 401` | GLM 家族块（`400–449`） | `glm_moe_router` | 2 / 3 | `glm_moe_router_sm100.cuh`（155 行） | `(1,1,1)`，`block 128` |

注意 GLM 路由**在自己的家族块 `400`**（`TASK_GLM_TASK_BEGIN=400` / `_END=449`，`runtime_header.h:218-223`），
而 sigmoid 路由落在通用 SM100 块（`:169`）——同一「sigmoid+bias 路由」的两支被划到不同 ID 分区，再次印证
`docs/24`/`docs/25` 的「`TaskType` 分区 = 家族元数据，不表示 device 身体的相似度」。

六层接线（均在快照可核）：

- **L1 登记（`TaskType`）**：`runtime_header.h:169`（sigmoid）、`:219`（GLM）。
- **L2 发射器映射**：`src/kernel/graph.cc`——
  `else if (name == "moe_topk_sigmoid_sm100")`（`:682-686`，写 `(2, 3, 280, variant)`）、
  `else if (name == "glm_moe_router")`（`:477-481`，写 `(2, 3, 401, variant)`）。
- **L3 编入（task_header include 闭包）**：sigmoid 在 `tasks/blackwell/task_header.cuh:66`（仅 Blackwell）；
  GLM router 在 `tasks/blackwell/task_header.cuh:33` **与** `tasks/hopper/task_header.cuh:28`（跨代，见 §5）。
- **发射器实现（`src/kernel/task_register.cc`）**：
  `register_moe_topk_sigmoid_sm100_task:2935-3001`（声明 `task_register.h:126-127`）、
  `register_glm_moe_router_sm100_task:258-315`（声明 `task_register.h:213-214`）。
- **前端 layer**：`python/mirage/mpk/persistent_kernel.py` 的
  `moe_topk_sigmoid_routing_layer:1995-2029`、`glm_moe_router_layer:1165-1202`。
- **builder（真实生产生产者）**：`python/mirage/mpk/models/deepseek_v3/builder.py:905`（主 MoE 链）与
  `:1501`（MTP 旁路）；`python/mirage/mpk/models/glm4_moe/builder.py:305`。
- **单测**：`tests/runtime_python/blackwell/sm100_moe_sigmoid/`、`…/sm100_glm4_moe/`（见 §6）。

三方 ABI 一致（`glm_moe_router_sm100.cuh:38-41` 显式写「same ABI as topk_sigmoid_sm100 /
inkling_moe_router」），故下游 `moe_w13_linear_layer` / `moe_silu_mul` / `moe_w2_linear` /
`moe_mul_sum_add` 不需要区分来源路由。

---

## 2. `topk_sigmoid_sm100.cuh` 逐行（364 行，DeepSeek 系 group-aware 向量化）

**参考数学与七步**（文件头 `:44-58`）：sigmoid → 加 `e_score_correction_bias` → **每组（
`EXPERTS_PER_GROUP=32`）取 top-2 求和得组分数** → top-K（`TOPK_GROUP=4`）选组 → 掩码非选中组 →
剩余里选 top-K（`TOPK_EXPERTS=8`）专家 → **取原始未加 bias 的 sigmoid** 作权重 → 按 `sum` 归一化乘
`routed_scaling_factor`。这是 DeepSeek-V3 `DeepseekV3TopkRouter` + `get_topk_indices` 的
group-limiting 语义。

**模板与签名**（`:75-94`）：模板 `<T, VPT=8, NUM_EXPERTS, WARPS_PER_CTA=8, BYTES_PER_LDG=16,
NUM_GROUPS=8, TOPK_GROUP=4, EXPERTS_PER_GROUP=32, TOPK_EXPERTS=8>`（发射器实参见
`task_register.cc:2980-2989`）；运行期入参除 `input/bias/output` 外还有 **`finished`（bool 数组，
可空）**、`num_rows`、两个簿记输出、**`start_expert`/`end_expert`**（节点内专家切片）与
`routed_scaling_factor`。**一个易漏点**：发射器把 `finished` 实参写成 **`nullptr`**、`start_expert`
写 `0`、`end_expert` 写 `num_experts`（`task_register.cc:2992/2997/2998`），故生产路径
`row_is_active` 恒真（`:167` 的 `finished ? … : true`）、专家切片恒为全集——`finished`/切片是给
**投机解码多节点共享路由**预留的接口，本路径未用。

**线程划分（`:128-163`）**：`ELTS_PER_ROW=NUM_EXPERTS`、`THREADS_PER_ROW=ELTS_PER_ROW/VPT`（256 专家
⇒ 32，满 warp）、`ELTS_PER_WARP=WARP_SIZE*VPT=256`、`ROWS_PER_WARP=1`——即**一个 warp 一整行**，
`warp_idx` 定行（`:157`），lane 定列段（`thread_group_idx=lane%32`）。奇数行时末行用半 warp mask
（`:161-163`）。一个 block 8 warp 处理 8 行（`block 256`）。

**Phase 0（`:103-118`）清路由结构**：`indices[expert*num_rows+row]=0`、`active[expert-start_expert]=-1`，
`active[NUM_EXPERTS]=0` 为计数器。

**Phase 1（`:169-206`）载入 + sigmoid + 清零**：向量化 16B 载入（`cutlass::AlignedArray<T, ELTS_PER_LDG>`
= 8 个 bf16），`sigmoid` 后**把 logits 原地写 0**（`:197`/`:204-206`，供 split-k gate linear 复用，
与 GLM router、Inkling router 同惯例）；`biased_chunk = sig + bias`。

**Phase 2（`:208-230`）组内 top-2**：先线程本地 top-2，再 `__shfl_xor_sync` 在 `THREADS_PER_GROUP=4`
个线程（即一组 32 专家由 4 线程持）内蝶形归约（`merge_top2`，`:65-73`），得 `group_score=top1+top2`。

**Phase 3（`:232-255`）广播 + 选 top-K 组**：把每组的 `group_score` 用 `__shfl_sync` 从 source lane
广播给全 warp，再 `O(TOPK_GROUP×NUM_GROUPS)` 迭代选组，`group_selected[best_g]=true` 防重。

**Phase 4（`:257-263`）掩码非选中组**：本线程若属未选组，把 `biased_chunk` 全置 `-10000`。

**Phase 5（`:265-337`）组内 top-K 专家**：迭代 `TOPK_EXPERTS` 次，每次本地 argmax + subgroup
`__shfl_xor_sync` 归约（**带 `other_expert < expert` 的小 index tie-breaker**，`:291-295`）；选中专家的
**未加 bias 原始分**通过 `__shfl_sync` 从**拥有该专家的线程**取回（`owning_thread=expert/VPT`，
`:298-306`）——这是「选择用 biased、权重用 unbiased」的实现关键；仅 `thread_group_idx==0` 的线程写
`output`/`indices`（`k_idx+1`）并置 `active`；选择下一个前把该专家在 `biased_chunk` 里置 `-10000`
（`:326-336`，需定位到持有它的 lane 才清）。

**Phase 6（`:339-346`）归一化 + 缩放**：`inv_sum=1/(Σorig_score+1e-20)`，乘 `routed_scaling_factor`。

**Phase 7（`:350-361`）紧凑化活跃专家**：`active[local_expert]>=0` 者 `atomicAdd(active+NUM_EXPERTS,1)`
取位写回——与 GLM/Inkling 同构。

> **硬约束**：专家数、`VPT`、`BYTES_PER_LDG`、`THREADS_PER_ROW` 都须为 2 的幂（`static_assert`
> `:121-126`/`:137`），`NUM_GROUPS*EXPERTS_PER_GROUP==NUM_EXPERTS`（`:146`）——这正是 GLM 160 专家
> **不能复用**本 kernel 的根因（GLM 文件 `:33-36` 自陈）。

---

## 3. `glm_moe_router_sm100.cuh` 逐行（155 行，GLM-4.x n_group=1 标量）

**参考数学**（文件头 `:18-44`，取自 `zai-org/GLM-4.6` 的 `modeling_glm4_moe.py`，`n_group==1` 使
group limiting 退化为 no-op）：`scores=sigmoid(logits)`；`topk_idx=top-K(scores+bias)`；
`w=scores.gather(topk_idx)`（**未加 bias**）；`w=w/(sum+1e-20)*routed_scaling_factor`。共享专家
（`n_shared==1`，与 routed 专家同 intermediate 尺寸）**折进专家张量的 expert `NUM_ROUTED` 槽**，
恒选中、权重恒 1.0（HF 参考把 `shared_experts(x)` 无权重相加），故标准
`moe_w13/silu_mul/moe_w2/mul_sum_add` 管线原样跑（`:27-31`）。

**为什么标量**（`:33-36`）：直接写「DeepSeek 的 `topk_sigmoid_sm100` 不能复用——它的向量化布局静态
要求专家数是 2 的幂，而 GLM 有 160 个 routed 专家」；decode 行数很小，故用与
`inkling_moe_router_sm100` 相同的**一线程一行**方案。

**模板与签名**（`:48-62`）：`<T, NUM_ROUTED=160, N_SHARED=1, TOPK=8, LOGITS_STRIDE=192>`；入参 2
（`logits [num_rows, LOGITS_STRIDE]`、`bias [NUM_ROUTED]` fp32）+ 出参 3，另有 `num_rows`、
`routed_scaling_factor`。`static_assert(LOGITS_STRIDE >= NUM_ROUTED)`（`:65`）——`LOGITS_STRIDE` 是
**gate linear 的输出行宽**（builder pad 到 192），router 只读前 160 列（`:92`）。

**Phase 0（`:73-83`）清结构**：与 sigmoid 版同构（`indices` 清 0、`active` 清 -1、计数器置 0）。

**Phase 1（`:85-134`）一线程一行**：
- `for (row = threadIdx.x; row < num_rows; row += blockDim.x)`（`:86`）——行数少，一个 block 128 线程
  轮流处理；
- 遍历 160 专家：读 `logit`，**立即把 `lrow[e]` 写 0**（`:93`，split-k gate linear 复用），
  `sig[e]=sigmoid(logit)`（**最终权重**）、`choice[e]=sig[e]+bias[e]`（**仅选择用**）（`:91-96`）；
- **`O(TOPK×NUM_ROUTED)` 朴素选 top-8**：每轮线性扫最大 `choice`，选中后把 `choice[best]` 置 `-1e30`
  防重，权重取 `sig[best]`、累加 `denom`（`:101-114`）；
- `scale = routed_scaling_factor/(denom+1e-20)`，缩放 routed 权重（`:115-119`）；
- 共享专家：`sel[TOPK+s]=NUM_ROUTED+s`、`w[TOPK+s]=1.0f`（**不缩放**，`:120-125`）；
- 写 `weights[row*K_OUT+k]`、`indices[e*num_rows+row]=k+1`、`active[e]=e`
  （`:127-133`；`active` 那行是**良性写竞争**——同名 `e` 由多个写者写，注释 `:132`）。

**Phase 2（`:137-153`）紧凑化活跃专家**：与 Inkling 同构、也是本家族强调的一点——**先把各线程认领的
活跃 id 快照进寄存器 `owned[MAX_OWNED]`**（`MAX_OWNED=(NUM_TOTAL+127)/128+1`，`:140-147`），再
`atomicAdd(active+NUM_TOTAL,1)` 取位写回（`:149-152`）；注释 `:138-139` 明确「先快照再写，避免
`active[pos]` 的写覆盖尚未读到的标记」。

**发射器**（`task_register.cc:258-315`）：从 `params=[scale_bits, n_shared]` 解出缩放与共享数
（`:266-269`），断言 5 个张量维度：logits 2 维、bias 1 维、weights 2 维、indices 2 维、active 1 维
（`:283-287`），反推 `num_routed=bias.dim[0]`、`topk=weights.dim[1]-n_shared`（`:290-292`），
断言 `grid (1,1,1)`（`:299`）；JIT 发射 `kernel::glm_moe_router_task_impl<bfloat16, R, S, K, STRIDE>`
（`:302-313`）。

---

## 4. 两实现的差异与「折平」结论

| 维度 | `topk_sigmoid_sm100` | `glm_moe_router_sm100` |
|---|---|---|
| 选择口径 | **group-aware**（每组 top-2 求和 → 选组 → 组内 top-K） | `n_group==1`，**直接 top-K** |
| 实现形态 | 向量化：warp 一行、每线程 `VPT=8` 专家、`shfl` 归约 | 标量：一线程一行、线性扫 |
| 专家数约束 | 必须 2 的幂（`static_assert`） | 任意（GLM 160） |
| 权重 | 未加 bias 的 sigmoid 归一化 × scale | 同（未加 bias 的 sigmoid） |
| 共享专家 | 无 | 折成 expert `NUM_ROUTED`，权重 1.0 |
| 同步 | 块内 `__syncthreads` ×2（`:118`/`:348`） | `__syncthreads` ×2（`:83`/`:135`） |
| 清零 logits | 是（split-k 复用） | 是 |
| 活跃专家紧凑化 | Phase 7 | Phase 2（寄存器快照法） |
| block | 256（8 warp × 1 行） | 128 |

**折平结论**：两条路由的**产物 ABI 完全一致**，且都把「共享专家 / 不同归一化口径 / 不同选择策略」
关在 device task 内部；下游 MoE 管线（w13/silu/w2/mul_sum_add）完全不知道上游是哪支路由。这与
`docs/27` 的「家族差异在 builder 层被折平」互为表里——**折平有两条路**：要么 builder 折（同一 task
不同参数），要么 device 折（不同 task、同 ABI）。本文这两支走的是后者。

---

## 5. 跨代 / 跨家族调用（GLM router 的 `_sm100` 命名反例）

- **同名文件被两条 `task_header` include**：`tasks/blackwell/task_header.cuh:33` 与
  `tasks/hopper/task_header.cuh:28`。Hopper 头在 `:22` 起是「Blackwell task impls」段落，`:27` 对
  GLM router 单独注释 *"arch-agnostic scalar kernel (GLM-4.x router), shared with the sm100 build"*。
  对照：`topk_sigmoid`/`inkling`/`dflash` 等只在 Blackwell 头出现（`docs/53`/`docs/54`）。
- **原因链**：GLM-4.6 在本专题前序里已被证明走 `cc≥90` 路径（`docs/45` §A：partial RoPE 的
  Ampere/Hopper/Blackwell 不对称；GLM partial-rotary 只能 `cc≥90`），所以它的路由身体**必须**能被
  Hopper 闭包编入。而该身体没有任何 SM100 专用原语，因此直接跨代共享同一文件，无需重写。
- **命名启示**：不能从 `*_sm100.cuh` 后缀反推「只能 SM100 编译」——它是**创建渠道**的标记而非**硬件
  约束**。核 `TaskType` 表时同理（`docs/24`）：`TASK_MOE_TOPK_SIGMOID_SM100` 与
  `TASK_GLM_MOE_ROUTER_SM100` 都在「SM100」块，但前者真 SM100-only，后者 arch-agnostic。

---

## 6. 单测层（接线有效性的证据）

**`topk_sigmoid`**（`tests/runtime_python/blackwell/sm100_moe_sigmoid/`，3 文件 + 1 源）：

- `test_topk_sigmoid_testmode.py`（210 行）：用 `PersistentKernel` test_mode 建单层图（`block 256`），
  对 DeepSeek-V3 配置（256 专家 / 8 组 / topk_group 4 / top-8 / scale 2.5）跑一次，与 PyTorch 参考
  比 `topk_weights`（`atol=0.01`）、`routing_indices`（逐位）、`active_expert_ids`（集合相等 +
  计数）三项。
- `test_gate_topk_sigmoid.py`（约 190 行）+ `runtime_kernel_wrapper_sm100.cu` + `setup.py`：
  **遗留 standalone wrapper 路线**（`import runtime_kernel_moe_sigmoid`），走「一个 task = 一颗
  pybind extension kernel」；对照 `docs/30`/`docs/31` 的单 kernel 单测层——即该路由**同时**落在
  test_mode 与 wrapper 两套回归里。

**`glm_moe_router`**（`tests/runtime_python/blackwell/sm100_glm4_moe/`，4 文件）：

- `test_glm_moe_router_testmode.py`（144 行）：生产 shape（160 routed / top-8 / scale 2.5 / 1 共享 /
  `GATE_PADDED=192`），四项校验——权重（`atol=rtol=1e-4`）、共享槽恒 1.0、`indices` 逐位、活跃 id
  集合 + 计数，并**额外验证 logits 前 160 列被清零、pad 列（160..192）未被改动**（`:125-135`）。
- `test_glm4_moe_builder_testmode.py`（约 110 行，**端到端**）：建 4 层 GLM-4.6（hidden 5120、96/8
  头、3 dense + 1 MoE），随机权重跑一次 decode step，与 **HF `Glm4MoeForCausalLM`** 参考比 lm-head
  logits 与 argmax token——这是本批路由里**最强的接线见证**（覆盖 embed/rmsnorm/qkv+residual/paged
  attention(qk-norm+partial RoPE)/o_proj/dense MLP/MoE 全链）。
- `pytorch_reference.py` 提供 `glm_moe_router_ref` 等参考。
- 另有 `test_glm_attention_testmode.py`（GLM attention 身体）。

结论：两条路由都「登记 → 发射 → 编入 → 前端 → builder → 单测」六层全通，**无 `docs/49`/`docs/50`
记的「已实现未接线」问题**；GLM 路由还多一层端到端 builder 证据。

---

## 7. 判据映射

| 判据 | 本文两条路由是否贡献 |
|---|---|
| C1 单次 launch | 间接（两个 task 只是同一常驻图里的节点，launch 由 `PersistentKernel` 统一发） |
| C2 常驻 | 间接（同 C1） |
| C3 多算子范围 | **是**——给出 MoE 路由段「两条 device 身体 + 一致 ABI + 折平边界」的清单 |
| C4 设备内协调 | 间接（跨 task 依赖由 MPK event 承担，`docs/06`/`docs/09`） |
| C5 SM 级调度 | 否（device body 层不涉 worker/scheduler 划分） |

结论：**本文贡献 C3 的「MoE 路由段的范围与 ABI 契约」**，与 `docs/37` §2（摘要）、
`docs/54`（Inkling 第三支路由）互补。两条路由把「选择策略 / 归一化口径 / 共享专家 / 硬件约束」全部
关在 task 内部、对外只暴露一个统一 ABI，是「薄 task + 统一接口」这一 megakernel 组装范式的又一组实证
（对照 `docs/27` builder 层折平）。

---

## 8. 局限（诚实标注）

- **纯源码走读，未构建 / 未运行**：两条路由都编在 SM100 闭包里（GLM router 虽 arch-agnostic，本文仍
  未单独在 sm90 编它），未取任何 device 计数；所有行号为快照 `f9eb70c` 静态事实，不构成上游 bug 判定。
- **数值等价未验证**：单测声明的 `atol/rtol` 只作了**声明**转述，本文未跑；Phase 2 寄存器快照在竞争下
  的正确性、`active` 的良性写竞争都只按源码读。
- **`finished`/专家切片未走**：`topk_sigmoid` 的 `finished` 生产传 `nullptr`、`start/end_expert` 传
  `0/num_experts`，本文只记「接口预留、本路径未用」，未追它在投机解码多节点共享路由里的实际用法。
- **`WARPS_PER_CTA` 是悬空模板参数**：`topk_sigmoid` 的 `WARPS_PER_CTA=8` 在函数体内**无任何引用**
  （`grep` 仅 `:78` 声明处），本文只记不判其意图。
- **模型侧常量未核**：GLM `NUM_ROUTED_EXPERTS=160`/`N_SHARED_EXPERTS=1`/`TOPK=8`/`GATE_PADDED=192`
  取自 `builder.py:55-61` 与 docstring，本文未加载真实 ckpt、未逐层核权重名与 HF 对应；DeepSeek 侧
  配置取自单测常量。

---

## 9. 参考坐标（可核对）

- 快照：`mirage-project/mirage` 分支 `mpk`，commit `f9eb70c`（与 `docs/06`–`docs/54` 同快照）。
- **device 身体（sigmoid）**：`include/mirage/persistent_kernel/tasks/blackwell/topk_sigmoid_sm100.cuh`
  （364 行；七步注释 `:44-58`；`merge_top2:65-73`；模板/签名 `:75-94`；Phase 0 `:103-118`；
  static_assert `:120-147`；线程划分 `:149-163`；Phase 1 `:169-206`；Phase 2 组 top-2 `:208-230`；
  Phase 3 选组 `:232-255`；Phase 4 掩码 `:257-263`；Phase 5 专家 top-K `:265-337`（tie-breaker
  `:286-296`、取回未加 bias 分 `:298-306`、置 `-10000` `:326-336`）；Phase 6 `:339-346`；
  Phase 7 紧凑化 `:350-361`）。
- **device 身体（GLM）**：`…/blackwell/glm_moe_router_sm100.cuh`（155 行；参考数学 `:18-44`；
  模板/签名 `:48-62`；Phase 0 `:73-83`；Phase 1 `:85-134`（sigmoid+清零 `:91-96`、top-8 `:101-114`、
  缩放 `:115-119`、共享专家 `:120-125`、写 `:127-133`）；Phase 2 `:137-153`）。
- **登记 / 编入 / 布线**：`runtime_header.h:169/219`（`_BEGIN/_END` `:218/:223`）；
  `tasks/blackwell/task_header.cuh:33/66`、`tasks/hopper/task_header.cuh:27-28`（共享注释）；
  `src/kernel/graph.cc:477-481/682-686`；`include/mirage/kernel/task_register.h:126-127/213-214`；
  `src/kernel/task_register.cc:258-315/2935-3001`；`src/kernel/runtime.cc:1815/1855`（名字映射）。
- **前端**：`python/mirage/mpk/persistent_kernel.py:1165-1202`（GLM）、`:1995-2029`（sigmoid）。
- **builder**：`python/mirage/mpk/models/glm4_moe/builder.py`（常量 `:55-61`；gate pad `:284-293`；
  router 调用 `:305-311`；w13 折叠共享专家 `:313-343`）；`…/deepseek_v3/builder.py:905/1501`。
- **单测**：`tests/runtime_python/blackwell/sm100_moe_sigmoid/`（`test_topk_sigmoid_testmode.py` 210 行、
  `test_gate_topk_sigmoid.py`、`runtime_kernel_wrapper_sm100.cu`、`setup.py`）；
  `…/sm100_glm4_moe/`（`test_glm_moe_router_testmode.py` 144 行、
  `test_glm4_moe_builder_testmode.py`、`test_glm_attention_testmode.py`、`pytorch_reference.py`）。
- **本专题交叉引用**：`docs/37` §2（路由三兄弟摘要）、`docs/54`（Inkling logsigmoid 第三支）、
  `docs/27`（builder 层家族折平）、`docs/24`/`docs/25`（`TaskType` 分区口径）、`docs/45` §A（GLM
  cc≥90 路径）、`docs/49`/`docs/50`（接线审计，本文两条路由均无孤儿问题）。
