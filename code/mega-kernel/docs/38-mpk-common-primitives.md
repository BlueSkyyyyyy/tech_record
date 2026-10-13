# MPK 共享原语层走读：`tasks/common/*` 与 `ampere/mma.cuh`（P124-1）

> 定位：本文补掉 `docs/37-mpk-sm100-misc-tasks.md` **§8「局限」明留的两条尾巴**（原话）——
>
> > `sampling_from_logits_kernel`（`tasks/common/sampling.cuh` 的 CUB BlockReduce 内部）未展开、
> > `mma_m16n16k16`（`tasks/ampere/mma.cuh`）汇编未逐条；
> >
> > `mla_prefill_host.cu` 的原型来源仍未走。
>
> `docs/06`–`docs/37` 一路走读了 MPK「一个 task = 一个 device 身体」的各代际实现
> （`hopper/*`、`blackwell/*`、`cute/`、`speculative_decoding/*`），但**这些 device 身体全部站在同一层
> 「共享原语库」上**：`include/mirage/persistent_kernel/tasks/common/`（CUB 版采样、`cp.async`/`ldmatrix`、
> 逻辑布局包装、bf16 fragment 组装、SMEM 清零）与 `tasks/ampere/mma.cuh`（跨代共用的 `m16n16k16` 手写 MMA）。
> 本文把这层**被反复 `#include` 却从未逐行打开**的库走通。快照 `mirage-project/mirage` 分支 `mpk`、
> commit `f9eb70c`（与 `docs/06`–`docs/37` 同快照）。**纯源码走读（资料分析），无新实验**——
> 这些原语是 arch-agnostic/Hopper 的，且本文只读代码、不构建。

---

## 0. 三条 headline

1. **`tasks/common/` 是 megakernel 全部 device task 的「arch-agnostic 地基」，覆盖度可量**：
   7 个头文件、919 行，其中 `common_header.cuh` 是伞状入口（`:17-21`），被 **41 个 device task
   文件 `#include`**；`utils.cuh`（bf16/SMEM/PTX 助手）单独被 **33 个**文件直接引用。`ampere/mma.cuh`
   被 `blackwell/attention_sm100.cuh:18` 与 `blackwell/dflash_attention_sm100.cuh:16` 两个**非 MLA 的
   SM100 attention** 引用。即：**「Blackwell」目录里的通用 attention 其实还在用 Ampere 时代的手写
   `mma.sync` + `cp.async` + CUB**——megakernel 的 task 目录按代际分文件夹，但底层原语是**跨代共享**的。
2. **`common/sampling.cuh` 是 FlashInfer 移植的「一趟式 Gumbel-Max 采样器」，复用 CUB `BlockReduce`
   的 `Sum` 做 argmax**：`SamplingDataAndIndex::operator+` 定义成「取较大者」`sampling.cuh:76-95`，
   于是 CUB 的 `BlockReduce<...>::Sum<VEC_SIZE>(cur_data)`（`:212-215`）实际**在跑 max+argmax 归约**
   （CUB 只认 `operator+`，作者用 max-plus 语义借 `Sum` 的名字）。它一次 block 内循环跑完**所有 batch**
   （`:177`），Gumbel 噪声用 curand Philox4x32（`:104-121`），**只写回 argmax 的 index**（`:219-221`），
   是「不 materialize 全分布」的最简采样。生产上它登记为 `TASK_SAMPLING_SM100`（`task_register.cc:2543/2550`），
   与 SM100 分块版 `sampling_partial`/`sampling_reduce` 并存（`docs/37` §3）。
3. **`ampere/mma.cuh` 的 44 行里藏着一条 load-bearing 的 codegen 历史坑**：唯一函数
   `mma_m16n16k16_bf16bf16bf32`（`:20-43`）把两次 `mma.sync.m16n8k16` 拼成一个 `16×16` 累加，
   其注释（`:22-30`）明确记录：早期把累加器声明为**只写** `"=f"(D[i])` + 独立 `"f"(C[i])` 输入，
   在 `#pragma unroll` 下编译器会**复用已「死亡」累加器的寄存器**、跨 fragment 污染
   （即「K≥2 / MMA_ITERS_M≥2 bug」）；修复是把累加器改成**单一读写 `"+f"(C[i])`**，让每个
   `x_frag_f[m]` 有独立的 def-use 链。`attention_sm100.cuh:506-516` 的注释正记着这个 bug 的下游后果：
   `MMA_ITERS_M>1` 时曾用 `if (m>0) continue;` 跳过后续 tile、**损坏 EAGLE3 投机解码的 token 流**。
   即一条「共享原语」的正确性直接决定上层推理正确性——对应「已知坑」里「常驻共驻 device 适配」的一族。

---

## 1. 这层库在目录里的位置与包含关系

`tasks/` 下的目录分工（`ls` 快照）：

| 目录 | 角色 | 本文 |
|---|---|---|
| `common/` | **arch-agnostic 地基**：CUB 采样、`cp.async`/`ldmatrix`、逻辑布局、bf16 助手、SMEM 清零、worker 常数 | 走读 |
| `ampere/` | Ampere 代 device 身体 + 跨代 `mma.cuh`/`smem_layout.cuh` | 只走 `mma.cuh` |
| `hopper/`、`blackwell/`、`cute/`、`speculative_decoding/` | 各代/家族的 device 身体 | `docs/06`–`docs/37` 已覆盖 |
| `deprecated/` | 退役实现 | 不涉 |

包含关系（伞状 + 直引）：

```
blackwell/attention_sm100.cuh
  ├─ tasks/ampere/mma.cuh                   (手写 m16n16k16)
  ├─ tasks/ampere/smem_layout.cuh
  └─ tasks/common/common_header.cuh
        ├─ common/bfloat16.h
        ├─ common/copy_sm80.cuh  (cp.async / ldmatrix)
        ├─ common/dmem_layout.cuh (dmem_row/col 逻辑布局)
        ├─ runtime_header.h
        └─ common/utils.cuh      (+ worker_config.h)
```

`common_header.cuh:17-21` 是伞状入口；**41 个文件**引用它，**33 个**直接引用 `utils.cuh`。`worker_config.h`
提供 `NUM_THREADS = 128` / `NUM_THREADS_PER_WARP = 32` / `NUM_WARPS = 4`（`:18-21`），并在
`MIRAGE_GRACE_HOPPER`/`MIRAGE_GRACE_BLACKWELL` 下把 `WORKER_NUM_THREADS` 设成 **256**（`:26-29`）——
**这个 128/256 的张力见 §2.4**。

---

## 2. `common/sampling.cuh`：CUB + curand 的一趟式 Gumbel-Max 采样器

文件头（`:1-2, 17-20`）自陈：**CMU + FlashInfer 双版权**（Apache-2.0），
*"Based on FlashInfer's sampling kernel"*——是 FlashInfer 采样核的 MPK 移植。

### 2.1 数据结构与「用 max-plus 冒充 `Sum`」

`SamplingDataAndIndex<DType, IdType>`（`:71-95`）是 `{data, index}` 对，其 `operator+`（`:76-83`）
和 `operator+=`（`:85-94`）语义都是**「保留 data 较大的一对」**。CUB 的 `BlockReduce` 只要求类型有
`operator+`，于是采样核把 `BlockReduce<...>::Sum<VEC_SIZE>(cur_data)`（`:212-215`）当**argmax 归约**用：
`Sum` 只是入口名字，真正语义由 max-plus 决定。这是「借 CUB 的 block 归约骨架、不写自定义 reducer」的
省事做法，也解释了为什么 `SAMPLING_REDUCE_ALGO = BLOCK_REDUCE_WARP_REDUCTIONS`（`:154-155`，注释里的
`SAMPLING_SCAN_ALGO` 在核里**未被使用**）。

### 2.2 Gumbel 噪声与向量化

`GenerateSamplingGumbelNoise<DType, VEC_SIZE>`（`:99-149`）：`curand_init(philox_seed, subsequence+i,
philox_offset)` → `curand_uniform4` → `uniform2gumbel`（`:109-111`，`-log2e·log2(-log2(u+ε)+ε)`），
按 `VEC_SIZE` 以 4 为步长展开、尾部按 `VEC_SIZE%4` 单独补（`:123-137`），最后按需转成目标 dtype（`:139-148`）。
`kEPSILON=1e-20` 防 `log2(0)`。采样核（`:157-226`）对每个 batch 做
「向量 load logits → 加 Gumbel → block argmax」的 chunk 循环（`:183-216`），**越界位置填 `-inf`**（`:185/207`），
只有 `tx == 0` 写回 index（`:219-221`），batch 间 `__syncthreads()` 复用 SMEM 的 `TempStorage`（`:173-174/224`）。

### 2.3 生产布线与「同名不同实现」

`register_sampling_sm100_task`（`task_register.cc:2518-2550`）发射
`kernel::sampling_from_logits_kernel<256, 4, bfloat16, int>(... , seed, 0 /*philox_offset*/, batch_size)`
（`:2543-2549`），登记到 **`TASK_SAMPLING_SM100 = 265`**（`runtime_header.h:152`；`register_task_variant`
在 `:2550`）。而 SM100 分块版 `sampling_partial_sm100_kernel`/`sampling_reduce_sm100_kernel`（`sampling_sm100.cuh`）
分别登记为 `TASK_SAMPLING_PARTIAL_SM100 = 279`（`:2603`）与**同一个 `TASK_SAMPLING_SM100`**（`:2663`，
`runtime_header.h:167-168` 注释 *"The reduce stage is a variant of TASK_SAMPLING_SM100"*）。
即：**同一个 `TaskType` 下挂着两个 device 身体（一趟式 common 版 + 两段式 reduce 版）**，
由前端 layer 选路——`docs/37` §3 从采样链角度、本文从 register 行号角度各记一次。
函数名里的 `sm100` 是「SM100 块里登记的 task」，**不代表 device 身体用了 SM100 特性**——
`sampling_from_logits_kernel` 本体全是 CUB/curand，**arch-agnostic**。

### 2.4 一个口径落差：`NUM_THREADS=128` vs 256 线程 worker

`common/utils.cuh` 的清零助手 `clear_smem_buffer`（`:97-116`）以 `i += NUM_THREADS`（=`128`，
`worker_config.h:18`）跨步，而 `vec_zero_t::fill_zero`（`:124-145`）用 `blockDim.x`。MPK 的 Blackwell
worker 是 **256 线程**（`docs/34` §3.5；`sampling_from_logits_kernel` 也显式实例化为 `<256,…>`）。
故 `clear_smem_buffer` 在 256 线程 block 里只让前 128 个线程参与 bulk 清零循环——**是否安全取决于调用点
是否另有覆盖**，本文**只记录口径、不判对错**（同 `docs/36` 对源码自陈的态度）。这是「共享原语层按
Ampere 128 线程写、上层按 256 线程跑」的又一处代际张力，与「已知坑」的线程守卫族同源。

---

## 3. `common/utils.cuh`：bf16 / PTX / SMEM 助手全集

147 行、被 33 个文件直引。四组：

1. **编译期常数/小工具**：`log2e`（`:22`）、`log2_constexpr`（`:24-26`）、`max_power_of_two_le`（`:28-37`，
   路由/attention 用来把非 2 的幂规模向上取整，呼应 `topk_sigmoid` 对「专家数是 2 的幂」的要求，`docs/37` §2）。
2. **bf16 fragment 组装**：`convert_32_f32_to_16_bf16_uint32`（`:39-48`）与 `convert_f32_to_bf16_uint32`
   （`:50-59`）把 `float[32]`/`float[8]` 两两打包成 `uint32`（low/high 各 16 bit）——这是 `mma.sync`
   的 bf16 寄存器 fragment 约定，被各 attention/GEMM body 用来把 fp32 结果喂回 tensor core。
3. **PTX intrinsic 包装**：`shfl_xor_sync`（`:61-67`，`shfl.sync.bfly.b32`）、`ptx_exp2`（`:73-77`，
   `ex2.approx.ftz.f32`）、`ptx_log2`（`:83-87`，`lg2.approx.ftz.f32`）——softmax 的 `exp/log` 快路径。
4. **lane/warp 与 SMEM 清零**：`lane_id`（`:89-91`）、`warp_id`（`:93-95`，`__shfl_sync` 广播
   `threadIdx.x/NUM_THREADS_PER_WARP`）、`clear_smem_buffer`（`:97-116`）、`clear_8_floats`（`:118-121`）、
   `vec_zero_t`（`:124-145`，带「`sizeof(T)*N` 须 16 字节倍数」的 `static_assert`）。

`bfloat16.h`（278 行）定义 `type::bfloat16_t`（`storage:uint16_t`，`:25`），带 f32↔bf16 的 `cvt.rn.bf16.f32`
（`:50`）与 `to_nv_bfloat16`（`:85-86`）——是 `utils.cuh:20` 的 `using bfloat16 = type::bfloat16_t;` 之源。

---

## 4. `common/copy_sm80.cuh`：`cp.async` / `ldmatrix` 原语（本专题实验的「研究对象本体」）

104 行，全部用 `#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800))` 保护的 `CP_ASYNC_SM80_ENABLED`
（`:19-21`）——又一次「按代际标注、跨代可用」。四组：

- `cp_async_wait<N>()`（`:25-34`）：`N==0` 走 `cp.async.wait_all`，否则 `cp.async.wait_group N`（立即数）。
  **这正是「已知坑」里「`cp.async.wait_group N` 的 `N` 必须是立即数」的落点**，也是 `docs/04` §40.8
  的「尾批 `wait_group 1` 会漏等」坑所对的 PTX。
- `cp_async_fence()`（`:36-40`）：`cp.async.commit_group`。
- `load_smem<T,BYTES=16>(smem, gmem)`（`:43-57`）：`cp.async.ca.shared.global.L2::128B [smem], [gmem],
  BYTES, BYTES;`——16 B 拷贝 + **预取 128 B**。这是 MPK `persistent_kernel.cuh:1063-1078` 取 `TaskDesc`
  用的同一原语（`docs/04` §39/§40 把它的「成本与描述符大小无关」量出来）；`load_smem_with_predict`
  （`:59-74`）用 `pred ? BYTES : 0` 做谓词化零填充。
- `ldsm`/`ldsm_t`/`ldsm_divergence`（`:76-103`）：`ldmatrix.sync.aligned.m8n8.x4[.trans].shared.b16`
  取 4 个 8×8 b16 矩阵到 4 个 `uint32` 寄存器——`mma.sync` 的 A/B fragment 装载。

即：**本专题 `docs/04` §39–§56 花了十几节实测的「取 task / 数据面载体」，其 device 侧本体就是这 104 行**。
从 megakernel 视角，这层把「跨 SM 也只用 16 B 非阻塞拷贝 + 软计数」写死，是 MPK「描述符预取」与
DeepGEMM「TMA + 硬件 mbarrier」两条路线在快照里的**分界点**（对比 `hopper/tma.cuh`、`blackwell/sm100_utils.cuh`
的 TMA/mbarrier）。

---

## 5. `common/dmem_layout.cuh`：全局显存的「逻辑布局」包装

107 行，三个只读/读写包装（`STRIDE` 与 `COL` 解耦，允许 padded layout）：

- `dmem_row<T,ROW,COL,STRIDE>`（`:17-52`）：行主序，`operator()(r,c) = &base[r*STRIDE + c]`（`:27-32`），
  另有 `offset`（`:34-39`）与 `at`（`:41-51`）。
- `dmem_row_const`（`:54-81`）：如上但 `const`。
- `dmem_col_const`（`:83-107`）：列主序，索引改成 `c*STRIDE + r`（`:95/104`）。

这些是 device body 里「把 `Q/K/V/O` 等张量当二维逻辑矩阵读写」的统一入口——`attention_sm100.cuh`
的 `q_dmem(src_row,src_col)`/`paged_k_cache_dmem(...)`（`:279/295/351`）即此。把 `STRIDE≠COL` 单列出来
意味着原语层**显式支持 padded/带 stride 的全局布局**，不必在最内层手算地址。

---

## 6. `ampere/mma.cuh`：跨代共用的手写 `m16n16k16`（含 in-place 累加器修复）

44 行、单函数 `mma_m16n16k16_bf16bf16bf32(float *C, uint32_t *A, uint32_t *B, float *D)`（`:20-43`）。
实现是**两条 `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`**（`:32-36` 覆盖 C[0..3]、`:38-42`
覆盖 C[4..7]），合起来一个 `16×16` 输出，`B` 的两半分别吃 `B[0..1]`/`B[2..3]`。

**关键在注释（`:22-30`）**：函数**永远 in-place**（`(void)D; // C == D`，`:31`），累加器写成一个**读写
`"+f"(C[i])`** 操作数（`:35/41`）。注释解释：早期版本用只写 `"=f"(D[i])` + 独立 `"f"(C[i])` 输入，
两者名义上别名同一 C++ 对象、但**不把源/目的寄存器绑在一起**，于是 `#pragma unroll` 下编译器可以把某个
已「死亡」累加器的寄存器**转给后续迭代**、跨 fragment 污染——即注释所称的 *"the K>=2 / MMA_ITERS_M>=2
bug"*。

下游证据（`attention_sm100.cuh:506-516`）把这条 bug 的因果链补全：`MMA_ITERS_M>1` 时曾用
`if (m>0) continue;` **跳过第 2 个及之后的 m-tile** 作为 workaround，结果 query 槽 ≥ `16/NUM_QO_PER_KV`
没被计算，**在 K≥2 时损坏已提交的 EAGLE3 token 流**（`ac=K+1` 的 bonus token 从没算过的槽读出）；
改用 `"+f"` 单一累加器后每个 `x_frag_f[m]` 有独立 def-use 链、不再跳过任何 tile。消费点：
`attention_sm100.cuh:529`（QK^T）与 `:637`（PV）、`dflash_attention_sm100.cuh:443/528`。

从 megakernel 视角，这是「共享原语」与「上层推理正确性」直接耦合的样本：一条 44 行的内联汇编修复，
决定了投机解码 token 流对不对。也与「已知坑」里 SM100 共驻 device 适配族（TMEM-warp、命名屏障 ID、
半 CTA 提前 return）同属「常驻思想下 device 细节决定成败」的一类。

---

## 7. `mla_prefill_host.cu` 原型来源的收口

`docs/35` §9 / `docs/36` §8 / `docs/37` §8 反复留的同一条尾巴——「`mla_prefill_host.cu` 这个 prefill 的
原型来源未走」——**静态结论是：它不在快照里**。全仓 `grep` 只有一处提及：`mla_prefill_sm100.cuh:201`
的注释 *"Adapted from `mla_prefill_host.cu`. Changes: `__device__`, `blockIdx->{head, q_block}` params.
No TMA — uses `cp.async`, same as original. 256 threads = matches MPK Blackwell worker thread count exactly."*
即 device 版是**从一份 host 版原型机械改写的**（去掉 host 启动、把 `blockIdx` 换成显式 `{head,q_block}` 参数、
保持 `cp.async` 无 TMA、线程数对齐 256 worker）。

这不是孤例：MPK 的快照里有一族「**adapted from 外部/standalone 原型**」注释，且**原型源文件均不在仓库**：

| device 文件 | 注释所指原型 | 在快照？ |
|---|---|---|
| `mla_prefill_sm100.cuh:201` | `mla_prefill_host.cu` | 否 |
| `mla_prefill_tp8_sm100.cuh:17` | `src/kernel/mla_prefill_tp8.cu`（standalone kernel） | 否 |
| `mla_mtp_decode_tp8_sm100.cuh:3` | `ferret/examples/.../v001_tp8_swapab.cu` | 否（无 `ferret/` 目录） |
| `mla_mtp_decode_tp2/tp4_sm100.cuh:3` | `ferret/examples/.../v007_tp4_swapab.cu` 等 | 否 |

故这条尾巴的**可核对结论是**：MPK 的若干 attention prefill/decode device 身体是**把外部 standalone 原型
port 进「一个 task = 一个 device 函数」的形态**（同 `docs/31`/`docs/32` 对 TP standalone 的判断：
「对 ferret standalone 的机械 port」），**原型本体不在快照、无法逐行对照**——只能核 port 后的结构
（`docs/35`/`docs/36` 已做）。这与 `docs/35` §9「`mla_prefill_tp8` 是对外部 standalone 的机械 port，
只核了 port 后结构」的记录一致，二者合起来把该尾巴**收口为「外部来源、快照内不可核」**。

---

## 8. 判据与生产位置

- **判据贡献：只 C3（融合范围）**。这些原语不引入任何 host 侧提交/常驻/设备内协调/去中心化调度——
  它们是 device body **内部**的算子实现砖块，`TaskType`/注册/builder 在生产层别处（`docs/24`/`docs/25`）。
  `TASK_SAMPLING_SM100` 的登记只说明「这个身体在 SM100 task 表里被编进常驻内核」，不贡献 C1/C2/C4/C5。
- **生产位置**：`common/sampling.cuh` → `TASK_SAMPLING_SM100`（`task_register.cc:2550`），真实在前端
  （`docs/37` §3 的 qwen3 采样链）；`common/copy_sm80.cuh`/`dm_layout.cuh`/`utils.cuh` 是**所有** device
  body 的公共依赖（41/33 引用）；`ampere/mma.cuh` 只被两个非 MLA SM100 attention 消费。
- **与 `mega-moe` 专题的交叉引用**：DeepGEMM MoE 前/反向的 tile 级实现见 `../mega-moe/docs/`；本专题
  聚焦 megakernel 范式与 MPK 的 task/原语分层，**不重复** MoE 数值细节。

---

## 9. 局限（诚实标注）

- **纯源码走读，未构建 / 未运行**：这些原语虽是 arch-agnostic/Hopper 可编译的，但本文只读代码、未取
  device 计数；所有行号为快照 `f9eb70c` 静态事实。
- **CUB 内部未展开**：`sampling.cuh` 只走到 CUB `BlockReduce` 的**调用契约**（自定义 `operator+` 充当
  max+argmax），CUB 的 `BlockReduce`/`BlockScan` 内部 warp-reduction 实现未逐行（属 CUB 库侧）。
- **`bfloat16.h` 只读接口**：278 行只核了 `storage`/`cvt`/`to_nv_bfloat16` 的坐标，其余转换重载未逐条。
- **`clear_smem_buffer` 的 128/256 线程口径**只记录不判对错（§2.4）。
- **原型源文件缺失**（§7）：`mla_prefill_host.cu`/`ferret/*` 不在快照，只能核「port 后结构」。
- 未与生产端到端对跑：本机无 SM100。

---

## 10. 参考坐标（可核对）

- 快照：`mirage-project/mirage`，分支 `mpk`，commit `f9eb70c`。
- `include/mirage/persistent_kernel/tasks/common/`：`sampling.cuh`（228 行）、`utils.cuh`（147）、
  `copy_sm80.cuh`（104）、`dmem_layout.cuh`（107）、`bfloat16.h`（278）、`worker_config.h`（29）、
  `common_header.cuh`（26）。
- `include/mirage/persistent_kernel/tasks/ampere/mma.cuh`（44 行）。
- 消费者：`blackwell/attention_sm100.cuh:18/506-516/529/637`、
  `blackwell/dflash_attention_sm100.cuh:16/443/528`、`blackwell/sampling_sm100.cuh:36`。
- 生产布线：`src/kernel/task_register.cc:2518-2663`；`include/mirage/persistent_kernel/runtime_header.h:152/167-168`。
- 前置走读：`docs/06`（运行层）、`docs/07`（Hopper task）、`docs/34` §3.5（共驻 device 适配）、
  `docs/35`/`docs/36`/`docs/37`（SM100 task 家族）。实验交叉引用：`docs/04` §39–§56（取 task / 数据面载体，
  其 device 本体即 `common/copy_sm80.cuh`）。
- 相关坑（`ROADMAP.md`「已知坑」）：`cp.async.wait_group` 立即数、16 B `cp.async` SMEM 对齐
  （`docs/04` §65.9）、多载体 kernel 寄存器溢出族。
