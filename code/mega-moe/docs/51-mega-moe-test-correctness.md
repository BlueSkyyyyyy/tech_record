# Mega MoE 的端到端测试与逐位正确性判据（`tests/test_mega_moe.py`）

> 本专题到第 50 篇为止，已经把 Mega MoE 的 host 编排（第 42 篇）、调度器（第 34/44 篇）、主 kernel
> （第 03/09/10/11 篇）、共享专家（第 44 篇）、测试基座（第 40 篇）与操作数生成器（第 41 篇）逐个拆开，
> 却始终没有正面回答一个问题：**这个 kernel 的「正确」到底由谁定义？** 答案就是本篇要拆的
> `tests/test_mega_moe.py`（467 行）——它是 Mega MoE 唯一的端到端验收测试，也是整个专题里
> **正确性判据（逐位相等）与性能账目（TFLOPS/GB/s/overlap）的来源**。
>
> 它有两部分：**能跑的部分**（测试侧的 SF 布局变换、逐位判据、单 rank 参考数学、开销公式）架构无关，
> H100 真机实测；**跑不了的部分**（真正端到端的 fused kernel 对比）是 SM100 专属，只做源码精读。

---

## 1. 测试全貌：一次 launch 比两套实现

测试主函数 `test()`（`tests/test_mega_moe.py:75`）默认配置（`:429-449`）：8 进程、每 rank 最多 8192
token、`hidden=7168`、`intermediate=3072`、`num_experts=384`、`topk=6`、`num_shared_experts=1`、
`activation_clamp=10`、`fast_math=1`、`mma_type` 三选一（`fp8xfp4` 默认 / `fp8xfp8` / `bf16xbf16`）。

它同时构造**两条路径**：

| 路径 | 实现 | 说明 |
|------|------|------|
| `run_fused()` (`:202-219`) | **单个 mega-kernel** `deep_gemm.fp8_fp4_mega_moe` / `bf16_mega_moe` | dispatch+L1+SwiGLU+L2+combine 融合、通信-计算重叠 |
| `run_baseline()` (`:304-335`) | **非重叠基线**：DeepEP `dispatch` → `m_grouped` GEMM ×2 → tilelang SwiGLU → DeepEP `combine` | 先通信后计算的分步实现 |

两者共享同一份权重与输入，最后用 **`torch.equal` 逐位相等**（`:345-346`）验收。这个「两套实现、
逐位判据」的结构，正是 NVIDIA 系库惯用的验收方式：只要有一路是可信的、且两路在确定性下必须位等，
就把「内核 bug」与「量化误差」彻底分离——**没有任何容差，任何误差都是 bug**。

### 1.1 三个必须搞对的测试细节

- **`use_deterministic_algorithms(True)`（`:77`）**：DeepGEMM 的确定性开关（第 48 篇 P38-1）。
  它关掉 cuBLASLt 回退、禁 Mega Gate split-K、钉死 mHC splits——都是为了让 baseline 复现。
  没有它，baseline 的 `m_grouped` 可能走 cuBLASLt（`nvjet_*`），与 fused kernel 的累加顺序不同，
  逐位判据必然失败。
- **`get_block_m_for_mega_moe`（`:157-158`）**：测试要自己算一次 `block_m`，因为共享专家的输入 SF
  要**按 kernel 实际会用的 BLOCK_M** 预排（见 §3.2）。这个值是 host 侧同一个启发式的输出
  （`csrc/apis/mega_moe.hpp:21-31`，实参 `num_ranks/num_experts/num_max_tokens_per_rank/num_topk/
  num_tokens/mma_type`）。
- **debug 模式会清零整个 buffer**（`:190` 注释）：所以 `run_fused` 每次调用前都要
  `copy_inputs_to_buffer()`（`:191-200`）把 `x/x_sf/topk_idx/topk_weights`（以及 shared 的
  `shared_l1_acts_sf`）重新拷进对称缓冲区。`x` 和 `shared_l1_acts` 在缓冲区里是**同一个 view**
  （第 42 篇：`layout/mega_moe.cuh:114`），所以共享输入的 SF 是单独一份 `shared_l1_acts_sf`。

### 1.2 baseline 依赖 DeepEP，所以端到端在 H100 上还跑不了

正确性对比被 `is_legacy_loaded and num_correctness_tests > 0` 两层门控（`:339`）：

```python
if is_legacy_loaded and num_correctness_tests > 0:
    ...
    assert torch.equal(fused_stats, baseline_stats)
    assert torch.equal(fused_y, baseline_y)
```

`is_legacy_loaded` 来自 `import_baseline()`（`:15-33`），它尝试 import **DeepEP**（`deep_ep`）与
**tilelang_ops**（`third-party/tilelang_ops`）；失败就只打印一行并**跳过正确性对比**（只跑 fused
+ benchmark）。也就是说：**这个验收测试的 oracle 本身依赖 DeepEP 的 dispatch/combine**。而本专题第 0
篇就记过（`docs/08`、ROADMAP「阻塞」）：本机 PyTorch 链的 NCCL 2.29.7 与 DeepEP 2.5 需要的 ≥2.31 API
不兼容，`EPBuffer` 构造段错误。因此在 H100 上：① 端到端 fused kernel 本就 SM100 专属；
② 连它的 oracle（DeepEP baseline）也因环境受阻。本篇只做**测试侧可架构无关验证的部分**。

---

## 2. 正确性判据：`torch.equal` 而非 `calc_diff`

测试用 `torch.equal`（`torch.equal(fused_y, baseline_y)`，`:346`），而不是本专题其它测试常用的
`calc_diff`（第 40 篇：`calc_diff` 恒等于归一化平方误差，`=0` 当且仅当逐元素相等）。两者在本测试里
等价但**严格程度不同**：`torch.equal` 是逐位（bitwise），`calc_diff` 是数值。Mega MoE 选逐位，
因为它把「通信-计算重叠」当成**纯调度变换**——重叠只改变**谁在什么时候算**，不改变任何一次乘加，
所以结果必须逐位不变（这也是为什么 split-K/顺序相关的优化都被确定性开关关掉）。

H100 实测（用真实 SM90 grouped BF16 kernel `m_grouped_bf16_gemm_nt_contiguous` 演示判据）：

- 同一 kernel 两次调用 `torch.equal == True`、`calc_diff == 0.000e+00`；
- 只改动 1 个输入 bf16 元素 → `torch.equal == False` 且 `calc_diff = 1.4e-05`（坐实「有误差就能被
  逐位判据抓到」）；
- **组内 token 重排后，对应行输出逐位不变**——每行的 K 归约顺序与其它行无关，这是「重叠/重排不改变
  数值」的可执行证据。

---

## 3. 测试侧的三个 SF 布局变换（纯 torch，H100 真机验证）

Mega MoE 的激活是 **per-token FP8 + gran_k=32 packed UE8M0**（第 10/25 篇）。`create_inputs()`（`:123-187`）
在把 `x` 塞进缓冲区前，要做三次与 kernel 内部布局严格对齐的变换。

### 3.1 `_cast_fp8_for_mega_moe`（`:53-59`）：量化 + TMA 对齐 SF

```python
x_fp8, x_sf = per_token_cast_to_fp8(x, use_ue8m0=True, gran_k=32, use_packed_ue8m0=True)
x_sf_tma = torch.empty_strided((mn, packed_sf_k), (1, align(mn, 4)), ...)
x_sf_tma.copy_(x_sf)
```

`per_token_cast_to_fp8`（`deep_gemm/utils/math.py:26-52`，第 27 篇 P27-1 拆过）：per-token amax/448、
`ceil_to_ue8m0`、gran_k=32；`use_packed_ue8m0=True` 时把 4 个 UE8M0 字节 LSB-first 打包成 1 个 int32
（`pack_ue8m0_to_int`，`math.py:19-23`）。所以输出 `x_sf` 的形状是 `(mn, k/32/4)`。H100 实测
`k=7168`：`x_sf` 为 `(200, 56)` 的 **int32**，`56 = 7168/32/4`。

`x_sf_tma` 是给 TMA 用的**列主序**副本：shape 相同、stride 为 `(1, align(mn,4))`。H100 实测
`mn=200` → stride `(1, 200)`，且内容与 `x_sf` 逐位相同。为什么要单独复制：TMA 描述符需要 SF 沿行
方向 16B 对齐的列主序（第 15/25 篇的 `get_tma_aligned_size`），`x_sf` 本身是行主序紧凑排列。

### 3.2 `_to_shared_mega_moe_sf_layout`（`:36-50`）：共享专家的 UTCCP 4×32 交错

这是**最容易被忽略的一处**。只有共享专家（`num_shared_experts>0`）需要它（`:163`），因为它要把
共享输入的 SF 预排成 kernel 里 UTCCP（`SM100_UTCCP_4x32dp128bit_2cta`，第 10 篇）读 smem→TMEM
时的 **4×32 交错行序**：

```python
transposed_m_idx = (m_idx // 128) * 128 + (m_idx % 32) * 4 + (m_idx % 128) // 32
result[block_idx * aligned_block_m + transposed_m_idx].copy_(sf[block_idx * block_m + m_idx])
```

对照 kernel 的 `transform_sf_token_idx`（`impls/sm100_fp8_fp4_mega_moe.cuh:145-149`）：

```cpp
const uint32_t idx = token_idx_in_expert % BLOCK_M;
return token_idx_in_expert / BLOCK_M * SF_BLOCK_M +
       (idx & ~127u) + (idx & 31u) * 4 + ((idx >> 5) & 3u);
```

两者**逐项等价**：`SF_BLOCK_M == align(BLOCK_M,128) == aligned_block_m`、`idx & ~127 = (idx/128)*128`、
`(idx&31)*4 = (idx%32)*4`、`((idx>>5)&3) = (idx%128)//32`（因为 `&3` 只保留 128 内的 32 分组位）。
H100 实测对 `block_m ∈ {8, 64, 128, 240}` 逐点 `== kernel 公式`，且**未被占用的 padding 行全零**、
输出张量是列主序 `(1, num_max_sf_tokens)`。

> 一个坑（本实验踩到）：`_to_shared_mega_moe_sf_layout` 的 `num_max_sf_tokens` 必须 ≥
> `ceil(num_tokens/block_m) * align(block_m,128)`，否则写越界（`test:40-43` 的 `empty_strided`
> 不会帮你算）。对 137 token、`block_m=64` 需要 `3*128=384`。真实测试里这个上界来自
> `buffer.shared_l1_acts_sf.shape[0]`（对称缓冲区按所有候选 BLOCK_M 取 max，第 42/44 篇）。

### 3.3 `_copy_fp8_sf`（`:62-70`）：padding 行「广播最后一行」

```python
dst[:num_tokens].copy_(src)
if num_tokens < dst.shape[0]:
    dst[num_tokens:].copy_(src[-1:].expand(dst.shape[0] - num_tokens, -1))
```

为什么是**广播最后一行**而不是填零：kernel 读取的行数固定为 `num_max_tokens`，尾部 padding 行的
SF 若为零或脏值可能触发非法缩放（0 不是合法 UE8M0 幂），而复制最后一行有效 SF 是安全且确定的
（这些行的输出被 token 掩码丢弃）。H100 实测：前段 copy、后段广播、`num_tokens==0` 完全 no-op、
shape 相同时直接 copy——三种分支全对。

---

## 4. 性能账目：TFLOPS / HBM / NVLink / overlap

测试末尾的性能报告（`:368-420`）是整个专题里出现过多次的「重叠收益」数字的出处。它不测中间量，
而是**按语义闭式算字节与 FLOP**，再除以实测时间：

- **FLOP**（`:370-372`）：`2·M_recv·H·I·3 + 2·M_tok·H·I_shared·3`。系数 **3 = L1 的 gate 与 up 各算
  一次 + L2 一次**（L1 输出 2I 列，L2 输出 H 列但收缩维 I）。`M_recv` 是本 rank 实际收到的 token 数
  （`:353-357` 用 `uneven_all_gather` 汇总全局 `topk_idx` 后数落在本 rank 专家区间的条目）。
- **HBM 字节**（`:374-395`）：routed = 「被访问的专家权重（L1 `2·I·H` + L2 `H·I`）+ 激活读写（L1 读
  H、L1 写 I、L2 读 I、L2 写 H）+ 输出写（恒 bf16，`2` 字节）」；shared 同构。注意
  `routed_weight_elem_size` 随 mma_type 变（bf16=2 / fp8=1 / **fp4=0.5**，`:377`）。
- **NVLink 字节**（`:398`）：`M_recv·H·3` = dispatch 拉取激活（1×）+ combine 写回（2×，因为 topk 贡献）。
- **串行归约时间**（`:401`）：`t_reduction = M_tok·H·2·(1+topk) / 6.5e12`——一个**经验常数 6.5e12 B/s**
  的串行 combine 归约模型。
- **overlap factor**（`:408`）：`elapsed / (elapsed - t_reduction)`，把「若归约不重叠会多花
  `t_reduction`」折算成 TFLOPs/GB/s 的放大倍数。

H100 实测（默认形状，`T=8192`、`topk=6`、`I=3072`、取 `M_recv=8192`、`touched=48` 作示范）：

| mma_type | routed HBM | shared HBM | total | NVLink | FLOP |
|----------|-----------|-----------|-------|--------|------|
| bf16xbf16 | 6368 MiB | 446 MiB | 6.65 GiB | 168 MiB | 2.16 T |
| fp8xfp8 | 3240 MiB | 279 MiB | 3.44 GiB | 168 MiB | 2.16 T |
| fp8xfp4 | 1728 MiB | 279 MiB | 1.96 GiB | 168 MiB | 2.16 T |

`t_reduction = 126.5 us`，于是 `elapsed=500us` 时 `overlap_factor=1.339`（raw 4329 → overlap 5795
TFLOPS）；`elapsed` 越小倍数越大（**:409-417** 打印的 `overlap:` 一栏）。另外实测 H100 一次大张量
`sum` 的读带宽 ~2.6 TB/s，用来对照：**6.5e12 不是 HBM 带宽**，而是 combine 归约（跨 NVLink 拉 topk
贡献再 fp32 累加）的经验串行吞吐常数。

---

## 5. H100 实测汇总（`harness/exp_mega_moe_test.{py,sh,out.txt}`）

脚本**直接 import 真实 `tests/test_mega_moe.py`** 复用它的 helper，而不是重写；逐位判据用真实
SM90 kernel。**37 PASS / 0 FAIL**：

- **A. 测试侧 SF 变换**：`_cast_fp8_for_mega_moe` 形状 stride 全对（`x_sf` `(200,56)` int32、
  `x_sf_tma` stride `(1,200)`、内容逐位同）；`_to_shared_mega_moe_sf_layout` 对 4 组
  `block_m ∈ {8,64,128,240}` 与 kernel 公式**逐点相等**、padding 全零、列主序 stride 对；
  `_copy_fp8_sf` 三分支全对。
- **B. 逐位判据**：同一 SM90 grouped BF16 kernel 两次调用逐位相等 / `calc_diff=0`；改 1 元素即
  逐位不等（`calc_diff=1.4e-05`）；组内重排对应行逐位不变。
- **C. 单 rank 参考数学**：手写「dispatch→L1→SwiGLU→L2→combine(+shared)」的**朴素循环**与
  **分组向量化**在 fp64 下 `relerr=2.7e-16`（语义等价）；共享项**无权重、只加一次**（若错乘
  `Σ topk 权重` 则 relerr=0.687）；masked `-1` 索引权重清零；`activation_clamp=1.0` 生效。
- **D. 账目公式**：三种 mma_type 的字节/FLOP 复算；overlap factor 随 elapsed 单调；H100 实测
  归约读带宽 ~2.6 TB/s。
- **E. 骨架事实**：源码断言 `use_deterministic_algorithms(True)` / `torch.equal(fused_y,baseline_y)`
  / `bench_kineto(run_fused,'mega_moe')` / `ep_buffer.dispatch` / legacy 门控 均命中。

---

## 6. 边界与「仅分析」

- **端到端 fused vs baseline 的逐位对比：SM100 专属，仅分析**。fused kernel 是 SM100（第 03/10 篇），
  且 oracle 依赖 DeepEP（本机 NCCL 版本受阻，ROADMAP「阻塞」）。
- **bench_kineto 抓 `mega_moe` 内核名**（`:363`）与「overlap TFLOPs」是官方 PR 基准表的同源指标
  （第 02 篇 §8 已引 PR #316 的 EP8 基准）。本篇只复核其**公式与测试侧变换**，不复核 SM100 上的
  实测耗时。
- 测试还演示了几个共享专家/BF16 变体的分支（`:205-218` 的 `bf16_mega_moe`、`:277-302` 的 shared
  bias、`:377-378` 的 fp4 权重半字节），与第 20 篇（BF16 Mega MoE）、第 44 篇（共享专家）一致。

## 7. 参考链接

- DeepGEMM 仓库（Mega MoE 引入）：<https://github.com/deepseek-ai/DeepGEMM>
- Mega MoE 基准 PR #316：<https://github.com/deepseek-ai/DeepGEMM/pull/316>
- DeepEP（baseline dispatch/combine）：<https://github.com/deepseek-ai/DeepEP>
- 测试文件（本地核对版本 `057ca59`）：`tests/test_mega_moe.py`
