# 52 · DeepGEMM 的 kernel 发射层：PDL / cluster / 动态共享内存与设备端 grid 依赖同步

> 前面 51 篇把 DeepGEMM 从 GEMM 主 kernel、attention、grouped 反向、Mega MoE 一路拆到 PTX 原语
> （第 38 篇）与 DeepJIT 编译缓存（第 33/47 篇）。但所有 kernel 真正落到 GPU 上的那一下——
> **`cuLaunchKernelEx` 的 launch attribute**——一直是从未单独讲过的一层：
> `deep_jit/backend/cuda/kernel.hpp:92-170` 决定了 cluster、Programmatic Dependent Launch（PDL）、
> cooperative、动态共享内存；而 24 个 device kernel 里都埋着一句 `cudaGridDependencySynchronize()`
> 作为它的设备端对偶。
>
> 本文拆这一层，并在 H100（sm90）上把**架构相关但非 SM100 专属**的 PDL / cluster / 动态 smem
> **真机跑出来**：PDL 的 prologue 重叠、触发器位置、以及「缺设备端同步就读到 stale」的反例。

---

## 1. 为什么单开一篇：发射层夹在「host 配置」与「kernel 正文」之间

前面各篇讲过的两层：

- **host 配置**（第 5/28 篇）：block_m/n/k、cluster、stage、线程数由启发式算出来，进入 `GemmConfig`；
- **kernel 正文**（第 7/9/10 篇）：warp specialization、mbarrier 流水、UMMA/WGMMA。

但 `GemmConfig` 到「真的产生一个 CUDA launch」之间还有一步：每个 kernel 的
`compile_and_launch` 把配置翻译成 **`deep_jit::cuda::LaunchOptions`**，再由 DeepJIT 的
`Kernel::launch` 用 **driver API** `cuLaunchKernelEx` + `CUlaunchAttribute` 发射。
本文就补这一步。

一句话结论（先给）：**DeepGEMM 的 kernel 发射永远是同一个 driver 调用**，装配 **至多 3 个属性**
（cooperative / cluster dimension / PDL）；其中只有 **PDL 是可被用户全局切换的**
（`deep_gemm.set_pdl`），另两个（cooperative、non-portable cluster）在当前代码里**从不启用**，
只是「支持但未用」。而 24 个 device kernel 开头都有一句「等前一个 kernel」的
`cudaGridDependencySynchronize()`，它是 PDL 正确性的设备端半边。

---

## 2. host 侧：`LaunchOptions` 与 `Runtime::launch`

### 2.1 结构、默认值与 merge（`options.hpp:178-233`）

```cpp
// third-party/deep_jit/include/deep_jit/backend/cuda/options.hpp
struct LaunchOptions {                       // :178
    std::optional<CUstream> stream;          // :180
    std::optional<int>      num_smem_bytes;  // :183  动态共享内存
    std::optional<dim3>     grid_dim;        // :186
    std::optional<dim3>     block_dim;       // :187
    std::optional<dim3>     cluster_dim;     // :188
    std::optional<bool>     cooperative;     // :191
    std::optional<bool>     enable_pdl;      // :194
    std::optional<bool>     nonportable_cluster_size_allowed; // :197

    static LaunchOptions default_options(const Env&) {        // :199
        return { .stream = std::nullopt, .num_smem_bytes = 0,
                 .grid_dim = std::nullopt, .block_dim = std::nullopt,
                 .cluster_dim = dim3(1, 1, 1),
                 .cooperative = false, .enable_pdl = false,
                 .nonportable_cluster_size_allowed = false };
    }
    LaunchOptions override_with(const LaunchOptions& overrides) const { // :212
        ...
    }
};
```

两个关键语义：

- **`std::optional` 表示「是否指定」**：`override_with` 只在 `overrides.字段` **有值**时才覆盖
  （`if (overrides.cooperative) ...`，`:224-229`），所以**显式写 `false` / `0` 也是「有值」**，
  会真正覆盖默认；而「没写」表示继承。
- **`cluster_dim` 默认是 `(1,1,1)`**（不是 `nullopt`），是唯一有非空默认的 launch 维度。

### 2.2 每个 kernel 只填四个字段，PDL 是唯一全局可调项

DeepGEMM 里所有 `compile_and_launch` 构造的 `LaunchOptions` **一律只填 4 个字段**
（以 `sm90_fp8_gemm_1d2d.hpp:119-128` 与 `sm100_bf16_gemm.hpp:124-128` 为例）：

```cpp
.options = {
    .num_smem_bytes = config.pipeline_config.smem_size,             // 流水线 smem
    .grid_dim       = dim3(config.launch_config.num_sms, 1, 1),     // = SM 数
    .block_dim      = dim3(config.launch_config.num_threads, 1, 1),
    .cluster_dim    = dim3(config.layout.get_cluster_size(), 1, 1), // 1/2/... CTA
},
```

`cooperative`、`enable_pdl`、`nonportable_cluster_size_allowed` **从不被设置**，因此始终继承
`Runtime` 的 `default_launch_options`（`runtime/runtime.hpp:31` 声明、`:42` 初始化）。唯一会改它的是
用户开关 `set_pdl`（`csrc/apis/config.hpp:27-31`，README `152` 行有说明）：

```cpp
m.def("set_pdl", [](const bool& new_enable_pdl) {
    jit->default_launch_options.enable_pdl = new_enable_pdl;   // :28
});
```

发射时（`runtime/runtime.hpp:71-73`）：

```cpp
void launch(const std::shared_ptr<Kernel>& kernel, const LaunchOptions& override_options, ...) {
    kernel->launch(default_launch_options.override_with(override_options), args...); // :72
}
```

即 **`全局默认 → 每 kernel override`** 的两级合并。H100 上直接 include 真实头文件验证（见
`src/pdl_launch_host.cpp`、`harness/exp_pdl_launch.out.txt`）：默认值全对；每 kernel 四字段正确覆盖；
`set_pdl(true)` 后 per-kernel 不吃 `enable_pdl` 也变 true；显式 `false/0` 能覆盖掉 base 的 `true`。

> **为什么 nonportable / cooperative 从不启用**：DeepGEMM 的 cluster 都 ≤ 8（H100 便携上限），
> 不需要 non-portable 属性；而 cooperative launch（`cudaLaunchCooperativeKernel` 语义、允许
> grid-wide sync）会让 grid 必须一次性全部驻留，与 DeepGEMM「按 SM 数开 grid + 可被多任务共享」
> 的部署方式冲突，故从不使用。

---

## 3. `Kernel::launch`：装配 `CUlaunchAttribute`（`kernel.hpp:92-170`）

DeepJIT 的发射是纯 driver API（不经过 runtime `<<<>>>`），逐段：

1. **8 个 `has_value` 断言**（`:97-105`）：`kernel_handle`、`num_smem_bytes`、`grid_dim`、
   `block_dim`、`cluster_dim`、`cooperative`、`enable_pdl`、`nonportable_cluster_size_allowed`
   都必须先被指定（optional 有值）——这就是为什么 `default_options` 必须给它们全填上值。
   随后是正数检查（grid/block > 0、cluster.x > 0、cluster.y/z == 1「只支持一维 cluster」，`:106-113`）。
2. **动态共享内存上限**（`:115-118`）：`num_smem_bytes > 0` 时先
   `cuFuncSetAttribute(MAX_DYNAMIC_SHARED_SIZE_BYTES, num_smem_bytes)`——把每个 kernel 需要的
   smem（Mega MoE 是 232448 B）设为上限，否则超过 48 KiB 的 launch 会失败。
3. **non-portable cluster 的「只设不撤」坑**（`:124-131`）：注释明确写道——不能写成
   `if (x) set(1); else set(0);`，因为第一次用 `=true` 设过属性后，第二次 `=false` 若不再显式设，
   属性会**残留**；DeepGEMM 选择「一直在 true 侧设」，因为 launch 开销重要、而 `=false` 侧副作用可忽略。
4. **属性数组**（`:133-153`）：`std::array<CUlaunchAttribute, 3>`，按需追加：
   - `cooperative` → `CU_LAUNCH_ATTRIBUTE_COOPERATIVE`（`:136-140`）；
   - `cluster_dim->x > 1` → `CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION`（`:141-147`）；
   - `enable_pdl` → `CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION`（`:148-153`）。
     注意属性名是「**programmatic stream serialization allowed**」——PDL 打开的是「允许下一个
     kernel 不等前一个跑完就先发」。
5. **`cuLaunchKernelEx`**（`:155-170`）：填 `CUlaunchConfig`（grid/block/smem/stream/attrs），
   kernel 参数用 `kernel_arg_pointer`（支持 `NoRefPtr` 解引用，`:24-31`），最后 driver 调用发射。

一句话：**发射层不改变任何数学**，它只把「这块 grid 要几个 CTA 组成 cluster、要不要和其它 kernel
重叠发射、要多少动态 smem」告诉硬件。

---

## 4. 设备侧：24 处 `cudaGridDependencySynchronize()`，且从不显式触发

PDL 的设备端对偶是一条 `griddepcontrol.wait`，CUDA C++ 里叫 `cudaGridDependencySynchronize()`。
DeepGEMM 在 **24 处**调用它（22 个 device kernel，其中 `smxx_layout.cuh` 有三个 kernel 各一处）：

| 位置 | 行 | 位置 | 行 |
|---|---|---|---|
| `impls/sm100_fp8_fp4_mega_moe.cuh` | 289 | `impls/sm100_bf16_mega_moe.cuh` | 244 |
| `impls/sm100_fp8_fp4_gemm_1d1d.cuh` | 195 | `impls/sm100_bf16_gemm.cuh` | 153 |
| `impls/sm90_fp8_gemm_1d2d.cuh` | 149 | `impls/sm90_fp8_gemm_1d1d.cuh` | 157 |
| `impls/sm90_bf16_gemm.cuh` | 131 | `impls/sm90_bmk_bnk_mn.cuh` | 97 |
| `impls/sm100_bmk_bnk_mn.cuh` | 106 | `impls/sm100_mega_mhc.cuh` | 170 |
| `impls/sm100_bf16_mega_gate.cuh` | 120 | `impls/sm100_tf32_hc_prenorm_gemm.cuh` | 137 |
| `impls/sm90_tf32_hc_prenorm_gemm.cuh` | 120 | `impls/sm100_mqa_logits.cuh` | 148 |
| `impls/sm100_sparse_mqa_logits.cuh` | 259 | `impls/sm90_fp8_mqa_logits.cuh` | 155 |
| `impls/sm90_fp8_paged_mqa_logits.cuh` | 135 | `impls/smxx_layout.cuh` | 36 / 99 / 222 |
| `scheduler/sm100_mqa_logits.cuh` | 108 | `scheduler/sm100_sparse_mqa_logits_metadata.cuh` | 191 |
| `scheduler/sm100_paged_mqa_logits.cuh` | 61 | `scheduler/sm90_paged_mqa_logits.cuh` | 17 |

**放置位置极有讲究**：它总在「共享内存 / mbarrier 初始化 + cluster 同步」**之后**、
「block scheduler 开始消费前一个 kernel 的数据」**之前**。以 `sm90_fp8_gemm_1d2d.cuh:138-157` 为例：

```cpp
cutlass::arch::fence_barrier_init();                              // :139
(kNumTMAMulticast > 1) ? comm::cluster_sync_with_relaxed_arrive() // :142
                       : __syncthreads();
// Register reconfigurations ...                                  // :146-147
cudaGridDependencySynchronize();                                  // :149  ←「等前一个 kernel」
auto scheduler = sched::Scheduler<...>(...);                      // :152  之后才开始取数据
```

Mega MoE 主 kernel 同构：`cluster_sync()` 之后、`MegaMoEScheduler` 之前（`sm100_fp8_fp4_mega_moe.cuh:283-289`）。
**这就是 PDL 的重叠点**——barrier init、cluster sync、TMEM 分配这些「不依赖上一个 kernel 输出」的
**prologue** 可以在前一个 kernel 还在跑时先做，直到真正要读它的输出才 `griddepcontrol.wait`。

**与 `cudaTriggerProgrammaticLaunchCompletion` 的不对称**：DeepGEMM **一处都不调用**
显式触发（全仓库 grep 为空），即依赖 **隐式触发**（primary grid 结束时自动触发）。这一点我们
用实验证实了它的含义（见下节）。

---

## 5. PDL 语义实机验证（H100 / sm90）

先看编译产物（`nvcc -ptx`）确认两条 intrinsic 对应的 PTX：

```
cudaTriggerProgrammaticLaunchCompletion()  ->  griddepcontrol.launch_dependents;   // 提前解锁后继
cudaGridDependencySynchronize()            ->  griddepcontrol.wait;                // 等前驱真正完成
```

两者都**不需要 `-rdc=true`**，sm90 可编译可运行。

### 5.1 重叠收益取决于「是否有空闲 SM」

实验用一对 producer/consumer（`src/pdl_launch.cu`）：producer 先写 output 再进入 tail（≈1.04 ms），
consumer 先做一段等长 prologue 再 `griddepcontrol.wait` 再读校验。40 次取最小值：

| 组 | producer 网格 | serial（无 PDL） | PDL（提前触发） | speedup | 正确性 |
|---|---|---|---|---|---|
| l 全格 | 132（=SM 数） | 2.028 ms | 1.971 ms | **1.03×** | fail=0 |
| 半格 | 66 | 2.028 ms | 1.042 ms | **1.95×** | fail=0 |

- **全格时几乎无收益**：producer 占满 132 个 SM，consumer 的 block 根本没有 SM 可调度，
  即使 PDL 已把 launch 提前，prologue 也只能等 producer 的 wave 退去才跑。
- **半格时接近 2×**：producer 只占 66 个 SM，其余空闲 SM 立刻跑 consumer 的 prologue，
  与 producer 的 tail **真正并发**。

> 这解释了 DeepGEMM 用 `grid = num_sms` 时 PDL 的定位：它不是把一对 kernel 整体并行，
> 而是**消掉 stream 串行化 + 与前一个 kernel 的 epilogue/退场重叠 prologue**；在 MoE 推理
> 那种「几十个 kernel 首尾相接」的链里，每一次省下的是发射延迟与前序尾部的边角时间。

### 5.2 触发器位置：不提前触发就没有 block 级重叠

同半格、PDL 打开，但 producer **不提前触发**（触发点落在 kernel 结束）：

```
PDL trigger early_ms=1.042   implicit_end_ms=2.027
```

隐式触发 ≈ 串行。也就是说：**PDL 要产生 block 级重叠，primary 必须显式
`cudaTriggerProgrammaticLaunchCompletion()`**；DeepGEMM 不调用它，故其 kernel 对之间的
重叠是「退场边界级」的。这与它把 `griddepcontrol.wait` 放在 prologue 之后是自洽的：
无论触发何时发生，`wait` 都保证读到的是完整的主 kernel 结果。

### 5.3 设备端同步的必要性：缺它就读到 stale

做一个「坏用法」producer：**提前触发，但数据在 tail 之后才写**（模拟「以为触发=数据已就绪」）。
半格、PDL、consumer 用极短 prologue，跑 100 轮（每轮换新值），统计读到的值不等于本轮值的元素数：

| consumer | 读错元素 / 总数 |
|---|---|
| 有 `cudaGridDependencySynchronize()` | **0** / 104857600 |
| 缺 `cudaGridDependencySynchronize()` | **1,689,600** / 104857600（≈1.6%） |

这是「`griddepcontrol.wait` 不是可有可无」的硬证据：没有它，consumer 会在 producer 数据写回之前
就读，产生静默错误。DeepGEMM 把 `wait` 写进**每一个** kernel，正是为了在 `set_pdl(true)` 时
无论前序是何种 kernel 都正确。

---

## 6. cluster / 动态共享内存（`CUlaunchAttribute` 的另外两项）

### 6.1 cluster

`CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION`（`kernel.hpp:141-147`）把 `grid` 里的相邻 CTA 打成 cluster；
device 侧用 `%cluster_ctarank` / `%cluster_nctarank` 取簇内 rank 与簇大小。H100 实测（grid 对齐到
cluster 的整数倍）：

```
CLUSTER cs=2 grid=132 launch_err=0 bad=0 sample=2,1002      ← 每簇 rank∈{0,1}，簇大小=2
CLUSTER cs=4 grid=132 launch_err=0 bad=0 sample=4,1004
CLUSTER nonportable cs=16 without_attr_err=912 (cluster misconfiguration)
                          set_attr_err=0  with_attr_err=0
```

- cs=2/4 的 `rank*1000+ncta` 与 host 期望**逐块 0 不符**；
- H100 便携 cluster 上限是 **8**；cs=16 不设
  `cudaFuncAttributeNonPortableClusterSizeAllowed` 时返回 **error 912**（cluster 配置错误），
  设了之后成功——对应 `kernel.hpp:124-131` 那段「只设不撤」的注释。

### 6.2 动态共享内存

H100 上 `k_dynsmem`（100 KiB 动态 smem，`src/pdl_launch.cu`）：

```
DYN maxDynamicSharedMemorySize_default = 49152                 ← 默认只有 48 KiB
DYN launch_without_attr err=1 (invalid argument)               ← 不设属性直接挂
DYN set_attr_err=0  maxDynamicSharedMemorySize_after=102400
    launch_err=0  readback=25599 expect=25599                  ← 设后可跑、结果对
```

这正是 `kernel.hpp:115-118` 必须 `cuFuncSetAttribute(MAX_DYNAMIC_SHARED_SIZE_BYTES, ...)` 的原因：
Mega MoE 每个 kernel 要 232448 B 动态 smem，不显式抬高上限根本发不出去。

---

## 7. H100 实测汇总（`harness/exp_pdl_launch.{sh,out.txt}`、`src/pdl_launch.{cu,_host.cpp}`）

真机：H100 80GB HBM3 / sm90 / 132 SM / driver 570.211.01 / CUDA 13.2。总判定 **ALL PASS**。

| 组 | 测什么 | 结果 |
|---|---|---|
| PTX | 两条 intrinsic 的编译产物 | `launch_dependents` / `griddepcontrol.wait`，无需 `-rdc` |
| PDL 全格 | 132+132，prologue 无空闲 SM | 2.028 → 1.971 ms（1.03×），fail=0 |
| PDL 半格 | 66+66，有空闲 SM | 2.028 → 1.042 ms（**1.95×**），fail=0 |
| 触发器 | 提前触发 vs 隐式（末尾） | early 1.042 ms vs implicit 2.027 ms |
| stale | 缺 `griddepcontrol.wait` | fail **1,689,600 / 104857600**（有则 0） |
| cluster | cs=2/4 rank/size | bad=0；sample 2,1002 / 4,1004 |
| cluster 非便携 | cs=16，无/有属性 | error 912 → 成功 |
| dyn smem | 100 KiB，无/有属性 | invalid argument → 成功、readback 正确 |
| LaunchOptions | default / override / set_pdl | 真实头文件 host 测试 **ALL PASS** |

> 引用：DeepGEMM `README.md:152`（`set_pdl`/`get_pdl`）；News `README.md:16`（2026.04.16 引入 PDL，
> 见 [PR #304](https://github.com/deepseek-ai/DeepGEMM/pull/304)）；仓库
> <https://github.com/deepseek-ai/DeepGEMM>。

---

## 8. 结论与边界

1. **发射层是「一个 driver 调用 + 至多 3 个属性」**：DeepGEMM 的每个 kernel 都走
   `cuLaunchKernelEx`（`kernel.hpp:92-170`），只可能带 cooperative / cluster / PDL 三个属性，
   且只有 PDL 由用户全局切换（`set_pdl`），另两个当前从不启用。
2. **PDL = host 属性 + device 同步成对使用**：host 侧
   `CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION` 允许后继提前发射；device 侧
   `griddepcontrol.wait`（24 处）保证读到前驱的完整结果。缺 device 侧会静默读 stale（实测 1.6%）。
3. **重叠收益有物理上界**：PDL 的 block 级重叠需要「primary 显式提前触发 + 有空闲 SM」；
   DeepGEMM 不显式触发、且 grid=num_sms，所以它的收益是「消除 stream 串行化 + 与前序尾部重叠
   prologue」，在长 kernel 链里累积。
4. **cluster / 动态 smem 是「必须显式设属性」的硬件开关**：cs>8 要 non-portable 属性（error 912 反例），
   >48 KiB dyn smem 要 `cuFuncSetAttribute`（invalid argument 反例）——两者都在 `Kernel::launch`
   里被处理。
5. **架构边界**：PDL（`griddepcontrol`）、cluster、动态 smem 在 **sm90 即支持**，本文全部在 H100 真机验证；
   与 SM100 无关，故不受「仅分析」限制。

> 配套实验：`code/mega-moe/src/pdl_launch.cu`、`src/pdl_launch_host.cpp`、
> `code/mega-moe/harness/exp_pdl_launch.{sh,out.txt}`。
