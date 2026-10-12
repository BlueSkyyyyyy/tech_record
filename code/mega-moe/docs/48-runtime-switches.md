# DeepGEMM 的全局运行时/编译开关面：`config.hpp` + `HeuristicsRuntime` + `Runtime`（H100 实测）

> 本文是「DeepSeek Mega MoE 前向 / 反向算子实现解析」专题的 P38 增量（第 48 篇）。
> 前面 47 篇把一个个 kernel 拆得很细，但有一层**每个 kernel 都踩、却从没单独讲过**的东西：
> DeepGEMM 顶部那组**全局开关**——`deep_gemm.set_num_sms / set_tc_util / set_pdl /
> use_deterministic_algorithms / set_ignore_compile_dims / set_block_size_multiple_of /
> set_mk_alignment_for_contiguous_layout`，以及它们背后的两个单例 `Runtime` / `HeuristicsRuntime`
> 和一整套 `DG_*` 环境变量。本篇把这层开关面拆透：每个开关**改的是什么状态、被哪段代码消费、
> 为什么这样设计**，并在 H100 上给出可观测的实测证据。
>
> 关键点：这一层**与 UMMA/TMA/WGMMA 完全无关**，全部是 host 侧状态与启发式输入/编译维/发射属性，
> 所以 H100（sm90）能真跑并逐项观测。参考仓库 `~/github/DeepGEMM`，核对 commit `057ca59`
> （v2.8.1）。本文行号均在该 commit 上核对过。

---

## 1. 为什么单独讲这层：一个「状态 singletons + pybind 面」

DeepGEMM 不是一个无状态函数库：它在进程内维护**两份可变全局状态**，所有 host 入口都读它：

- **`Runtime`**（`csrc/runtime/runtime.hpp:20-102`）：设备相关资源——`num_sms`、`tc_util`、
  cuBLASLt handle 与 workspace、SM 局部域张量。singleton 定义在 `:104`
  （`inline auto runtime = deep_jit::LazyInit<Runtime>(...)`）。
- **`HeuristicsRuntime`**（`csrc/jit_kernels/heuristics/runtime.hpp:10-65`）：选核/编译相关——
  `deterministic_algorithms`、`ignore_compile_dims`、`block_m/n_multiple_of`、
  `mk_alignment_for_contiguous_layout`。singleton 在 `:67`。

两者通过 `csrc/apis/config.hpp:8-48` 的 `register_apis` 直接 pybind 到 `deep_gemm.*`
（`csrc/python_api.cpp:31` 调用 `config::register_apis`），于是 Python 侧看起来是一组平铺的
`set_*`/`get_*`。本文第 2 节先给全景表，第 3~9 节逐个拆「状态→消费点→为什么」，第 10 节给实测。

---

## 2. 全景：七个开关与它们的真实消费点

| Python API | 状态（文件:行） | 关键消费点 | 效果 |
|---|---|---|---|
| `set_num_sms` / `get_num_sms` | `Runtime::num_sms`（`runtime.hpp:22,78-87`） | `is_cublaslt_available`（`:89-91`）、各 impl 填 `GemmDesc.num_sms`（如 `sm100_bf16_gemm.hpp:98`） | 限制 kernel 用几个 SM；改了会**禁用 cuBLASLt 回退** |
| `set_tc_util` / `get_tc_util` | `Runtime::tc_util`（`:22,93-100`） | 只有 SM100 BF16 主 kernel 把它编进模板（第 36 篇） | tensor-core 占空比节流；默认/越界回 100 |
| `set_pdl` / `get_pdl` | `deep_jit` `LaunchOptions.enable_pdl`（`config.hpp:27-32`） | 发射属性（`deep_jit/backend/cuda/kernel.hpp:148-151`） | 开启 `CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION` |
| `use_deterministic_algorithms` | `HeuristicsRuntime::deterministic_algorithms`（`runtime.hpp:14,20-26`） | `gemm.hpp:519`、`einsum.hpp:81,115`、`mega_gate.hpp:111`、`mega_mhc.hpp:15-17` | 关 cuBLASLt 回退 + 禁 split-K（batch invariance） |
| `set_ignore_compile_dims` | `HeuristicsRuntime::ignore_compile_dims`（`:14,28-34`） | `get_compiled_dim`（`runtime_utils.hpp:31-40`） | 编译维不写进 kernel 模板 → 一个 kernel 服务所有形状 |
| `set_block_size_multiple_of` | `block_m/n_multiple_of`（`:16-17,36-47`） | `sm90.hpp:40`、`sm100.hpp:55,70-72` | 强制选核时 block 大小是给定值的倍数 |
| `set_mk_alignment_for_contiguous_layout` | `mk_alignment...`（`:18,49-55`） | `sm90.hpp:33`、`sm100.hpp:40`、`gemm.hpp:336,449,690` | grouped 连续布局的 M/K 对齐（默认 128，SM100 理论 256） |

外加 `set_mk_alignment_for_contiguous_layout` 的「理论值」查询
`get_theoretical_mk_alignment_for_contiguous_layout`（`runtime.hpp:57-64`，pybind 在
`layout.hpp:139-141`）和 `config.hpp` 里的 `init` / `shutdown`（`:9-14`）。

---

## 3. `Runtime`：`num_sms` 与 `tc_util` 语义

`num_sms` 是一个「**上限**」而非「目标」：

```
:78  void set_num_sms(const int& new_num_sms) {
:79      DG_HOST_ASSERT(0 < new_num_sms and new_num_sms <= jit->device.get_num_sms());
:80      num_sms = new_num_sms;
:81  }
:83  int get_num_sms() {
:84      if (num_sms == 0) num_sms = jit->device.get_num_sms();
:85      return num_sms;
:86  }
```

`:84-85` 的懒初始化很关键：**默认值 `0` 表示「还没设过」**，第一次读时才取当前设备的 SM 数；
所以 `get_num_sms` 不是常量，而是「当前设备」。`set_num_sms` 在 `:79` 断言 `0 < k <= 设备数`，
`0`/负数/超过设备数都会被 `DG_HOST_ASSERT` 抛 `RuntimeError`。

`num_sms` 最隐蔽的副作用在 `is_cublaslt_available`（`:89-91`）：

```
:89  bool is_cublaslt_available() {
:90      return get_num_sms() == jit->device.get_num_sms();
:91  }
```

只要用户把 SM 数**改小一点**（比如为了给别的 kernel 让路），cuBLASLt 回退就整个失效——
因为 cuBLASLt 无法被限制 SM 数，DeepGEMM 出于「尊重用户 SM 预算」的考虑直接不复用它。
第 27 篇讲的「三与门」（非确定性 ∧ 未限制 SM 数 ∧ plain-store epilogue）里的第二项正是这里。

`tc_util` 的语义同样有个「0 即默认」的坑（`:93-100`）：

```
:93  void set_tc_util(const int& new_tc_util) {
:94      DG_HOST_ASSERT(0 <= new_tc_util and new_tc_util <= 100);
:95      tc_util = new_tc_util;
:96  }
:98  int get_tc_util() const { return tc_util == 0 ? 100 : tc_util; }
```

`set_tc_util(0)` 不是「关掉 tensor core」，而是在 `:99` 被解释回默认 `100`；越界（`>100`/`<0`）
在 `:94` 被拒。这个值最终只被 **SM100 BF16 主 kernel** 当作模板参数消费（第 36 篇：
`kTensorCoreUtilControl`，`sm100_bf16_gemm.cuh:35`），对 SM90 上的 bf16/fp8 完全无影响——这也是
第 36 篇实测过的事实。

---

## 4. `HeuristicsRuntime`：四个「选核/编译」开关

`HeuristicsRuntime` 的字段默认值在 `runtime.hpp:14-18`：

```
:14  bool ignore_compile_dims = false;
:15  bool deterministic_algorithms = false;
:16  int  block_m_multiple_of = 1;
:17  int  block_n_multiple_of = 1;
:18  int  mk_alignment_for_contiguous_layout = kLegacyMKAlignmentForContiguousLayout;  // =128 (:12)
```

它们都没进 JIT 缓存键（第 33 篇的五段式键里没有这些），而是**在每次调用时决定「生成哪个微源码/
选哪个 layout」**——所以改这些开关不需要清 JIT 缓存，只会让下次调用重新走一遍 host 启发式。

### 4.1 `deterministic_algorithms`：一次开关，三个落点

这个开关的名字容易让人以为只是「让 GEMM 可复现」，实际它在三个地方分别是不同含义：

1. **关掉 cuBLASLt 回退**（`gemm.hpp:519`、`einsum.hpp:81,115`）：

   ```
   :519  if (not heuristics_runtime->get_deterministic_algorithms() and runtime->is_cublaslt_available()
   :520      and epilogue_alpha.has_value()) {
   :521      cublaslt_gemm(...); return;
   :522  }
   ```

   cuBLASLt 的 split-K/归约顺序不保证确定，所以确定性模式下直接不走它，改用自研 kernel
   （自研的 split-K 只出现在显式 `atomicAdd` 路径，且分组 kernel 一般不用）。

2. **禁止 Mega Gate 的 split-K**（`mega_gate.hpp:111,126`）：注释写得很明白
   「deterministic mode is also batch invariant, so K is never split」（`:110`），
   于是 `num_split_k` 的上界从 `kNumMaxSplitK=8`（`:54`）退化成 `1`：

   ```
   :126  for (int num_split_k = 1; num_split_k <= (deterministic ? 1 : kNumMaxSplitK); num_split_k *= 2)
   ```

   **为什么 split-K 破坏 batch invariance**：同一条 token 的 K 被切成若干段、由不同 CTA 部分求和
   再用浮点加法合并；沿 K 做 split 会让「同一 token 在不同 batch 批大小下」落入不同的求和树，
   于是相同输入得到不同 bit 输出。禁掉 split 后 token 的归约顺序与 batch 无关。

3. **固定 Mega mHC 的 splits**（`mega_mhc.hpp:15-17`）：

   ```
   :15  if (heuristics_runtime->get_deterministic_algorithms())
   :16      return mhc_layout::kDefaultNumSplits;   // =16，layout/mega_mhc.cuh:36
   ```

   非确定性时按 SM 数与 M block 数动态取 splits（`:19-23`，上限 `kNumMaxSplits=64`），
   同样会破坏 batch invariance；确定性时钉死 16。

所以 `use_deterministic_algorithms(True)` 的完整语义是「**cuBLASLt 回退关闭 + 所有 split-K 归约
路径关闭**」，成本是可能损失一点并行度——这是个典型的「确定性 vs 吞吐」显式取舍。

### 4.2 `block_size_multiple_of`：约束候选集

`set_block_size_multiple_of` 接受 `int`（同时设 m/n）或 `(m,n)` 元组（`config.hpp:39-47`）。
它只影响**选核时候选 block 的步长**，不改 kernel 本身。SM90 与 SM100 各有一处：

- SM90（`sm90.hpp:40`）：`int step = std::lcm(16, get_block_n_multiple_of());`——N 候选步长
  从 16 抬到给定值的倍数。
- SM100（`sm100.hpp:55/70/72`）：
  ```
  :55  int step = std::lcm(16, get_block_m_multiple_of());   // swap-A/B 分支的 M 候选
  :70  if (16 % get_block_n_multiple_of() == 0) block_n_candidates.push_back(16);
  :72  int step = std::lcm(32, get_block_n_multiple_of());    // 非 swap 分支的 N 候选
  ```
  注意 swap 分支用 `block_m_multiple_of`、非 swap 分支用 `block_n_multiple_of`——因为 swap 后
  「block_m」实际是原 N 维。

**为什么留这个开关**：某些下游框架要求 GEMM 的 tile 边界与自己的内存/流水约定对齐（例如与
另一个 kernel 共享 smem 布局），这时强制 block 尺寸是给定值的倍数能避免额外的边界处理。

### 4.3 `mk_alignment_for_contiguous_layout`：grouped 布局的对齐

默认 `128`（`runtime.hpp:12`，名为 `kLegacyMKAlignmentForContiguousLayout`）。它被
**grouped 布局**消费：SM90 的 `MGroupedContiguous` 直接把它当唯一 block_m 候选
（`sm90.hpp:33`），SM100 的 m-grouped 同理（`sm100.hpp:40`）；host 侧在非连续轴的边界计算里也用它
（`gemm.hpp:336/449/690`）。理论值查询 `get_theoretical_mk_alignment_for_contiguous_layout`
（`runtime.hpp:57-64`）只看架构：

```
:58  if (jit->device.get_arch_major() != 10) return kLegacyMKAlignmentForContiguousLayout;  // 128
:63  return 256;  // SM100：UMMA_N=256 允许固定 256 对齐，对 MoE cast / grouped 更友好
```

注释（`:61-62`）说 SM100 之所以能上 256，是因为新支持的 `UMMA_N=256` 让 256 对齐「对 MoE cast、
M-grouped、K-grouped 更友好」，且**忽略了 `expected_m`**（小形状可能因此掉性能，官方建议这类
workload 直接用 Mega MoE）。这解释了第 41 篇发现的「枚举器会改写全局 mk-alignment」这一坑：
测试里每迭代一次就 `set_mk_alignment`，末值取决于枚举顺序。

### 4.4 `ignore_compile_dims`：「一核多形状」的开关

每个 host impl 会把 M/N/K 里的某几维当**模板参数**编进 kernel（默认 `compiled_dims="nk"`，
即 N/K 是编译期常量、M 是运行期）。`get_compiled_dim`（`runtime_utils.hpp:31-40`）统一决定：
某个维度是「用真实值」还是「用 0」：

```
:31  static int get_compiled_dim(const int& dim, const char& name, const std::string& compiled_dims) {
:32      if (heuristics_runtime->get_ignore_compile_dims()) return 0;   // ← 全部置 0
:35      for (const char& c: compiled_dims) if (name == c) return dim;
:39      return 0;
:40  }
```

`ignore_compile_dims=True` 时所有编译维都变 0——同一个 kernel 二进制于是能吃任意形状，
省掉「每个形状一份 cubin」的 JIT 编译开销，代价是丢掉按形状特化的常量折叠。它对**字节码层面**
的可见证据就是模板实参：默认 `sm90_bf16_gemm_impl<..., 0u, 4096u, 4096u, ...>`（M=0 被忽略、
N/K=4096），打开后变成 `<..., 0u, 0u, 0u, ...>`。第 46 篇讲的「`compiled_dims` 随入口变化」
正是这一机制的输入。

---

## 5. `set_pdl`：不是状态、是发射属性

`set_pdl` 改的是 `deep_jit` 的 `default_launch_options.enable_pdl`（`config.hpp:27-32`），
不是 DeepGEMM 自己的状态。它落到 kernel 发射属性上（`deep_jit/backend/cuda/kernel.hpp:148-151`）：

```
:148  if (*launch_options.enable_pdl) {
:149      ...
:150      attribute.id = CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION;
:151      attribute.value.programmaticStreamSerializationAllowed = 1;
```

开了之后，同一 stream 里的下一个 kernel 可以**在依赖满足前就启动**（Programmatic Dependent
Launch）；kernel 内用 `cudaGridDependencySynchronize()` 显式等待前驱（例如
`sm90_bf16_gemm.cuh:131`、`sm100_bf16_gemm.cuh:153`）。Mega MoE / MQA logits 系列正是靠它把
前一 kernel 的尾部和后一 kernel 的头部重叠起来。`get_pdl` 读回这个布尔（`:30-32`）。

---

## 6. 环境变量面：`DG_*` 的两级解析 + 打包时固化

DeepGEMM/DeepJIT 的运行时开关有一半走环境变量，实现是 `deep_jit::Env`（`deep_jit/utils/env.hpp:56-84`）：

```
:66  std::optional<T> get(const std::string_view suffix) const {
:67      if (const auto value = get_with_prefix<T>(prefix, suffix)) return value;
:69      return get_with_prefix<T>("DJ", suffix);
:70  }
```

`prefix` 由 DeepGEMM 在建 JIT 时给定——`csrc/runtime/jit.hpp:19` 传 `"DG"`，所以
`env.get("JIT_DEBUG")` 先找 `DG_JIT_DEBUG`，找不到退回全局 `DJ_JIT_DEBUG`（`DJ` 被保留为全局前缀，
`env.hpp:62`）。这解释了一个真实行为：**`DG_JIT_DEBUG=0` 能盖过 `DJ_JIT_DEBUG=1`**。

值解析在 `get_env`（`:28-54`）：布尔接受 `true/false/yes/no` 或整数（非零即真，`:35-45`），
整数用 `std::from_chars` 严格解析、部分/畸形直接抛（`:22-25`）。

DeepGEMM 代码里直接调 `deep_jit::get_env<int>("DG_PRINT_CONFIGS")`（全名，非 Env 前缀路径）的地方：

- `DG_PRINT_CONFIGS`：`heuristics/common.hpp:44`、`heuristics/mega_moe.hpp:252`、
  `mega_gate.hpp:177`、各 impl（`runtime_utils.hpp:147/191/236/283` 的 TMA 描述符打印等）。
  它把选中配置打印到 stdout（第 28 篇用它抓选核）。
- `DG_COMM_KERNEL_DEBUG`：`csrc/apis/mega_moe.hpp:281,388`（Mega MoE 通信段调试）。
- `DG_USE_PYTORCH_CUBLASLT_HANDLE` / `DG_USE_TEMP_CUBLASLT_WORKSPACE`：
  `runtime.hpp:44,48`（第 27 篇讲过：前者用 PyTorch 的 cuBLASLt handle 省 CPU 开销，
  后者每次调用新建 workspace 以便配合 compute-sanitizer）。

还有一类是**安装时固化**的：`setup.py:148-155` 把构建机上的
`DG_JIT_CACHE_DIR` / `DG_JIT_PRINT_COMPILER_COMMAND` / `DG_JIT_CPP_STANDARD` 写进生成的
`deep_gemm/envs.py`（`persistent_envs`），`deep_gemm/__init__.py:6-13` 在 import 时把它们
注入 `os.environ`（**仅当用户没设过**，「用户显式设置优先」）。

DeepJIT 自己那批 `DG_JIT_*`（`JIT_DEBUG`/`JIT_DUMP_ASM`/`JIT_DUMP_PTX`/`JIT_CHECK_NO_SPILLS`/
`JIT_CHECK_NO_LOCAL_MEMORY`/`JIT_WITH_LINEINFO`/`JIT_CPP_STANDARD`/`JIT_NVCC_COMPILER`……）
在 `deep_jit/backend/cuda/options.hpp:40-54` 被读；它们大多**进缓存键**（第 33 篇），
而 DeepGEMM 层的 `DG_PRINT_CONFIGS` 等**不进键**。

---

## 7. H100 实测：25 项逐条可观测

实验 `harness/exp_runtime_switches.py`（driver `harness/exp_runtime_switches.sh`，原始输出
`harness/exp_runtime_switches.out.txt`）。全部是 host 侧行为，H100（sm90）真机可跑，25/25 PASS。

**§1 `set_num_sms`**：默认 `get_num_sms()==132==设备 SM 数`；`0/-1/133` 三个越界输入全被拒；
`set_num_sms(64)` 生效。最直接的可见证据是把 `use_deterministic_algorithms(True)` 打开后
（强制走自研、否则默认是 `nvjet`）用 profiler 抓到的模板实参：`num_sms=64` 时模板里出现 `64u`、
恢复 `132` 后回到 `132u`——**用户设的 SM 数真的编进了 kernel**。

**§2 `tc_util`**：默认 100；`set(50)→50`；`set(0)→getter 100`（`:99`）；`101/-1` 被拒（`:94`）。

**§3 `pdl`**：默认 `False`；`set(True)/set(False)` 回读一致。

**§4 `mk_alignment`**：默认 128；`theoretical(None)==128`（非 arch10 恒 128，`:58`）；
`set(256)` 回读 256。

**§5 `block_size_multiple_of`**：`int` 与 `(m,n)` 两种入参都被接受。

**§6 `use_deterministic_algorithms`**：默认下 `bf16_gemm_nt` 走 cuBLASLt（profiler 抓到
`nvjet_sm90_tst_256x128_64x4_1x2_h_bz_coopA_TNT`）；打开后改成自研
`deep_gemm::sm90_bf16_gemm_impl`。`einsum("bhr,hdr->bhd")` 同样从 `nvjet` 切回
`sm90_bf16_gemm_impl`——**一次开关同时改 GEMM 与 einsum 的选路**，证实三个落点里的第一个。

**§7 `DG_PRINT_CONFIGS`**（子进程）：默认 `bf16_gemm_nt(4096³)` 选中
`Layout(block_m=256, block_n=128)`；`set_block_size_multiple_of((256,256))` 后变成
`block_m=128, block_n=256`（N 是 256 的倍数）；`set_num_sms(32)` 后打印的
`LaunchConfig(num_sms=32)`。**开关对选核的影响可见、可复现**。

**§8 `ignore_compile_dims`**：默认 compiled_dims=`nk` → 模板实参 `<M=0,N=4096,K=4096>`；
打开后 → `<0,0,0>`。**编译维开关在二进制层的直接证据**。

```
TOTAL: 25 PASS / 0 FAIL
```

---

## 8. 小结与前后篇关系

- 这层开关面是「**全局可变状态 → 每次调用的 host 启发式/编译维/发射属性**」的通道：
  `Runtime` 管设备资源（`num_sms`/`tc_util`/cuBLASLt），`HeuristicsRuntime` 管选核与编译
  （`deterministic`/`block_size_multiple_of`/`mk_alignment`/`ignore_compile_dims`），
  `deep_jit` 侧管发射（`set_pdl`）与编译（`DG_JIT_*`）。
- 三个最容易被忽视的语义：`num_sms` 改小会**顺带关掉 cuBLASLt 回退**（`:89-91`）；
  `tc_util==0` 当默认 100（`:99`）；`deterministic` 是「关回退 + 关 split-K」而非单纯 GEMM 标志。
- 与前面各篇的接口：第 27 篇的「三与门」第二项 = `is_cublaslt_available`；第 28/46 篇的选核与
  `compiled_dims` 的输入 = `block_size_multiple_of`/`ignore_compile_dims`；第 36 篇的 `set_tc_util`
  是本层唯一进模板的开关；第 33 篇的 JIT 缓存键与本层（除 `DG_JIT_*` 外）**互不影响**。
- 全部 25 项 H100 真机可观测，无需 SM100。

## 参考

- 源码：`csrc/apis/config.hpp`、`csrc/apis/layout.hpp`、`csrc/runtime/runtime.hpp`、
  `csrc/runtime/jit.hpp`、`csrc/jit_kernels/heuristics/{runtime,sm90,sm100,common,mega_gate}.hpp`、
  `csrc/jit_kernels/impls/runtime_utils.hpp`、`csrc/apis/{gemm,einsum,mega_mhc}.hpp`、`setup.py`；
  `third-party/deep_jit/include/deep_jit/{utils/env.hpp,backend/cuda/{options,kernel}.hpp}`。
- 线上：DeepGEMM GitHub <https://github.com/deepseek-ai/DeepGEMM>。
