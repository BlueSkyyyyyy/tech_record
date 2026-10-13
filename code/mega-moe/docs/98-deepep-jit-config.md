# DeepEP 的编译期配置面与 DeepJIT 接入：`compiled.cuh` + `init_jit` 的八个开关

> 本文对应路线图 **P88-1**，是「DeepSeek Mega MoE 前向 / 反向算子实现解析」的**第 98 篇**。
> 目标：把 DeepEP 里**「任何设备代码跑起来之前」的两层编译配置**讲透——
> ① `deep_ep/include/deep_ep/common/compiled.cuh`（编译期 ABI 常量：`topk_idx_t` 位宽、SF 打包单元、
> 三档对齐、通信上限），② `csrc/runtime/jit.hpp` 的 `init_jit`（把 DeepEP 的源码树、NCCL 根目录与
> 8 条 nvcc flag 交给 **DeepJIT**，并在**缓存键**里体现）。
>
> **可跑性**：两者都**不是 SM100 专属**，也不碰 UMMA/TMA/NCCL 设备 API——是完全 host / 编译期的地基。
> `compiled.cuh` 用 nvcc 直接 include 真实头文件编译运行；`init_jit` 的 `Config` 与 8 条 flag 用
> deep_jit 真实的 `Config` / FNV1a 哈希在 H100(sm90) 上逐条验证。**H100 真机 48 PASS / 0 FAIL**。
>
> **与专题其余篇目的关系**：第 33 篇拆了 DeepGEMM 的 DeepJIT 接入（`csrc/runtime/jit.hpp`）、
> 第 47 篇拆了 DeepJIT 的 post-hook、第 86 篇拆了 DeepEP 的 Python 引导层（`__init__.py` 的
> `init_jit` **调用点**）。本篇补的是**同一个 `init_jit` 的 C++ 实现**以及它引出的 `compiled.cuh`——
> 也就是第 86 篇 §11 反复当坐标用、却从没系统读过的两个文件。
>
> 源码坐标（`~/github/DeepEP`，本地 commit `93eb6eb`、`deep_ep.__version__ == 2.5.0`；
> DeepJIT `2efdab4`）：
> - 编译期常量：`deep_ep/include/deep_ep/common/compiled.cuh`（84 行）
> - JIT 接入：`csrc/runtime/jit.hpp`（54 行）、`csrc/utils/system.hpp`（31 行）
> - host `_C` 编译面：`setup.py`（`persistent_env_names:14-15`、`cxx_flags:92-98`、`generate_default_envs:63-73`）
> - 调用点：`deep_ep/__init__.py:8-16`（persistent_envs）、`:44-51`（`init_jit`）、`:19-41`（`check_nccl_so`）
> - Python 绑定：`csrc/python_api.cpp:19-51`
> - 实验：`src/deepep_compiled_abi.cu` + `harness/deepep_jit_config_probe.cpp` +
>   `harness/exp_deepep_jit_config.{sh,out.txt}`（H100）

---

## 1. 两个编译面：host `_C` 与 device JIT

DeepEP 和 DeepGEMM 一样是「装的时候只编译一个很小的 host 扩展，设备 kernel 首次调用才现编」。
但它的编译被**切成两半**，两半必须看到**同一个** `topk_idx_t`：

| 编译面 | 谁编 | 何时编 | 关键开关 | 落点 |
|---|---|---|---|---|
| **host `_C`** | `setup.py`（setuptools + torch C++ 扩展） | `pip install` | cxx `-DEP_NUM_TOPK_IDX_BITS=<bits>` | `setup.py:96-98` |
| **device JIT** | DeepJIT `Runtime<CUDA>`（运行时 nvcc） | 首次发射某个 kernel | nvcc `-DEP_NUM_TOPK_IDX_BITS=<bits>` + 7 条 | `csrc/runtime/jit.hpp:38` |

也就是说 `EP_NUM_TOPK_IDX_BITS` 这条「top-k 索引是 32 位还是 64 位」的宏被**同时**塞进 host
pybind（`csrc/python_api.cpp:22` 的 `m.attr("topk_idx_t")` 由它推出）和每个 device kernel 的
nvcc 命令行。若两边不一致，Python 侧看到的 `topk_idx_t` 与 kernel 实际解释的 `topk_idx` 就会错位。
`compiled.cuh:45-47` 给出缺省值 **64**（`#ifndef ... #define ... 64`），所以两边都不设时是一致的。

host 面还有一条并行的「persistent env」通道：`setup.py:14-15` 把
`EP_JIT_CACHE_DIR / EP_JIT_PRINT_COMPILER_COMMAND / EP_JIT_CPP_STANDARD / EP_NUM_TOPK_IDX_BITS /
EP_NCCL_ROOT_DIR / EP_DEFAULT_RDMA_SL / EP_OVERRIDE_RDMA_SL` 在**安装时**固化进
`deep_ep/envs.py`（`setup.py:63-73` 的 `generate_default_envs`），运行时
`__init__.py:8-16` 只在「环境里没设」时补上——这解释了为什么第 86 篇实测里 `persistent_envs`
里能看到这些键。这些键里 `EP_JIT_*` 由 DeepJIT 的 `Env`（`Config.env_prefix == "EP"`）消费。

---

## 2. `compiled.cuh`：所有 kernel 共享的编译期 ABI 常量

这个头只有 84 行，却被几乎每个文件 include（`csrc/python_api.cpp:4`、`jit.hpp:7`、所有 kernel）。
它做五件事。

### 2.1 抹平 torch 的 half/bf16 运算符限制（`:10-25`）

torch 的编译选项会定义 `__CUDA_NO_HALF_CONVERSIONS__` / `__CUDA_NO_HALF_OPERATORS__` /
`__CUDA_NO_HALF2_OPERATORS__` / `__CUDA_NO_BFLOAT16_CONVERSIONS__` / `__CUDA_NO_BFLOAT162_OPERATORS__`，
效果是**禁用** `__half`/`__nv_bfloat16` 的隐式转换与算术运算符。`compiled.cuh` 在 torch 扩展里
只能在这些宏生效的前提下工作，于是逐个 `#undef` 把它们**改回可用**。这是 DeepEP 设备代码能写
`bf16 + bf16` 的前提，也是为什么这个头必须在任何 CUDA 类型运算之前 include。

### 2.2 256-bit LD/ST 兼容类型（`:33-43`）

```cpp
#if defined(CUDART_VERSION) and CUDART_VERSION >= 13000
using longlong4_t = longlong4_32a;          // CUDA 13 的内置 32B 对齐向量
#define make_longlong4_t make_longlong4_32a
#else
struct alignas(32) longlong4_t { long long x, y, z, w; };   // 自己造一个 32B 对齐的
__device__ __forceinline__ longlong4_t make_longlong4_t(...) { ... }
#endif
```

CUDART ≥ 13.0 提供 `longlong4_32a`（32 字节对齐的 4×long long，可映射到 256-bit `LDG.256`/`STG.256`）；
旧版没有，就手写一个 `alignas(32)` 的等价结构。DeepEP 的拷贝 epilogue / bucket copy-engine 用
`longlong4_t` 做**一次 32 字节的向量化搬运**（`copy_engine` 一类的宽拷贝），这个头保证跨
CUDA 版本都是 32B 对齐。本机 CUDART 13020 → 走第一支，实测 `sizeof=32 alignof=32`。

### 2.3 `topk_idx_t` 与 `int_with_bits`（`:45-57`）

```cpp
#ifndef EP_NUM_TOPK_IDX_BITS
#define EP_NUM_TOPK_IDX_BITS 64
#endif
template <int kNumBits> struct int_with_bits;
template <> struct int_with_bits<8>  { using type = int8_t;  };
template <> struct int_with_bits<16> { using type = int16_t; };
template <> struct int_with_bits<32> { using type = int32_t; };
template <> struct int_with_bits<64> { using type = int64_t; };
using topk_idx_t = int_with_bits<EP_NUM_TOPK_IDX_BITS>::type;
```

`topk_idx_t` 是 dispatch / combine / MQA 等所有「token→专家」索引的类型。32 位省一半显存与带宽，
但 `num_experts` 超过 2³¹（或与 `torch.int64` 混用）时不够；默认 64 位更安全。`int_with_bits`
只特化了 8/16/32/64——**给别的位数就直接编译错**，把「只允许这两个值」变成类型系统约束。
实验里我们用 `-DEP_NUM_TOPK_IDX_BITS=32` 重新编译同一份源，验证 `topk_idx_t` 从 `int64_t` 切成
`int32_t`（同一份 `static_assert`，两个编译单元各过一次）。

### 2.4 SF 打包单元 `sf_pack_t` 与 `kNumAlignedSFPacks`（`:59-64`）

```cpp
union sf_pack_t { float fp32; int ue8m0x4; };
constexpr int kNumAlignedSFPacks = 16 / sizeof(sf_pack_t);   // == 4
```

DeepEP 的 FP8 激活带一份 **block-scale**：每个 block 一个 UE8M0 指数。`sf_pack_t` 是一个 4 字节
union——要么是一个 fp32 scale、要么是**打包好的 4 个 UE8M0 指数**（正是第 10/25/26 篇的
`pack_fp32_into_ue8m0` 的输出形态）。这意味着 DeepEP 的 SF 张量**元素类型是 4 字节**，
`csrc/buffers/ep.hpp:355` 用 `EP_HOST_ASSERT(sf->element_size() == sizeof(sf_pack_t))` 硬卡，
而 `csrc/buffers/ep.hpp:693` / `engram.hpp:185` 用 `math::align(num_tokens, kNumAlignedSFPacks)`
把 SF 的 token 维**按 4 对齐**——因为 16 字节 TMA/向量化搬运一次要 4 个 pack，
`kNumAlignedSFPacks == 16/sizeof == 4` 就是这个「16 字节一搬」的常量。

### 2.5 对齐三档与通信上限（`:66-82`）

```cpp
constexpr int kNumTMAAlignmentBytes        = 32;              // TMA 描述符
constexpr int kNumRDMAAlignmentBytes       = 64;              // RDMA 注册
constexpr int kNumAllocationAlignmentBytes = 2 * 1024 * 1024; // 对称 buffer 分配
static_assert(kNumRDMAAlignmentBytes % kNumTMAAlignmentBytes == 0);
static_assert(kNumAllocationAlignmentBytes % kNumRDMAAlignmentBytes == 0);
```

三档是对称内存的**层层加严**：TMA 要 32B、RDMA 注册要 64B、VMM 分配（`ncclCommWindowRegister`
或 CUDA VMM）要 2 MiB。两条 `static_assert` 保证「严的能被松的整除」，于是**一块 2 MiB 对齐的
分配天然同时满足 32/64B**，不用在运行期再补对齐。`csrc/buffers/bucket.hpp:78` 就靠这条把
`num_storage_bytes % kNumAllocationAlignmentBytes == 0` 当契约。

通信上限（`:73-82`）是「对称显存里信号/计数数组的静态上界」：`kNumMaxRanks = 1024`、
`kNumMaxContexts = 8`、`kNumMaxQPs = 1024`、`kDefaultQPDepth = 1024`、
`kNumMaxSignalBytes = 16 MiB`。它们是**编译期数组维度**——`layout/ep/workspace.cuh:15-20`
把 `notify_reduction[kNumMaxRanks + kNumMaxExperts]` 等做成定长数组，`:44` 再断言
`num_ranks <= kNumMaxRanks`；`layout/bucket/workspace.cuh:21` 断言
`sizeof(BucketSignals) * kNumMaxContexts <= kNumMaxSignalBytes`、`:24` 用这个维度放
`signals[kNumMaxContexts]`。`kNumMaxQPs` 则决定每个 RDMA rank 分多少 QP
（`pp.hpp:55` / `bucket.hpp:86`：`kNumMaxQPs / num_rdma_ranks`），`kDefaultQPDepth` 是
GIN 队列深度缺省（`context.cpp:137`），`context.cpp:95` 断言 `num_allocated_qps <= kNumMaxQPs`。
也就是说：这些「上限」不是可调的运行时参数，而是**写死在 ABI 里的数组尺寸**，改它们要重编全部。

---

## 3. `init_jit`：Config 五参与 8 条 nvcc flag

`csrc/runtime/jit.hpp` 是第 86 篇说的 `init_jit` 的**实现**（Python 侧 `__init__.py:44-51` 只是
把 `library_root` 与 `find_nccl_root()` 传进来）。

### 3.1 全局单例与 Config 五参（`:15-27`）

```cpp
inline deep_jit::LazyInit<deep_jit::Runtime<deep_jit::CUDA>> jit(nullptr);  // :15

inline void init_jit(const std::string& library_root_path, const std::string& nccl_root_path) {
    const auto library_root       = std::filesystem::absolute(library_root_path);
    const auto library_include_dir = library_root / "include";
    const auto nccl_include_dir   = std::filesystem::absolute(
        std::filesystem::path(nccl_root_path) / "include");
    const auto config = deep_jit::Config(
        library_root,          // ① python_library_root
        "EP",                  // ② env_prefix
        {},                    // ③ extra_signature（空！）
        {library_include_dir}, // ④ include_dirs
        {"deep_ep/"});         // ⑤ include_prefixes
```

五个参数（`deep_jit/runtime/config.hpp:12-30`）的含义：

- **① `python_library_root`** = Python 包目录（`.../deep_ep`）。DeepJIT 用它拼 `post_hook`
  的路径（`config.hpp:44` 的 `get_python_path`）。构造时断言**必须是绝对路径且非空**（`:32-36`）。
- **② `env_prefix = "EP"`** = DeepJIT `Env` 读环境变量的前缀（`EP_JIT_CACHE_DIR` 等的来源）。
  构造时断言非空、且**不许叫 `"DJ"`**（`:40-41`，`"DJ"` 是 DeepJIT 全局保留）。
- **③ `extra_signature = {}`** = 空字符串。这是与 DeepGEMM 最显眼的不同（见 §5）。
- **④ `include_dirs = {<deep_ep>/include}`** = nvcc 的 `-I` 来源（DeepEP 的源码树根）。
- **⑤ `include_prefixes = {"deep_ep/"}`** = **只跟踪** `#include "deep_ep/..."` 这些头的内容哈希
  （见 §4）。

### 3.2 8 条 flag（`:32-44`）

`Runtime` 构造后往 `default_compiler_options.nvcc_flags` 追加（注意是**追加进缺省选项**，
而非覆盖）：

```cpp
nvcc_flags.emplace_back("--diag-suppress=39,161,174,177,186,940,3012");
nvcc_flags.emplace_back("--compiler-options=-Wno-deprecated-declarations,-Wno-abi");
nvcc_flags.emplace_back("-I" + nccl_include_dir.string());
nvcc_flags.emplace_back("-DEP_NUM_TOPK_IDX_BITS=" + std::to_string(EP_NUM_TOPK_IDX_BITS));
nvcc_flags.emplace_back("-DNCCL_GIN_GDAKI_ENABLE=1");
nvcc_flags.emplace_back("-DNCCL_GIN_PROXY_ENABLE=0");
nvcc_flags.emplace_back("-DNCCL_GIN_GPI_ENABLE=0");
nvcc_flags.emplace_back("-DNCCL_GIN_EFA_GDA_ENABLE=0");
if (get_env<int>("EP_GIN_GDAKI_DEBUG", 0))
    nvcc_flags.emplace_back("-DNCCL_DEVICE_GIN_GDAKI_ENABLE_DEBUG=1");
```

逐条「为什么」：

1. `--diag-suppress=39,161,174,177,186,940,3012`：屏蔽 nvcc 对「声明使用但未定义」「`__host__`
   里用了 device 符号」等类模板实例化噪声（DeepGEMM 版只屏蔽到 `940`；DeepEP 多屏蔽
   `3012`，因为它的 `NCCLGin`/layout 模板更复杂）。
2. `--compiler-options=-Wno-deprecated-declarations,-Wno-abi`：宿主编译器静音（与 DeepGEMM 同）。
3. **`-I<nccl_include_dir>`**：DeepEP 的 device kernel 直接 `#include <nccl_device.h>` 调
   **NCCL 设备 API（GIN）**，所以必须把 NCCL 头加进 nvcc 搜索路径。**关键**：它进的是
   `nvcc_flags`，而 `nvcc_flags` 会进**缓存键**（§4），所以这条路径不是「随便指哪都行」。
4. `-DEP_NUM_TOPK_IDX_BITS=<bits>`：把 host 面同一个宏透传到每个 device kernel（§1）。
5. `-DNCCL_GIN_GDAKI_ENABLE=1`：打开 NCCL 的 **GDAKI**（GPU-Direct Async Kernel-Initiated）
   后端——DeepEP 的 RDMA 走的就是它（`README:334` 的 GIN）。
6. `-DNCCL_GIN_PROXY_ENABLE=0` / `-DNCCL_GIN_GPI_ENABLE=0` / `-DNCCL_GIN_EFA_GDA_ENABLE=0`：
   **显式关掉** GIN 的其它三条后端（proxy / GPI / EFA-GDA），保证只留 GDAKI，避免选到
   未测/不匹配的路径。
7. 条件 flag `-DNCCL_DEVICE_GIN_GDAKI_ENABLE_DEBUG=1`：由 `get_env<int>("EP_GIN_GDAKI_DEBUG", 0)`
   守卫（`system.hpp:12-29` 的 `get_env`），打开 GDAKI 设备侧调试。

这 8 条（+1 条件）全部走 `Runtime::default_compiler_options`，对**所有** DeepEP kernel 生效。

### 3.3 `get_env`：`system.hpp` 的 22 行（`:12-29`）

```cpp
template <typename dtype_t>
static dtype_t get_env(const std::string& name, const dtype_t& default_value = dtype_t()) {
    const auto c_str = std::getenv(name.c_str());
    if (c_str == nullptr) return default_value;
    if constexpr (std::is_same_v<dtype_t, std::string>) return std::string(c_str);
    else if constexpr (std::is_same_v<dtype_t, int>) { int value; std::sscanf(c_str, "%d", &value); return value; }
    else EP_HOST_ASSERT(false and "Unexpected type");
}
```

三个语义细节值得记：

- **只支持 `std::string` 与 `int`**；传别的类型（如 `bool`）走 `EP_HOST_ASSERT(false)` 抛
  `EPException`（实测消息 `system.hpp:27`）。这避免 `get_env<bool>("...")` 这类「想当然」调用。
- **int 用 `sscanf("%d")`**：`"123abc"` 会**取到 123**（不报错），所以环境变量写成 `"1x"` 也算 1。
- **未设时返回 `default_value`**，而不是 0/空串——所以 `get_env<int>("EP_AVOID_RECORD_STREAM", 0)`
  （`csrc/buffers/bucket.hpp:50`）这种「默认关」的开关是天然的。

### 3.4 注册进 `_C`（`:49-52`）

`register_apis` 调 `deep_jit::register_python_api(module, jit)`（暴露 `get_jit`）并 `m.def("init_jit", ...)`；
`csrc/python_api.cpp:29` 再调 `deep_ep::register_apis(m)`。于是 Python 侧可以 `_C.get_jit()`
拿到运行时做预热。

---

## 4. 缓存键：为什么 NCCL include 进键、DeepEP 自身 include 不进

`jit.hpp:32-33` 有一句关键注释（原文）：

> Keep the NCCL include path in compiler options so it participates in the cache key.
> DeepEP's own include path stays path-independent because its contents are tracked by the parser.

要理解它，得看 DeepJIT 的键怎么算。`deep_jit/runtime/runtime.hpp:92-97`：

```cpp
std::string cache_key(const std::string& source, const CompilerOptions& options) {
    auto hash = hash_base;                       // ① extra_signature + ② 编译器版本
    options.update_hash(hash);                   // ③ 编译选项（含 nvcc_flags）
    return hash.update(options.get_post_hook_hash(config))  // ④ post hook
               .update(parser.parse_into_hash(source))       // ⑤ 源码 + 被跟踪的头内容
               .get_hex_digest();
}
```

`hash_base` 在构造时只吃了两段（`runtime.hpp:52-53`）：`config.extra_signature`（DeepEP 传空）
与 `backend.compiler_info.get_hash()`（nvcc 版本）。**注意这里没有 `include_dirs` 本身**。

第 ③ 段 `options.update_hash` 的实际动作是 `hasher.update(str::join(get_flags()))`
（`backend/cuda/options.hpp:136-138`），而 `get_flags()` 会把 `nvcc_flags` **逐条拼进去**
（`options.hpp:129-133`）。于是：

- **NCCL include 路径在 `nvcc_flags` 里 → 进第 ③ 段 → 进缓存键。** 换一个 NCCL 根目录，
  缓存键就变（我们实测 `/opt/nccl-A/include` → `7b47f9c3…`、`/opt/nccl-B/include` → `1a00ac53…`）。
  这是**故意的**：第 3/6 条 GIN 开关必须匹配「编译 kernel 时看到的 `nccl_device.h` 版本」，
  不同 NCCL 头编出的 kernel 不能互相复用。
- **DeepEP 自身的 include 路径不进 `nvcc_flags`**，所以不进第 ③ 段——它进的是第 ④/⑤ 段：
  `parser.parse_into_hash(source)` 会沿 `include_prefixes = {"deep_ep/"}` 递归读取
  `#include "deep_ep/..."` 的**文件内容**并哈希。因为哈希的是**内容**不是路径，所以把 DeepEP
  装到哪个目录**都不影响键**（路径无关），但只要改动任何一个 `deep_ep/` 下的头，键立刻变。

一句话：**NCCL 头按「路径」进键（因为版本/内容不由 DeepEP 维护，路径是唯一可表达的信息），
DeepEP 自己的头按「内容」进键（因为它就在 `library_root/include` 下，parser 能读到）。**

---

## 5. 与 DeepGEMM 的 `init_jit` 对照

| | DeepGEMM `csrc/runtime/jit.hpp:15-33` | DeepEP `csrc/runtime/jit.hpp:17-47` |
|---|---|---|
| `env_prefix` | `"DG"` | `"EP"` |
| `extra_signature` | `"cutlass-" + CUTLASS_VERSION` | **`{}`（空）** |
| `include_dirs` | `{<deep_gemm>/include}` | `{<deep_ep>/include}` |
| `include_prefixes` | `{"deep_gemm/"}` | `{"deep_ep/"}` |
| `nvcc_flags`（无条件） | 2 条（diag-suppress…940 / compiler-options） | 8 条（多 `-I<nccl>`、`-DEP_NUM_TOPK_IDX_BITS`、4 条 GIN、…） |
| 条件 flag | 无 | `EP_GIN_GDAKI_DEBUG` → GDAKI debug |
| 外部依赖头 | cutlass（**编进 extra_signature**） | NCCL（**编进 nvcc_flags 的 `-I`**） |

两处差异都是「外部依赖怎么进缓存键」的选择：DeepGEMM 直接把 **CUTLASS 版本号**当
`extra_signature` 编进第 ① 段；DeepEP 的 CUTLASS 依赖不强，但它依赖 **NCCL 设备头**，于是改成把
`-I<nccl>` 塞进 `nvcc_flags`（第 ③ 段）。两者都保证「换了外部依赖头 → 缓存不复用」。

---

## 6. H100(sm90) 实测

实验 `harness/exp_deepep_jit_config.{sh,out.txt}`（真机 H100 80GB、CUDA 13.2、g++ 13.3；
DeepEP `93eb6eb` / DeepGEMM `057ca59` / DeepJIT `2efdab4`）。**总判定 48 PASS / 0 FAIL**。

- **Part 1（`src/deepep_compiled_abi.cu`，nvcc 直接 include 真实 `compiled.cuh`）**：默认
  `EP_NUM_TOPK_IDX_BITS=64` → `topk_idx_t == int64_t`（15 PASS）；用
  `-DEP_NUM_TOPK_IDX_BITS=32` 重编同一份源 → `topk_idx_t == int32_t`（15 PASS）。两类共验证：
  `int_with_bits` 四档映射（8/16/32/64 → 1/2/4/8 字节）、`sf_pack_t == 4B` 且
  `kNumAlignedSFPacks == 4`、`longlong4_t == 32B/32-align`（本机 `CUDART_VERSION=13020` 走
  `longlong4_32a` 支）、对齐三档 32/64/2MiB 及两条整除、通信上限（ranks 1024、ctx 8、QPs 1024、
  QPDepth 1024、signal 16 MiB）。另外 `1.0f` 按 `sf_pack_t` 位重解释得 `0x3f800000`（坐实
  union 是「按位重解释」而非数值转换）。
- **Part 2（`harness/deepep_jit_config_probe.cpp`，g++ + deep_jit 真实头，16 PASS）**：
  `get_env` 的四条正常路径（unset→默认 7、int→42、`"123abc"`→123、string→`"hello world"`）
  与「`get_env<bool>` 抛 `EPException`」（实测消息带真实行号 `system.hpp:27`）；`Config` 五参
  构造 + `to_json()` 逐字段（`extra_signature` 为空、`include_prefixes=["deep_ep/"]`）+
  `get_python_path` 相对拼接；三条负样本（相对根 / 空前缀 / 保留字 `"DJ"`）全被 `DJ_HOST_ASSERT`
  拦下；8 条 flag 复刻 + `EP_GIN_GDAKI_DEBUG` 打开后 9 条；用 deep_jit 真实 FNV1a 算 flag 哈希：
  `nccl-A 7b47f9c349771301c78afe4b96fb8a82`、`nccl-B 1a00ac53b21d91388fa541a06d5136d1`、
  `debug 63643454f9f46bbbacf44068468be36c`——三者互不相等，坐实「NCCL include 路径 + debug 开关
  都进缓存键」，且 flags 串里**不含** DeepEP 自身 include 路径。
- **Part 3（Python 解析真实 `jit.hpp`，2 PASS）**：正则抽出全部 `nvcc_flags.emplace_back("...")`
  字面量共 9 条（7 常量 + 2 动态 `-I` / `-DEP_NUM_TOPK_IDX_BITS=`），与期望逐字一致；并断言条件
  flag 由 `if (get_env<int>("EP_GIN_GDAKI_DEBUG", 0))` 守卫。这条把 Part 2 里「人肉复刻」的
  flag 清单锚回**真实源码**。

环境坐标（文件 sha256）：`compiled.cuh 5adc25fa…`、`jit.hpp 582d04c1…`、`system.hpp 6bd24a20…`。

---

## 7. 小结

- DeepEP 的编译配置被切成 **host `_C`（setup.py）** 与 **device JIT（`init_jit`）** 两面，
  `EP_NUM_TOPK_IDX_BITS` 是横跨两面、必须一致的那个宏。
- `compiled.cuh` 是**所有 kernel 共享的编译期 ABI**：`topk_idx_t` 位宽、`sf_pack_t` 4 字节
  UE8M0 打包单元、32/64/2MiB 三档对齐、以及写死成数组维度的通信上限。
- `init_jit` 的 `Config` 五参与 8 条 nvcc flag 决定了 DeepJIT 怎么编、怎么缓存 DeepEP 的 kernel；
  其中 **NCCL include 路径进缓存键**（因为 `nvcc_flags` 进第 ③ 段哈希），**DeepEP 自身头按内容
  进键**（`include_prefixes={"deep_ep/"}` 的 parser），这就是它与 DeepGEMM（用
  `extra_signature="cutlass-<ver>"`）的对照点。
- 这些层**完全 host / 编译期、架构无关**，H100 真机可逐条验证（48 PASS / 0 FAIL）；真正需要
  NCCL 设备 API 的 kernel 路径仍是「仅分析」（见 ROADMAP「阻塞」）。
