# 33. DeepGEMM 的 JIT 运行时：DeepJIT 的编译 / 缓存 / 加载 / 发射管线

> 本篇补上前面 32 篇里**一直当作黑盒**的那一层：DeepGEMM 为什么能「`pip install` 时**不编译任何 CUDA**，
> 首次调用某个形状时才现编 kernel」，以及这个现编过程怎么保证**跨进程、跨节点、跨用户复用**编译产物。
> 这一层是 DeepGEMM 独立的 **DeepJIT** 子模块（`third-party/deep_jit`，commit `2efdab4`，header-only C++20），
> DeepGEMM 通过 `csrc/runtime/jit.hpp` 把它接进来。
>
> **架构无关**：编译 / 缓存 / 加载 / 发射全是主机侧逻辑，**H100 上真机可跑**（本篇所有数字都是 H100 实测）。
> 行号核对：DeepGEMM `057ca59` + DeepJIT `2efdab4`。
> 实验：`harness/exp_deepjit.{py,sh,out.txt}`。

本篇和其它篇的关系：第 18 篇讲了「选哪个 block config」，第 27 篇讲了「什么时候不用自家 kernel」，
第 21 篇讲了「用自家 kernel 时尾部干什么」。它们讨论的都是**生成出来的 device 代码**。
本篇讲**这份代码是怎么被拼出来、编译、找到、加载、发射**的——即 `get_best_config` 选出的 config
如何变成一段源码字符串，源码字符串如何变成 `cubin`，`cubin` 如何被缓存命中，最后如何 `cuLaunchKernelEx`。

---

## 1. DeepJIT 是什么，DeepGEMM 怎么接进来

DeepJIT 的定位（`third-party/deep_jit/README.md`）：一个 header-only 的运行时，给 C++/Python 扩展作者
提供「**运行时编译 kernel 源码 → 缓存二进制 → 加载到设备 → 按后端参数发射**」的统一接口。
它同时支持 CUDA 与华为昇腾（Ascend）两个后端，共享：配置、源码/头文件哈希、内存缓存、磁盘缓存、惰性初始化。

DeepGEMM 的接入点只有一处——`csrc/runtime/jit.hpp:13-33`：

```cpp
inline deep_jit::LazyInit<deep_jit::Runtime<deep_jit::CUDA>> jit(nullptr);   // :13

inline void init_jit(const std::string& library_root_path) {                 // :15
    const auto library_root = std::filesystem::absolute(library_root_path);
    const auto include_dir = library_root / "include";
    const auto config = deep_jit::Config(
        library_root,
        "DG",                                                                // 环境变量前缀
        "cutlass-" + std::to_string(CUTLASS_VERSION),                        // extra_signature
        {include_dir},                                                       // include 目录
        {"deep_gemm/"});                                                     // 只跟踪这些前缀的 include

    jit = deep_jit::LazyInit<...>([config] {
        auto runtime = std::make_shared<...>(config);
        runtime->default_compiler_options.nvcc_flags->
            emplace_back("--diag-suppress=39,161,174,177,186,940");          // :29
        runtime->default_compiler_options.nvcc_flags->
            emplace_back("--compiler-options=-Wno-deprecated-declarations,-Wno-abi");  // :31
        return runtime;
    });
}
```

四个关键约定：

1. **`library_root` = Python 包目录**：`deep_gemm/__init__.py:122` 在 import 时执行
   `_C.init(os.path.dirname(os.path.abspath(__file__)))`，即把 `.../site-packages/deep_gemm`（或
   editable 安装下的仓库目录）作为根。include 目录就是它下面的 `include/`，里面放着
   `deep_gemm/` 这个源码前缀树（第 8 篇讲过 editable 安装要手动 `cp` cute/cutlass 进去，正是为了这里的
   `--include-path`）。
2. **`extra_signature = "cutlass-<ver>"`**：CUTLASS 版本是源码行为的一部分（第 29/32 篇的 BF16/FP8 kernel
   大量直接 include `cute/`、`cutlass/`），所以把它编进缓存键。本机 CUTLASS 4.2.1 →
   `CUTLASS_VERSION = 4*100+2*10+1 = 421` → `"cutlass-421"`（实测 `meta.json` 里就是这个值）。
3. **`include_prefixes = {"deep_gemm/"}`**：缓存键只递归追踪 `#include <deep_gemm/...>`。
   系统库、cute/cutlass 头**不进**哈希——它们的变化由 `extra_signature`（cutlass 版本）间接覆盖。
4. **环境变量前缀 `"DG"`**：所有 `DG_JIT_*` 变量都映射到 DeepJIT 的 `JIT_*`，
   并可回退到全局 `DJ_JIT_*`（见 §7）。

`jit` 本身是 `LazyInit`（`lazy.hpp:15-38`）：工厂只在**第一次解引用**时执行，所以
import `deep_gemm` 不会触发设备探测、nvcc 定位、缓存目录解析——这就是「惰性初始化」。
`csrc/python_api.cpp:31` 用 `deep_jit::register_python_api(m, deep_gemm::jit)` 把它以
`get_jit()` 暴露给 Python（`python_api.hpp:12-20`），供用户做 warmup；`shutdown()` 则把
`runtime`/`jit` 复位成 `nullptr` 的 LazyInit（`csrc/apis/config.hpp:12-14`）。

---

## 2. Runtime：五段式缓存键 + 内存/磁盘两级缓存

`deep_jit::Runtime<Backend>`（`runtime.hpp:19-98`）是整个管线的核心，成员（`:27-36`）：
`config`、`env`、`device`、`default_compiler_options`、`default_launch_options`、
`disk_cache`、`backend`、`parser`、`mem_cache`、`hash_base`。

### 2.1 缓存键的五段构成

构造时先填 `hash_base`（`runtime.hpp:46-53`），注释写得很清楚——注释指出 5 段，但
其中「compile options / post hook / code」是**每次编译时**才追加的：

| 段 | 内容 | 何时加入 |
|---|---|---|
| ① extra_signature | `"cutlass-421"` | 构造时（`hash_base`） |
| ② compiler version | `FNV1a(nvcc --version 原文)` | 构造时（`hash_base`） |
| ③ compiler options | `str::join(get_flags())`（空格连接） | `cache_key`（`:94`） |
| ④ post hook | post-hook 脚本路径 + 内容哈希（无则 `""`） | `cache_key`（`:95`） |
| ⑤ code | `parser.parse_into_hash(source)`（含递归 include） | `cache_key`（`:96`） |

`cache_key`（`:92-98`）：

```cpp
std::string cache_key(const std::string& source, const CompilerOptions& options) {
    auto hash = hash_base;                              // ①②
    options.update_hash(hash);                          // ③ = hash.update(join(get_flags()))
    return hash.update(options.get_post_hook_hash(config))  // ④
               .update(parser.parse_into_hash(source))       // ⑤
               .get_hex_digest();
}
```

注意 ⑤ 里 `parse_into_hash` 返回的是**十六进制字符串**，它又被外层再哈希一次——所以最终摘要
是「源码+include 的摘要」的摘要。

### 2.2 两级缓存

`compile`（`:56-62`）先算 key，查**内存缓存** `mem_cache`（`memory.hpp:9-22`，一个
`unordered_map<key, shared_ptr<Kernel>>` 的 `get_or_create`），未命中才走磁盘 + 编译 + 加载：

```cpp
std::shared_ptr<Kernel> compile(name, source, override_options) {
    const auto options = default_compiler_options.override_with(override_options);
    const auto key = cache_key(source, options);
    return mem_cache.get_or_create(key, [&] {
        return Backend::load(compile(name, source, key, options), env);   // 磁盘 → 加载
    });
}
```

磁盘那层（`:75-90`）**先查磁盘，命中直接返回路径；否则编译后 `commit()` 发布**：

```cpp
auto entry = disk_cache.entry(name, key);
if (entry.hit) return entry.path;               // 命中：直接复用
backend.compile(source, entry.path, env, config, options);
GilScopedRelease gil_release;                   // 发布时放 GIL，避免阻塞其它 Python 线程
return entry.commit();
```

磁盘布局（`disk.hpp:79-133`）：条目在 `<root>/cache/<tag>.<digest>/`（`entry_name = tag.digest`，`:120`），
其中 `tag` 是 kernel 名（`jit->compile("sm90_fp8_gemm_1d2d", ...)` 的第一个参数），`digest` 就是 cache key。
命中判据是看该目录下有没有 `.committed` 标记文件（`:19`、`:123`）。

**多根与共享缓存**（`from_env`，`:87-109`）：`DG_JIT_CACHE_DIR` 支持 `:` 分隔的路径列表，
**查找按顺序（第一个命中即用），编译产物只写第一个（可写）根**；不设则默认 `$HOME/.dj`（`:105`）。
这允许「个人可写缓存 + 团队只读共享缓存」：`export DG_JIT_CACHE_DIR="$HOME/.dj:/shared/deep_jit"`。

**发布是原子的**（`DiskCacheEntry::commit`，`:42-67`）：先写 `.committed`、`fsync` 整棵树
（`write_file_sync` + `fsync_dir`，`filesystem.hpp`），再 `std::filesystem::rename` 整个临时目录到最终路径。
并发场景（多进程/多节点同时 miss 同一 kernel）下，谁先 rename 谁赢，输的一方 `safe_remove_all` 删掉自己的临时目录
——注释特别说明这里**故意不用 `std::remove_all`**，它在分布式文件系统上并发删除父目录会段错误（`:56-60`）。
`DiskCacheEntry` 析构时若既没命中也没提交，会尽力清理临时目录（`:69-73`）。`backend.compile` 写的是
`<root>/tmp/<uuid>/`（`:130`），正是「临时构建目录 → 原子发布」的前半段。

---

## 3. 哈希：FNV1a + 递归 include 解析

### 3.1 FNV1a（`utils/hash.hpp:10-42`）

DeepJIT 用了一个**双状态**的 FNV1a 变体：两个 64-bit state（`:12-13`），
每个字节分别用不同素数滚动（`0x100000001b3` 与 `0x9e3779b97f4a7c15`，`:17-18`）。
关键在 `update` 的**长度前缀**（`:21-24`）：

```cpp
const int64_t size = data.size();
for (int shift = 0; shift < 64; shift += 8)
    update_byte(static_cast<uint8_t>(size >> shift));   // 先喂 8 字节小端长度
for (const char value: data) update_byte(value);         // 再喂内容
```

前缀长度是为了让**链式拼接无歧义**：`update("ab") + update("c")` 与 `update("a") + update("bc")`
必须得到不同结果（否则 `hash("ab")` 里再拼 `"c"` 会和 `"a"+"bc"` 撞）。摘要（`:30-37`）对两个 state
各做一次 splitmix 混淆（`:31-34`），拼成 **32 位十六进制**（`:36`）。这些性质在实验 §7 里逐条验证。

### 3.2 递归 include 解析（`utils/parser.hpp`）

`parse_into_hash(code)`（`:105-120`）先哈希源码本身，再**逐行扫描**找 include：

```cpp
hash.update(code);
for (auto line : code 的每一行) {
    if (auto filename = parse_include(line); not filename.empty())
        hash.update(parse_includes_into_hash(filename));   // 递归，用子文件摘要
}
```

`parse_include`（`:37-66`）只认 `#` + `include` + `<...>`，且**文件名必须以 `include_prefixes` 里的前缀开头**
（DeepGEMM 就是 `deep_gemm/`）才返回。`parse_includes_into_hash`（`:68-103`）在 `include_dirs` 里找文件、
读取、递归，并做**循环 include 检测**（`visiting` 集合，`:75-77`；缓存 `cache` 避免重复哈希，`:70-72`）。

这正是「源码只有 400 字节，但缓存键覆盖了 16 个头文件」的原因：`transpose_fp32` 的
`kernel.cu` 只 `#include <deep_gemm/impls/smxx_layout.cuh>`，但这个头又 include 了 `math.cuh`、
`ptx/*.cuh`……解析器递归进去，**实测哈希了 16 个 `deep_gemm/` 头文件**。
换句话说：**改任何一个被 kernel 间接 include 的 `.cuh`，都会改变缓存键、触发重新编译**——
这是热修复 kernel 后缓存自动失效的机制来源。

### 3.3 实测：逐位复刻缓存键

`harness/exp_deepjit.py` 用纯 Python 逐行重写了上面两节（FNV1a + Parser + `CompilerOptions::get_flags`），
从磁盘条目的 `meta.json`（配置、nvcc 版本、编译选项）与 `kernel.cu`（源码）出发重算摘要。
结果（`out.txt` §1/§2）：

```
磁盘目录 digest  : 8fbcaf5d3944657facea4004f6e7c383
Python 复刻 digest: 8fbcaf5d3944657facea4004f6e7c383
MATCH : True
递归哈希的 deep_gemm/ 头文件数 : 16
```

`nvcc --version` 的原文**以换行结尾**也被算进去了（`meta.json` 里 `version` 字段的末段是
`compiler.37668154_0\n`），复刻时必须保留——这解释了为什么「编译器版本」是整段几十字符的原文哈希，
而不是解析后的 `13.2`。

---

## 4. CUDA 后端：nvcc → cubin

### 4.1 编译（`backend/cuda/backend.hpp:77-168`）

`CUDA::compile` 的步骤：

1. **写源码** `kernel.cu`（`:93`，`write_file_sync`）。
2. **拼命令行**（`:96-109`）：`nvcc <src> --cubin --output-file <cubin>` + `get_flags()` + 每个 include 目录一个 `--include-path`。
3. **`cd` 到临时目录再编译**（`:111-112`）——注释说明这是防止同名 include 文件冲突。
4. **编译并可选打印**（`:115`，`print_compiler_command = JIT_DEBUG or JIT_PRINT_COMPILER_COMMAND`）。
5. **溢出/本地内存正则断言**（`:120-127`）：`JIT_CHECK_NO_SPILLS` 时若 ptxas 输出匹配
   `spilled\s+to\s+local\s+memory` 就抛错；`JIT_CHECK_NO_LOCAL_MEMORY` 同理匹配 `local\s+memory\s+used`。
6. **校验 cubin 非空**（`:128-130`）。
7. **可选 post-hook**（`:132-138`）：起一个 `python <hook> <cubin>` 子进程（DeepGEMM 默认不用）。
8. **可选 dump PTX / SASS**（`:140-158`）：`--ptx` 再编一遍，或 `cuobjdump --dump-sass`。
9. **写 `meta.json`**（`:160-167`）：命令行、config、compiler_info、compiler_options——这就是我们做缓存键复刻的原料。

`get_compiler_info`（`:63-75`）跑 `nvcc --version`，用正则 `release (\d+)\.(\d+)` 解析，**断言 ≥ 12.9**
（`--gpu-architecture=sm_XX[f/a]`、`cvt.rs` 等需要新 ptxas）。
`find_cuda_toolkit`（`:174-217`）按 `CUDA_HOME`/`CUDA_PATH` → `which nvcc` → `/usr/local/cuda` 的顺序找，
再定位 `bin/nvcc` 与 `bin/cuobjdump`；`DG_JIT_NVCC_COMPILER` 可覆盖。

### 4.2 编译选项（`backend/cuda/options.hpp:24-175`）

`CompilerOptions` 是一组 `optional`→「unset 时继承默认」。默认值（`default_options`，`:39-62`）：

```
optimize_level = "3",  fast_math = false,
ptxas_register_usage_level = 10,
arch = device.get_arch(),                       // sm_90a / sm_100f ...
nvcc_flags = ["-std=c++20",
              "--compiler-options=-fPIC", "--compiler-options=-fconcepts",
              "--expt-relaxed-constexpr", "--expt-extended-lambda"]
```

`init_jit` 再往 `nvcc_flags` 追加两条（§1）。`get_flags`（`:95-134`）把这些展开成真正的 flag 序列
（顺序：arch → `-O` → compiler `-O` → fast-math → ptxas verbose → register-usage-level →
warn-on-spills → warn-on-local-memory → line-info → `nvcc_flags` → `extra`）。
`update_hash`（`:136-139`）**只把 `str::join(get_flags())` 喂进哈希**——所以**任何 flag 变化都换缓存键**。
实验 §2 抓到本机的 11 个 flag：

```
--gpu-architecture=sm_90a  -O3  --compiler-options=-O3
--ptxas-options=--register-usage-level=10
-std=c++20  --compiler-options=-fPIC  --compiler-options=-fconcepts
--expt-relaxed-constexpr  --expt-extended-lambda
--diag-suppress=39,161,174,177,186,940
--compiler-options=-Wno-deprecated-declarations,-Wno-abi
```

一个有趣的副作用：**开 `DG_JIT_DEBUG` 会加 `--generate-line-info` 和 `--ptxas-options=--verbose`，
从而改变缓存键**——debug 编译出来的产物和非 debug 的**不共享缓存**（实验 §4b 的条目 digest
`1b3deb25…` 与 §1 的 `8fbcaf5d…` 不同，正因如此）。

`arch` 由 `device.hpp:68-80` 生成：sm90 → `sm_90a`，sm100 → `sm_100f`（family，也支持 `sm_100a`）。

---

## 5. 加载与发射：Driver API

### 5.1 加载 cubin（`backend/cuda/kernel.hpp:48-89`）

`Kernel::load`：`cuLibraryLoadFromFile` 载入 `kernel.cubin`（`:67-68`），
**断言 cubin 里恰好有一个 kernel**（`:70-73`，所以每个 JIT 单元只实例化一个 kernel），
再 `cuLibraryEnumerateKernels` + `cuKernelGetFunction` 拿到 `CUfunction`（`:75-77`）。
`Kernel` 用 `shared_ptr` 持有，析构时 `cuLibraryUnload`（`:42-46`、`:173-183`）。
`JIT_PRINT_LOAD_TIME` 会打印从开始到加载完的毫秒（`:84-87`）——实测两个 kernel 各 ~0.5–2.3 ms。

### 5.2 发射（`kernel.hpp:91-171`）

`Kernel::launch` 用 `cuLaunchKernelEx`（`:169-170`）。要点：

- **参数**: 把 `Args...` 逐个取地址成 `void*` 数组（`:155`，末位 `nullptr`）；对继承 `NoRefPtr` 的类型取 `.ptr`。
- **动态 smem**: `num_smem_bytes` 先 `cuFuncSetAttribute(MAX_DYNAMIC_SHARED_SIZE_BYTES)`（`:116-120`）。
- **cluster**: `cluster_dim.x > 1` 时加 `CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION` 属性（`:141-147`）；
  `nonportable_cluster_size_allowed` 单独 `cuFuncSetAttribute`（`:122-131`，注释解释了「不能写成 `if`」的原因）。
- **PDL**: `enable_pdl` → `CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION`（`:148-152`）。
- **stream**: 不指定则用 PyTorch 当前流 `at::cuda::getCurrentCUDAStream()`（`:164-166`）。

`LaunchOptions` 是每个 launch 的选项（`options.hpp:178-232`，`grid/block/cluster`、smem、
`cooperative`、`enable_pdl`…），经 `default_launch_options.override_with(...)` 合并。
`DG_JIT_*` 里没有直接改 launch 选项的；DeepGEMM 通过 `deep_gemm.set_pdl()`（`config.hpp:27-32`）
改 `jit->default_launch_options.enable_pdl`。

### 5.3 Driver API 的惰性绑定（`lazy.hpp:42-71`）

所有 `cuXXX` 都通过 `DJ_DECL_LAZY_DL_FUNCTION` 包装成 `driver::lazy_cuXXX`：`dlopen("libcuda.so")`
+ `dlsym` 在首次调用时完成（`lazy.hpp`）。这样扩展在**没有 CUDA driver 的环境**里也能 import，
只有真正用到才失败。`device.hpp` 里靠一次 `cudaFree(nullptr)` 确保 context 存在后再查设备属性（`:33-43`）。

---

## 6. DeepGEMM 的调用形态：一个 kernel 一段微源码

看一个具体例子（第 22/32 篇的 1D2D）：`jit_kernels/impls/sm90_fp8_gemm_1d2d.hpp:30-56`：

```cpp
const auto kernel = jit->compile(tag, std::format(R"(
#include <deep_gemm/impls/sm90_fp8_gemm_1d2d.cuh>
using namespace deep_gemm;
static void __instantiate_kernel() {
    auto ptr = reinterpret_cast<void*>(&sm90_fp8_gemm_1d2d_impl<
        {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}
    >);
};
)",
    to_string(args.major_sfb),
    get_compiled_dim(m,'m',compiled_dims), get_compiled_dim(n,'n',...), get_compiled_dim(k,'k',...),
    num_groups, block_m, block_n, block_k, swizzle..., num_stages,
    num_tma_threads, num_math_threads, cluster_size, cluster_n>1, num_sms, gemm_type, cd_dtype));
```

即：**每个配置组合都生成一段「只 include 头文件 + 强制实例化一个模板」的微源码**，交给 DeepJIT 编译。
这正是实验 §1 看到的 407 字节 `kernel.cu`。`__instantiate_kernel` 这个无用函数存在的唯一意义是
**强制模板实例化**（否则 nvcc 不会生成代码）；「恰好一个 kernel」的断言（§5.1）由此满足。

`get_compiled_dim`（`runtime_utils.hpp:31-40`）：

```cpp
static int get_compiled_dim(const int& dim, const char& name, const std::string& compiled_dims) {
    if (heuristics_runtime->get_ignore_compile_dims()) return 0;   // 强制忽略所有维度
    for (char c: compiled_dims) if (name == c) return dim;          // 名字在 compiled_dims 里才编进去
    return 0;
}
```

`compiled_dims` 的默认值按算子在 pybind 里定死（`csrc/apis/gemm.hpp`）：稠密/grouped 的 `nt` 是 `"nk"`
（m 不进模板），`tn`/`tt` 与 K-grouped 是 `"mn"`（k 不进模板）。这解释了两个现象（实验 §5）：

- `fp8_gemm_nt` 的实例化维度行是 `0, 512, 1024`（m 位是 0），即 **m 不作为编译期维度**；
- 但 (1024,512,1024) 与 (3072,512,1024) **仍是两个不同的缓存条目**——因为 **m 会改变启发式选出的
  block config**（第 18 篇），而 block config 也被写进了模板。实测这两条的模板分别是
  `… 64, 64, 128 … num_stages 13 …` 和 `… 128, 112, 128 … num_stages 6 …`。
  所以准确的说法是：**cache key 覆盖「完全实例化后的模板」（编译维度 + 启发式 block config + launch 配置），
  而不只是形状**。
- `deep_gemm.set_ignore_compile_dims(True)`（`config.hpp:36-38`）把 `get_compiled_dim` 一律返回 0，
  于是实例化维度行全变 `0, 0, 0`；但 4 个形状仍产生 3 个条目（block config 不同）——再次印证上面这点。
  `set_block_size_multiple_of`（`config.hpp:39-47`）则在启发式里约束 block 大小对齐。

调用形态的设计收益：**JIT 单元极小**（一个 include + 一行实例化），所以编译/哈希的对象是稳定的微源码 +
稳定的头文件树，而不是把大量模板参数塞进命令行；同时**新形状只多一个模板实例**，不会重编整个扩展。

> 附带一个冷知识：`csrc/indexing/main.cu` 把所有 kernel 头都 include 一遍，`main()` 直接 `return 0`
> ——它的唯一用途是让 IDE / 语言服务器（clangd）能索引到所有 device 代码。

---

## 7. 环境变量矩阵（`README.md` §Utilities + `utils/env.hpp`）

`Env` 的解析规则（`env.hpp:56-84`）：先查 `DG_<suffix>`，没有再查全局 `DJ_<suffix>`（`:65-70`）；
布尔接受 `true/false/yes/no/整数`（`:35-46`）。常用变量：

| 变量 | 作用 | 源码 |
|---|---|---|
| `DG_JIT_DEBUG` | 打印编译命令 + ptxas 输出 + 加载报告 + 行号 | `options.hpp:40-41`、`backend.hpp:89-90`、`kernel.hpp:58-60` |
| `DG_JIT_CACHE_DIR` | 磁盘缓存根（`:` 列表，命中按序、只写第一个） | `disk.hpp:87-109` |
| `DG_JIT_PRINT_COMPILER_COMMAND` | 打印 nvcc 命令 | `backend.hpp:90` |
| `DG_JIT_PTXAS_VERBOSE` | 打印寄存器/溢出 | `options.hpp:45`、`backend.hpp:116-119` |
| `DG_JIT_CHECK_NO_SPILLS` / `_NO_LOCAL_MEMORY` | ptxas 输出正则，命中即报错 | `backend.hpp:120-127` |
| `DG_JIT_DUMP_ASM` / `_PTX` / `_SASS` | dump PTX/SASS 到缓存目录 | `backend.hpp:140-158` |
| `DG_JIT_WITH_LINEINFO` | `--generate-line-info`（进缓存键！） | `options.hpp:49` |
| `DG_JIT_NVCC_COMPILER` / `_CPP_STANDARD` | 指定 nvcc / C++ 标准 | `backend.hpp:205-209`、`options.hpp:54` |
| `DG_JIT_PRINT_LOAD_TIME` | 打印 cubin 加载耗时 | `kernel.hpp:59` |
| `DG_PRINT_CONFIGS` | 打印选定 config 与 TMA 描述符 | `runtime_utils.hpp:147` 等 |

---

## 8. H100 实测小结（`harness/exp_deepjit.out.txt`）

1. **缓存条目解剖**：跑一个 `fp8_gemm_nt(1024,512,1024)` 生成 **2 个**条目
   （SF 布局 `transpose_fp32` + 主 kernel `sm90_fp8_gemm_1d2d`）；每个条目 = `kernel.cu`（407 B）
   + `kernel.cubin` + `meta.json` + `.committed`。
2. **缓存键逐位复刻**：Python 重算 digest == 目录名（`8fbcaf5d…`，`MATCH True`），
   过程中递归哈希了 **16 个 `deep_gemm/` 头文件**；`nvcc --version` 原文（含行尾换行）也在键里。
3. **编译 vs 加载**：同一 shape，首进程 JIT（编+载 2 个 kernel）**~4.98 s**，
   第二进程（命中磁盘，只加载）**~0.20 s**，约 **25×**；单 kernel 加载 ~0.5–2.3 ms。
   总进程墙钟 7.60 s → 2.80 s（差额主要是 Python import，两边都有）。
4. **真实命令行**：抓到完整 `nvcc … --gpu-architecture=sm_90a -O3 … --include-path …`（§5 列出的 11 个 flag）。
5. **`DG_JIT_DEBUG`**：额外 dump `kernel.ptx`/`kernel.sass`（条目里多两个文件）；
   ptxas 报告 `sm90_fp8_gemm_1d2d` 用 **168 寄存器 / 10 barrier、0 spill**，
   `transpose_fp32` 用 32 寄存器 / 1 barrier；因 `--generate-line-info` 进入缓存键，digest 与非 debug 不同。
6. **digest 敏感性**：4 个形状 → 4 个条目（编译维度 `"nk"` 使 m 不显式进模板，
   但 m 经启发式 block config 仍区分）；`ignore_compile_dims=True` 使实例化维度行全为 `0,0,0`，条目降到 3。
7. **无溢出强制**：`DG_JIT_CHECK_NO_SPILLS=1` 在真 kernel 上 rc=0（ptxas 报 `0 bytes spill`）。
8. **FNV1a 性质**：`FNV1a("ab")+("c")` ≠ `FNV1a("a")+("bc")`（长度前缀消歧成立），digest 恒 32 hex。

---

## 9. 小结

- DeepGEMM **安装时零 CUDA 编译**：kernel 在**首次调用某个配置**时，由
  `jit->compile(tag, 微源码)` 现编成 `cubin`，加载进 CUDA Driver。
- **缓存键 = 五段 FNV1a**：`extra_signature(cutlass版本)` + `compiler_version(nvcc原文)` +
  `get_flags()` + `post_hook` + `源码及其递归 include 摘要`。任何一段变化都触发重编。
- **两级缓存**：进程内 `mem_cache`（key=cache_key）；磁盘 `<root>/cache/<kernel>.<digest>/`
  （`.committed` 标记 + 原子 rename 发布，支持 `:` 分隔的多根与只读共享缓存），
  面向多进程 / 多节点 / 多用户复用。
- **微源码 + 头文件树**：每个 JIT 单元只 include 一个 `deep_gemm/…cuh` 并强制实例化一个模板；
  缓存键覆盖「**完全实例化的模板**」——即形状中 `compiled_dims` 指定的维度 + 启发式 block config + launch 配置。
- 这一层把第 18 篇的 `get_best_config`、第 21 篇的 epilogue 算子、第 27 篇的 cuBLASLt 回退串了起来：
  **选完 config → 生成微源码 → 查/编/缓存 → 加载 → `cuLaunchKernelEx`**。
