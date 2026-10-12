# 47. DeepJIT 的 post_hook 后处理、磁盘产物与 CUDA↔Ascend 后端抽象

> 第 33 篇把 DeepJIT 的编译 / 缓存 / 加载 / 发射讲清楚了，但里面有一个**列表项**一直没展开：
> 缓存键的「第四段」——`post_hook`（后处理钩子）。它是 DeepJIT 提供给下游库的**唯一一个「在编译产物
> 发布之前注入用户代码」的接口**，DeepGEMM 默认**不用**（`post_hook = std::nullopt`），所以前面所有篇里
> 它一直是空的、也就没被注意。本篇把它拆透，并顺带把 **CUDA↔Ascend 后端抽象**（同一套 `Runtime<Backend>`
> 如何服务两个完全不同的工具链）与 **Python API / WIP 特性**补齐。
>
> **架构无关**：post_hook 是纯主机侧「编译后跑一个 Python 脚本」的机制，**H100 上真机可跑**。
> 行号核对：DeepGEMM `057ca59` + DeepJIT `2efdab4`（`third-party/deep_jit`）。
> 实验：`harness/exp_deepjit_posthook.{sh,out.txt}`（源码 `src/deepjit_posthook.cpp`，直接 include 真实头文件，
> 在 H100 上编译并运行真实 DeepJIT 代码）。

---

## 1. post_hook 是什么：编译后、发布前的可插拔后处理

DeepJIT 的编译流水线（`backend/cuda/backend.hpp:77-168`）在 `nvcc` 产出 `kernel.cubin` 之后、写 `meta.json`
并提交到缓存之前，留了一个钩子点（`backend.hpp:132-138`）：

```cpp
// Run post hook
if (options.post_hook) {                                                 // :133
    const auto hook_path = config.get_python_path(*options.post_hook);   // :134
    const auto hook_command = "cd " + dir.string() +                     // :135
                              " && python " + hook_path.string() + " " + cubin_path.string();
    call_external_command(hook_command, print_compiler_command);         // :137
}
```

也就是说：**在临时构建目录里，起一个 `python <hook> <cubin>` 子进程，脚本收到 cubin 的绝对路径，必须就地改写它**
（README §Post hook）。`hook` 的相对路径由 `Config::python_library_root` 解析（`runtime/config.hpp:48-50`，
`python_library_root / rel_path`），DeepGEMM 里就是 Python 包目录。

它存在的意义：下游库（如 DeepGEMM）如果需要「对编译器输出做一层确定性后处理」——嵌入签名、合规改写、
二次校验、加壳——不需要 fork DeepJIT，只要提供脚本 + 把脚本路径塞进 `CompilerOptions::post_hook`。
DeepGEMM **没有用**这个能力（`csrc/runtime/jit.hpp` 里只加了 nvcc flags，没设 hook），但接口在那里。

**契约 / 安全边界**（README:153, 264, 273）：

- `post_hook` 相对路径必须由调用方提供**可信**路径；路径与自由 flag 一样走简单 shell 拼接，
  因此**不得含空白或 shell 元字符**。
- 脚本必须对「给定 cubin + 已跟踪签名」**确定性**：它 import 的模块、辅助文件、Python/包版本、环境变量、
  随机状态、时间、网络数据**都不会被自动发现**；任何会影响输出的依赖都要自己写进 `Config::extra_signature`
  （`runtime/config.hpp:16`、`runtime/runtime.hpp:52`）。
- 这个钩子是**摘要键之外**的唯一用户注入点——换句话说，缓存键信任模型里，hook 文件本身被哈希（见 §2），
  但 hook 所依赖的一切外部状态不被哈希。

---

## 2. post_hook 如何进入缓存键：路径 + 内容哈希，且线程内缓存

`Runtime::cache_key`（`runtime/runtime.hpp:92-98`）的五段中，第四段（`:95`）就是
`options.get_post_hook_hash(config)`。实现（`backend/cuda/options.hpp:141-156`）：

```cpp
std::string get_post_hook_hash(const Config& config) const {
    if (not post_hook) return "";                                        // :142-143 无 hook → 空串
    thread_local std::unordered_map<std::string, std::string> post_hook_hashes;  // :146
    const auto path = config.get_python_path(*post_hook);                // :147
    if (iterator = post_hook_hashes.find(path); ...) return iterator->second;    // :148-149 命中
    const auto digest = hash::FNV1a()
        .update(*post_hook)          // 相对路径本身                              // :153
        .update(read(path))          // 文件内容                                  // :153
        .get_hex_digest();
    return post_hook_hashes.emplace(path, digest).first->second;         // :155
}
```

三个要点：

1. **哈希 = 相对路径 + 文件内容**（`:152-154`）。所以「换内容」和「换路径」都会换键，二者都进缓存键。
2. **线程内按绝对路径缓存**（`thread_local` map，`:146-149`）。这与 README:271 一致：**改完 hook 文件必须重启进程**
   才生效——同一进程内再算哈希会命中旧值。
3. 无 hook 时返回**空串**（`:143`），照样被喂进 FNV1a（`hasher.update("")` 也带长度前缀，是良定义的），
   只是第四段恒为一个固定值。

第 33 篇的 Python 复刻里第四段一直是 `""`，正是因为这个实验从未配 hook。配上 hook 后，缓存键
（实测）：

```
无 hook           noop.9a0a899a27f9d4ac614484e0a891cfbe
hook h_mark.py    noop.068c54b35021b9434285943fcd8dbbdf   （h_mark.py 内容参与键）
hook h_mark2.py   noop.b604b167c85aecd297255ef7d99adcfc   （新路径 → 新键）
```

线程内缓存与跨进程冷读取的对照（实测）：

- **同进程**改 `h_mark.py` 内容后，`get_post_hook_hash` 前后**完全相等**（`before == after`，
  `threadlocal_cached=1`）——旧的哈希被复用。
- **跨进程**（两次独立调用，同一路径 `h_cold.py`，内容 `AAA` vs `BBB`）→ 哈希与缓存键都不同
  （`73360241…` vs `5dcc60df…`；digest `5bb29276…` vs `e11ef00e…`）——冷进程会重新读文件。

---

## 3. override 语义：nullopt 继承、空串仍是「已配置」

`CompilerOptions` 是「unset 字段继承默认」的模型（`options.hpp:64-93`）。post_hook 这一段（`:90-91`）：

```cpp
if (overrides.post_hook)          // 注意：只判断 optional 是否有值，不判断内容
    result.post_hook = overrides.post_hook;
```

由此得到三条容易踩的行为（README:264 + 实验 §4）：

| 场景 | 结果 | 原因 |
|---|---|---|
| 默认没设、本次设 `"hooks/x.py"` | 生效 | override 有值 |
| 默认设了、本次 override `std::nullopt` | **继承默认**（无法关闭） | `if (overrides.post_hook)` 为假 |
| 默认设了、本次 override `""` | 结果 `post_hook = ""`（**算已配置**） | 空串是「有值的 optional」，覆盖生效 |

也就是说：**`nullopt` 是「不改」，不是「关掉」；空串是「配置了一个空路径的 hook」**。空串的后果是
`get_python_path("")`（`config.hpp:49`）返回 `python_library_root` 本身，接着 `read(它)`（`options.hpp:153`）
会去读一个目录 → 报错。实验里 `nullopt_inherits=1`、`empty_is_configured=1`、`empty_value=''` 全部符合。

**结论**：一旦 runtime 默认设了 post_hook，就无法对单个 kernel 关闭它（README:264 明确写了这点）。
要 per-kernel 关闭，只能不设默认、每次显式传 hook。

---

## 4. 磁盘产物与发布顺序：为什么 hook 改的 cubin 就是最终产物

磁盘条目（`cache/disk.hpp:111-133`）位于 `<root>/cache/<tag>.<digest>/`（`:120`），
`tag` = `jit->compile("noop", …)` 的第一个参数（只能含字母数字下划线，`:116-117`），
`digest` = 缓存键。命中判据是 `.committed` 标记文件存在（`:123`）。

未命中时 `entry()` 建 `<root>/tmp/<uuid>/` 临时目录（`:130-132`），`CUDA::compile` 把
`kernel.cu`（`:93`）、`kernel.cubin`（nvcc）、`meta.json`（`:160-167`）都写在这个临时目录里，
然后 `DiskCacheEntry::commit`（`:42-67`）**先写 `.committed` + `fsync` 整棵树**（`:47-48`），
再 `std::filesystem::rename` 整目录到最终路径（`:54`）；并发 loser 用 `safe_remove_all` 清理自己的临时目录
（`:56-60`，注释解释为何故意不用 `remove_all`——分布式 FS 上并发删父目录会段错误）。

关键在**顺序**：hook（`backend.hpp:132-138`）在 `meta.json`（`:160-167`）和 `commit` **之前**执行。
因此：

1. **hook 改的就地 cubin 就是发布出去的 cubin**。实测：破坏 hook `h_corrupt.py`（`open(cubin,'wb').write(b'HOOK')`）
   → 产物 `kernel.cubin` 头就是 `HOOK`，随后 `CUDA::load` 直接失败（`load_threw=1`）。
2. **hook 失败 → 编译失败 → 不得发布**。失败 hook `h_fail.py`（`sys.exit(7)`）让 `call_external_command`
   抛异常（`utils/command.hpp:35` 断言 `exit_code == 0`），异常向上冒泡，`DiskCacheEntry` 析构时
   best-effort 清理临时目录（`disk.hpp:69-73`），磁盘上**没有**该条目。实测 `threw=1`、`published_hit=0`。
3. **缓存命中根本不会进入 `compile()` 的编译段**（`runtime/runtime.hpp:80-82`：`if (entry.hit) return entry.path;`），
   所以 hook **不会重跑**。实测：`compile()` 加载 + `launch` 后副作用文件仍只有 1 行；清空内存缓存后走磁盘命中，
   行数仍是 1。

`meta.json`（`backend.hpp:160-167`）记录 `command / config / compiler_info / compiler_options` 四块，
其中 `compiler_options.post_hook` 记录**选中的相对路径**（`options.hpp:172`）。实测条目（sm90 上）：

```json
{
  "command": "/usr/local/cuda/bin/nvcc <tmp>/kernel.cu --cubin --output-file <tmp>/kernel.cubin
              --gpu-architecture=sm_90a -O3 --compiler-options=-O3
              --ptxas-options=--register-usage-level=10 -std=c++20 --compiler-options=-fPIC
              --compiler-options=-fconcepts --expt-relaxed-constexpr --expt-extended-lambda",
  "config": {"python_library_root": "...", "extra_signature": "sig", "include_dirs": [], "include_prefixes": []},
  "compiler_info": {"path": "/usr/local/cuda/bin/nvcc", "version": "nvcc: … release 13.2 …\n"},
  "compiler_options": { …, "post_hook": "hooks/h_mark.py" }
}
```

注意 `command` 里的路径是**临时构建目录**的路径（发布后可能已不存在，README:295），且**不记录**外围的
`cd`、stderr 重定向、hook 命令与 dump 命令。第 33 篇做缓存键复刻时用的正是这份 `meta.json`。

---

## 5. CUDA ↔ Ascend：同一套 `Runtime<Backend>`，两个工具链

DeepJIT 的卖点是「一个运行时接口服务 CUDA 与昇腾」。抽象点在 `Runtime<Backend>`
（`runtime/runtime.hpp:19-25`）只依赖 Backend 暴露的五个类型别名：

```cpp
using Device          = Backend::Device;
using Kernel          = Backend::Kernel;
using CompilerOptions = Backend::CompilerOptions;
using LaunchOptions   = Backend::LaunchOptions;
```

以及 `Backend::load(dir, env)` 与 `backend.compile(source, dir, env, config, options)`、
`backend.compiler_info`。**共享的部分**：`Config`、`Env`（同样的 `PREFIX_ > DJ_ > default` 解析）、
`Parser`、`MemCache`、`DiskCache`（同一份 `disk.hpp`）、五段式缓存键、`.committed` 原子发布、
`register_python_api`。**分化的部分**（逐条对照真实源码）：

| 关注点 | CUDA（`backend/cuda/*`） | Ascend（`backend/ascend/*`） |
|---|---|---|
| CompilerInfo hash | nvcc 版本原文（`cuda/backend.hpp:39-41`） | bisheng 版本 **+** linker 版本（`ascend/backend.hpp:37-39`，`:64-71` 跑两个 `--version`） |
| 编译步骤 | 一步 `nvcc --cubin`（`cuda/backend.hpp:96-115`） | 两步：bisheng 编 `kernel.asc→kernel.rel.o`，再 `ld.lld` 链接成 `kernel.o`（`ascend/backend.hpp:100-124`，中间 `.rel.o` 删掉 `:122`） |
| dump | `dump_ptx` / `dump_sass`（`cuda/backend.hpp:141-158`，cuobjdump） | `dump_asm`：`-save-temps` 存到 `asm/`（`ascend/backend.hpp:126-139`） |
| 发射 | grid/block/cluster/smem/PDL（`cuda/options.hpp:178-232`） | `num_blocks` / `num_ubuf_bytes` / `num_launch_timeout_secs`（`ascend/options.hpp:129-158`） |
| **post_hook** | **支持**（`cuda/options.hpp:37` 字段 + `:141-156` 哈希 + `cuda/backend.hpp:132-138` 执行） | **不支持**：`get_post_hook_hash` 恒返回 `""`（`ascend/options.hpp:112`），`CompilerOptions` 里根本没有 `post_hook` 字段 |

所以「缓存键第四段」在 Ascend 上**恒为空**——这不是 bug，而是 post_hook 被设计成 CUDA 专属能力。
两个后端的 dump 都**不进缓存键**，且只在**真正发生编译**时生成，命中不回填（README:297, 403）。

Ascend 的 `Runtime<Ascend>` 逻辑上可编译（头文件在），但本机没有 CANN/torch_npu，**本篇不跑 Ascend**，
仅做源码对照（与前面「SM100 专属，仅分析」同性质）。

---

## 6. Python API 与两个 WIP 特性

`python_api.hpp:12-20` 的 `register_python_api(module, jit)` 做两件事：

```cpp
pybind11::class_<Runtime<Backend>, std::shared_ptr<Runtime<Backend>>>(module, "Runtime", …);  // :14
module.def("get_jit", [&jit]() { return jit.get(); });                                        // :17-19
```

`get_jit()` 返回的是**进程内唯一的 runtime 对象**，并且调用它会**触发 `LazyInit` 初始化**
（`jit.get()` 解引用才跑工厂，`utils/lazy.hpp`）。DeepGEMM 用它在 `csrc/python_api.cpp:31` 暴露
`deep_gemm._C.get_jit()` 供 warmup（第 33 篇 §1 讲过）。注意 `Runtime` 以 `pybind11::module_local()`
注册，避免多个扩展之间的类型冲突。

README §In development（`:18-21`）列出两个**当前不可用**的特性：

1. **Cache warmup from history**：用历史缓存条目预测未来需要哪些 kernel，提前预热——减少运行期编译延迟。
2. **Python compilation API**：直接从 Python 传源码编 CUDA/Ascend kernel。

这两个正是「DeepGEMM 目前只能让用户在 Python 侧调一次真实算子来触发编译，而不能主动预编」的原因。
DeepGEMM 自己的替代方案是 `set_ignore_compile_dims` / `set_block_size_multiple_of`
（第 33 篇 §6），用「少编几个模板实例」来压编译面，而不是预热。

---

## 7. H100 实测小结（`harness/exp_deepjit_posthook.out.txt`，总判定 ALL PASS）

在 H100（sm90，CUDA 13.2）上**直接链接真实 DeepJIT 头文件**编译宿主程序（`g++ -std=c++20`，链接 torch +
libpython）并运行，8 组结论：

1. **基线产物**：无 hook 时条目含 `kernel.cu / kernel.cubin / meta.json / .committed`（4/4），
   `meta.compiler_options.post_hook = null`，`command` 首段是真实 nvcc。
2. **hook 改变缓存键**：加 `h_mark.py` 后 digest `068c54b3… ≠ 9a0a899a…`，hook 副作用文件被写入
   （证明脚本真的跑了），`meta.post_hook = "hooks/h_mark.py"`。
3. **线程内哈希缓存**：同进程改 hook 文件后哈希不变（`threadlocal_cached=1`）；新路径 `h_mark2.py` → 新键
   `b604b167…`。
4. **override 语义**：`nullopt` 继承默认（`=1`）、空串仍算已配置（`=1`，值为 `''`）。
5. **失败 hook**：抛异常（`threw=1`）且**未被发布**（`published_hit=0`）。
6. **破坏 hook**：发布产物里的 cubin 头就是 `HOOK`（hook 就地改写在发布之前生效），随后 `load` 失败
   （`load_threw=1`）。
7. **缓存命中不重跑 hook**：`compile()` 加载 + `launch` 后副作用仍 1 行；清内存缓存后磁盘命中对象非空、
   副作用仍 1 行。
8. **跨进程冷读取**：两个独立进程读同一路径不同内容 → hookhash `73360241…` vs `5dcc60df…`、
   digest `5bb29276…` vs `e11ef00e…`，均不同。

---

## 8. 小结

- `post_hook` 是 DeepJIT 流水线里**唯一在「编译产物发布前」注入用户代码**的点：`nvcc` 出 cubin 后、
  写 meta / 提交缓存前，跑 `python <hook> <cubin>`，脚本就地改 cubin（`cuda/backend.hpp:132-138`）。
- 它进缓存键的方式是**「相对路径 + 文件内容」的 FNV1a**（`cuda/options.hpp:141-156`），并按**绝对路径做
  线程内缓存**——所以改 hook 要重启进程；跨进程则会重新读取、区分内容。
- override 模型里 `nullopt` 是「继承」不是「关闭」，空串是「已配置」；默认设了 hook 后无法 per-kernel 关闭。
- 磁盘产物与发布顺序解释了三条实证：hook 改的 cubin 就是最终产物、失败 hook 不发布、命中不重跑 hook。
- `Runtime<Backend>` 用五个类型别名 + `load`/`compile` 抽象 CUDA 与 Ascend；**post_hook 是 CUDA 专属**
  （Ascend 恒返回空哈希），Ascend 的编译是 bisheng + ld.lld 两步、dump 走 `-save-temps`。
- Python 侧只有 `get_jit()`（进程内单例 + 触发 LazyInit）；「历史预热」与「Python 编译 API」两个特性
  **当前是 WIP、不可用**——这解释了 DeepGEMM 为何只能靠「跑一次真实算子」触发编译、以及为何要靠
  `set_ignore_compile_dims` 缩编译面。

至此，第 33 篇那条列表里的 `post_hook` 第四段被补上，DeepJIT 面向 DeepGEMM 的全部可配置面
（配置 / 环境变量 / 编译选项 / 缓存 / 发射 / 后处理 / 后端）都已讲清。
