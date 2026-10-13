# DeepJIT 的支撑工具层：文件系统 / 子进程 / 序列化 / GIL / ABI

> 本文是「DeepSeek Mega MoE 前向 / 反向算子实现解析」专题的 **P160-1 增量（第 170 篇）**。
> 第 33 篇拆了 DeepJIT 的编译 / 缓存 / 加载 / 发射管线，第 47 篇拆了 post_hook 后处理与磁盘产物，
> 第 38 篇拆了 `env` / `lazy`，第 40 篇拆了 `LazyInit` 生命周期，第 116 篇拆了错误 / 断言层。
> 这些篇目里反复出现的 `meta.json`、`<uuid>` 临时目录、`call_external_command`、`str::join`、
> `GilScopedRelease` 一直当黑盒。本篇把夹在它们下面、**任何一行设备代码跑起来之前**就已定型的那层
> 支撑工具拆透：`deep_jit/utils/{filesystem,command,str,uuid,json,gil,no_ref_ptr}.hpp`（共 377 行）。
>
> **纯 host C++ / 与 GPU 架构无关**：本机 H100(sm90) 上直接把真实头文件 `#include` 进两个程序真机跑
> （一个纯 host 程序 + 一个 pybind 探针），64 项判定 **ALL PASS**。源码坐标：
> `~/github/DeepGEMM` commit `057ca59`，子模块 `third-party/deep_jit` commit `2efdab4`。实验：
> `src/deepjit_utils.cpp`、`src/deepjit_gil.cpp`、`harness/exp_deepjit_utils.{sh,py}` →
> `harness/exp_deepjit_utils.out.txt`。

---

## 1. 这层在哪：DeepJIT 工具层的完整清单

`third-party/deep_jit/include/deep_jit/utils/` 下有 13 个头。到本篇之前，被专篇覆盖过的是：

| 文件 | 行数 | 覆盖篇目 | 干什么 |
|---|---|---|---|
| `hash.hpp` | 44 | 第 33 篇 | FNV1a 双状态哈希（缓存键） |
| `parser.hpp` | 123 | 第 33 篇 | 递归 include 扫描（源码摘要） |
| `env.hpp` | 86 | 第 38 / 117 篇 | `DG_*` / `DJ_*` 环境变量解析 |
| `lazy.hpp` | 71 | 第 40 / 80 篇 | `LazyInit<T>` 惰性单例 + 惰性驱动符号 |
| `exception.hpp` | 197 | 第 116 篇 | `Exception` + `DJ_HOST_ASSERT` / `DJ_PANIC` + libdw 栈回溯 |
| **`filesystem.hpp`** | **120** | **本篇** | 分布式 FS 上的读写 / fsync / 原子发布 |
| **`command.hpp`** | **39** | **本篇** | `popen` 子进程 + 退出码判定 |
| **`str.hpp`** | **28** | **本篇** | `join` 命令行拼接 |
| **`uuid.hpp`** | **25** | **本篇** | 进程唯一临时名 |
| **`json.hpp`** | **131** | **本篇** | `meta.json` 极简序列化器 |
| **`gil.hpp`** | **24** | **本篇** | 编译 / 加载时释放 Python GIL |
| **`no_ref_ptr.hpp`** | **10** | **本篇** | CUDA kernel 参数指针的 ABI 标记 |

它们是**两条主线的地基**：① `cache/disk.hpp` 的目录级缓存用 `filesystem` + `uuid` 把一次编译原子
发布到磁盘；② `backend/cuda/backend.hpp` 用 `command` + `str` 跑 `nvcc`、用 `json` 写 `meta.json`、
用 `gil` 在编译时放出 GIL。本篇的 `harness/exp_deepjit_utils.sh` 会把所有消费点的真实 grep 命中打出来。

## 2. `filesystem.hpp`：在（可能是分布式的）文件系统上做缓存发布

DeepJIT 的磁盘缓存是 `cache/<tag>.<digest>/` 目录 + `.committed` 标记（第 33 篇）。目录级原子发布
对文件系统操作的要求和「写一个文件」完全不同，这 120 行就是围绕这件事的：

- **`read:18-22`**：二进制 `ifstream` + `istreambuf_iterator` 读整文件；打不开就 `DJ_HOST_ASSERT`
  抛 `Exception`（`:20`）。被 `backend/cuda/options.hpp:153` 用来读 post_hook 源码算哈希。
- **`fsync_file:24-39`**：`open(O_RDONLY)` → `fsync` → `close`，三步各自检查 errno（`:27/:31/:36`）。
  注释 `:41-42` 点明动机——**分布式文件系统上 `close()` 本身不保证数据对其它进程可见**。
- **`fsync_dir:43-51`**：**自底向上**递归（先子后父，`:44-49` 递归、`:50` 收尾 fsync 目录本身），
  让目录项也对其它进程可见。`cache/disk.hpp:48` 在写 `.committed` 后调它。
- **`make_dirs:53-59`**：`create_directories` + error_code 判定，「已存在」不算错（`:57` 只在
  `not created and error_code != 0` 时报错）。
- **`try_update_mtime:65-69`**：`noexcept`（`:65` 显式标注，实测 `static_assert(noexcept(...))` 通过），
  且**吞掉失败**——注释 `:61-64` 说明它是「缓存访问记账」，在只读查询缓存上更新 mtime 失败**绝不能**
  把一次合法命中变成 miss。`cache/disk.hpp:124` 就是命中路径调的它。
- **`write_file_sync:73-83`**：写 + `close` + 再 `fsync_file`（`:82`），任何一步失败抛异常。
- **`safe_remove_all:88-108`**：**不用 `std::filesystem::remove_all`**——注释 `:85-87` 说它在分布式
  文件系统上、多进程并发操作同一父目录时**会 segfault**（残留目录项）。实现关键：`:99` 用
  `skip_permission_denied` 打开迭代器，`:102` **先 `increment` 再递归**（避免迭代器悬垂），
  `:107` 最后删目录本身。`cache/disk.hpp:60/72` 与 Ascend 后端 `backend/ascend/backend.hpp:122` 都用它。
- **`normalize_path:110-114`**：`nullopt` 或空路径 → `nullopt`；否则 `absolute(...).lexically_normal()`。
- **`is_executable:116-118`**：`is_regular_file` 且 `access(X_OK)==0`。
  在 `backend/cuda/backend.hpp:210/215` 用它校验 `nvcc` / `cuobjdump` 可执行。

## 3. `command.hpp`：把 `nvcc` 当成黑盒子进程

整层只有 39 行，核心是 `call_external_command:13-37`：

- **命令拼接**：`:14` 空命令断言；`:18` **无条件追加 `" 2>&1"`**——把子进程 stderr 并进 stdout，
  因为 `nvcc` 的诊断（以及 `DG_JIT_CHECK_NO_SPILLS` 扫的 `spilled to local memory`，第 60 篇）都在
  stderr。
- **`popen` + RAII 收尾**：`:19-23` 用 `unique_ptr<FILE, deleter>`，deleter 里 `pclose`（`:20-21`）；
  `:28-29` 逐行 `fgets` 累积输出。
- **退出码判定**：`:33` `pclose` 取 `status`，`:34` `WIFEXITED(status) ? WEXITSTATUS(status) :
  128 + WTERMSIG(status)`。注释 `:31-32` 解释**为什么要特判信号**——子进程被信号杀死（如 Ctrl+C 的
  SIGINT）时 `WEXITSTATUS` 会**错误地返回 0**，必须当成失败。`:35` 非零即断言抛异常，消息里带上完整命令与输出。

消费点：`backend/cuda/backend.hpp:64`（`nvcc` 版本进 `CompilerInfo` 哈希）、`:115`（真正编译）、
`:137`（post_hook 子进程，第 47 篇）、`:145`（dump PTX）、`:155`（`cuobjdump --dump-sass`）、`:187`
（`which nvcc`）；Ascend 后端 `backend/ascend/backend.hpp:67/69/120/121/137` 同构。

## 4. `str.hpp`：给缓存键和命令行做拼接

`str::join:10-26` 是典型的「先算总长再 reserve」实现：`:11` 空返回空串；`:14-16` 总长 =
分隔符 × (n-1) + 各元素长度；`:19-25` 拼接。默认分隔符 `" "`（`:10`）。两处关键消费：
`backend/cuda/options.hpp:138` 的 `hasher.update(str::join(get_flags()))`——**编译器 flag 的拼接
参与缓存键**（第 33 篇「第四段」）；`backend/cuda/backend.hpp:109` 把参数经 `str::join` 拼成命令行。

## 5. `uuid.hpp`：进程唯一的临时目录名

`get_uuid:14-23` 生成 `<pid>-<rand>-<rand>-<rand>`（`:22` 的 `std::format`）。三个随机字来自
**`thread_local` 的 `mt19937`**：`:15-20` 的种子 = `random_device()` ⊕ `steady_clock` ⊕
`std::hash<std::thread::id>`（`:17-19`）——注释 `:12-13` 明说「**每线程种子**，并发调用者绝不碰撞」。
唯一消费点 `cache/disk.hpp:130` 的 `tmp/<uuid>` 临时目录。`uuid` 的 `<chrono>` / `<format>` 依赖是
deep_jit 要求 `-std=c++20` 的原因之一。

## 6. `json.hpp`：131 行的 `meta.json` 序列化器

`meta.json`（`backend/cuda/backend.hpp:161-167`）记录 `command` / `config` / `compiler_info` /
`compiler_options`（含 post_hook 路径，第 47 篇）。这份极简 JSON **只序列化、不解析**：

- **类型系统**：`:18` 的 `value_t = variant<nullptr_t, bool, int64_t, uint64_t, string, array_t,
  object_t>`；`array_t = vector<json>`（`:16`）、**`object_t = vector<pair<string,json>>`（`:17`）**。
- **构造**：`:22-36` 一组重载——注意 `:27-32` 把**有符号 `int64_t` / 无符号 `uint64_t` 分开**（避免
  唯一大整数被截断到 `int64`）；`:25` 的 `bool` 重载用 `is_same_v<remove_cv_t<T>, bool>` 把 `bool` 从
  整型里摘出来；`:38-39` 的 `optional<T>`（`nullopt` → `null`）；`:41-47` 的 `vector<T>`。
- **序列化**：`dump:49-53` → `dump_to:94-128`——`null`/`true`/`false`/整数/字符串/数组/对象逐分支
  （`:95-127`）；字符串走 `dump_string:55-92`：转义 `" \ \b \f \n \r \t`（`:60-80`），**其余控制字符
  (<0x20) 写成 `\u00xx`**（`:81-88`，十六进制表 `:56`），≥0x20 的字节**原样透传**（UTF-8 多字节不受影响）。
- **两个反直觉点**：① `object_t` 是**有序 vector**，故**保留插入序、允许重复键**（与 `std::map` 语义不同）；
  ② 没有 `parse`——`meta.json` 只写不读。

消费点：`runtime/config.hpp:52-63`（`Config` 的 `to_json`）、`backend/cuda/options.hpp:158-...`（`CompilerOptions`）、
`backend/cuda/backend.hpp:43-...`（`CompilerInfo` 的 `to_json`）与 `:161-167`（顶层 `meta.json`）；Ascend 侧
`backend/ascend/backend.hpp:41-.../142-148`、`backend/ascend/options.hpp:114-...` 同构。

## 7. `gil.hpp`：编译 / 加载时放出 Python GIL

`GilScopedRelease:9-22` 只有 24 行，但要讲清它的**条件性**：

- `:13-16` 构造函数：**只有** `Py_IsInitialized() and PyGILState_Check()` 都为真时才
  `release.emplace()`（`:14-15`）——即「解释器活着**且**我当前持 GIL」才释放。
- `:11` 的 `std::optional<pybind11::gil_scoped_release> release` 是成员而非继承，靠 `optional` 的
  析构自动恢复 GIL。
- `:18-21` 删除拷贝/移动——RAII 对象不应被搬运。
- 消费点都在**重活**上：`backend/cuda/backend.hpp:83`（`GilScopedRelease` 包着跑 `nvcc` 编译，可能几秒）、
  `runtime/runtime.hpp:88`（`GilScopedRelease` 包缓存提交 / fsync）、`backend/cuda/kernel.hpp:50/94`
  （`cuLibraryLoadFromFile` 与 `cuLaunchKernelEx` 前的准备）。动机是「编译 / 加载是 host 重活，释放 GIL 让别的 Python 线程跑」，
  而「GIL 已释放 / 无解释器」时必须安全地什么都不做。

## 8. `no_ref_ptr.hpp`：一个休眠的内核参数 ABI 标记

`struct NoRefPtr { void* ptr = nullptr; }`（`:6-8`）。唯一引用点在
`backend/cuda/kernel.hpp:24-30` 的 `kernel_arg_pointer`：`if constexpr (is_base_of_v<NoRefPtr, T>)`
就直接把 `value.ptr` 当参数（`:25-26`），否则取 `&value`（`:28`）。注释 `:5` 说得很直白——
「**指针本身就是参数存储，不要取它的地址**」。

但**公开仓库里没有任何 `NoRefPtr` 的派生类型**（grep 全仓只有定义与本处 `is_base_of`）——
它和第 55 篇的 `common/probe.cuh` 一样是个**休眠设施**：为「有些 kernel 参数本身就是 `void*`、
再取一次地址会传错」这种场景预留的逃生舱，当前路径用不到。

## 9. H100 真机实验：64 项 ALL PASS

`harness/exp_deepjit_utils.sh` 在 `kernel_lab`（H100/sm90）里：① 打印七个头的 sha256（见 `.out.txt`）；
② `g++ -std=c++20 -O2 -I<deep_jit/include> src/deepjit_utils.cpp -ldw -ldl` **直接 include 真实头**编出纯 host
程序，跑 json/str/uuid/command/filesystem/no_ref_ptr；③ 把 `src/deepjit_gil.cpp` 编成 pybind 模块，在
Python 进程里 import 调 `GilScopedRelease`。C++ 段 **43/43**、Python 段 **21/21**，总判定 **ALL PASS**。

关键真实数字 / 发现：

- **json**：`{"a":1,"b":"x","c":[1,2,3],"d":null,"e":true}` 用 Python `json.loads` 解析一致；有符号
  `-5`、无符号 `18446744073709551615` 各按原值输出；转义串 `"\"\\\b\f\n\r\t\u0001"` 与 Python 一致
  （控制字符确为 `\u0001`）；`{"k":1,"k":2}` 保留重复键、Python 解析保留末值（`{"k":2}`）。
- **str::join**：`-O3 --use_fast_math -std=c++17` 与 Python `" ".join` 逐字相同；空/单元素/自定义分隔符全对。
- **uuid**：格式 `<pid>-8hex-8hex-8hex`；同线程 **10 万次唯一**、8 线程各 2 万次共 **160000/160000 无碰撞**
  （坐实 thread_local 种子）。
- **command**：`echo hi` → `"hi\n"`；`sh -c "echo to_stderr >&2"` 被 `2>&1` 捕获；`exit 3` 抛异常且消息含
  `exit code 3`；**`kill -TERM $$` → 退出码 143**（= 128+15，坐实信号特判 `:34`）；空命令断言。
  **一个真坑**：直接写 `echo x 1>&2` **测不到**合并——追加的 `2>&1` 在内层重定向之后，必须用 `sh -c`
  把 `>&2` 关在子 shell 里。
- **filesystem**：写读往返 14 字节；读缺失抛断言；`try_update_mtime` 存在→真、缺失→假且 `noexcept`；
  `fsync_dir` 递归成功；`safe_remove_all` 递归删净、不存在时静默；`normalize_path("a/b/../c")` →
  绝对且以 `/c` 结尾，`nullopt`/空串 → `nullopt`；`/bin/true` 可执行、非执行文件 / 缺失为假。
  **另一个真坑**：`write_file_sync(f, "hello \x00 binary")` 只写 **6** 字节——字符串字面量退化成
  `string_view(const char*)` 会**在 NUL 处截断**，必须显式给长度（首版实验因此得到 `len=6` 的假 PASS）。
- **no_ref_ptr**：`NoRefPtr{&x}` 的参数指针 == `&x`；普通 `int` 的参数指针 == `&x`（取地址）；
  `sizeof(NoRefPtr)==sizeof(void*)`。
- **gil**：Python 持 GIL 进 C++ → `before=1`、**构造后 `inside=0`**、析构后 `after=1`；用
  `py::call_guard<py::gil_scoped_release>()` 预释放进入 → `before=0/inside=0`（**不重复释放**）。
- **编译期警告（真实）**：`gil.hpp:9` 报 `'GilScopedRelease' declared with greater visibility than the
  type of its field 'release'`（`-Wattributes`）——成员是 pybind11 类型、类在默认可见性下导出，属无害告警。

## 10. 结论 / 覆盖清单

- 至此 `deep_jit/utils/` **13 个头全部被专篇覆盖**（hash/parser 第 33、env 第 38/117、lazy 第 40/80、
  exception 第 116、本篇 7 个）。
- 这层的设计主题只有一句话：**「让多次编译在多进程 / 分布式文件系统上安全、可复现、不阻塞 Python」**——
  `filesystem` 的 fsync/原子 rename、`command` 的信号特判、`uuid` 的线程局部唯一名、`json` 的只写不读、
  `gil` 的条件释放，全是为这一句话服务的。
- 两个休眠 / 欠用设施：`no_ref_ptr.hpp`（全仓无派生类型）、以及 `json.hpp` 无 `parse`（`meta.json` 只写不读）。
- 对专题主线的意义：Mega MoE 的**每一次首次 JIT 编译**（无论 SM90 的 grouped GEMM、还是 SM100 的
  mega kernel）都会走这层；跑不了的目标卡 kernel，其编译产物正是靠这层被原子写进缓存并被后续进程复用
  （第 33 篇量到命中 ~25×）。本篇不涉及任何设备指令，故 H100 与本层完全同构。

---

### 覆盖坐标（均真实核对）

| 主题 | 文件 | 行 |
|---|---|---|
| 读 / fsync / 写 | `utils/filesystem.hpp` | `read:18-22`、`fsync_file:24-39`、`fsync_dir:43-51`、`write_file_sync:73-83` |
| 目录 / 记账 / 删除 | 同上 | `make_dirs:53-59`、`try_update_mtime:65-69`、`safe_remove_all:88-108` |
| 路径 / 可执行 | 同上 | `normalize_path:110-114`、`is_executable:116-118` |
| 子进程 | `utils/command.hpp` | `call_external_command:13-37`（`2>&1 :18`、信号 `:31-35`） |
| 拼接 | `utils/str.hpp` | `join:10-26` |
| 临时名 | `utils/uuid.hpp` | `get_uuid:14-23`（thread_local 种子 `:15-20`） |
| 序列化 | `utils/json.hpp` | `value_t:18`、`object_t:17`、构造 `:22-47`、`dump_string:55-92`、`dump_to:94-128` |
| GIL | `utils/gil.hpp` | `GilScopedRelease:9-22`（条件释放 `:13-16`、禁拷贝 `:18-21`） |
| ABI 标记 | `utils/no_ref_ptr.hpp` | `NoRefPtr:6-8`；消费 `backend/cuda/kernel.hpp:24-30` |
| 消费（缓存） | `cache/disk.hpp` | `:47-48`、`:60`、`:72`、`:124`、`:130` |
| 消费（CUDA 后端） | `backend/cuda/backend.hpp` | `:64/83/93/109/115/137/145/155/157/161/167/187/210/215` |
