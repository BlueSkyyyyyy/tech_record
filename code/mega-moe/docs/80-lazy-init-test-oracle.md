# DeepGEMM 顶层运行时的验收测试即正确性规范（`tests/test_lazy_init.py`，H100 实测）

> 本文是「DeepSeek Mega MoE 前向 / 反向算子实现解析」专题的 P70 增量（第 80 篇）。
> 第 23 篇（P23-1）拆了 DeepJIT 的编译 / 缓存 / 加载 / 发射管线；第 40 篇（P40-1，`docs/50`）
> 拆了顶层 `Runtime` 单例的 **lazy init / shutdown / cuBLASLt handle 与 per-stream workspace 的生命周期**，
> 并在 §6.5 只用一句「直接跑上游测试：Ran 3 tests / OK」引用了它。
> 本篇把 `tests/test_lazy_init.py`（199 行）单独当**可执行规格**拆透：它定义了
> 「lazy init / fork safety / 原生退出清理」到底**算什么对**，也是第 40 篇那些
> 「handle 恰好 create/destroy 各 1 次」结论的**判据出处**。
>
> 与第 67/79 篇的 mega-mHC / prenorm 测试不同，这份测试**与 GPU 架构完全无关**：
> 它测的是 host 侧单例的对象生命周期与宿主进程的退出行为，**H100（sm90）上整份可跑**。
> 源码坐标：`~/github/DeepGEMM`，公开版 commit `057ca59`（v2.8.1）。

## 1. 它为什么是一种「规范」而不是一个「单元测试」

`tests/test_lazy_init.py` 只有一个 `unittest` 类（`:56 TestNativeRuntimeShutdown`），
带一个 Linux 门控 `@unittest.skipUnless(sys.platform.startswith('linux'), 'LD_PRELOAD requires Linux')`
（`:55`）——因为它的核心手段是 **`LD_PRELOAD` 探针**。它有三个用例：

| 用例 | 行 | 测什么 |
|---|---|---|
| `test_native_shutdown` | `:73` | 自建 cuBLASLt handle 的 **create/destroy 次数**与退出期顺序（6 个用例矩阵） |
| `test_real_cublaslt_shutdown` | `:116` | **真实** cuBLASLt 在原生退出期的 **use-after-free** 是否被规避 |
| `test_cuda_workspaces_are_released` | `:141` | per-stream workspace 在退出时**确实被释放**（2×2 矩阵） |

`setUpClass`（`:57-71`）先把探针编译成 `.so`（`c++ -shared -fPIC -std=c++17`，`:70-71`），
但它先做两道 **skip 闸门**：没有本地构建的扩展 `deep_gemm/_C*.so`（`:60-61`）、
或没有 C++ 编译器（`:62-64`）就整类跳过。这保证它测的**永远是当前源码树**，而不是 pip 装的另一份。

`__main__`（`:187-199`）做的事很有信息量：

```
test_lazy_init.py:189: parser.add_argument('--num-processes', type=int, default=8, ...)
test_lazy_init.py:192: procs = [mp.Process(target=main, args=(i, ), ) for i in range(args.num_processes)]
test_lazy_init.py:197: assert p.exitcode == 0, f'Lazy initialization worker exited with code {p.exitcode}'
test_lazy_init.py:199: unittest.main(argv=[sys.argv[0], *unittest_args])
```

即**先用 8 个 `mp.Process` 各自 `torch.cuda.set_device(local_rank)`（`main`，`:18-19`）跑一遍、
断言全部 `exit 0`，再跑 unittest**。前面那段与 DeepGEMM 逻辑无关，测的是
**fork 后 CUDA 上下文 + DeepJIT 单例的 fork safety**——一个在 Python 多进程训练栈里
（DDP / DeepEP 起 mp worker）必须成立的前提。所以这份文件回答的是：
**「进程从 fork、到首次用 GEMM、再到退出」整条生命线上，DeepGEMM 顶层单例是否无泄漏、无崩溃、可复现。**

## 2. 探针：用 `LD_PRELOAD` 把 cuBLASLt 换成一枚可计数的假库

真实 cuBLASLt 的 create/destroy 是黑盒，没法在 Python 侧数次数。测试改成
**在动态链接层截胡符号**：探针导出同名 `cublasLtCreate` / `cublasLtDestroy`（`extern "C"`，
`:38` / `:45`），`LD_PRELOAD` 预加载后 DeepGEMM 调到的就是探针版本。

```
test_lazy_init.py:27: static uint32_t live_handles = 0, handle_token = 0;
test_lazy_init.py:28: static bool backend_dead = false;
test_lazy_init.py:29: static void verify_cleanup() {
test_lazy_init.py:30:     if (live_handles != 0) std::_Exit(92);
test_lazy_init.py:31:     std::fputs("DG_PROBE_CLEAN\n", stderr);
test_lazy_init.py:33: } // Registered before the extension's static destructors, so this runs last.
test_lazy_init.py:34: __attribute__((constructor)) static void initialize_probe() {
test_lazy_init.py:35:     std::atexit(verify_cleanup);
test_lazy_init.py:36: }
test_lazy_init.py:37: static void stop_backend() { backend_dead = true; }
test_lazy_init.py:38: extern "C" uint32_t cublasLtCreate(void** handle) {
test_lazy_init.py:39:     std::atexit(stop_backend);
test_lazy_init.py:40:     *handle = &handle_token;               // 返回一个「本探针认得」的假 handle
test_lazy_init.py:41:     ++live_handles;
test_lazy_init.py:42:     std::fputs("DG_PROBE_CREATE\n", stderr);
test_lazy_init.py:43:     return 0;
test_lazy_init.py:44: }
test_lazy_init.py:45: extern "C" uint32_t cublasLtDestroy(void* handle) {
test_lazy_init.py:46:     if (backend_dead) std::_Exit(91);       // 退出期才发生的销毁 = use-after-free
test_lazy_init.py:47:     if (handle != &handle_token or live_handles != 1) std::_Exit(93);
test_lazy_init.py:48:     --live_handles;
test_lazy_init.py:49:     std::fputs("DG_PROBE_DESTROY\n", stderr);
test_lazy_init.py:50:     return 0;
test_lazy_init.py:51: }
```

三个 magic exit code 就是这条规范的**三把尺子**：

| code | 触发 | 判定的失败 |
|---|---|---|
| `91` | `backend_dead` 后仍调 `cublasLtDestroy`（`:46`） | **退出期 use-after-free**：库已在退出回调里把自己关了 |
| `92` | `verify_cleanup` 时 `live_handles != 0`（`:30`） | **handle 泄漏**：建了没销 |
| `93` | 销毁了一个陌生 handle、或 `live_handles != 1`（`:47`） | **重复销毁 / 销毁错对象** |

关键的两处时序设计：

- `initialize_probe` 是 `__attribute__((constructor))`（`:34`）：动态库加载即注册
  `std::atexit(verify_cleanup)`。注释（`:33`）点明「在扩展的静态析构器之前注册，所以它最后跑」——
  **它检查的是「整个退出序列走完后」的最终状态**，而不是中途。
- `cublasLtCreate` 里再 `std::atexit(stop_backend)`（`:39`）：第一次建 handle 时给进程挂一个
  「后端已死」标志，于是后续任何**发生在它之后**的 destroy 都会踩 `_Exit(91)`。这正是
  `test_real_cublaslt_shutdown` 想暴露的真实 bug（第 40 篇 §3 的 cuBLASLt 13.1 退出竞态）。

探针把「对象生命周期」变成「stderr 上的事件流 + 退出码」，于是在 Python 里可以精确断言。

## 3. 6 用例矩阵：handle 的 create/destroy 次数

`test_native_shutdown`（`:73-114`）把「是否触发 Runtime 构造」「是否显式 shutdown」「handle 归谁管」
三个变量组合成 6 个用例：

```
test_lazy_init.py:74: use_runtime   = 'assert deep_gemm.get_tc_util() == deep_gemm.get_tc_util() == 100\n'
test_lazy_init.py:75-84:  close_runtime = shutdown(); shutdown(); 再取 get_tc_util() 必须抛 RuntimeError
test_lazy_init.py:85-92:  cases = (
    ('unused',           '0', '',                          0),
    ('owned',            '0', use_runtime,                 1),
    ('managed',          '1', use_runtime,                 0),
    ('explicit',         '0', use_runtime + close_runtime, 1),
    ('managed_explicit', '1', use_runtime + close_runtime, 0),
    ('unused_explicit',  '0', close_runtime,               0),
)
```

每个用例是一个**独立子进程**（`:101-108`），环境三件事：`CUDA_VISIBLE_DEVICES=''`（`:104`，
**故意不给 GPU**）、`DG_USE_PYTORCH_CUBLASLT_HANDLE=managed`（`:105`）、`LD_PRELOAD=探针`（`:106`）。
子进程 preamble（`:93-100`）先 `sys.path.insert(repo)` 再 `import deep_gemm`，并断言
`deep_gemm._C.__file__.parent == repo/deep_gemm`（`:99`）——**坐实导入的是本地构建**。
断言三连（`:110-114`）：`returncode == 0`、`CREATE == expected`、`DESTROY == expected`、`CLEAN == 1`。

这张矩阵的语义：

| 用例 | 构造 Runtime | 显式 shutdown | handle 归属 | create/destroy |
|---|---|---|---|---|
| `unused` | 否 | 否 | 自建 | 0 / 0 —— 什么都不做就不该建库 |
| `owned` | 是 | 否 | 自建 | 1 / 1 —— 建了，退出时由 `~Runtime`/atexit 销 |
| `managed` | 是 | 否 | PyTorch 托管 | 0 / 0 —— 托管就不该碰 cuBLASLt |
| `explicit` | 是 | 是 | 自建 | 1 / 1 —— 显式 shutdown 提前销，退出不再销 |
| `managed_explicit` | 是 | 是 | 托管 | 0 / 0 |
| `unused_explicit` | 否 | 是 | 自建 | 0 / 0 —— shutdown 在未构造时是安全的 no-op |

**为什么可以不给 GPU**：探针的 `cublasLtCreate` 直接返回一个假 handle（`:40`），
绕开了真实 CUDA 初始化。所以这 6 个用例纯粹在测 host 侧的对象计数，与设备无关——
这也是它 fork 8 进程也能稳过（§6）的原因。`managed` 行揭示了一个非显然点：
`DG_USE_PYTORCH_CUBLASLT_HANDLE > 0` 时**连接口都不建**（`runtime.hpp:44/51`），
而不是「建了再换用托管」。

## 4. `test_real_cublaslt_shutdown`：真实库的退出竞态

第二个用例（`:116-139`）不再用探针，而是跑**真实** cuBLASLt，故意把退出期的
use-after-free 逼出来：

```
test_lazy_init.py:128: env['DG_USE_PYTORCH_CUBLASLT_HANDLE'] = '0'
test_lazy_init.py:129: # Expose the real cuBLASLt 13.1 use-after-free during native shutdown.
test_lazy_init.py:130: # Freed storage must not retain values that hide the invalid second destroy.
test_lazy_init.py:131: env['MALLOC_PERTURB_'] = '165'
test_lazy_init.py:132: env['GLIBC_TUNABLES'] = env.get(...) + ':glibc.malloc.tcache_count=0'
```

子进程只做两件事（`:117-124`）：`assert deep_gemm.get_tc_util() == 100`（触发建库）、
打印 `DG_REAL_CUBLASLT_CREATED`，然后**直接退出**，让退出序列去走 destroy。
`MALLOC_PERTURB_=165`（`:131`）让 glibc 在 `free` 时把内存填成确定字节、
`glibc.malloc.tcache_count=0`（`:132`）关掉线程缓存——注释（`:130`）说得很清楚：
**被释放的存储不允许残留能「掩盖」第二次非法销毁的值**，否则 use-after-free 会静默通过。
这就是「用确定性的堆扰动把竞态变成可复现的崩溃」的标准手法。

对照第 40 篇：`Runtime` 的 `~Runtime` 会 `cublasLtDestroy`（`runtime.hpp:54-56`），
而 `deep_gemm/__init__.py:125` 注册了 `atexit.register(shutdown)`，**让 DeepGEMM 在
Python/PyTorch/CUDA 尚存时先把自己的 handle 释放掉**，避免原生 cuBLASLt 的退出回调
与 C++ 静态析构顺序打架。这个用例就是「这条设计是否真的生效」的验收。

## 5. `test_cuda_workspaces_are_released`：workspace 的退出释放

第三个用例（`:141-184`）测 per-stream cuBLASLt workspace：子进程开**两个 stream**、
各跑 2 次 `cublaslt_gemm_nt`（`:160-165`）——正是第 27 篇讲的「workspace 每 stream 一份，
所以并发 stream 不能共享」。跑完记录 `torch.cuda.memory_allocated()`（`:169`），
再注册一个**在 DeepGEMM 的 shutdown 之后**才跑的 `atexit`（`:154-155` 注释）：

```
test_lazy_init.py:150:  def check_cleanup():
test_lazy_init.py:151:      if os.environ['DG_USE_TEMP_CUBLASLT_WORKSPACE'] == '0':
test_lazy_init.py:152:          assert allocated[0] - torch.cuda.memory_allocated() >= 2 * 16 * 1024 * 1024
test_lazy_init.py:153:      print('DG_CUDA_CLEAN', flush=True)
```

即：默认路径下（`temp == '0'`，常驻 workspace），退出时相对峰值应**至少释放 2×16 MiB**
（两个 stream 各一份，对应 `Runtime::kCublasLtWorkspaceSize = 16 MiB`）。
外层是 `managed × temporary` 的 2×2 矩阵（`:171-172`），断言 `returncode==0` 且输出含
`DG_CUDA_CLEAN`（`:181/184`）。这验证的正是第 40 篇「显存账 16/16/32 MiB」那条结论的**退出侧**。

## 6. H100 实测

实验 `harness/exp_lazy_init_test_oracle.py`（+`.sh` / `.out.txt`）在 `kernel_lab`（H100 sm90）上跑，
**7 PASS / 0 FAIL**。四部分：

**Part C —— 原样跑真实测试模块**（与官方 `__main__` 同款命令 `--num-processes 8`）：

```
Ran 3 tests in 25.142s
OK
```

即 8 个 mp worker 全部 `exit 0`，三个用例全过。**注意 unittest 的输出在 stderr**（这是 unittest
的默认行为），所以判据要把 stdout+stderr 合并看——一个容易误判的小坑。

**Part D —— 独立复刻 6 用例矩阵**：把测试源码里的 `_PROBE_SOURCE` **逐字取出**（断言它确实是
`test_lazy_init.py` 源码子串），自己编译成 `.so`，用同一套 `LD_PRELOAD` 跑 6 个用例、逐用例数事件：

| 用例 | managed | get_tc_util | shutdown | 期望 create | 实测 create | 实测 destroy | clean | 判定 |
|---|---|---|---|---|---|---|---|---|
| unused | 0 | – | – | 0 | 0 | 0 | 1 | PASS |
| owned | 0 | Y | – | 1 | 1 | 1 | 1 | PASS |
| managed | 1 | Y | – | 0 | 0 | 0 | 1 | PASS |
| explicit | 0 | Y | Y | 1 | 1 | 1 | 1 | PASS |
| managed_explicit | 1 | Y | Y | 0 | 0 | 0 | 1 | PASS |
| unused_explicit | 0 | Y | Y | 0 | 0 | 0 | 1 | PASS |

这与测试自带断言**互为独立佐证**（我们用自己的子进程驱动、自己的计数逻辑）。

**Part E —— 利齿**：

- **E1** `shutdown` 幂等且终态：`get_tc_util()` 先返 100，两次 `deep_gemm.shutdown()` 都不报错，
  之后 `get_tc_util()` 抛 `RuntimeError`（实测 `rc=0, out='RAISED'`）。异常正文直接带源码座标
  `third-party/deep_jit/include/deep_jit/utils/lazy.hpp:27: factory != nullptr: lazy object must be
  initialized before use`——这就是第 40 篇「`shutdown` 一行 null-factory 赋值、`init` 不可恢复」的运行时证据。
  **踩坑**：E1 必须真给 GPU。若像 6 用例矩阵那样设 `CUDA_VISIBLE_DEVICES=''`，没有探针假造
  `cublasLtCreate` 时真库会报 `the library was not initialized`（`runtime.hpp:51`）。
  矩阵用例能空设备跑，恰恰是因为探针替它造了假库——这一点反过来证明了 §3 的设计意图。
- **E2** `DG_USE_PYTORCH_CUBLASLT_HANDLE` 是**整数**而不是布尔（与第 48 篇的开关面对照）：
  `'0'→create 1`、`'1'/'2'→create 0`、`'-1'→create 1`（`> 0` 为真），全部符合；
  而 `'true'/'false'/'abc'` 直接 `rc=1`（非整数被 `deep_jit::get_env<int>` 的 `DJ_PANIC` 拒）。

**Part F —— fork safety**：直接调用真实模块的 `tl.main` 起 8 个 `mp.Process`，
`rc=0` 且 `ALL_WORKERS_OK`——fork safety 成立。

## 7. 与第 40 篇的关系、小结

| | 第 40 篇（P40-1，`docs/50`） | 本篇（P70-1） |
|---|---|---|
| 对象 | `Runtime` / `LazyInit` 的**实现** | `tests/test_lazy_init.py` 的**规范** |
| 回答 | 「生命周期怎么实现、为什么这样」 | 「生命周期算什么对、判据怎么编码」 |
| 手段 | 自建探针 + 显存账 + 手动复刻 | 逐字复用上游 `LD_PRELOAD` 探针 + 跑真实测试体 |
| 硬结论 | null factory = 终态；handle 恰好 1/1；workspace 每 stream 一份 | 6 用例 create/destroy 次数 = 规格；91/92/93 三把尺子；env 是 int |

**三个可复用结论**：

1. **规范用「退出码 + stderr 事件流」把对象生命周期变成可断言的东西**。`LD_PRELOAD` 截胡
   `cublasLtCreate/Destroy`，三个 magic exit code（91/92/93）分别锁死 use-after-free / 泄漏 / 重复销毁。
   看这三个数就能反推「这个库最怕什么」。
2. **判据会编码实现事实**：`CUDA_VISIBLE_DEVICES=''` 能跑，是因为探针 `cublasLtCreate` 返回假 handle
   （`:40`）——这恰好说明这 6 个用例测的是**纯 host 对象计数**；而 E1 必须真给 GPU，
   因为那里没有探针。`unused_explicit` 期望 0/0 则编码了「`shutdown` 在未构造时是安全的 no-op」。
3. **`MALLOC_PERTURB_=165` + 关 tcache 是「把退出竞态变成可复现崩溃」的钥匙**：被 free 的存储
   必须留确定字节，否则第二次非法 `cublasLtDestroy` 会静默通过。这与第 60 篇
   （compute-sanitizer）的「缓存分配器会屏蔽越界」是同一种「让隐藏 bug 现形」的思路。

**边界**：本篇**没有任何 SM100 专属内容**——`Runtime` 的 handle / workspace / 生命周期与 GPU 架构无关，
H100（sm90）上整份测试可跑。它与第 23 篇（DeepJIT 管线）、第 40 篇（生命周期实现）
合起来，把「从 fork、到首次 GEMM 触发 JIT、再到退出清理」这条顶层运行时链路讲完。

**本篇实测脚本**：`harness/exp_lazy_init_test_oracle.py` / `.sh` / `.out.txt`
（H100 真机，原样跑真实测试模块 + 独立复刻 6 用例探针矩阵，7 PASS / 0 FAIL）。
