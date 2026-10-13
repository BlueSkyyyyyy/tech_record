# 70. DeepGEMM 的内存安全验证层：`tests/test_sanitizer.py`（compute-sanitizer 驱动）

> 前面第 40 篇拆了「判对/计时」基座 `deep_gemm/testing/`、第 41 篇拆了「怎么造操作数」
> `tests/generators.py`、第 43 篇拆了「更硬的逐位断言」`tests/utils.py`——它们回答的是
> **数值对不对**。本篇补上另一个维度的验收：**内存访问安不安全、同步用没用错**，
> 由 `tests/test_sanitizer.py` 借 NVIDIA `compute-sanitizer` 驱动。
>
> 本文不是新 kernel，而是一个**测试驱动器 + 一组开关的「为什么」**。全部 host/Python 与
> 内联 PTX，**架构无关，H100 真机可跑**；只有被它驱动的那几个测试函数本身有 SM100 门控。
> 所有行号都对着 `~/github/DeepGEMM` 真实核对过；实测见 `harness/exp_sanitizer.{py,sh,out.txt}`
> 与 `src/sanitizer_negctl.cu`。

---

## 1. 它是什么：一个驱动器，而不是一个测试用例

`tests/test_sanitizer.py` 只有 79 行，本身不含任何 GEMM 断言。它的全部工作是：
**把 `tests/` 下每个测试函数，放到 `compute-sanitizer` 下、以独立子进程重新跑一遍**。

### 1.1 子进程模板

```
# tests/test_sanitizer.py:12-26
script_dir = os.path.dirname(os.path.abspath(__file__))
test_template = """
import random
import sys
import torch

# Necessary for `generators.py`
sys.path.append('{script_dir}')

torch.manual_seed(0)
random.seed(0)

from {module_name} import {func_name}
{func_name}()
"""
```

模板每次都**重新固定随机种子**（`torch.manual_seed(0)` / `random.seed(0)`）并
**把 `tests/` 加进 `sys.path`**（否则 `from generators import ...` 会失败）。
为什么每个测试函数一定要**单开一个进程**（`:71-77` 用 `subprocess.run`）而不是在一个进程里
循环调用：

1. `compute-sanitizer` 一旦命中**精确设备错误**，在默认 `--destroy-on-device-error=context`
   下会**销毁 CUDA context**，后续所有调用都是「sticky error」而失真——必须隔离；
2. DeepGEMM 的 JIT 有全局磁盘/内存缓存、`Runtime` 是 singleton（第 48/50 篇），
   sanitizer 又改了一堆环境变量，跨用例复用同一进程状态不可控；
3. 单个用例崩溃/超时不会连累其余用例。

### 1.2 自动发现：遍历所有 `test*` 函数

```
# tests/test_sanitizer.py:40-49
files = [f for f in os.listdir(script_dir) if f.endswith('.py')]
exclude_files = ['test_sanitizer.py', 'generators.py', 'test_mega_moe.py', 'test_locality_domain.py']
funcs = [
    (module_name, name)
    for module_name in [os.path.splitext(f)[0] for f in files if f not in exclude_files]
    for name, obj in inspect.getmembers(importlib.import_module(module_name))
    if inspect.isfunction(obj) and name.startswith('test') and 'test_filter' not in name
]
```

用 `inspect.getmembers` 取「名字以 `test` 开头、且不叫 `test_filter`」的函数——
也就是**所有测试入口本身**（`test_bf16.test_gemm`、`test_layout.test_sf_layout_kernels`……）。
注意它并不理解 `@test_filter`：被门控的 `test_fp8_bhd_hdr_bhr`（`test_einsum.py:195`）、
`test_mega_gate`（`test_mega_gate.py:204/256`）都会进列表，只是运行期由
`test_filter`（`deep_gemm/testing/utils.py:11-22`）在 SM90 上打印 "Filtered by ..." 后直接返回，
子进程仍以 0 退出。

排除的四个文件各有原因：

| 文件 | 为什么排除 |
|---|---|
| `test_sanitizer.py` | 驱动器自己，不是测试 |
| `generators.py` | 操作数生成器，无 `test` 函数 |
| `test_mega_moe.py` | 端到端融合测试需要 **DeepEP + 多 rank**，且是 **SM100** 专属  |
| `test_locality_domain.py` | 需要 **MPS MLOPart** 双 device instance / 多进程（第 24 篇） |

实测（`exp_sanitizer.out.txt` §A）：本机自动发现 **31 个**测试函数，覆盖
`test_attention/test_bf16/test_einsum/test_fp8_fp4/test_hyperconnection/test_layout/
test_lazy_init/test_legacy/test_mega_gate/test_mega_mhc`（`utils.py` 无 `test*`）。

### 1.3 两个命令行参数

```
# tests/test_sanitizer.py:30-39
parser.add_argument('--funcs', type=str, default='all')     # 或 a.b,c.d 指定具体函数
parser.add_argument('--tools', type=str, default='memcheck,synccheck')
```

`--funcs all` 跑全部；`--funcs test_layout.test_sf_layout_kernels` 只跑一个；
`--tools` 默认 **memcheck + synccheck**（不含 racecheck / initcheck，见 §5 盲区）。

---

## 2. 六个环境变量：每个的「为什么」

```
# tests/test_sanitizer.py:52-58
env['CUDA_LAUNCH_BLOCKING'] = '1'
env['DG_JIT_CHECK_NO_SPILLS'] = '1'
env['DG_USE_NVIDIA_TOOLS'] = '1'
env['DG_USE_TEMP_CUBLASLT_WORKSPACE'] = '1'   # Avoid holding CUDA tensor that crashes during shutdown
env['PYTORCH_NO_CUDA_MEMORY_CACHING'] = '1'
env['TORCH_SHOW_CPP_STACKTRACES'] = '1'
```

### ① `CUDA_LAUNCH_BLOCKING=1` —— 让报错落到「刚提交的那个 kernel」

默认 CUDA launch 是异步的：一个错误可能在下一次 `synchronize` 才浮出，堆栈会指到无关代码。
`CUDA_LAUNCH_BLOCKING=1` 把每次 launch 变成同步，**配合 `--force-blocking-launches`
（见 §3）**让 sanitizer 的「by thread/block」信息能对准触发它那次 launch。

### ② `DG_JIT_CHECK_NO_SPILLS=1` —— 有寄存器溢出就直接编译失败

DeepGEMM 的 JIT（第 33/37 篇）把它翻成 nvcc 选项并**把 ptxas 警告当错误**：

```
# third-party/deep_jit/include/deep_jit/backend/cuda/options.hpp:47
.check_no_spills = env.get<bool>("JIT_CHECK_NO_SPILLS", false),
# options.hpp:117-118
if (check_no_spills.value_or(false))
    flags.emplace_back("--ptxas-options=--warn-on-spills");
# backend.hpp:120-122：扫描 nvcc 输出，命中 "spilled to local memory" 就断言失败
DJ_HOST_ASSERT(not options.check_no_spills.value_or(false) or
               not std::regex_search(compiler_output, std::regex(R"(spilled\s+to\s+local\s+memory)", std::regex::icase)),
               "PTXAS reported register spills:\n{}", compiler_output);
```

为什么放在 sanitizer 里：spill 会把寄存器流量挪到 **local memory**，既拖慢又可能把真正的
访存问题掩盖掉；在「一切从零编译」的 sanitizer 流水里加这道闸，保证被验证的 kernel
本身是干净的高性能版本。（环境变量名 `DG_JIT_CHECK_NO_SPILLS` 由 `DG_` 前缀 + `JIT_CHECK_NO_SPILLS`
拼出，`deep_jit/utils/env.hpp:66-81`；`DG_` 找不到再退回 `DJ_`。）

### ③ `DG_USE_NVIDIA_TOOLS=1` —— 跳过 kineto profiling

```
# deep_gemm/testing/bench.py:87-90
# Conflict with Nsight Systems, Nsight Compute and Compute Sanitizer
if int(os.environ.get('DG_USE_NVIDIA_TOOLS', 0)):
    return (1, ) * len(kernel_names) if is_tuple else 1
```

`bench_kineto` 靠 `torch.profiler`（kineto）抓 kernel 名与耗时，而 kineto/nsys/ncu 与
compute-sanitizer **抢同一套性能计数器/插桩**，会互相干扰。打开这个开关后计时直接返回 `1`，
测试里的 `Perf` 一律打印 `1000000 us / 0 TFLOPS`（实测见 out.txt §C），**只验正确性、不验性能**。

### ④ `DG_USE_TEMP_CUBLASLT_WORKSPACE=1` —— 躲开退出期的 `cudaErrorCudartUnloading`

`Runtime` 默认按 stream 持有一块 16 MiB 的 cuBLASLt workspace（第 40 篇）。sanitizer 下
**driver 可能先于 C++ 静态析构被卸载**，持有到那时的 workspace tensor 析构会触发
`cudaErrorCudartUnloading`：

```
# csrc/runtime/runtime.hpp:45-48
// Whether to create workspace tensor on each call instead of holding one.
// Enabled by compute-sanitizer tests, which trigger `cudaErrorCudartUnloading`
// when the workspace tensor is destructed after CUDA driver shutdown.
use_temp_cublaslt_workspace = deep_jit::get_env<int>("DG_USE_TEMP_CUBLASLT_WORKSPACE", 0) > 0;
```

打开后每次调用**现开现弃**（`runtime.hpp:69-70` 直接 `torch::empty`），不再有跨 shutdown 的持有。

### ⑤ `PYTORCH_NO_CUDA_MEMORY_CACHING=1` —— 让越界真的越界

这是**最容易被忽略、但最要命**的一个。PyTorch 的缓存分配器从**大段（segment）**里切块给张量，
一个「多写一个元素」的越界很可能**落进同一段里另一块已分配内存**，对 memcheck 而言
「地址在某个 cudaMalloc 段内」→ **判定为合法访问**。关闭缓存后每个 tensor 各自
`cudaMalloc`，越界才落到分配之外。

实测（out.txt §D）：
- 缓存开：`reserved=20971520 >> allocated=8388608`（从 20 MiB 段里切 8 MiB）；
- 缓存关：`reserved=0`（每 tensor 独立 cudaMalloc）。

再用 `src/sanitizer_negctl.cu` 的 `mask_big` / `mask_exact` 两个模式把整条因果做实（§4）：
- `mask_big`：一次 `cudaMalloc(2n)`、把 `[0,n)` 当 x、写 `x[n]`（落在同段内）→ **memcheck 0 errors**；
- `mask_exact`：`cudaMalloc(n)` 后写 `x[n]` → **memcheck 命中**「out of bounds」。

### ⑥ `TORCH_SHOW_CPP_STACKTRACES=1` —— PyTorch 算子报错时打印 C++ 栈

当某个 `torch`/扩展算子内部抛错，打印 C++ 调用栈，便于定位是哪个 op、哪一层。

---

## 3. 六个 sanitizer flag：每个的「为什么」

```
# tests/test_sanitizer.py:63-74
'/usr/local/cuda/bin/compute-sanitizer',
f'--tool={tool}',
'--target-processes=application-only',
'--destroy-on-device-error=context',
'--force-blocking-launches',
'--check-api-memory-access=no',
'--kernel-name-exclude', 'kns=nvjet',
'python', '-c', test_template.format(...)
```

| flag | 默认 | 作用与「为什么」 |
|---|---|---|
| `--tool=memcheck,synccheck` | memcheck | memcheck 查**访存合法性**（越界/非对齐/未映射）；synccheck 查**同步误用**（发散 barrier / 缺 `__syncthreads` / mbarrier 误用）。**不含 racecheck** |
| `--target-processes=application-only` | **all** | 只插桩根进程，**不追踪子进程**。DeepGEMM 的 JIT 会用 `subprocess` 调 `nvcc`、post_hook 会调 `python`（第 37 篇），这些子进程不该被插桩（否则慢且噪声大）。实测中不加会有 "Tracking kernels launched by child processes requires `--target-processes all`" 的提示 |
| `--destroy-on-device-error=context` | context | 首个精确设备错误即销毁 CUDA context，**斩断级联**（实测 `oob` 模式：1 个 invalid write 之后又跟出 sticky error + launch failure，共 3 条） |
| `--force-blocking-launches` | off | 强制阻塞 launch，和 `CUDA_LAUNCH_BLOCKING` 双保险 |
| `--check-api-memory-access=no` | **yes** | 关掉对 `cudaMemcpy`/`cudaMemset` **API 级**参数的检查，避免与内核级 memcheck 重叠刷屏 |
| `--kernel-name-exclude kns=nvjet` | — | `kns=`=**mangled name 子串匹配**（`compute-sanitizer --help`）。排除名字含 `nvjet` 的 kernel——即 **cuBLASLt 预编译核**（无行信息、且大量测试拿它做 baseline，第 27/45 篇），否则假阳性/噪声淹没真问题 |

---

## 4. H100 实测：负样本 + 真实测试

全部见 `harness/exp_sanitizer.{sh,out.txt}`（`src/sanitizer_negctl.cu` 负样本 + Python driver）。

### 4.1 负样本控制（证明「这把尺子真的能测出问题」）

`src/sanitizer_negctl.cu` 故意制造各类错误，用**与 test_sanitizer 相同的 flag** 跑：

| tool | 模式 | 结果 | 关键输出 |
|---|---|---|---|
| memcheck | `clean` | **0 errors** | — |
| memcheck | `oob` | **3** | `Invalid __global__ write of size 4 bytes` … `1 bytes after the nearest allocation at … of size 1024 bytes` |
| memcheck | `misalign` | **130** | `Invalid __shared__ write of size 8 bytes` … `Access to 0x404 is misaligned` |
| memcheck | `mask_big` | **0** | 越界落在同段内 → **被屏蔽** |
| memcheck | `mask_exact` | **3** | `Access to … is out of bounds` → **命中** |
| synccheck | `clean` | **0** | — |
| synccheck | `sync` | **64** | `Barrier error detected. Divergent thread(s) in warp.` |
| racecheck | `race` | **1** | `Race reported between Write access … and Write access` |

三处值得记的「为什么」：

- **`oob` 那句 `1 bytes after the nearest allocation of size 1024 bytes`**：memcheck 追踪的是
  `cudaMalloc` 的**请求大小**（`n*4=1024`），不是它向下取整后的物理页——所以精确分配下的越界必被抓。
- **`sync` 必须先让 ptxas 无法优化**：直接写 `if (tid<16) __syncthreads();`，ptxas 会 if-convert 成
  「`tid>15` 的线程提前 `EXIT`，剩下 16 个线程无分支地执行 `BAR.SYNC`」——**反而没有发散**（实测 SASS：
  `@P0 EXIT` 后跟着无谓词的 `BAR.SYNC.DEFER_BLOCKING`）。只有让**两个分支都有副作用**，barrier 才会真的落在
  发散的路径上、被 synccheck 抓住。（一个「读源码看不出来、只有看 SASS/真机才现」的坑。）
- **`race` 用 `racecheck` 才报**：`synccheck` 只查同步指令的误用，**不查共享内存数据竞争**；
  数据竞争是 `racecheck` 的职责。而 `test_sanitizer.py` 默认只跑 memcheck+synccheck（§5 盲区）。

### 4.2 真实 `test_sanitizer.py` 驱动（H100 可跑的两个测试函数）

`--funcs` 选中本机 SM90 真能跑的：
`test_hyperconnection.test_hc_prenorm_gemm`（SM90 TF32 prenorm，第 16 篇）与
`test_layout.test_sf_layout_kernels`（SF 转置/打包，第 25 篇）：

| 测试函数 | memcheck | synccheck |
|---|---|---|
| `test_hyperconnection.test_hc_prenorm_gemm` | `ERROR SUMMARY: 0 errors` (rc=0) | `0 errors` (rc=0) |
| `test_layout.test_sf_layout_kernels` | `0 errors` (rc=0) | `0 errors` (rc=0) |

即：**这两个函数的全部 SM90 路径在真机 memcheck + synccheck 下零告警**。
（`DG_USE_NVIDIA_TOOLS=1` 生效，测试内所有 `bench_kineto` 只打印 `1000000 us / 0 TFLOPS`。）

---

## 5. 与前面各篇的关系，以及这层验收的盲区

**测试栈的四层**（本篇补最外一层）：

1. `deep_gemm/testing/`（第 40 篇）：`calc_diff` 判据 + `bench_kineto` 计时；
2. `tests/generators.py`（第 41 篇）：操作数/布局生成；
3. `tests/utils.py`（第 43 篇）：比 relerr 更硬的**逐位** oracle；
4. **`tests/test_sanitizer.py`（本篇）**：**内存安全 + 同步正确性**。

前三层问「结果对不对」，第四层问「**没读越界、没错用 barrier**」——它不产生数值断言，
却能在 kernel 只在边界形状下越界、或 barrier 只在某种 warp 分工下发散时，把 bug 提前抓住。

**已知盲区（值得注意）**：

- **不跑 racecheck**：共享内存数据竞争没有 CI 覆盖（实测 `synccheck` 对数据竞争 0 报告，
  换 `racecheck` 才报 1）。要用 `--tools racecheck` 手动补。
- **不跑 initcheck**：全局内存「读到未初始化」也没覆盖。
- **排除 `test_mega_moe.py` / `test_locality_domain.py`**：前者是多 rank + SM100 的融合测试，
  后者需 MPS MLOPart；两者都不在 sanitizer 覆盖面内。
- **`kns=nvjet` 排除 cuBLASLt**：所以「cuBLASLt 分支」本身不进 sanitizer（那是 NVIDIA 的核，
  不该由 DeepGEMM 的 CI 负责）。
- sanitizer 强制 `--force-blocking-launches`，与 **PDL（第 52 篇）**的异步语义天然冲突：
  这里验的是内存安全，不是 launch 重叠，两者不要混为一谈。

---

## 6. 复现

```
# 负样本控制 + 静态核对 + 真实测试 + 缓存分配器/旁路实证（约 4 分钟）
bash code/mega-moe/harness/exp_sanitizer.sh > code/mega-moe/harness/exp_sanitizer.out.txt

# 直接驱动真实测试函数（在 ~/github/DeepGEMM/tests 下）
python test_sanitizer.py --funcs test_hyperconnection.test_hc_prenorm_gemm --tools memcheck,synccheck
python test_sanitizer.py --funcs all --tools memcheck,synccheck            # 全量（含被门控的用例）
```

## 7. 链接

- DeepGEMM 仓库：<https://github.com/deepseek-ai/DeepGEMM>（`tests/test_sanitizer.py`）
- NVIDIA Compute Sanitizer 官方文档（`--tool` / `--kernel-name-exclude kns=` 等语义）：
  <https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html>
