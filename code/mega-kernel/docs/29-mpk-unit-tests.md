# MPK 「单 kernel 单测」层源码走读：`tests/runtime_python/` 与 `runtime_kernel_wrapper*.cu`

> 本文补齐 `docs/28`（P114-1）**§8 明留的尾巴**：「**未覆盖 `tests/runtime_python/` 的非 test_mode
> 部分**（`test_rmsnorm.py` 等走 `runtime_kernel` pybind 直接调 device kernel，不经 `PersistentKernel`
> ——那是『单 kernel 单测』，与 megakernel 端到端是两层，其 wrapper `runtime_kernel_wrapper.cu` 未走读）」。
>
> 快照：`mirage-project/mirage` 分支 `mpk` commit `f9eb70c`（2026-10-05，与 `docs/06`–`docs/28` 同一快照）。
> **纯源码走读（资料分析），本轮无新实验**——本层是 host 手写 `<<<>>>` 直接调 device task 的回归网，
> 在目标卡 H100/sm90 上并非必须运行；结论均基于核对过的快照行号，不采信其性能数。

## 0. 定位：megakernel 编译栈上的「第三层验证面」

本专题此前把 MPK 的验证面分了两层：

- **device task 层**（`docs/07`/`docs/24`）：一个 task 内部是一段怎样的 device 代码（`linear_hopper`、
  `rmsnorm_hopper`、`allreduce` …就是 `tasks/hopper/*.cuh` / `tasks/blackwell/*.cuh` 里的模板函数）。
- **megakernel 端到端层**（`docs/28` 的 `test_mode`）：`PersistentKernel` 把整张图编成一张常驻
  megakernel、单次 launch、「跑一次就退」，和 PyTorch 端到端参考比。

本文打开夹在中间的**第三层**：`tests/runtime_python/`（非 `test_mode` 部分）——它把**一个 device task
当成一整颗 kernel** 直接 host launch，和一个逐张量构造的 PyTorch/numpy 参考比数值。这一层**不经过
`PersistentKernel`、不经过 event/tGraph、不经过 worker/scheduler**。

论文（`~/github/ai-infra-book/experiments/ch05/05-08/sources/mpk-v2.txt`）**从未提及**这一层：它是
「实现质量 / 数值回归」层面的东西，不进 §6 评估（同 `docs/28` §8 对 `test_mode` 的判断）。它存在的
意义是：**在把 task 塞进常驻 megakernel 之前，先证明这个 task 的 device 实现本身算得对**。

## 1. 目录与构建单元：五套彼此独立的 pybind extension

`tests/runtime_python/` 不是一个包，而是**一组各自带 `setup.py`、各自编译成一个 `.so` 的测试目录**：

| 构建单元 | 主源文件 | extension 名 | 目标 arch / 宏 |
|---|---|---|---|
| 主（默认） | `runtime_kernel_wrapper.cu`（1895 行） | `runtime_kernel` | `sm_80` + `sm_90a` + `sm_100`；`-DMIRAGE_UNIT_TEST`、`-DMIRAGE_GRACE_HOPPER` |
| Hopper | `hopper/runtime_kernel_wrapper_hopper.cu`（1405 行） | `runtime_kernel_hopper` | `sm_90a`；`-DMPK_TARGET_CC=90` |
| Hopper-MoE | `hopper-moe/runtime_kernel_wrapper_moe_hopper.cu`（492 行） | `runtime_kernel_moe_hopper` | `sm_90a` |
| Blackwell 家族 | `blackwell/<子目录>/runtime_kernel_wrapper*.cu` | 12 个（见 §2.3） | `sm_100` |
| CUTE-Hopper | `cute/hopper/gemm.cu`（556 行） | `runtime_kernel_cute_hopper` | `sm_90a` |

主 `setup.py`（`tests/runtime_python/setup.py:23-51`）语气很清楚：`sources=` 只编
`runtime_kernel_wrapper.cu` 一个文件，`depends=` 把 `include/mirage/persistent_kernel/*.cuh`、
`deps/cutlass/include/*.cuh`、`speculative_decoding/*.cuh` 列为「改了才重编」的依赖；`extra_compile_args.nvcc`
同时下 `-gencode=arch=compute_80,code=sm_80` / `code=sm_90a` / `code=sm_100` 三档（`:41-43`），并定义
`-DMIRAGE_UNIT_TEST`（`:44`）与 `-DMIRAGE_GRACE_HOPPER`（`:45`）。

`MIRAGE_UNIT_TEST` 是整个仓库里**只在单测层出现**的宏（全仓仅两处引用）：

- `tests/runtime_python/setup.py:44` 定义它；
- `include/mirage/persistent_kernel/tasks/ampere/moe_linear.cuh:256` 消费它。

被消费的那段（`moe_linear.cuh:256-263`）注释写得很直白——*"Below advance don't need in real MPK."*：
单测用一个二维 grid，用 `blockIdx.y` 当「**真实的 expert 权重块 id**」手动把 `d_weight`/`d_residual`/
`d_output` 指针推进到该 expert 的偏移；真实 MPK 里这层偏移由运行层/描述符负责，task 代码看不见 blockIdx.y。
**这是本层「把 task 当整核用」这一事实在 device 代码里留下的唯一痕迹**。

`DEPRECATED_TESTS` 是另一个纯单测开关：主 wrapper 里 7 处 `#ifdef DEPRECATED_TESTS`
（`:30/:342/:801/:1147/:1228/:1487/:1596`），包住 `single_batch_gqa`、旧 `paged_attention`、
`prompt_lookup`、`verify`、`argmax` 等旧 wrapper。**没有任何 `setup.py` 定义它**，故这些函数默认
根本不参与编译（见 §2.2 的后果）。

## 2. pybind 接线现状：大面积「导入即失败」（headline）

### 2.1 主 wrapper：注册 3 个，测试调用 14 个

`runtime_kernel_wrapper.cu` 的 `PYBIND11_MODULE`（`:1844-1895`）几乎整段被注释，**实际注册只有 3 个**：

```
m.def("moe_linear", ...)                    // :1848
m.def("multitoken_paged_attention", ...)    // :1887
m.def("sampling_from_logits", ...)          // :1890
```

而被注释掉的注册（`:1845-1894`）覆盖 `prompt_lookup`/`embedding`/`linear`/`argmax`/`verify`/
`norm_linear`/`silu_mul_linear`/`single_batch_decoding`/`single_batch_extend`/`paged_attention`/
`rms_norm`/`rope`。与之对照，主目录测试文件实际调用的符号有 14 个（按 `runtime_kernel.<sym>`
统计）：

| 测试调用的符号 | 是否注册 | 症状 |
|---|---|---|
| `moe_linear` | ✅ | `test_moe_linear.py` 可用 |
| `multitoken_paged_attention` | ✅ | `test_multitoken_paged_attention.py` 可用 |
| `sampling_from_logits` | ✅ | `test_sampling.py` 可用 |
| `from runtime_kernel import rms_norm` | ❌（`:1893` 注释） | `test_rmsnorm.py` **导入即 `ImportError`** |
| `from runtime_kernel import norm_linear` | ❌（`:1851`） | `test_norm_linear.py`/`test_alignment_norm_linear.py` 导入失败 |
| `from runtime_kernel import rope` | ❌（`:1894`） | `test_rotary_embedding.py` 导入失败 |
| `linear` | ❌（`:1847`） | `test_linear.py` 调用点 `AttributeError` |
| `argmax` | ❌（`:1849`） | `test_argmax.py` |
| `embedding` | ❌（`:1846`） | `test_embedding.py` |
| `silu_mul_linear` | ❌（`:1852`） | `test_silu_mul_linear.py` |
| `paged_attention` | ❌（`:1886`） | `test_paged_attention.py` |
| `prompt_lookup` | ❌（`:1845`，且在 `DEPRECATED_TESTS` 内） | `test_prompt_lookup.py` |
| `verify` | ❌（`:1850`，且在 `DEPRECATED_TESTS` 内） | `test_verify.py` |
| `single_batch_decoding` / `single_batch_extend` / `single_batch_gqa` | ❌（`:1853-1885` 注释 / `DEPRECATED_TESTS`） | `test_decoding_*`、`test_extend_w_norm.py` |

即：**19 个主目录测试里只有 3 个（`test_moe_linear`/`test_multitoken_paged_attention`/`test_sampling`）
与当前 wrapper 对得上**，其余 16 个在收集/调用阶段就会失败。

### 2.2 Hopper wrapper：注册 1 个，测试调用 3 个

`hopper/runtime_kernel_wrapper_hopper.cu` 的 `PYBIND11_MODULE`（`:1397-1402`）只注册
`linear_swapAB`（`:1400`）；`linear`、`norm_linear`、`multitoken_paged_attention` 的注册被注释
（`:1399/:1401/:1402`）。而 hopper 目录测试实际调用 `linear`、`norm_linear`、`linear_swapAB`
三个符号，故 `test_norm_linear_hopper.py`/`test_linear_hopper.py` 的非 swapAB 分支同样导入失配。
另有 `hopper/attention_tail_wrapper.cu`（91 行）**不通过 pybind**，而是由
`test_attention_tail_hopper.py` 用 `ctypes` + 临时 `nvcc` 共享库调（该测试自己起子进程编译）。

### 2.3 Hopper-MoE 与 Blackwell 家族：符号基本对齐

- `hopper-moe/runtime_kernel_wrapper_moe_hopper.cu:486-491` 注册 `moe_w13_linear_sm90`、
  `moe_w2_linear_sm90` 两个符号，与 `test_moe_w13_linear_hopper.py`/`test_moe_w2_linear_hopper.py`
  调用一致。
- Blackwell 家族**每个子目录自带 `setup.py`**（12 个：`sm100_moe`/`sm100_moe_sigmoid`/
  `sm100_fp8_moe`/`sm100_linear`/`sm100_linear_fp8`/`sm100_quantize_fp8`/`sm100_paged_attention`/
  `sm100_mla`（另含 `setup_mla_decode/mtp/prefill.py`）/`sm100_mla_prefill_tp8` …），extension 名各异
  （`runtime_kernel_blackwell`、`runtime_kernel_blackwell_mla`、`runtime_kernel_fp8_moe`、
  `runtime_kernel_moe_sigmoid` …）。这些 wrapper 里注册的符号（`moe_w13_linear_sm100`、
  `topk_softmax_sm100`、`mla_attention`、`quantize_fp8_sm100`、`silu_mul`、`mul_sum_add_sm100`、
  `supported_dense_gemm_shapes` …）与对应测试的调用**逐条对得上**——它们是**持续维护**的
  （Blackwell 是当前主推代际，见 `docs/17`/`docs/21`）。

**小结**：同一层里「通用/历史测试（主目录 + hopper）」大面积失配、「当前代际测试（blackwell 家族 +
hopper-moe）」整齐对齐。这正是本专题反复出现的「**同一套惯用法/验证实现了两遍**」母题在测试层的
翻版——主 wrapper 保留了少数仍在用的符号（`moe_linear`/`multitoken_paged_attention`/
`sampling_from_logits`），其余随代际迁移被**注释而非删除**。

## 3. 单测形态解剖：一个 wrapper 如何把 task 当整核 launch

以主 wrapper 的 `moe_linear` 为例（这是本层最典型的形态）。

**(a) 外层 `__global__` 只做一次转发。** `moe_linear_kernel_wrapper<T,...>`（`:1318-1339`）体内
唯一一句话就是调用 device task：

```
moe_linear_kernel<T, BATCH_SIZE, OUTPUT_SIZE, OUTPUT_STRIDE, REDUCTION_SIZE,
                  NUM_EXPERTS, NUM_TOPK, EXPERT_STRIDE, NOBIAS>(
        input_ptr, weight_ptr, residual_ptr, expert_routing_ptr, expert_mask_ptr,
        output_ptr, blockIdx.x);        // 把 blockIdx.x 当 worker id 传给 task
```

注意最后那个实参 `blockIdx.x`——**真实 megakernel 里「这个 CTA 是几号 worker」由运行层填**（`docs/06`
§3 的 worker 循环），单测层直接拿 blockIdx 顶上。

**(b) `launch_*` 手算 grid/block/smem，并设 attribute。** `launch_moe_linear<T,...>`（`:1348-1384`）：

```
static_assert(OUTPUT_STRIDE % 64 == 0);
dim3 grid_dim(EXPERT_STRIDE, OUTPUT_STRIDE / 64, 1);
dim3 block_dim(128, 1, 1);
size_t smem_size = 96 * 1024;
cudaFuncSetAttribute(..., cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
moe_linear_kernel_wrapper<...><<<grid_dim, block_dim, smem_size>>>(...);
```

grid 的**第二维** `OUTPUT_STRIDE/64` 就是 §1 说的「expert 内偏移」维度（配合 `MIRAGE_UNIT_TEST` 的
`blockIdx.y` 指针推进）；`96 KiB` 动态 SMEM、128 线程/block 都是**手写常量**——在 megakernel 里这些
是**编译器/运行层按 task 图算出来的**（`docs/11`/`docs/12` 的 worker 数据面/SMEM 预算）。

**(c) host 侧按形状派发模板实例。** `moe_linear()`（`:1411-1440`）用两层宏展开
（`MOE_LINEAR_DISPATCH_BATCH_SIZE`/`..._OUTPUT_STRIDE`，`:1388-1409`）按 `input.size(0)` 与
`output.size(2)` 选模板参数，并**硬编码** `2048/128/8/5`（reduction size / experts / topk / expert
stride，`:1390`）；不支持的形状只 `printf` 提示（`:1405/:1432`）。末尾 `cudaDeviceSynchronize()`
同步并检查错误（`:1436-1439`）——**单测是一次同步 launch，没有 event、没有流**。

**(d) 另外两个「在用」符号形态一致**：`multitoken_paged_attention`（`launch_*` 在 `:558`，
`:501` 是它的 `__global__` 转发）与 `sampling_from_logits`（`:1793-1841`，`grid=1`/`block=256`，
`smem` 直接取 `cub::BlockReduce<SamplingDataAndIndex<float,int>,256,...>::TempStorage` 大小，只支持
`vocab_size==50257`，否则 `printf`）——即 **wrapper 里塞满了「只支持测试用例那一组形状」的硬编码分支**。

## 4. 对照 megakernel 端到端层（`docs/28`）

| 维度 | 本层（单 kernel 单测） | megakernel 端到端（`docs/28`） |
|---|---|---|
| 驱动者 | host 手写 `<<<>>>` + `cudaDeviceSynchronize` | `PersistentKernel.compile/__call__`（`:3207`/`:3574`） |
| task 的角色 | **一个 task = 一整颗 kernel** | 一个 task = **CTA 内一段** device 代码 |
| 依赖 | 无（单 task，无 DAG） | event/tGraph + worker/scheduler 事件计数器（`docs/06`/`docs/08`） |
| grid/block/smem | **手写常量**（`dim3`/`96*1024`） | 编译器/运行层按 task 图算（`docs/11`/`docs/12`） |
| 参考 | 逐张量 PyTorch/numpy | PyTorch 端到端（`test_inference_output.py`） |
| 形状灵活性 | 硬编码 dispatch（batch∈{1,2,4,8} 等） | 编译期定型（batch≤16，`docs/09` §12） |
| 覆盖的 C 判据 | **无** | C1–C5（`docs/28` §7） |

一句话：**单测层验证的是「task 内数值」，megakernel 层验证的是「task 间/整图」**——两者的 device
代码是同一份（`kernel::moe_linear<...>` 就是 megakernel 里 `_execute_task` 会调的那个模板函数），
差别只在**谁负责 launch、依赖由谁保证、尺寸由谁算**。

## 5. CI 现状：本层已不在覆盖范围内

- **`gpu-tests.yml` 已失效**：文件头注释 *"The gpu-nvidia runner ... is no longer functional as of
  April 2026."*，Qwen3 CI 已迁到 self-hosted B200。
- **当前在跑的 `mpk-qwen-ci`（`ci-tests-qwen3.yml`）不碰本层**：它 `pip install -e .` → 跑
  `tests/ci-tests/run_ci_tests_qwen3.sh`（`demo/qwen3/demo.py --save-tokens` 的 torch vs
  `--use-mirage` MPK 端到端、`test_inference_output.py`、`perf_comparison.py`）+ `run_batch_perf.py`
  扫 batch 1/2/4/8。**没有任何一步编译或运行 `tests/runtime_python/` 的单测**。
- `build-test.yml` 只做 `python -c "import mirage"`。
- 结论：§2 的大面积失配**与 CI 不覆盖本层互为因果**——只有被现役测试/CI 引用的符号（blackwell 家族、
  hopper-moe，以及主 wrapper 的 3 个）保持可导入，其余随 CI 迁移自然腐化。这也解释了 `docs/28` 为何
  以 `test_mode/` 而非本层为「可执行规范」。

## 6. 判据 C1–C5 映射

本层**不贡献任何 C 判据**（与 `docs/22` §8、`docs/28` §7 一脉相承）：

- 不是 C1（每次 host 一次 launch，且常常多次/无提交边界）；
- 不是 C2（无持久化 kernel）；
- 只是 C3（融合范围）里「**单个 device task 数值正确**」的回归网；
- 无 C4（无设备内协调）、无 C5（无 worker/scheduler）。

它是**编译栈最底层的数值可信度来源**：`docs/07`/`docs/24` 走读的 device task、`docs/28` 走的
megakernel 端到端，其「算得对」的最终依据都指回这一层（或 CI 的端到端 torch 对照）。

## 7. 局限

- **纯源码走读，未构建/未运行**：所有「导入即失败」判断基于 pybind 注册与测试调用的**静态对照**
  （`setup.py` 是否定义 `DEPRECATED_TESTS`、宏展开后的注册表）；未实际 `pip install` 各 extension、
  未跑 `pytest` 复现失败。**不构成「这些测试当前一定失败」的运行时判定**——只记录快照里的接线条目。
- **失配的口径**：无法从快照区分「有意裁剪（只保留在用的符号）」与「随代际漂移的腐化」，本文只记录
  事实与分布（主/hopper 失配、blackwell 对齐），**不判对错**。
- **只抽样解剖**：§3 只精读 `moe_linear`（并旁及 `sampling_from_logits`/`multitoken_paged_attention`
  的 launch 形态），未逐行走读 12 个 blackwell wrapper、`cute/hopper/gemm.cu` 与
  `attention_tail_wrapper.cu` 的内部。
- **`test_mode/` 侧的对应关系**见 `docs/28`（本文不重复）；本层与 device task 的对应见 `docs/07`/
  `docs/24`。

## 8. 参考坐标

- 快照：`mirage-project/mirage` 分支 `mpk` commit `f9eb70c`（2026-10-05）。
  主路径：`tests/runtime_python/runtime_kernel_wrapper.cu`（1895 行；`PYBIND11_MODULE:1844-1895`，
  在用注册 `moe_linear:1848`/`multitoken_paged_attention:1887`/`sampling_from_logits:1890`；
  `moe_linear_kernel_wrapper:1318`、`launch_moe_linear:1348`、`moe_linear:1411`、
  `sampling_from_logits:1793`；`DEPRECATED_TESTS` 七处 `:30/:342/:801/:1147/:1228/:1487/:1596`）。
  `tests/runtime_python/setup.py`（51 行；`sources:28`、三档 gencode `:41-43`、
  `-DMIRAGE_UNIT_TEST:44`、`-DMIRAGE_GRACE_HOPPER:45`）。
  `include/mirage/persistent_kernel/tasks/ampere/moe_linear.cuh:256-263`（`MIRAGE_UNIT_TEST` 指针推进）。
  `tests/runtime_python/hopper/runtime_kernel_wrapper_hopper.cu`（1405 行；注册 `linear_swapAB:1400` 唯一）；
  `hopper/attention_tail_wrapper.cu`（91 行）；`hopper-moe/runtime_kernel_wrapper_moe_hopper.cu`（492 行；
  注册 `:486-491`）；`cute/hopper/gemm.cu`（556 行）。
  `tests/runtime_python/blackwell/`（12 个 `setup*.py`，扩展名见 §2.3）。
- CI：`.github/workflows/gpu-tests.yml`（失效，文件头注释）、`.github/workflows/ci-tests-qwen3.yml`
  （现役 `mpk-qwen-ci`）、`tests/ci-tests/run_ci_tests_qwen3.sh`、`tests/ci-tests/test_inference_output.py`、
  `tests/ci-tests/run_batch_perf.py`、`.github/workflows/build-test.yml`。
- 论文快照：`~/github/ai-infra-book/experiments/ch05/05-08/sources/mpk-v2.txt`——**未提及本层**（实现质量
  层面，同 `docs/28` 对 `test_mode` 的判断）。
- 同专题相关篇：`docs/28`（本文补其 §8 尾巴——非 `test_mode` 的单测层）、`docs/07`/`docs/24`
  （本层 wrapper 调用的 device task）、`docs/06`/`docs/08`（megakernel 路径的 task 驱动）、
  `docs/22` §8（C 判据总映射，本层不贡献）、`docs/09` §12（batch≤16 编译期约束）。
- 交叉专题：DeepGEMM mega 家族的反向/前向验证见 `../mega-moe/`（本专题不重复）。
