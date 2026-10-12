# H100 环境探测：DeepGEMM / DeepEP 在本机可跑到哪一步

> 本文对应路线图 **P0-4**，是「DeepSeek Mega MoE 前向 / 反向算子实现解析」的环境基线篇。
> 目的是把「Mega MoE 主体是 SM100 专属、H100 跑不了」这条约束**落到可执行的清单上**：
> 哪些 DeepGEMM kernel 在本机（H100 / sm90）真能编译并跑出正确数字，哪些一调用就因架构被拒，
> 以及 DeepEP 通信实验在本机是否可行。结论直接决定后续 P4 数值实验与 DeepEP 实验的边界。
>
> 实验脚本：`harness/probe_h100_env.py`、`harness/probe_deepep.py`；
> 原始输出：`harness/probe_h100_env.out.txt`、`harness/probe_deepep.out.txt`（均为真机运行结果）。

## 1. 环境基线

`kernel_lab` 容器（`scripts/lab.sh`）内实测：

| 项 | 值 |
|---|---|
| GPU | 8 × NVIDIA H100 80GB HBM3，compute capability **9.0**（sm90） |
| 驱动 | 570.211.01 |
| CUDA Toolkit | **13.2**（V13.2.78） |
| PyTorch | 2.12.0a0（NVIDIA 构建 `+nv26.04`），CUDA 13.2 |
| 系统 NCCL | **2.29.7**（`/usr/lib/x86_64-linux-gnu/libnccl.so.2.29.7`） |
| DeepGEMM | 本地 `~/github/DeepGEMM`，commit `057ca59`（Public release 26/09/30） |
| DeepEP | 本地 `~/github/DeepEP`，commit `93eb6eb`（v2.5.0） |

关键点：PyTorch 链接的 **NCCL 是 2.29.7**，而 DeepEP 2.5 的 host 扩展要求 **NCCL ≥ 2.31 的设备通信 API**——
这在本机埋了一个硬约束，见 §4。

## 2. 安装 DeepGEMM：三个非显然的坑

DeepGEMM 采用 **DeepJIT**：安装时只编译一个很小的 host C++ 扩展（`csrc/python_api.cpp`），
GPU kernel 在**首次调用时用 nvcc 现场编译**（见 README「no CUDA compilation during installation」）。
因此「装得上」不等于「跑得起来」，本机踩了三个坑（都记在下方，后续实验环境已修复）：

1. **缺 `elfutils/libdwfl.h`**：DeepJIT 的 host 扩展依赖 libdw 做设备端栈回溯。
   `apt-get install -y libdw-dev`（先 `apt-get update`）后解决。
2. **子模块 `third-party/deep_jit` 与实际 gitlink 不一致**：仓库记录的是
   `2efdab4`（2026-09-30），但工作区 checkout 成了旧的 `e5bdee2`（2026-09-08）；
   旧版 `deep_jit/backend/cuda/driver.hpp` 没有 `lazy_cuMemRelease` / `lazy_cuMemAddressFree`，
   编译报「not a member of `deep_jit::cuda::driver`」。用
   `git -C third-party/deep_jit checkout 2efdab4` 对齐后解决。
3. **editable 安装缺 cutlass/cute 头**：`setup.py` 的 `CustomBuildPy.prepare_includes`
   （`~/github/DeepGEMM/setup.py:137` 起）本来会把 `third-party/cutlass/include/{cute,cutlass}`
   拷进 `deep_gemm/include/`；但 editable 安装走临时 build-lib，源码树里没有。
   运行期 JIT 的 `--include-path` 只指向 `deep_gemm/include`，于是首次编译报
   `cute/numeric/math.hpp: No such file or directory`。**手动**
   `cp -r third-party/cutlass/include/{cute,cutlass} deep_gemm/include/` 后解决。

装完 `import deep_gemm` 得到版本 **2.8.1+local**。注意 2.8.1 的 API 已把 FP8/FP4 统一为
`fp8_fp4_gemm_{nt,nn,tn,tt}`（`fp8_gemm_nt` 是别名），量化参数走 `recipe/recipe_a/recipe_b`，
`(data, sf)` 打包成 tuple 传入（`stubs/_C.pyi:218`）。

## 3. H100 上实测可用的 kernel（真机数字）

`harness/probe_h100_env.py` 复用 DeepGEMM 自带 `tests/generators.py` 造数据，逐个试跑。
结果（`arch_major=9`，即 sm90）：

| kernel | 用途 | 结果 | 备注（实测残差） |
|---|---|---|---|
| `fp8_gemm_nt`（SM90 1D2D） | dense FP8 前向 | **OK** | m512 n1024 k2048，相对残差 ~3.7e-2（FP8 per-128 块量化） |
| `bf16_gemm_nt` | dense BF16 前向 | **OK** | relerr ~1.1e-6 |
| `m_grouped_fp8_gemm_nt_contiguous` | MoE linear1/linear2 前向 | **OK** | 8 组、m=4224、n1024、k2048 |
| `k_grouped_fp8_gemm_nt_contiguous` | **MoE wgrad**（K 轴分组） | **OK** | 4 组、ks=[512,384,640,256]，relerr ~2.0e-2（accumulate 模式） |
| `k_grouped_fp8_gemm_tn_contiguous` | MoE wgrad（TN 版） | **FAIL** | `csrc/apis/gemm.hpp:369`：`Unsupported architecture`（SM100 专属） |
| `mega.SymmBuffer` | Mega MoE 对称缓冲 | 可构造对象 | 真正的 `fp8_fp4_mega_moe` 端到端需 SM100 |

### 3.1 为什么 wgrad 的 `tn` 版在 H100 上直接拒绝

反向 wgrad 的数学是 $\mathrm{d}W_i = \mathrm{acts}_i^\top \mathrm{d}\text{out}_i$，
天然是 **TN** 布局（A 转置、B 非转置），且各专家 N 固定、K（token 数）随专家变化，
所以按 **K 轴分组**。`k_grouped_fp8_gemm_tn_contiguous` 的分发逻辑
（`~/github/DeepGEMM/csrc/apis/gemm.hpp:360-370`）：

- `arch_major == 10`：走 `sm100_k_grouped_fp8_gemm_1d1d`，用 tcgen05 UMMA（`Major::MN`）；
- 否则：`DG_HOST_UNREACHABLE("Unsupported architecture")`。

也就是说 **SM100 的 wgrad 走 TN + tcgen05**；而 SM90 没有对应实现，DeepGEMM 为 Hopper 提供的是
**`k_grouped_fp8_gemm_nt_contiguous`**（NT 版，见上表第 4 行，实测 OK）。这解释了两点：

1. 上游 `tests/test_fp8_fp4.py:244-248` 会用 `get_arch_major()` 二选一：
   `arch_major == 9` 选 `k_grouped_fp8_gemm_nt_contiguous`，否则选 `..._tn_contiguous`；
2. 路线图 P4-2 要精读的 `k_grouped_fp8_gemm_tn_contiguous` 是 **SM100 专属，仅分析**；
   但在 H100 上可以用它的兄弟 `..._nt_contiguous` 做**数值对拍**（P4-4 的落点）。

`m_grouped_fp8_gemm_nt_contiguous`（dgrad / 前向）在两代都是 NT，sm90 可用。

### 3.2 对后续实验的意义

- **P4-4（数值对拍）可行**：dgrad（`m_grouped_*_nt_contiguous`）与 wgrad（SM90 的
  `k_grouped_*_nt_contiguous`）都能在 H100 上跑，可与 PyTorch 参考实现对比误差、测性能。
- **Mega MoE 主体（P3）仍只做源码精读**：`fp8_fp4_mega_moe` 需要 SM100 的 tcgen05 / UE8M0 /
  FP8xFP4，本机不具备，禁止尝试端到端跑。

## 4. DeepEP：能装能 import，但 EPBuffer 在本机不可用

DeepEP 2.5（`93eb6eb`）**已去掉 NVSHMEM**，host 扩展链接 NCCL、kernel 走 DeepJIT。本机探测：

1. **子模块未初始化**：`third-party/deep_jit` 状态为 `-`（未 checkout）。
   `git submodule update --init third-party/deep_jit` 拉取 `cba802d` 后解决。
2. **NCCL 依赖**：按 README 安装 `nvidia-nccl-cu13>=2.32.3`（装了 2.32.3）。
3. **构建/安装成功**：`bash install.sh` 得到 `deep_ep-2.5.0+local` 并 import 成功。
4. **运行期崩溃（阻塞）**：构造 `EPBuffer` 时在
   `deep_ep/buffers/ep.py:304` 调用 `_C.calculate_ep_buffer_size(...)` **段错误**。

根因是 **NCCL 版本不一致**：

- PyTorch 链接的是系统 **NCCL 2.29.7**；DeepEP 期望 **2.32.3**。
- `deep_ep/__init__.py:20` 的 `check_nccl_so()` 会比较 `/proc/self/maps` 里已加载的 libnccl
  与 `find_nccl_root()/lib/libnccl.so*` 是否同一文件；不一致直接 assert。
- 用 `LD_PRELOAD` 强制加载 2.32.3 可过版本检查，但 `EPBuffer` 构造仍段错误。
- 反向尝试：用系统 2.29.7 头文件重建 —— **编译失败**，因为 DeepEP 2.5 用到
  `ncclCommProperties::ginSupport` / `ginMinStride`、`ncclDevCommRequirements::ginType`
  等 **NCCL ≥ 2.31 才有的字段**（`csrc/kernels/comm/context.cpp:71-142`）。

结论：**DeepEP 的 H100 dispatch/combine 实验在本机当前软件栈下不可行**，需要一个
「PyTorch 链接的 NCCL ≥ 2.32.3」的环境（例如与 CUDA 13 匹配、自带新 NCCL 的 PyTorch 构建）。
这是环境问题，不是 DeepEP/内核问题；记为路线图「阻塞」。这可解释 `docs/07` §8 记录的现象。

## 5. 小结与下一步

- **可跑**：dense/grouped FP8（含 MoE 前向）、BF16、**SM90 版 MoE wgrad（`nt`）**——P4-4 有落点。
- **仅分析**：Mega MoE 主体 `fp8_fp4_mega_moe`、`k_grouped_fp8_gemm_tn_contiguous`（SM100/tcgen05）。
- **受阻**：DeepEP 运行期实验（NCCL 2.29.7 vs 需求 ≥2.31/2.32.3）。
- 下一步进入 **P1**（MoE/EP 基础补全）与 **P2-1 收尾**（`transform_weights_for_mega_moe` 权重布局），
  随后启动 **P4** 的 H100 数值对拍实验。
