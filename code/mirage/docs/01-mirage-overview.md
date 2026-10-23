# Mirage 是什么：一个仓库里的两条产品线

> 这是「Mirage 项目分析」的第 1 篇。我们先不急着读源码，先把**这个项目到底卖什么**讲清楚：
> 它有一个能自动搜 kernel 的编译器（Mirage），还有一个把整模型编成单个常驻 kernel 的运行时
> （MPK）。两者在同一个仓库里，共用一张图 IR 和同一套 codegen 基础设施。

## 0. 先记住一个画面

Mirage 像一个**代工厂**。

你把「要算什么」写成一张计算图（比如 attention），交给它；它不开模、不手工画图纸，而是在一个
巨大的「候选实现空间」里搜索，找到一个**算得一样、但更快**的写法，然后自动生成 CUDA 代码。

MPK 是这家工厂接的一个极端订单：客户不想再要「一个个零件」（一颗颗 kernel），而是要**整条产线
焊死在一个工位里**——一次 launch、一个常驻 kernel、内部自己调度所有工序。

> **类比的边界**：代工厂的搜索是「试出来」的，Mirage 的搜索靠 e-graph 等价变换 + 代价模型，
> 能保证**功能等价**（论文用形式化/概率指纹验证），但它不是万能的——搜索空间、模板、目标硬件
> 都写死在代码里；MPK 也不是「随便什么模型都能一键变 megakernel」，它需要你在 Python 里把模型
> 一层层手工搭出来（`demo/qwen3/demo.py` 就是范例），只是**调度**这件事被自动化了。

读完这一篇，你应该能回答：

1. Mirage 和 MPK 是什么关系？为什么在同一个仓库？
2. 这个仓库的代码大致分成哪几块？想改一处东西该去哪个目录？
3. 想跑起来，最少要做什么？

## 1. 两条产品线，一张图 IR

仓库有两个「产品」：

| 产品 | 分支 | 输入 | 输出 | 一句话 |
|---|---|---|---|---|
| **Mirage**（superoptimizer） | `main` | 一段 PyTorch 风格的算子图 | 一颗（或几颗）高性能 kernel | 自动搜索 kernel 实现 |
| **MPK**（Mirage Persistent Kernel） | `mpk`（**默认分支**） | 用 `PersistentKernel` API 搭出的整模型图 | 一次 launch 的常驻 megakernel | 把整模型编成一个 kernel |

它们能住在一起，是因为共享三层地基：

1. **图 IR**：`KNGraph` / `KNOperator`，描述张量怎么算（`src/kernel/`、`include/mirage/kernel/`）。
2. **搜索与变换**：e-graph、区间推导、线程块级 schedule（`src/search/`、`src/threadblock/`）。
3. **代码生成**：把线程块图翻译成 CUDA（`src/transpiler/`），或翻译成 Triton（`src/triton_transpiler/`）。

MPK 是在这三层之上又加了两层：**task/event 图**（把算子图细化为设备内可调度的任务）和
**常驻运行时**（worker + scheduler 的设备循环）。

## 2. 仓库地图：先认路，再读代码

按「你改什么」来找目录：

| 你想改… | 去这里 | 规模（本快照实测） |
|---|---|---|
| 算子怎么表示、图怎么建 | `src/kernel/`、`include/mirage/kernel/` | 13,842 + 1,410 行 |
| 自动搜索（等价变换、代价模型） | `src/search/` | 6,605 行 |
| 线程块 schedule 与 CUDA codegen | `src/transpiler/`、`src/threadblock/` | 8,863 + 2,319 行 |
| Triton 后端 | `src/triton_transpiler/` | — |
| **常驻 kernel 运行时** | `include/mirage/persistent_kernel/` | **44,337 行** |
| 每种 task 的 device 实现 | `include/mirage/persistent_kernel/tasks/{ampere,hopper,blackwell,cute,...}` | 含在上面 |
| Python 前端（构图 / 编译 / 启动） | `python/mirage/mpk/` | 11,886 行 |
| Qwen3 端到端样例 | `demo/qwen3/` | 14,853 行（整个 `demo/`） |

几个数字值得记住：

- **C++ 45,496 行**（`.cc/.cpp/.h`）、**CUDA 45,147 行**（`.cu/.cuh`）、**Python 14,894 行**
  （逐文件计数，口径：C++/CUDA 只算 `src/` + `include/`，Python 只算 `python/mirage/`；
  不含 `tests/`、`demo/`、`deps/`、`build/`）。加上 `tests/` 还有 1.4 万行 C++/CUDA。
- 其中 `include/mirage/persistent_kernel/` 一个目录就占 **44,337 行**——MPK 的重量不在「调度器」，
  而在**每个 task 的 device 身体**（各种 GEMM / attention / norm / MoE 的手写实现）。
- 论文里报的规模是 44K C++ / 42K CUDA / 10K Python，和快照实测同量级。

> 读源码时的经验：MPK 的「骨架」很小（`persistent_kernel.cuh` 2,115 行 + `runtime_header.h` 462 行），
> 其余 4 万行都是「肌肉」。先读骨架，再按需读肌肉。

## 3. MPK 对外承诺了什么

README 的原话是：把 LLM 推理变成**一次 kernel launch**，端到端延迟降低 **1.2× 到 6.7×**
（原文：*reduces LLM inference latency by 1.2× to 6.7×*）。

它的用户 API 长这样（`README.md` 的示例）：

```python
import mirage as mi

mpk = mi.PersistentKernel(
    world_size=world_size,
    mpi_rank=rank,
    num_workers=96,
    num_local_schedulers=48,
    num_remote_schedulers=0,
    meta_tensors=[step, tokens],
    profiler_tensor=profiler_tensor,
)
```

三个参数的含义要先记住，后面会反复出现：

- `world_size` / `mpi_rank`：几张卡、当前是第几张；
- `num_workers` / `num_local_schedulers` / `num_remote_schedulers`：**必须和物理 SM 对上**，
  关系是 `num_workers + (local + remote) / 4 = SM 数`（每个 scheduler 占 4 个 warp、正好 1 个 SM）；
- `meta_tensors`：两个特殊张量。`step` 记录当前解码步（MPK 每轮自增），`tokens` 存 prompt 和
  已生成的 token。

之后你用 `mpk.attach_input(...)` 把已有 PyTorch tensor 挂进来，用 `mpk.new_tensor(...)` 申请
中间张量，再调 `mpk.linear_layer(...)`、`mpk.attention_layer(...)` 这些「层 API」把模型搭出来。

**注意这不是 `torch.nn.Module`**：你在写一张静态图，张量是图上的边，不是会变的 Python 对象。

## 4. 五分钟上手

官方步骤（`README.md`）：

```bash
git clone --recursive --branch mpk https://github.com/mirage-project/mirage
cd mirage
pip install -e . -v
export MIRAGE_HOME=$(pwd)
```

跑 Qwen3-8B 的端到端 demo：

```bash
python demo/qwen3/demo.py            # 原生 Triton / FlashInfer 路径（对照组）
python demo/qwen3/demo.py --use-mirage   # 编译成 megakernel 并执行
```

想看图执行时间线，加 `--profiling`，会导出 Perfetto trace。

### 四个容易踩的坑

1. **必须 `--recursive`**：`deps/cutlass`、`deps/z3`、`deps/json` 都是 submodule，缺了直接编译失败。
2. **`MIRAGE_HOME` 要指到仓库根**：`compile()` 生成代码时要按这个路径找头文件。
3. **多卡需要 NVSHMEM ≥ 3.5.19**：`compile()` 会去 `/usr/include/nvshmem_12/` 和
   `/usr/lib/x86_64-linux-gnu/` 找 `nvshmem.h` 与 `libnvshmem_device.a`，找不到就报错；
   单卡路径不需要。
4. **`num_workers` 不是随便填的**：填错会出现「kernel 起来了但永远不结束」——它和 SM 数不匹配时，
   scheduler 和 worker 会互相等（`docs/04` 有真实案例）。

## 5. 它和「MegaKernel 专题」是什么关系

本仓库还有另一个专题「MegaKernel 资料与实现分析」（站内路径 `/posts/mega-kernel-00-index/`），
两者分工是：

| | MegaKernel 专题 | 本专题（Mirage） |
|---|---|---|
| 关注 | **范式**：为什么融合、怎么判据、MPK 论文 | **项目**：这个仓库怎么用、怎么编、怎么跑 |
| 内容 | 论文精读、生产案例、源码逐文件走读 | 端到端链路、关键 API、构建实录 |
| 适合 | 想理解 megakernel 这类系统 | 想上手 Mirage / 改 MPK |

如果你第一次接触 megakernel，建议先读 MegaKernel 专题的第 1 篇（三笔账与五条判据），再回来读本专题。

## 6. 三句话小结

- Mirage 是「自动搜 kernel 的编译器」，MPK 是它长出来的「整模型 megakernel 运行时」，共用图 IR 与 codegen。
- 仓库的骨架很小（运行时 2,115 行），肌肉很大（task 实现 4.4 万行）——先读骨架。
- 上手就三步：`pip install -e .` → 用 `PersistentKernel` 搭图 → `compile()` + 一次调用。

下一篇跟着 `demo/qwen3/demo.py` 走一遍：从 HuggingFace 权重到 `test.cu`，看看 `compile()`
那一行背后到底发生了什么。
