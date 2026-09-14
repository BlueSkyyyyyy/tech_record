---
title: "Flash Attention 精读（六）：官方仓库源码全景 —— FA2 / FA3 / FA4 三套实现怎么组织"
date: 2026-09-14
draft: false
weight: 6
series: ["flash-attention"]
tags: ["flash-attention", "cuda", "hopper", "blackwell", "cute", "系列"]
categories: ["算子开发"]
---

前五篇我们逐行读了算法：[原理]({{< relref "flash-attention-01-theory" >}})、[Triton 前向]({{< relref "flash-attention-02-triton-fwd" >}})、[CUDA 前向]({{< relref "flash-attention-03-cuda-fwd" >}})、[反向]({{< relref "flash-attention-04-bwd" >}})、[DSL 对比]({{< relref "flash-attention-05-dsl-zoo" >}})。但第一次 clone 下官方仓库 [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) 的人，往往会先被目录结构劝退：

```
csrc/     hopper/     flash_attn/cute/     flash_attn/     training/     benchmarks/ ...
```

同一个算法为什么有三套实现目录（`csrc/`、`hopper/`、`flash_attn/cute/`）？Python 侧的 `flash_attn_func` 到底调用的是哪一个？CUDA 扩展是怎么从这些 `.h` 文件变成几十个 `.cu` 编译单元的？本篇就回答这些问题——**不推导数学，只把仓库的"骨架"讲清楚**，为读源码建立一张地图。

> 所有行号以官方仓库 commit `8d3a3b8`（2026-09-13）为准。这个仓库同时容纳了三个硬件时代的实现，理解它们的分工，比记住任何单个 kernel 都重要。

## 1. 一次 attention 调用，代码走了多远

先把端到端链路摆出来。以 `flash_attn_func(q, k, v)` 为例（FA2 路径）：

```
用户代码
  flash_attn.flash_attn_interface.flash_attn_func          # flash_attn_interface.py:1156
    └─ FlashAttnFunc.apply(...)                            # autograd.Function, :828
         ├─ forward:  _flash_attn_forward(...)             # :85  → flash_attn_gpu.fwd(...)  :99
         │     └─ C++  mha_fwd                             # csrc/flash_attn/flash_api.cpp:368
         │           └─ run_mha_fwd                        # flash_api.cpp:261  （按 dtype/causal 分派）
         │                 └─ run_mha_fwd_hdim128          # flash_fwd_launch_template.h:246
         │                       └─ flash_fwd_kernel       # flash_fwd_kernel.h: compute_attn_1rowblock:55
         └─ backward: _flash_attn_backward(...)            # :253 → flash_attn_gpu.bwd(...) :280
               └─ C++  mha_bwd                             # flash_api.cpp:800
                     └─ run_mha_bwd → run_flash_bwd_seqk_parallel
                           └─ flash_bwd_kernel.h: compute_dq_dk_dv_1colblock:81
```

一次前向调用，从上到下要穿过 **6 层**：Python 公开函数 → autograd 封装 → torch custom op → C++ 分派 → host 侧 launch template → device 侧 kernel。这条链路在后文会反复出现，因为**每一层都在做"选择"**：这一章讲清楚各层选什么。

## 2. 三套实现：一个仓库，三个时代

官方仓库最反直觉的一点，是它同时维护了三套并不共享代码的 CUDA/Triton 实现。它们不是"新旧版本替换"，而是**按硬件代际并存**。

把仓库想成一座城市：`csrc/` 是**老城**——住的人最多、道路最成熟，大多数下游框架走的还是它；`hopper/` 是**新区**——专门为 Hopper 这种"新式住宅"重新规划；`flash_attn/cute/` 是**高科技园区**——图纸直接用 Python 画，只有最新的机器（Blackwell）才完全住得进去。

| 目录 | 名字 | 硬件 | 语言/框架 | 入口 | 状态 |
|---|---|---|---|---|---|
| `csrc/flash_attn/` | FA2 | sm80 / sm86 / sm89（Ampere 及消费级）| CUTLASS 2.x 风格 CUDA C++ | `flash_api.cpp` | 主力，覆盖最广 |
| `hopper/` | FA3 | sm90（Hopper）| CUTLASS 3.x C++（TMA + WGMMA）| `hopper/flash_api.cpp` | Hopper 专用 |
| `flash_attn/cute/` | FA4 | sm90 / sm100 / sm120（Hopper + Blackwell）| **CuTe DSL（Python）** | `cute/interface.py` | 最新，独立 pip 包 |
| `csrc/flash_attn_ck/` | — | AMD ROCm | Composable Kernel | — | AMD 后端 |

一句话区分三者：

- **FA2（`csrc/`）**：用 CUTLASS 的 CuTe 类型写 sm80 的 `mma.sync` kernel，靠手工编排 cp.async 流水。前三篇讲的就是它。
- **FA3（`hopper/`）**：为 Hopper 重写，把加载交给 TMA、GEMM 交给 WGMMA、用 warp specialization 把 softmax 与 GEMM 解耦。这是第 3 篇的后半部分。
- **FA4（`flash_attn/cute/`）**：不再写 C++，而是用 CUTLASS 的 **CuTe DSL** 写 Python，再由编译器降到 sm90/sm100/sm120 指令。它支持 Blackwell 的 tcgen05/TMEM，是当前唯一覆盖 sm100 的完整实现。

> 为什么 FA4 用 Python？第 3 篇的结论在这里显形：Hopper/Blackwell 上，"哪条指令等哪条指令"已经复杂到手写 C++ 难以维护。CuTe DSL 允许在 Python 里显式描述布局、mbarrier、TMA，同时保留编译期常量折叠。FA4 是"把 kernel 工程变成可组合的 Python 代码"的一次尝试。

## 3. FA2（`csrc/flash_attn/`）：最宽的覆盖面

### 3.1 目录构成

```
csrc/flash_attn/
├── flash_api.cpp              # 1542 行，唯一的 pybind11 绑定 + 所有 host 逻辑
├── src/
│   ├── flash_fwd_kernel.h     # 1301 行，前向 device 代码
│   ├── flash_bwd_kernel.h     #  841 行，反向 device 代码
│   ├── flash_fwd_launch_template.h   # 326 行，host 侧 tile 选择与 launch
│   ├── flash_bwd_launch_template.h   # 308 行
│   ├── flash_bwd_preprocess_kernel.h # 383 行，Δ 预处理 + dQ 转换
│   ├── kernel_traits.h        # 344 行，tile/MMA/smem 尺寸推导
│   ├── softmax.h / mask.h / utils.h / dropout.h / alibi.h / rotary.h
│   ├── static_switch.h        # 把运行时 bool 变成模板参数的分派宏
│   ├── flash.h                # 参数结构体 Flash_fwd_params / Flash_bwd_params
│   └── generate_kernels.py    # 生成 flash_*_sm80.cu 实例化文件
└── flash_*.cu                 # 约 96 个由脚本生成的 thin instantiation 文件
```

### 3.2 绑定层只暴露 5 个函数

`flash_api.cpp:1535-1541` 的 `PYBIND11_MODULE` 只注册 5 个入口：

```cpp
m.def("fwd",         &mha_fwd, ...);          // 普通前向
m.def("varlen_fwd",  &mha_varlen_fwd, ...);   // 变长（含 paged KV）
m.def("bwd",         &mha_bwd, ...);          // 普通反向
m.def("varlen_bwd",  &mha_varlen_bwd, ...);   // 变长反向
m.def("fwd_kvcache", &mha_fwd_kvcache, ...);  // 推理解码 / KV cache
```

Python 侧所有函数（`flash_attn_func`、`..._qkvpacked_func`、`..._varlen_func`、`flash_attn_with_kvcache`）最终都落到这 5 个中的某一个。函数名里的 `varlen` 表示支持 packed 变长序列，`kvcache` 表示推理时的增量解码。

注意：**FA2 没有单独的 `paged` 或 `mla` 入口**。paged KV 是通过给 `varlen_fwd` / `fwd_kvcache` 传 `block_table` 参数复用进来的（`flash_attn_interface.py:1407`）；MLA 完全不在 FA2 里，只在 FA3/FA4（见 §7）。

### 3.3 分派：`HEADDIM_SWITCH` 与 `BOOL_SWITCH`

真正决定"跑哪个 kernel"的是 `run_mha_fwd`（`flash_api.cpp:261-273`）：

```cpp
FP16_SWITCH(!params.is_bf16, [&] {                     // 1. dtype: fp16 / bf16
  HEADDIM_SWITCH(params.d, [&] {                       // 2. head_dim → 32/64/96/128/192/256
    BOOL_SWITCH(params.is_causal, Is_causal, [&] {     // 3. causal
      if (params.num_splits <= 1 && !force_split_kernel)
        run_mha_fwd_<T, kHeadDim, Is_causal>(params, stream);       // 标准 kernel
      else
        run_mha_fwd_splitkv_dispatch<T, kHeadDim, Is_causal>(...);  // split-KV kernel
    });
  });
});
```

这套 `XXX_SWITCH` 宏（定义在 `static_switch.h`）把 `if` 梯子变成**编译期模板特化**：每个组合实例化一份独立 kernel，运行时零分支。dtype/causal/head_dim 这三个维度在 `flash_api.cpp` 就定死了；dropout、local(window)、alibi、softcap 这几个则下沉到 launch template 里继续用 `DROPOUT_SWITCH` / `LOCAL_SWITCH` / `ALIBI_SWITCH` / `SOFTCAP_SWITCH` 分派（`flash_fwd_launch_template.h:68-73`）。

### 3.4 为什么有 96 个 `.cu` 文件

`generate_kernels.py` 干的事很朴素：把「4 类方向 × 2 dtype × 6 head_dim × 2 causal = 96」种组合，各生成一个只做显式实例化的 `.cu`。例如 `flash_fwd_hdim128_fp16_causal_sm80.cu` 内容就是：

```cpp
template<> void run_mha_fwd_<cutlass::half_t, 128, true>(Flash_fwd_params &params, cudaStream_t stream) {
    run_mha_fwd_hdim128<cutlass::half_t, true>(params, stream);
}
```

这样做的好处是**把编译并行化**：96 个独立编译单元可以由 ninja 并行编译，也方便按需裁剪。`setup.py` 的 arch 列表默认 `FLASH_ATTN_CUDA_ARCHS = "80;90;100;110;120"`（`setup.py:73-74`），通过 `-gencode` 控制每个 arch 的 SASS/PTX 产物。

> **读源码的实用建议**：不要从 `.cu` 文件开始读——它们只有一行实例化。直接看 `flash_fwd_kernel.h` / `flash_bwd_kernel.h` 这两个 device 头文件。

### 3.5 Kernel_traits：tile 尺寸从哪来

`kernel_traits.h` 是所有尺寸决策的集散地。前向 `Flash_fwd_kernel_traits`（`kernel_traits.h:49`）从模板参数推出 `kBlockM/kBlockN/kHeadDim`、MMA atom、smem 布局、总 smem 大小 `kSmemSize`。第 3 篇讲过的那些 tile 尺寸（hdim64→128×128、hdim128→128×64……）实际由 `flash_fwd_launch_template.h` 里的 `run_mha_fwd_hdim*` 函数选定，再喂给 traits。

## 4. FA3（`hopper/`）：Hopper 专属的第二套

FA3 不是 FA2 的一层 patch，而是整个 `hopper/` 子目录里的独立实现（连 host 侧都有自己的 `flash_api.cpp`，1769 行）。它的目录可以按职责分成四块：

```
hopper/
├── flash_api.cpp                     # host 分派 + 绑定（mha_fwd/mha_bwd/mha_fwd_combine）
├── flash_attn_interface.py           # Python 接口（1146 行）
├── flash_fwd_kernel_sm90.h           # 前向 kernel 骨架（线程/寄存器/pipeline）
├── mainloop_fwd_sm90_tma_gmma_ws.hpp # 1717 行，前向主循环
├── flash_bwd_kernel_sm90.h           # 反向骨架
├── mainloop_bwd_sm90_tma_gmma_ws.hpp # 1046 行，反向主循环
├── epilogue_fwd.hpp / epilogue_bwd.hpp
├── tile_scheduler.hpp                # 817 行，5 种调度器
├── flash_fwd_combine_kernel.h        # 728 行，split-KV 归并
├── softmax.h / mask.h / block.h / seqlen.h / pack_gqa.h / paged_kv.h
└── instantiations/                   # 451 个 .cu 实例化
```

几个和前几篇呼应的关键位置：

- Producer/consumer 的寄存器再分配在 `flash_fwd_kernel_sm90.h:309`（dealloc）和 `:361`（alloc）。
- L2-aware 的 persistent 调度在 `tile_scheduler.hpp:218`，L2 大小写死 `32MB`（`:255`）。
- 反向的 dQ 用 TMA 硬件归约加：`mainloop_bwd_sm90_tma_gmma_ws.hpp:647` 调 `SM90_BULK_REDUCE_ADD`，PTX 封装在 `copy_sm90_bulk_reduce.hpp:22`。
- split-KV 的 combine 数学在 `flash_fwd_combine_kernel.h:592-621`，就是第 1 篇 online softmax 递推在 split 粒度的复用。

FA3 的 Python 接口在 `hopper/flash_attn_interface.py`，走的是 `torch.library.custom_op` 注册（`:59`），和 FA2 的风格类似但 API 更丰富：多了 `flash_attn_combine`（`:938`）和 `get_scheduler_metadata`（`:1106`）。

## 5. FA4（`flash_attn/cute/`）：Python 写的 kernel

FA4 是仓库里最新、也最值得单独讲的部分。它是一个**独立的 pip 包** `flash-attn-4`（`flash_attn/cute/pyproject.toml:6`），但通过 `pkgutil.extend_path` 和 FA2 共享 `flash_attn.*` 命名空间（`flash_attn/__init__.py:1-4`）——所以 `pip install flash-attn` 和 `pip install flash-attn-4` 可以共存。

### 5.1 用户看到的入口

```python
from flash_attn.cute import flash_attn_func, flash_attn_varlen_func
out = flash_attn_func(q, k, v, causal=True)
```

`flash_attn/cute/__init__.py` 只导出这两个函数。它们在 `interface.py` 里定义（`flash_attn_func:3432`、`flash_attn_varlen_func:3480`），然后转入 `FlashAttnFunc` / `FlashAttnVarlenFunc` 两个 autograd 封装（`:3121`、`:3251`），最终调用内部编译缓存的 kernel。

### 5.2 一张"按架构选 kernel"的派发表

FA4 的派发逻辑比 FA2 干净得多，因为它在 Python 里直接把不同架构的实现类分开写。核心是 `_get_device_arch()`（`interface.py:92`，可用环境变量 `FLASH_ATTENTION_ARCH` 覆盖），然后：

**前向**（`interface.py:1267-1402`）：

| `arch//10` | 选中的类 |
|---|---|
| 8 | `FlashAttentionForwardSm80`（`:1270`） |
| 9 | `FlashAttentionForwardSm90`（`:1289`） |
| 10 / 11（Blackwell） | `qv != None` → MLA 前向（`:1315`）；`head_dim==256` → 专用 2CTA 前向（`:1348`）；否则 `FlashAttentionForwardSm100`（`:1350`） |
| 12 | `FlashAttentionForwardSm120`（`:1382`） |

**反向**（`interface.py:2436-2541`）：sm8x/sm12x → `...Sm80`/`...Sm120`；sm90 → `FlashAttentionBackwardSm90`（`:2462`）；sm100/110 → `head_dim==256` 走专用 2CTA（`:2502`），否则 `FlashAttentionBackwardSm100`（`:2523`）。

这种"Python 里 `if arch`"的写法在 CUDA 工程里很少见，但正是 CuTe DSL 的卖点：**kernel 是普通 Python 对象**，编译和缓存由 `cute.compile` 统一管理（缓存 key 在 `interface.py:2400-2409`）。

### 5.3 文件规模与分工

FA4 一共 52 个 `.py`、约 5.2 万行。按职责分：

| 类别 | 代表文件 | 说明 |
|---|---|---|
| 前向 kernel | `flash_fwd.py`（SM80 基类）、`flash_fwd_sm90.py`、`flash_fwd_sm100.py`（3392 行）、`flash_fwd_sm120.py`、`flash_fwd_mla_sm100.py`（3176 行） | 每代一个类 |
| 反向 kernel | `flash_bwd.py`、`flash_bwd_sm90.py`、`flash_bwd_sm100.py`（4345 行） | 第 7 篇细讲 |
| MLA 反向 | `flash_bwd_mla_sm100.py`、`flash_bwd_mla_dk_sm100.py`、`flash_bwd_mla_dq_dqv_sm100.py` | DeepSeek MLA |
| 公共算法 | `softmax.py`、`mask.py`、`block_info.py`、`seqlen_info.py`、`utils.py` | online softmax / 掩码 / 边界 |
| 硬件原语 | `blackwell_helpers.py`、`ampere_helpers.py`、`mma_sm100_desc.py`、`copy_utils.py`、`pipeline.py`、`barrier.py` | tcgen05 / MMA / TMA / mbarrier |
| 调度 | `tile_scheduler.py`（2120 行）、`prepare_scheduler.py` | LPT / varlen / persistent |
| 特性 | `pack_gqa.py`、`paged_kv.py`、`topk_gather_kv.py`、`block_sparsity.py` | GQA / paged / 稀疏 |

一句话看 FA4 的设计哲学：**把 FA2/FA3 里散落在 `if constexpr` 和宏里的架构差异，收敛成"每代一个类、Python 里选类"**。代价是文件更多，好处是每个架构的 kernel 能独立演化。

## 6. 构建：从源码到 wheel

三套实现的构建方式完全不同：

- **FA2**：`setup.py`（805 行）编译 `flash_attn_2_cuda` 扩展，源文件里显式列出了 fwd/bwd/split/split_align 各 24 个 `.cu`（`setup.py:351-446`），include 目录指向 `csrc/cutlass/include`。默认 arch `80;90;100;110;120`。
- **FA3**：`hopper/setup.py`（842 行）单独编译，产物是 `flash_attn_3_cuda`，`instantiations/` 下有 451 个 `.cu`。
- **FA4**：不编译 C++/CUDA，只是打包 Python，靠 CuTe DSL 在**运行时 JIT 编译**目标 kernel（并缓存到磁盘，见 `cache_utils.py`）。

所以安装 FA4 后第一次跑某个 shape 会明显变慢（在编译），之后走缓存。这也解释了为什么 FA4 的 `interface.py` 里到处是 `_flash_attn_fwd.compile_cache` 之类的缓存字典。

## 7. 一张功能矩阵

把三套实现的能力对齐，选型就清楚了：

| 能力 | FA2（`csrc/`） | FA3（`hopper/`） | FA4（`cute/`） |
|---|---|---|---|
| sm80 | ✅ 主力 | ❌ | ✅（`flash_fwd.py`） |
| sm90 | ✅（旧路径） | ✅ **最优** | ✅ |
| sm100 / sm120 | ❌ | ❌ | ✅ |
| fp16 / bf16 | ✅ | ✅ | ✅ |
| fp8 | ❌ | ✅（前向） | ✅ |
| 反向 | ✅ | ✅ | ✅ |
| dropout / alibi / softcap | ✅ | softcap ✅；dropout/alibi ❌（默认构建 `TORCH_CHECK(p_dropout == 0)`） | softcap ✅；dropout/alibi ❌ |
| paged KV | ✅（复用 varlen） | ✅ | ✅ |
| split-KV | ✅ | ✅ | ✅ |
| GQA / PackGQA | ✅ | ✅ | ✅ |
| MLA（吸收式） | ❌ | ❌ | ✅（sm100） |
| 块稀疏 | ❌ | ❌ | ✅ |
| 确定性反向 | ✅ | ✅ | ✅ |
| 多模态（HuggingFace 集成） | `flash_attn/models/` | — | — |

一个常见误区是"FA4 出了，FA2/FA3 就没用了"。实际上：

- **FA2 仍是覆盖最广的**：sm80 及消费级卡、以及 HuggingFace/vLLM 等下游框架默认走的还是它（顶层 `flash_attn.__version__ = "2.8.4"`）。
- **FA3 在 Hopper 上通常最快**（专门的 WGMMA + pingpong）。
- **FA4 是唯一支持 Blackwell 和 MLA 的**，但很多特性仍在补齐。

## 8. 读这个仓库的路线图

如果你是第一次读，建议按这个顺序：

1. **先看 Python 接口**：`flash_attn/flash_attn_interface.py` 的 `flash_attn_func` 到 `_flash_attn_forward`，搞清楚一个调用进 C++ 前经历了什么。
2. **再进 C++ host 层**：`flash_api.cpp` 的 `run_mha_fwd`，看清楚 dtype/head_dim/causal 怎么分派。**不要一上来读 kernel**。
3. **然后读 launch template**：`flash_fwd_launch_template.h` 里 tile 尺寸和 smem 的取舍，这是"算法 → 工程"的桥。
4. **最后读 device kernel**：`flash_fwd_kernel.h` 的 `compute_attn_1rowblock`——这就是第 1、2、3 篇一直在讲的 online softmax 循环。
5. 想读 Hopper/Blackwell，就直接从 `flash_attn/cute/` 的 Python kernel 入手，比 `hopper/` 的 C++ 模板友好得多。

## 9. 小结

- 官方仓库同时维护 **FA2（`csrc/`，sm80）、FA3（`hopper/`，sm90）、FA4（`flash_attn/cute/`，sm90/sm100/sm120）** 三套实现，按硬件代际并存而非替换。
- 一次 `flash_attn_func` 要穿过 **Python → autograd → torch op → C++ 分派 → launch template → device kernel** 六层；前几篇讲的 kernel 只是最底层。
- FA2 的 `XXX_SWITCH` 宏 + `generate_kernels.py` 把组合爆炸变成 **96 个编译期特化实例**；FA4 则把架构差异收敛成 **Python 里按 arch 选类**。
- `flash_attn_func` 默认走 FA2（`flash_attn/__init__.py` 只导出 FA2），FA3/FA4 需要显式 `from flash_attn.cute import ...` 或装对应包。
- MLA、块稀疏、Blackwell 只在 FA4；sm80 和大覆盖率仍是 FA2 的天下。

下一篇[（七）：反向传播深潜]({{< relref "flash-attention-07-bwd-deep" >}})，我们把这四套实现（Triton / FA2 / FA3 / FA4）的反向逐个拆开，再补上 MLA 反向与第 4 篇没来得及展开的反向数学（dropout、softcap、GQA 归约、确定性）。
