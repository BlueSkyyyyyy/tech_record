# 109 · DeepGEMM / DeepEP 的 CUDA Graph 兼容性面与 capture 前的 warm-up 契约

> 第 50/52/98 篇分别拆过 `Runtime` 生命周期、`cuLaunchKernelEx` 发射层、DeepEP 的编译配置面，
> 但**「把这些算子放进 CUDA Graph 会怎样」**这条贯穿训练/推理部署的线索一直没被单独讲过。
> 它恰好是 mHC / Mega Gate / 稀疏 MQA logits 三个 API 里**最不显眼、却最容易在部署时踩到**的一段：
> 它们都用一句 `DG_HOST_ASSERT(currentStreamCaptureStatusMayInitCtx() == CaptureStatus::None)`
> 强制「先 warm-up 再 capture」。本篇把这层契约单独拆透，并在 H100(sm90) 上把可跑的部分跑成真数字。
>
> **约束**：三个 API 的主体 kernel 都是 **SM100 专属**（`mega_mhc.hpp:223` / `mega_gate.hpp:159` /
> `attention.hpp:313` 断言 `arch_major == 10`），在 H100 上**整段不可达**（下面的断言在 arch 闸门之后），
> 故「契约语义」用**忠实复刻的 Python 版本**验证；而**发射层与真实 SM90 kernel 在 CUDA Graph 下的
> capture/replay 是 H100 真机实测**。

---

## 0. 一页结论

- **三层时间轴**是理解一切的关键：`host 一次性初始化`（分配 + 清零 scratch）／`graph 捕获`
  （把 stream 上的操作录成 DAG）／`replay`（重放 DAG）。懒初始化如果落在**捕获**里，就会把
  「一次性」的工作**录进 DAG 并每次 replay 重放**——这就是要断言的根因，而不是「捕获会报错」。
- DeepGEMM 有**三个** API 用同一套模式：按当前 `CUDAStream` 懒分配一块 **必须为零** 的 scratch
  （`torch::zeros`，会发射 zeroing kernel），并在首次分配前断言当前 stream **不在捕获中**。
- 反过来，**cuBLASLt workspace 的懒分配用 `torch::empty`（无 zeroing）**，所以**不需要**这个断言——
  真机实测：在一个全新 stream 上**首次调用就发生在捕获体内**也能成功（workspace 从 graph-private 内存池分配）。
- 发射层（`cuLaunchKernelEx` + cluster / PDL / cooperative 属性、每次 launch 的 `cuFuncSetAttribute`）
  **全部 capture-safe**：`bf16`（自研 / cuBLASLt）、`fp8 1d2d`、`set_pdl(True)` 三种路径在 H100 上
  capture + replay 与 eager **逐位相等**。
- 把 64 次 `256³` bf16 GEMM 放进一个 graph，replay 比 eager 连发快 **1.79×**（12.14 → 6.77 us/call）。
- DeepEP 走的是**另一条路线**：它不靠断言，而是用 `EP_AVOID_RECORD_STREAM` 把
  `tensor.record_stream()` 换成事件对象持有张量引用，因为 **`record_stream` 与 CUDA Graph 不兼容**。

---

## 1. 背景：懒初始化 × 捕获 = 一次性工作被录进 DAG

CUDA Graph 的语义是「把某条 stream 上**此刻发射的所有操作**录成一张 DAG，之后每次 replay 重放」。
任何「只在第一次做」的 host 侧初始化都可以分成两类：

| 初始化动作 | 落在捕获内会怎样 | 是否安全 |
|---|---|---|
| 纯 host 动作（`cuFuncSetAttribute`、JIT 加载 cubin） | 不是 stream 操作，**不被录进 DAG** | 安全 |
| 分配显存但**不发射 kernel**（`torch::empty`） | 从 graph-private 内存池分配，**录进去但无害** | 安全 |
| 分配显存**并发射清零 kernel**（`torch::zeros`） | 清零 kernel 被**录进 DAG，每次 replay 重放** | 有副作用 |

第三类就是三个 API 要防的：scratch 本身**只需清零一次**（后续靠 kernel 内部的状态编码免复位，
见第 92 篇 `ScoreBarrier`/`SplitBarrier` 的 `%gridid` 免复位设计），但如果清零被录进 graph，
**每次 replay 都会把 scratch 清零一遍**——既浪费，又可能与「跨 replay 保持状态」的设计冲突。

所以正确的用法是：**先在目标 stream 上跑一次（warm-up），让一次性分配 + 清零落地，再去 capture。**
三个 API 把这个「必须 warm-up」用一句断言**编码进契约**。

---

## 2. DeepGEMM 的三个「capture 前 warm-up」API

### 2.1 三个几乎逐字相同的 helper

**（a）Mega mHC 的分段屏障**（`csrc/apis/mega_mhc.hpp:26-39`）

```cpp
static const torch::Tensor& get_split_barriers(const torch::TensorOptions& options) {
    const auto stream = at::cuda::getCurrentCUDAStream();
    DG_HOST_ASSERT(options.device() == stream.device());
    static std::unordered_map<c10::cuda::CUDAStream, torch::Tensor> split_barriers_by_stream;
    auto& split_barriers = split_barriers_by_stream[stream];
    if (not split_barriers.defined()) {
        // Warm up each stream before capture so one-time zeroing is not replayed with the graph.
        DG_HOST_ASSERT(c10::cuda::currentStreamCaptureStatusMayInitCtx() == c10::cuda::CaptureStatus::None);
        split_barriers = torch::zeros(
            {mhc_layout::kNumSplitBarriers, mhc_layout::kNumMaxMBlocks, mhc_layout::kSplitBarrierLineBytes},
            options.dtype(torch::kByte));
    }
    return split_barriers;
}
```

**（b）Mega Gate 的打分屏障**（`csrc/apis/mega_gate.hpp:13-26`）与 **（c）稀疏 MQA logits 的 workspace**
（`csrc/apis/attention.hpp:284-301`）是同一模板的复制品，只是形状与变量名不同：

| API | helper | scratch 形状 | 断言行 | 分配行 |
|---|---|---|---|---|
| Mega mHC | `get_split_barriers` | `[kNumSplitBarriers, kNumMaxMBlocks, line]` | `mega_mhc.hpp:33` | `:34-36` |
| Mega Gate | `get_score_barriers` | `[kNumMaxTokenBlocks, line]` | `mega_gate.hpp:20` | `:21-23` |
| 稀疏 MQA logits | `get_sparse_mqa_logits_workspace` | `[sizeof(WorkspaceState)+2^20·sizeof(QBlockInfo)]` | `attention.hpp:297` | `:298` |

三处都在文件头 `#include <c10/cuda/CUDAGraphsC10Utils.h>`（`mega_mhc.hpp:4` / `mega_gate.hpp:5` /
`attention.hpp:5`），即抽头就是这个 `CaptureStatus` API。

三个 helper 的四个共同约定，缺一不可：

1. **按 stream 缓存**：`std::unordered_map<c10::cuda::CUDAStream, torch::Tensor>`（`mega_mhc.hpp:29`
   等）。graph 与 capture stream 绑定，一个 stream 一份 scratch 才有多 stream 正确性；
2. **只分配一次**：`if (not x.defined())`，之后复用同一块（后续调用不再动它）；
3. **必须为零**：`torch::zeros`（不是 `empty`）——屏障/状态初值必须是 0；
4. **分配前断言不在捕获中**：`currentStreamCaptureStatusMayInitCtx() == CaptureStatus::None`。

### 2.2 断言的确切语义

`c10::cuda::currentStreamCaptureStatusMayInitCtx()` 定义在 PyTorch 头
`c10/cuda/CUDAGraphsC10Utils.h:70`，实现只有一行核心：

```cpp
inline CaptureStatus currentStreamCaptureStatusMayInitCtx() {
  cudaStreamCaptureStatus is_capturing{cudaStreamCaptureStatusNone};
  C10_CUDA_CHECK(cudaStreamIsCapturing(c10::cuda::getCurrentCUDAStream(), &is_capturing));
  return CaptureStatus(is_capturing);
}
```

`CaptureStatus` 枚举（同头 `:45-49`）三态：`None / Active / Invalidated`。断言要求 **`None`**：
既不在捕获中、也没有一个被失效的捕获。它的 Python 等价物是
`torch.cuda.is_current_stream_capturing()`（同样查当前 torch stream），实测 eager 时 `False`、
`torch.cuda.graph(...)` 体内 `True`（实验 A1a/A1b）。

注意它查的是**当前 stream**——这正是「per-stream」这层设计的原因。

### 2.3 为什么是断言，而不是「让它录进去」

因为**它真的会录进去、真的会重放**。实验 A4 用一个最小复现证明：把 `buf.zero_()` 放进 capture 体内，
capture 后把 `buf` 改成 5，再 `replay`，`buf` **又变成 0**——清零被固化进 DAG 了。
而实验 A3 证明正确用法（先 warm-up 再 capture）下：同一 stream 复用同一对象、
capture 成功、且 graph 里**没有**这条 zeroing。

### 2.4 为什么 per-stream，而不是全局一份

两个理由，实验 A3d 都验证了：

- **不同 stream 各自独立**：新 stream 首次取会分配**不同**的对象（`is not` 为真，`alloc_count` 递增）；
- **graph 与 stream 绑定**：一条 stream 的 graph 只能 replay 在它捕获时的 stream 上，scratch 也跟着
  stream 走。第 67 篇的 `test_mega_mhc_api_contract` 还额外验证了「并发 kernel 在不同 stream 上
  用**互相独立**的 split barrier」（`tests/test_mega_mhc.py:298-315`）。

### 2.5 契约的可执行规范

`tests/test_mega_mhc.py:259-283` 就是这段契约的验收测试（`@test_filter(get_arch_major() == 10)`，
H100 整体跳过）：

```python
# A stream must initialize its split barriers before CUDA Graph capture
expected_outputs = make_outputs(inputs, 'bf16')
run_mega_mhc(inputs, expected_outputs)          # 先 warm-up（默认 stream）
torch.cuda.synchronize()
capture_stream = torch.cuda.Stream()
graph_outputs = make_outputs(inputs, 'bf16')
try:
    with torch.cuda.graph(torch.cuda.CUDAGraph(), stream=capture_stream):
        graph_outputs['y_bf16'].zero_()
        run_mega_mhc(inputs, graph_outputs)     # capture_stream 未 warm-up -> 应被拒绝
except RuntimeError as error:
    assert 'CaptureStatus::None' in str(error)  # 断言消息里带 CaptureStatus::None
else:
    raise AssertionError('Mega mHC must be warmed up before CUDA Graph capture')

with torch.cuda.stream(capture_stream):
    run_mega_mhc(inputs, graph_outputs)         # warm-up 该 stream
capture_stream.synchronize()
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph, stream=capture_stream):
    run_mega_mhc(inputs, graph_outputs)         # 现在可 capture
graph.replay()
capture_stream.synchronize()
# 与 eager 的 expected_outputs 逐位比较（assert_bitwise_equal）
```

它以 `python.cuda.graph` + 一个**全新 stream** 精确复刻了 §2.3 的两步：
**未 warm-up → 抛错且消息含 `CaptureStatus::None`；warm-up → 可 capture 且 replay 结果与 eager 逐位相等。**
本篇实验 A2/A3 用忠实 Python 复刻在 H100 上把这两步都跑通了（复刻的函数与 C++ 逐行对应）。

---

## 3. 为什么大多数 kernel 不需要这个断言

关键对照在 `csrc/runtime/runtime.hpp:67-76`——cuBLASLt 的 workspace 也是**按 stream 懒分配**的：

```cpp
torch::Tensor get_cublaslt_workspace(const c10::cuda::CUDAStream& stream) {
    const auto options = dtype(torch::kByte).device(stream.device());
    if (use_temp_cublaslt_workspace)
        return torch::empty({kCublasLtWorkspaceSize}, options);
    auto& workspace = cublaslt_workspaces[stream];
    if (not workspace.defined())
        workspace = torch::empty({kCublasLtWorkspaceSize}, options);  // empty，无零ing
    return workspace;
}
```

它和三个 helper 只差一点：用 **`torch::empty`** 而不是 `torch::zeros`。没有清零 kernel，
就没有「一次性工作被录进 graph」的问题，于是**不需要断言**——哪怕这次分配*恰好发生在捕获体内*，
也只是从 graph-private 内存池拿一块内存，replay 无害。

实验 A5 直接验证了这一点：`torch.empty` 在 capture 体内成功。
实验 B2b 更进一步，用**真实 DeepGEMM**在**全新 stream 上、首次调用就在 capture 体内**跑通了
cuBLASLt 回退路径（`bf16_gemm_nt` 默认走 `nvjet_sm90_*`，第 27 篇），capture + replay 与 eager
**逐位相等**——这里就发生了 `get_cublaslt_workspace` 对该新 stream 的懒分配。

> 一句话对照：**`zeros` 需要 warm-up 断言，`empty` 不需要。** 差别只在「有没有发射一次性 kernel」。

---

## 4. 发射层是否 capture-safe：`cuLaunchKernelEx` 与三个 launch 属性

每次 launch 都会走 `deep_jit` 的 `Kernel::launch`（`third-party/deep_jit/include/deep_jit/backend/cuda/kernel.hpp`），
里面有三类「可能挡住 capture」的操作，逐一核对：

- **`cuFuncSetAttribute`**（`:116-120` 设 `MAX_DYNAMIC_SHARED_SIZE_BYTES`；`:128-131` 设
  `NON_PORTABLE_CLUSTER_SIZE_ALLOWED`）：改的是 **function 属性**，是 host 侧对 kernel 句柄的操作，
  **不是 stream 操作**，不被录进 DAG，capture-safe。
- **`CUlaunchAttribute` 数组**（`:134-152`）：
  - `CU_LAUNCH_ATTRIBUTE_COOPERATIVE`（`:136-140`）；
  - `CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION`（`:141-147`，`cluster_dim->x > 1` 才加）；
  - `CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION`（PDL，`:148-152`）。
  这些是 `cuLaunchKernelEx` 的**属性参数**，在 graph capture 中被支持。
- **`cuLaunchKernelEx`** 本身（`:169-170`），stream 取 `launch_options.stream` 或当前 stream
  （`:164-166`）——capture 时只有 stream 上的 kernel 被录，属性随之带进 DAG。

实验 B1/B3/B4 在 H100 上验证：自研 `sm90_bf16_gemm`、`sm90_fp8_gemm_1d2d`、以及 `set_pdl(True)`
三种路径 capture + replay 与 eager **逐位相等**（`max|Δ|=0.000e+00`）。实验 D 用 profiler 抓到
replay 时真实发射的 kernel `deep_gemm::sm90_bf16_gemm_impl<(cute::UMMA::Major)0, (cute::UMMA::Major)0, 0u, 256u, ...>`
（64 次调用对应 graph 里 64 个节点）。

> 注：DeepGEMM 自己的 kernel 都不做 per-stream 懒分配（scratch 由调用方传入），**所以默认就没有 §2
> 那个 warm-up 断言**；只有那三个自带 scratch 的 API 才需要。

---

## 5. DeepEP 走的另一条路线：绕开 `record_stream`

DeepEP 不靠断言，但它有一个**与 CUDA Graph 直接相关**的开关。跨 stream 使用时，PyTorch 的
`tensor.record_stream()` 会把张量「钉」在某条 stream 上，但它是**分配器级别的副作用、对 graph 不感知**，
所以在 capture 中会出问题。DeepEP 的处理方式是把 `record_stream` 换成一个**事件对象持有张量引用**：

- Python 侧注释直说（`deep_ep/utils/event.py:27-29`）：
  「we use extra tensors to achieve stream recording, otherwise, **stream recording will be incompatible
  with CUDA graph**」；`EventOverlap` 的 docstring（`:14`）也点明 `extra_tensors` **may be useful with CUDA graph**。
- C++ 侧由环境变量 `EP_AVOID_RECORD_STREAM` 切换（`csrc/buffers/ep.hpp:163-170`）：
  ```cpp
  if (get_env<int>("EP_AVOID_RECORD_STREAM", 0)) {
      event->tensors_to_record = tensors;      // 事件里存张量引用，不用 record_stream
  } else {
      for (auto& t: tensors) if (t.has_value()) {
          t->record_stream(compute_stream);
          t->record_stream(comm_stream);
      }
  }
  ```
  `EventHandle` 的 `tensors_to_record` 字段（`csrc/utils/event.hpp:15`）。
- bucket 版本同款（`csrc/buffers/bucket.hpp:48-58`，`EP_AVOID_RECORD_STREAM` 走 `tensors_to_record.assign`）。

也就是说，两个库对「CUDA Graph 兼容性」给了两种不同的答案：

| | 触发点 | 手段 | 对付的问题 |
|---|---|---|---|
| DeepGEMM 三 API | `torch::zeros` 的**一次性清零** | 捕获前**断言**强制 warm-up | 一次性 kernel 被录进 DAG |
| DeepEP V2 路径 | `record_stream` 的**分配器副作用** | `EP_AVOID_RECORD_STREAM` 换事件持有引用 | `record_stream` 对 graph 不感知 |

---

## 6. 实测（H100/sm90，DeepGEMM `057ca59`，torch 2.12.0a0，CUDA 13.2）

`harness/exp_cuda_graph_capture.{py,sh}` + `harness/exp_cuda_graph_capture.out.txt`，
**18 PASS / 0 FAIL**（`c10/cuda/CUDAGraphsC10Utils.h` sha256 `89213765e613b7fc…`）：

**Part A（三个 API 契约的 Python 忠实复刻）**

- A1 捕获状态 API：eager `False` / capture 体内 `True`；
- A2 未 warm-up 的 stream 在 capture 中取 scratch → `RuntimeError`，消息含 `CaptureStatus::None`，
  且**未发生分配**（`alloc_count == 0`）；
- A3 warm-up 后同一 stream 可 capture，且复用**同一对象**（`is`）；新 stream 分配**独立**对象
  （`alloc_count` 1→2，stream 号 `150597056` vs `150652224`）；
- A4 **坐实**：把 zeroing 放进 capture，capture 后把 buffer 改成 5，replay 后**又变 0**——一次性
  清零被录进 DAG 并重放（这就是为什么必须 warm-up）；
- A5 `torch.empty` 在 capture 体内分配成功（无 zeroing，无需断言）。

**Part B（真实 DeepGEMM SM90 kernel）**

| 用例 | 结果 |
|---|---|
| B1 `bf16_gemm_nt`（`deterministic=True` → 自研 `sm90_bf16_gemm`） | capture/replay 逐位相等 `max|Δ|=0` |
| B2a `bf16_gemm_nt`（默认 → cuBLASLt 回退） | 逐位相等 `max|Δ|=0` |
| B2b cuBLASLt **全新 stream 首次调用即在 capture 内** | 逐位相等（workspace 懒分配 capture-safe） |
| B3 `fp8_gemm_nt`（`sm90_fp8_gemm_1d2d`） | 逐位相等 `max|Δ|=0` |
| B4 `set_pdl(True)` 下 capture | 逐位相等（PDL 属性 capture-safe） |
| B5 两条 stream 各自的 graph replay | 结果一致 |

**Part C（发射开销）**：64 次 `256×256×256` bf16 GEMM——
eager 连发 **776.7 us/iter（12.14 us/call）** vs graph replay **433.1 us/iter（6.77 us/call）**，
**1.79×**。规模越小，graph 摊掉的 host 发射开销（`cuLaunchKernelEx` + 包装层）占比越高。

**Part D（profiler 取证）**：replay 发射 64×
`deep_gemm::sm90_bf16_gemm_impl<(cute::UMMA::Major)0, (cute::UMMA::Major)0, 0u, 256u, …>`——
graph 里就是一个 GEMM 节点一张调用。

> 一个无害的旁注：A2 里被拒绝的 capture 会在析构时触发 torch 的
> `UserWarning: The CUDA Graph is empty`（捕获体抛异常 → 空图）；这是预期产物，不影响结论。

---

## 7. 边界与「SM100 专属，仅分析」

- **三个 API 本体 H100 不可达**：`get_split_barriers` / `get_score_barriers` /
  `get_sparse_mqa_logits_workspace` 都在各自 arch 闸门**之后**被调用
  （`mega_mhc.hpp:223 → :233`、`mega_gate.hpp:159 → :174`、`attention.hpp:313 → :326`），
  所以 H100 上走不到断言。本篇用**与 C++ 逐行对应的 Python 复刻**验证契约语义（Part A）。
- **可执行规范也不可直接跑**：`tests/test_mega_mhc.py` 被 `@test_filter(get_arch_major() == 10)` 门控
  （`:241`），H100 整体跳过；本篇按它的两步语义复刻。
- **真机实测的部分**：捕获状态 API 语义、捕获内 zeroing 被重放、`empty` capture-safe、
  真实 SM90 GEMM（自研 / cuBLASLt / FP8 / PDL）的 capture/replay 逐位、发射开销、profiler 取证——
  全部在 H100(sm90) 上真实运行。
- **未覆盖**：`Invalidated` 捕获态（需要一次失败的捕获使 stream 失效）在本机未构造；DeepEP 的
  `EP_AVOID_RECORD_STREAM` 需要 `_C` + NCCL ≥2.31（本机阻塞，见第 86 篇），仅源码分析。

---

## 8. 参考文献（真实链接）

- DeepGEMM：https://github.com/deepseek-ai/DeepGEMM （本篇坐标 `057ca59`，release #462）
- PyTorch CUDA Graph：https://docs.pytorch.org/docs/stable/notes/cuda.html#cuda-graphs
- `torch.cuda.CUDAGraph` / `torch.cuda.graph`：https://docs.pytorch.org/docs/stable/generated/torch.cuda.CUDAGraph.html
- DeepEP：https://github.com/deepseek-ai/DeepEP （本篇坐标 `93eb6eb`）
