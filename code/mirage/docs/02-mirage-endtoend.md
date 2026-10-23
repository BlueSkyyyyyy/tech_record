# 从 HF 模型到 megakernel：一次 `compile()` 到底做了什么

> 这是「Mirage 项目分析」的第 2 篇。主角是 `demo/qwen3/demo.py` 和
> `PersistentKernel.compile()`：看一个 HuggingFace 模型怎样被**画成一张图**、再被**浇成一块混凝土**
> （一份可执行文件 + 一份任务清单），最后变成一次 kernel launch。

## 0. 先记住一个画面

用 MPK 写模型，不像平时写 PyTorch——你不是在「执行代码」，而是在**画施工图**。

- `mpk.linear_layer(...)`、`mpk.attention_layer(...)` 这些调用，是在图纸上画一根梁，**不会真的算**；
- `compile()` 是把图纸送去预制厂：生成 C++/CUDA 源码、编译成 `.so`；
- 之后每次推理，只是**按一次电钮**（`mpk()`），整栋楼按图纸自己跑完。

> **类比的边界**：施工图是死的，MPK 的图也是静态的——形状、层数、并行切分在 `compile()` 时
> 全部定死。模型结构一变（改层数、改 batch 上限、改 TP 度），就要重新画图、重新编译。
> 换来的是运行期**没有任何 Python 解释器参与**，也没有 host 侧的 kernel 派发。

读完这一篇，你应该能回答：

1. `PersistentKernel` 里那些 `*_layer()` 调用，到底往图里放了什么？
2. `compile()` 生成了哪两个文件？它们各自负责什么？
3. 为什么一次 `mpk()` 调用能跑完整个 decode 循环？

## 1. 全链路：五个阶段

先看全景。从左到右是「编译期」，从 `init` 开始是「运行期」：

```
① 建图            ② 生成              ③ 编译 + 初始化            ④ 运行
PersistentKernel → test.cu          → nvcc → .so → import     → mpk()
  + layer 调用      + task_graph.json     → init_func（分配队列）      一次 launch
```

对应到代码：

| 阶段 | 位置 |
|---|---|
| ① 建图 | `demo/qwen3/demo.py:347`（`mpk = mi.PersistentKernel(...)`）到 `:840` |
| ② 生成 | `PersistentKernel.compile()`（`python/mirage/mpk/persistent_kernel.py:3207`），内部调 `generate_task_graph`（`:3223`） |
| ③ 编译 + 初始化 | 同一个 `compile()` 里：`nvcc`（`:3243` 之后）→ `importlib` 加载 `.so`（`:3366-3374`）→ 立刻调 `init_func`（`:3434`） |
| ④ 运行 | `demo/qwen3/demo.py:949` 的 `mpk()` |

**注意 ③ 这一步的特殊之处**：`compile()` 不只是编译，它在末尾直接把 `.so` import 进来、
把 `init_func` / `launch_func` / `wait_func` / `finalize_func` 绑到对象上，**并立刻调用
`init_func`**——加载 `task_graph.json`、`cudaMalloc` 所有队列、建两条常驻 stream，都在
`compile()` 里做完了。所以用户代码里看不到单独的 `init()`，`compile()` 之后直接 `mpk()` 就行。

下面逐段拆。

## 2. 阶段①：把模型画成图

`demo/qwen3/demo.py` 的前半段和普通推理脚本没区别：加载 Qwen3 权重、tokenizer、规划 KV cache、
准备 `tokens`/`step` 等元数据。真正不同的是从 `:347` 开始：

```python
mpk = mi.PersistentKernel(
    mode="offline",
    world_size=world_size,
    mpi_rank=rank,
    num_workers=num_workers,
    num_local_schedulers=num_schedulers,
    num_remote_schedulers=0,
    max_seq_length=args.max_seq_length,
    max_num_batched_requests=args.max_num_batched_requests,
    max_num_batched_tokens=args.max_num_batched_tokens,
    max_num_pages=max_num_pages,
    kv_groups=kv_plan.group_specs(),
    meta_tensors={"step": step, "tokens": tokens, ...},
    profiler_tensor=profiler_tensor,
)
```

这段在干什么：**声明这张图的边界条件**——几张卡、多少 worker/scheduler、序列和 batch 上限、
KV cache 怎么分页、哪些张量是「外部传进来的元数据」。

然后挂权重、建中间张量：

```python
w_q = mpk.attach_input(layer.self_attn.q_proj.weight, name=f"layer_{i}_q_proj")
w_qkv = mpk.shuffle_tensors(inputs=[w_q, w_k, w_v], shuffled_dim=0, ...)
attn_in = mpk.new_tensor(dims=(...), dtype=mi.bfloat16, name="attn_in", io_category="cuda_tensor")
```

`attach_input` 做了一件很关键的事：把 PyTorch tensor 的**裸 GPU 指针**登记进图
（`persistent_kernel.py:746-776`）。所以它必须把 tensor 的引用存下来防止被 GC——
否则指针悬空，运行期就是随机错误。

接着是「一层 Transformer」的图（Qwen3-8B 是 dense 模型，节选）：

```python
mpk.rmsnorm_layer(input=x, weight=w_norm, output=rmsnorm_out, grid_dim=(..., 1, 1), block_dim=(128,1,1))
mpk.linear_layer(input=rmsnorm_out, weight=w_qkv, output=attn_in, grid_dim=(...), block_dim=(128,1,1))
mpk.paged_attention_layer(input=(...), k_cache=..., v_cache=..., output=attn_out, ...)
mpk.linear_with_residual_layer(input=attn_out, weight=w_o, residual=x, output=x, ...)
mpk.rmsnorm_layer(...)
mpk.linear_layer(... gate/up ...)
mpk.silu_mul_layer(...)
mpk.linear_with_residual_layer(... down ...)
```

注意每个调用都带 `grid_dim` / `block_dim`：**并行度是画图时你自己定的**，不是编译器自动推的。
这也是 MPK 和「自动并行编译器」最大的区别——它把调度自由度交给你，只把**运行期派发**自动化。

> 读代码时的感觉：这一大段像在「手写一张 DAG 的邻接表」，每一行是图上的一个节点。

## 3. 阶段②：`compile()` 生成两个文件

`mpk.compile()`（`persistent_kernel.py:3207`）做的第一件事是：

```python
results = self.kn_graph.generate_task_graph(num_gpus=self.world_size, my_gpu_id=self.mpi_rank)
```

这一行进 C++（`src/kernel/runtime.cc:1915`），把整张算子图细化为 **task/event 图**，然后写出两个文件：

| 文件 | 内容 | 谁用 |
|---|---|---|
| `test.cu` | 运行时骨架 + task 派发器 + Python 模块胶水 | `nvcc` 编成 `.so` |
| `task_graph.json` | 全部 task / event 的描述（类型、依赖、张量指针） | 运行期 `init_func` 加载 |

`test.cu` 里其实只有两块「生成出来的东西」：

**第一块：图加载器** `_init_persistent_kernel(...)`（`runtime.cc:1215`）。它按 JSON 里的
`all_tasks` / `all_events` / `first_tasks` 逐条构造 `FullTaskDesc`、`EventDesc`，并把用户挂进来的
tensor 指针填进去。**task 的数量和类型在编译期就固定了**，运行期只是把指针填上。

**第二块：task 派发器** `_execute_task(task_desc, runtime_config)`（`runtime.cc:1886`）。它对
`TaskRegister` 里注册的每个 `(TaskType, variant_id)` 生成一个分支：

```cuda
if (task_desc->task_type == TASK_EMBEDDING && task_desc->variant_id == 0) { ... }
else if (task_desc->task_type == TASK_RMS_NORM && task_desc->variant_id == 0) { ... }
...
```

这就是论文 §4.2 说的「每个 task 的 CUDA 实现生成」：**所有算子的 device 实现都编进同一个
函数**，运行期用 `task_type` 做一次分支跳转。这也是 megakernel 的「代码足迹」来源——融合的
task 种类越多，这个函数越大（相关代价在 MegaKernel 专题的 `docs/04` §10 有实测）。

编译产物最后是一个 Python 扩展模块 `__mirage_launcher`，由 `HARD_CODE`
（`persistent_kernel.py:22-226`）提供四个入口：

| 入口 | 作用 |
|---|---|
| `init_func` | 调 `init_persistent_kernel(...)`：加载 JSON、分配队列/事件、建流 |
| `launch_func` | 调 `launch_persistent_kernel(stream)`：**唯一的一次 launch** |
| `wait_func` | 等两条常驻 stream 退出 |
| `finalize_func` | 释放 |

`nvcc` 的编译命令在 `get_compile_command`（`:248`）里拼：`-O3`、`-shared`、
带 `-I deps/cutlass/include`、`-I deps/json/include`，并按模式加 `-DMODE_OFFLINE` 之类的宏。

## 4. 阶段③④：编完就初始化，然后一按到底

`compile()` 的最后几步（`persistent_kernel.py:3366-3434`）：

```python
spec = importlib.util.spec_from_file_location("__mirage_launcher", so_path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
self.init_func = getattr(mod, "init_func")      # 四个入口
...
self.init_func(meta_tensors_ptr, profiler_buffer_ptr, self.mpi_rank,
               self.num_workers, ..., model_tensor_names, model_tensor_ptrs, ...)
```

也就是说：**编译完立刻初始化**，`init_func` 一次性把所有资源（队列、事件、task 表、stream）分配好。

真正的运行只有一行（`demo.py:949`）：

```python
mpk()
```

这一行背后是 `launch_func` → `launch_persistent_kernel(stream)`，起 **一颗** grid：

- `blockIdx.x < num_workers` 的 block 是 **worker**：循环取 task、等依赖、执行；
- 其余 block 是 **scheduler**：消费事件队列，把 task 派发给 worker。

关键点：**整个 decode 循环在 kernel 里跑**。`step` 是 meta tensor，kernel 每完成一轮解码就把它
加一；`tokens` 被原地追加。所以 Python 侧不需要 `for token in range(...)`——一次调用，
一直跑到 EOS 或长度上限。这就是 README 里「一次 launch」的字面含义。

> 反过来，这也解释了为什么 `mpk()` 之后可以直接打印 `tokens[r, :step[r]+1]`：结果是内核写回显存的。

## 5. 为什么值得这么麻烦

把「图」和「执行」拆开，换来三件事：

1. **运行期零 host 开销**：没有 Python、没有 kernel launch 循环，只有一次 launch + 设备内调度。
2. **跨算子流水成为可能**：worker 执行 task A 的同时，scheduler 已经在给别的 worker 派 task B
   （下一轮迭代的 task 也能提前进队）——这是 kernel-per-operator 做不到的。
3. **调度策略可编程**：task 的粒度、grid/block、依赖关系都是你画图时写的，想融合就融合。

代价也很直接：

- **静态**：形状、并行度、层数全定死，改一点要重编。
- **难调试**：出错时你面对的是一颗跑了整轮推理的常驻 kernel（`docs/04` 会看到真实案例）。
- **要自己算并行度**：`grid_dim` 填错不会报错，只会算错或变慢。

## 6. 三个必须知道的坑

**坑 1：`num_workers` 必须和 SM 数对上。**
关系是 `num_workers + (num_local_schedulers + num_remote_schedulers) / 4 = SM 数`。
填错的典型症状不是报错，而是**kernel 起来后永远不结束**——worker 和 scheduler 互相等。
（MegaKernel 专题的 `docs/09` 记录了这条排查过程。）

**坑 2：`attach_input` 的 tensor 必须活到运行结束。**
生成代码里存的是裸指针。`PersistentKernel` 用 `_torch_tensor_refs` 保活（`:773-776` 有注释），
但你如果在中途 `del` 掉原 tensor、或让它被替换，指针就悬空了。

**坑 3：dtype / layout 的断言很严。**
`attach_input` 只接受 row-major（允许外层 stride 有 padding）或二维 column-major，
且最内层 stride 必须是 1（`:755-762`）。FP8 的 scale 张量常常是列主序，所以专门开了后门。

## 7. 三句话小结

- 用 MPK 写模型 = 画一张静态图：`*_layer()` 只登记节点，不执行。
- `compile()` 生成 `test.cu`（运行时骨架 + task 派发器）和 `task_graph.json`（任务清单），
  再由 `nvcc` 编成 Python 扩展；图与代码分离。
- 运行期只有一次 launch，decode 循环在 kernel 内推进，Python 只负责按电钮。

下一篇进入这颗常驻 kernel 的内部：worker 怎么取 task、scheduler 怎么派活、事件计数器怎么保证
依赖正确。
