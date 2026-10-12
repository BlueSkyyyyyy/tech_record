# MPK 线上实现的构建与端到端冒烟测试

> 本文是 `docs/06`（运行层源码走读）/`docs/07`（单 task device）/`docs/08`（编译层依赖分析器）
> 的**构建侧补齐**。`docs/06` §0 当时明确写着「mirage 是一套 4.4 万行 C++ + 4.2 万行 CUDA 的完整
> 编译器/运行时，在目标卡 H100/sm90 容器里从零编译成本过高，本轮不做构建/运行实验」。本文把这条
> 尾巴补上：**在 `kernel_lab`（目标卡 H100/sm90，CUDA 13.2）容器里真把 mirage 编译出来**，并用
> `PersistentKernel` API 搭了一个**最小 megakernel**（单层 RMSNorm）跑到了 **JIT 编译 + launch**。
>
> 快照：`https://github.com/mirage-project/mirage` 分支 `mpk`，**commit `f9eb70c`**（与
> `docs/06`/`docs/07`/`docs/08` 同一快照）。行号均对着该快照核对。
>
> 定位：本文解决「线上 MPK 到底能不能在这台机器上跑起来、跑到哪一步」。**主机侧全链路成功**；
> **设备侧 worker/scheduler 事件循环未终止**，本轮停在 launch 之后（下文 §3–§4 给出原始证据与
> 假设），因此计为**部分完成 + 一条阻塞**（见 ROADMAP「阻塞」）。

## 0. 一句话结论

| 阶段 | 结果 | 证据 |
|---|---|---|
| 源码构建 `pip install -e .` | **成功**（`mirage_runtime` 原生库 + cython 扩展 + editable wheel） | `harness/mpk_build_run.out.txt` §一 |
| `import mirage` | **成功**（0.2.4） | 同上 |
| 生成 megakernel（tGraph → `test_rank0.cu`） | **成功**（18 tasks / 3 events） | 同上 §二 |
| nvcc JIT 编译（sm_90a） | **成功**（~27 s / 次） | `harness/mpk_smoke.out.txt` |
| 常驻 kernel launch（worker/scheduler 双 stream） | **成功**（host 打印 `Finished Launching Persistent Kernel (Async)`） | 同上 |
| kernel 执行完成 / 终止 | **未完成**（`torch.cuda.synchronize()` 不返回、GPU 自旋 100%） | 同上 |

判据映射：到 launch 为止已经能看到 MPK 的 **C1（单次 launch）** 落地；但**执行不收敛**意味着本轮
**无法给出端到端数值/延迟结论**，C2–C5 仍只有 `docs/06`–`docs/08` 的源码级证据。

## 1. 构建：环境、步骤与真实报错

### 1.1 环境

- 容器 `kernel_lab`（`scripts/lab.sh`），目标卡 **H100 80GB HBM3 / sm90 / 132 SM**。
- `nvcc` release 13.2（V13.2.78）；`torch 2.12.0a0`；`cmake 3.31.6`；`ninja`；`mpirun`（`/usr/local/mpi`）。
- 构建根目录（容器内）：`/ssd/mirage_mpk_build`（快照 `f9eb70c` 的拷贝）。

### 1.2 步骤

1. `git submodule update --init --recursive`：`deps/cutlass`、`deps/json`、`deps/z3`。
   github 直连**前两次 clone 超时**，git 自动重试后全部成功（cutlass 152M / json 17M / z3 33M）。
2. `python/mirage/_cython/core.pyx` 等由 `setup.py` 走 `cmake + make -j8` 编译出
   `libmirage_runtime.a`（含 `src/kernel/*.cc`、`src/threadblock/cuda/*.cu` 等）与两个 Rust
   静态验证库（`abstract_subexpr` / `formal_verifier`，`cargo` 由 `setup.py` 自动 `rustup` 安装）。
3. **唯一真实报错**：运行期依赖 `tg4perfetto @ git+https://github.com/flashinfer-ai/tg4perfetto.git`
   在 `pip` 隔离环境里 `git clone` 超时（`RPC failed; curl 28 ... Couldn't connect to github.com:443`）。
   绕过：`pip install tg4perfetto`（PyPI 有同名包，0.0.6）后，改用
   `pip install -e . --no-deps --no-build-isolation` 完成安装。
4. 为省编译时间，把 `CMakeLists.txt:28` 的 `CMAKE_CUDA_ARCHITECTURES` 由 `"75;80;86;89;90"`
   收窄为 `"90"`（本机只跑 sm90）。

### 1.3 结果

`build2.log` 尾部：`[100%] Built target mirage_runtime` → `Mirage runtime library built
successfully.` → `Successfully installed mirage-project-0.2.4`。校验 `python3 -c 'import mirage'`
输出 `mirage 0.2.4`。完整命令与输出见 `harness/mpk_build_run.out.txt`。

> **环境副作用（重要）**：mirage 的 `pyproject.toml` 把 `z3-solver` 钉死在 `4.16.0.0`
> （soname 与扩展链接一致），而容器里 `tilelang 0.1.10` 要求 `z3-solver<4.15.5`。二者互斥。
> 本轮在跑完 mirage 后**已把 `z3-solver` 恢复为 `4.15.4.0`**（`tilelang` 恢复可用），因此当前
> 状态下 `import mirage` 会因缺 `libz3.so.4.16` 失败；下次要用 mirage 需重新 `pip install
> 'z3-solver==4.16.0.0'`（用完再切回，避免影响其它并行专题）。

## 2. 最小 megakernel：单层 RMSNorm PersistentKernel

实验脚本 `harness/mpk_smoke.py`（模仿快照自带 `tests/runtime_python/test_mode/test_rmsnorm_testmode.py`，
但把 worker/scheduler 数量做成参数、并加计时与正确性校验）。核心：

- `PersistentKernel.get_default_init_parameters()`（`persistent_kernel.py:669`）取默认参数，
  置 `test_mode=True`（`persistent_kernel.py:318` 会加 `-DMPK_TEST_MODE`；`:535` 会自动补全
  `tokens/step/prompt_lengths/...` 等 meta 张量，`test_mode=True` 时缺失项传空指针，见 `:3413-3418`）。
- `pk.rmsnorm_layer(input, weight, output, grid_dim=(batch,1,1), block_dim=(256,1,1))`
  （`persistent_kernel.py:865`），`pk.compile()` 生成并 JIT 编译，`pk()` 走
  `launch_func`（`:3574`）→ `launch_persistent_kernel`（`persistent_kernel.cuh:1955`）。
- 便不依赖任何 HuggingFace 权重：输入用随机张量，与 PyTorch 的 RMSNorm 逐元素比对。

`compile()` 生成的 `test_rank0.cu` 与 `task_graph_rank0.json` 落在 `MIRAGE_HOME`（本机
`/ssd/mirage_mpk_build/`），nvcc 命令行：

```
-gencode=arch=compute_90a,code=sm_90a -DMPK_ENABLE_TMA -DMIRAGE_GRACE_HOPPER -DNDEBUG
-DMODE_OFFLINE -DMPK_TEST_MODE -DMIRAGE_USE_CUTLASS_KERNEL=0
-DMPK_MAX_NUM_BATCHED_REQUESTS=1 -DMPK_MAX_NUM_BATCHED_TOKENS=1 -DMPK_MAX_NUM_PAGES=1 ...
```

这与 `docs/07` 「一个 task = 一个 CTA」的编译层映射一致；`-DMPK_ENABLE_TMA` /
`-DMIRAGE_GRACE_HOPPER` 由 `persistent_kernel.py:365-370` 对 `target_cc==90` 固定加上。

顶点规模：**18 tasks / 3 events**。事件类型（`runtime_header.h:229-235`）与触发次数：

| event | 类型 | 含义 | num_triggers | 覆盖 task |
|---|---|---|---|---|
| 0 | `EVENT_TERMINATION`=911 | 终止 | 1 | task 0 |
| 1 | `EVENT_LAUNCH_DEPENDENT_TASKS`=903 | 派发后继 | 1 | task 2..18 |
| 2 | `EVENT_END_OF_TASK_GRAPH`=910 | 图末 | 16 | task 0 |

即：`task0`(type 0，begin) → `task1`(type 10) → 16 个 `type 154`(rmsnorm，`request_id` 0..15) → 图末。
16 个 rmsnorm task 各触发一次 `EVENT_END_OF_TASK_GRAPH`（故 `num_triggers=16`）。这与
`docs/06` §3 描述的「事件计数器阈值 `num_triggers × iteration` 精确等依赖」
（`persistent_kernel.cuh:1185`）完全对得上。

计时与输出见 `harness/mpk_smoke.out.txt`：`[compile] 27.1 s`。

## 3. 观察到的现象：主机构建/launch 成功，设备侧不终止

`harness/mpk_smoke.out.txt` 原始日志（`stdbuf -oL` 行缓冲，`timeout 90`，`EXIT=124`）：

```
[MPK INIT] Total tasks: 18, Total events: 3
[compile] 27.1 s
[run] launching persistent kernel ...
worker kernel & scheduler kernel
smem size: 205824
Finished Launching Persistent Kernel (Async)
<此后 torch.cuda.synchronize() 永不返回；nvidia-smi 显示该卡 100% 占用/自旋>
```

关键读法：

- `worker kernel & scheduler kernel` 来自 `persistent_kernel.cuh:1977` 分支，说明
  `split_worker_scheduler == true`（该字段在 `:1795` 恒置 true），走的是**worker/scheduler
  双 stream + `cudaEventRecord`/`cudaStreamWaitEvent`** 的路径（`:1990-2010`）。
- `Finished Launching Persistent Kernel (Async)` 打印在 **两个 grid 都已 launch、event 已 record、
  default stream 已 wait 之后**（`:2015` 附近），故 **launch 本身是成功的**。
- 随后 `pk()` 返回，脚本调用 `torch.cuda.synchronize()`：它在 default stream 上等待，而 default
  stream 已 `cudaStreamWaitEvent(worker_done_event / scheduler_done_event)`（`:2002-2008`）。
  这两个 event 只有在 worker/scheduler 两个常驻 kernel **退出后**才会 ready。脚本永不返回 ⇒
  **两个常驻 kernel 没有退出**（GPU 100% 自旋佐证）。

只用 4 worker / 4 scheduler（总 5 个 block）也照样不退出，说明**不是 SM 占用不足/block 不可
共驻**导致的经典死锁；问题在**事件与终止协议本身**（或该快照的 offline/test 路径）。

用 `CUDA_LAUNCH_BLOCKING=1` 复现（`harness/mpk_build_run.out.txt` 未收录，见本轮 commit 前
的临时日志）会卡在 `worker kernel & scheduler kernel` 之后、`Finished Launching` 之前——
这正是把异步 launch 改成同步后、worker 等不到尚未 launch 的 scheduler 的表现，**与结论一致**：
死锁发生在设备侧 worker/scheduler 的相互等待，而不是 host。

## 4. 假设与定位（标注「待验证」）

按 `docs/06`/`docs/08` 的走读，offline 模式下 kernel 的收敛路径应为：

1. `launch_persistent_kernel` 里的 `prepare_kernel`（`persistent_kernel.cuh:205-231`）把
   **图末事件**（`end_of_task_graph_event_pos = num_events-1`）塞进 `sched_queue[0]`，作为
   「开跑第一批」的种子。
2. scheduler 收到 `EVENT_END_OF_TASK_GRAPH`（`:1379`）→ `prepare_next_batch(config)`
   （MODE_OFFLINE 版在 `:278-528`）构造 batch：空 batch 时先装 prefill 请求（`next_request_id
   < total_num_requests`，默认 `tokens.shape[0]=1`）→ `num_tokens>0` → `return true` →
   推 `compute_task_id(iteration+1, 1)` 启动图（`:1405-1415`）。
3. 16 个 rmsnorm task 跑到 `EVENT_END_OF_TASK_GRAPH` 的 16 次触发（`:1185`）→ 图末事件再次
   派发 → `prepare_next_batch` 第二次：finalize 掉已完成的请求（step 推进、`max_seq_length=1`
   时命中 `step+step_advance+1 >= max_seq_length`→`request_ids[i]=-1`，`:334-340`）→ 无活跃请求、
   `next_request_id >= total_num_requests` → `num_tokens==0` → `return false`（`:526`）→
   `terminate_schedulers`（`:888`）向所有 sched 队列压 `EVENT_TERMINATION` → 各 scheduler 在
   `:1361` 的 `is_termination_event` 分支终止自己并向所属 worker 写终止 task → worker 退出 →
   `worker_done_event`/`scheduler_done_event` ready。

逻辑上应当收敛，实际未收敛。可能的落点（**均为待验证假设，非结论**）：

- **H1：任务派发/事件计数的 iteration 数不匹配**。`prepare_kernel` 注入的图末事件不计入
  `all_event_counters`，而 worker 侧的触发阈值是 `num_triggers × get_task_iteration_num(task_id)`
  （`:1185`）；若注入事件与首批 task 的 iteration 编号（`compute_task_id(iteration+1, ·)`）错位，
  图末事件可能永远凑不满 16 次触发，或首批 task 根本没被派发。
- **H2：sm90 + `-DMIRAGE_GRACE_HOPPER` + `MPK_ENABLE_TMA` 路径的 task 单体内同步问题**。
  本轮唯一没能在 H100 上交叉验证的编译分支就是这套 Hopper 专属宏（`:365-370` 对
  `target_cc==90` 固定打开），而 `docs/07` §1 的 `linear/attention` task 大量依赖
  mbarrier/TMA transaction-count——一个 task 卡住即可让整图不收敛。
- **H3：快照 offline/test 路径本身在该 commit 上不完整**（test_mode 代码在 `persistent_kernel.py:535`
  才补全 meta 张量、`:3413` 允许空指针，属于较新的「测试基建」，不排除与 runtime 存在版本错位）。

排除项：**不是**构建错误（原生库/扩展都成功）、**不是** launch 失败（host 已打印 launch 完成）、
**不是** SM 数量不足（4 block 也复现）。

## 5. 与论文/源码走读的呼应

- 论文 §6.1（快照 `mpk-v2.txt` 行 1400+ 一带）描述的实现规模与「worker/scheduler 分工」在本机
  以 `worker kernel & scheduler kernel` 双 stream 形态**真实出现**，与 `docs/06` §4 的
  `execute_scheduler`（`persistent_kernel.cuh:1254-1523`）一致。
- `docs/06` §4「scheduler 收到 `END_OF_TASK_GRAPH` → `prepare_next_batch` 连续服务」
  （`:1379-1420`）正是本轮挂起所处的代码路径；`docs/08` §6 的 fork/join event fusion 产出
  （event 1 的 `LAUNCH_DEPENDENT_TASKS` 扇出、event 2 的 16 路 join）也在 `task_graph_rank0.json`
  里被真实生成。
- 因此本轮的增量价值是**把「代码怎么写」推进到「代码在这台机器上跑到哪」**：源码级 C1–C5
  （`docs/06` §7）不变，但**端到端可运行性**出现了新的、可复现的阻塞点，这是纯走读看不到的。

## 6. 结论与后续

**做到**：在目标卡 H100/sm90 上从零构建 mirage（`f9eb70c`）成功；用 `PersistentKernel` API
生成并 JIT 编译了一个 18 task / 3 event 的最小 megakernel，成功 launch（双 stream worker/scheduler）。

**未做到**：kernel 未收敛，无法取端到端延迟/正确性数字。

**后续（下轮候选，任选其一）**：

1. 在 `test_rank0.cu` 上开 `-DMPK_ENABLE_VERBOSE`（或加计数打印）逐事件 trace，定位是哪个
   event 的 `all_event_counters` 永远到不了阈值（验 H1）；必要时把 `num_workers/schedulers`
   固化成 `get_configurations_from_gpu` 的 128/16 再复现一次。
2. 换一个**非 Hopper-TMA** 的 task（例如纯逐元素的 `silu_mul`，若该快照的 task 编译器支持）
   建同构最小图，隔离 H2。
3. 直接用官方 CI 入口 `tests/ci-tests/run_python_tests.sh` / `run_ci_tests_qwen3.sh` 的路径跑
   （后者需要 Qwen 权重，本轮不具备）。

**阻塞**（已记入 ROADMAP「阻塞」）：MPK 在本容器共享 GPU 上**执行不收敛**，本轮无法给出端到端实测。
