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
> **§7（续做 P25-1）**进一步用官方 test_mode 用例复现挂起（3/3），并以插桩消融把根因定位到
> **`execute_worker` 取 task 轮询循环的时序竞态**：加一处无害的自旋计数即可让官方用例 3/3 收敛
> （正确性 `max|out-ref| ≤ 0.03125`）。**§8（再续 P25-1）**把「端到端延迟」补上：在收敛构建上测出
> 单层 RMSNorm megakernel 的**每次调用 36.4–39.1 µs**（对照 PyTorch 朴素 54.0 µs），128/16 连跑 3/3
> 收敛、plain 构建复现挂起（`EXIT=124`）。**§9（三续 P25-1）**把根因从「worker 轮询循环内存序」
> 降到「launch/握手时序敏感竞态」并给出 dtype 转换 kernel 的强签名。**§10（四续 P25-1）**再把 §9 的
> 两条候选（默认流历史 / host 排序）**都证伪**：挂起是**设备侧 worker 与 scheduler 两个常驻 kernel
> 同时自旋的双向死锁**，只被「前置 dtype 转换 kernel」这一个开关避开（与流 / 完成 / 规模 / 频率无关）。

## 0. 一句话结论

| 阶段 | 结果 | 证据 |
|---|---|---|
| 源码构建 `pip install -e .` | **成功**（`mirage_runtime` 原生库 + cython 扩展 + editable wheel） | `harness/mpk_build_run.out.txt` §一 |
| `import mirage` | **成功**（0.2.4） | 同上 |
| 生成 megakernel（tGraph → `test_rank0.cu`） | **成功**（18 tasks / 3 events） | 同上 §二 |
| nvcc JIT 编译（sm_90a） | **成功**（~27 s / 次） | `harness/mpk_smoke.out.txt` |
| 常驻 kernel launch（worker/scheduler 双 stream） | **成功**（host 打印 `Finished Launching Persistent Kernel (Async)`） | 同上 |
| kernel 执行完成 / 终止 | **未完成**（`torch.cuda.synchronize()` 不返回、GPU 自旋 100%） | 同上 |
| §7：官方用例收敛性 | plain 3/3 挂起；加 worker 轮询插桩后 3/3 收敛（`max abs diff ≤0.03125`） | `harness/mpk_converge_*.out.txt` |
| §8：端到端延迟（收敛构建） | **36.4–39.1 µs / 次调用**（1×RMSNorm 16×4096），对照 PyTorch 朴素 54.0 µs；128/16 连跑 3/3 收敛 | `harness/mpk_latency.out.txt` / `harness/mpk_latency_repeat.out.txt` |

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

## 7. 收敛性诊断（续做 P25-1）：官方用例复现挂起 + 时序敏感性的定位

§3–§6 的结论来自**自建** `mpk_smoke.py`（4 worker / 4 scheduler）。本轮（续做）先把"挂起"这件事
钉死，再用插桩把它从「玄学死锁」推进到「可复现、可定位的时序竞态」。所有实验：
目标卡 H100/sm90、**`CUDA_VISIBLE_DEVICES=7` 隔离到空闲卡**、快照 `f9eb70c`、每组用例内**顺序**跑
若干次，单次 `timeout 200s`。原始输出见 `harness/mpk_converge_plain.out.txt`、
`harness/mpk_converge_instrumented.out.txt`、`harness/mpk_converge_attribution.out.txt` 与
`harness/mpk_dbg_trace.out.txt`。

### 7.1 用快照自己的测试复现（排除自建脚本因素）

直接跑快照自带用例 `tests/runtime_python/test_mode/test_rmsnorm_testmode.py`——它用官方
`get_configurations_from_gpu(0)` 取 **128 worker / 16 scheduler**（与 `docs/06` §7 的 H100→128/16
一致），构建同样的单层 RMSNorm 图。结果：

- **[A] plain 组：3/3 挂起**（`EXIT=124`，GPU 100% 自旋、`torch.cuda.synchronize()` 不返回）。

即挂起**与本专题自建脚本、与 4/4 的小配置无关**，是「快照 `f9eb70c` + 本机 H100/sm90」的行为。

### 7.2 插桩：守门狗 + 逐事件 trace（patch 见 `harness/mpk_watchdog.patch`）

在 `persistent_kernel.cuh` 两处轮询循环各加一个"自旋计数守门狗"（超阈值 `printf` 诊断后 `__trap()`），
并在 scheduler 0 上打印每次取到的 event 与 `prepare_next_batch` 的返回：

- `execute_worker` 取 task 等待循环（`persistent_kernel.cuh:1013` 一带，等
  `worker_queue_last_ready_task_id`）；
- `execute_scheduler` 事件等待循环（`:1335` 一带，等 `sched_queue_last_ready_event_id`）；
- 通过 `get_compile_command`（`persistent_kernel.py:365-371`，`target_cc==90` 分支）加编译宏
  `-DMPK_DBG_WATCHDOG` 打开。

**[E] full instrumented 组：3/3 收敛**（`EXIT=0`），与 PyTorch 参考的 `max|out-ref|` =
0.015625 / 0.015625 / 0.03125（bf16 容忍 0.05 内）。**守门狗 3 次都没触发**——说明收敛时两条
轮询循环都没有"永久自旋等一个恒定不来的事件"。

### 7.3 完整事件序列（trace 实证）

收敛时的 scheduler 0 逐事件 trace（`harness/mpk_dbg_trace.out.txt`）：

```
[SDBG] sched0 got event_id(2) etype(910) nt(16) iter(0) cur0(0) last0(1)   # 种子: 图末事件
[SDBG] sched0 got event_id(1) etype(903) nt(1)  iter(0) cur0(1) last0(1)   # LAUNCH_DEPENDENT
[SDBG] sched0 got event_id(2) etype(910) nt(16) iter(1) cur0(1) last0(2)   # 16 个 rmsnorm 全部触发后
[SDBG] prepare_next_batch -> FALSE (terminate)
[SDBG] sched0 got event_id(0) etype(911) nt(1)  iter(1) cur0(2) last0(3)   # 终止
```

逐步对上 `docs/09` §2 的 18 task / 3 event 与 §4 的收敛路径：`prepare_kernel` 把图末事件
（`end_of_task_graph_event_pos = num_events-1 = 2`）种进 `sched_queue[0]`（`:226-233`）→ scheduler 0
取到它、`prepare_next_batch` 返回 true、派发 `compute_task_id(1, 1)`（begin task，`:1399-1405`）→
begin task 触发 event 1（`EVENT_LAUNCH_DEPENDENT_TASKS`，`nt=1`）→ 派发 16 个 `type 154` rmsnorm
task（`:1424-1448`）→ 16 个 task 各触发一次 event 2（`nt=16`）凑满阈值（`:1185`）→ scheduler 再次
处理 event 2、`prepare_next_batch` 返回 **false**、`terminate_schedulers`（`:1394`）→ 各 scheduler
在 `:1361` 终止、给 worker 写终止 task、双 kernel 退出。**与纯源码走读的预测完全一致**。

### 7.4 判别：挂起到底是哪一类

- **排除 H2（task 单体内 TMA/mbarrier 卡住）**：若某个 `type 154` 卡在 TMA/mbarrier，event 2 的
  `all_event_counters` 到不了 `num_triggers=16`，就不会第二次派发 event 2，也就不会打印
  `prepare_next_batch -> FALSE`。收敛组里它被打印，说明**16 个 task 全部跑完并触发**。
- **排除 H1 的"阈值永远凑不满"形态**：收敛组事件计数完整到阈值。
- **结论**：挂起是 **worker↔scheduler 之间的时序敏感同步竞态**，不是某个 task 体能卡住。

### 7.5 归因：哪一段插桩才是决定性的（variation 实验）

把插桩拆成三个正交部件，做消融（每组 2–3 次，见 `harness/mpk_converge_attribution.out.txt`）：

| 组 | worker 轮询自旋计数 | scheduler 自旋计数 | scheduler 逐事件 printf | 结果 |
|---|---|---|---|---|
| A plain | – | – | – | **3/3 挂起** |
| B prints-only | – | – | ✔ | **2/2 挂起** |
| C worker-spin-only | ✔ | – | – | **2/2 收敛** |
| D worker-spin + prints | ✔ | – | ✔ | 2/2 收敛 |
| E full | ✔ | ✔ | ✔ | 3/3 收敛 |

**决定性变量只有一个：`execute_worker` 取 task 等待循环（`:1013`）里"每轮多一次自旋自增+分支"
这点时序扰动**。scheduler 侧的逐事件 `printf`（B 组，无任何自旋计数）**不足以**修复——对照组 C
（只有 worker 侧自旋计数、**连 printf 都没有**）就已经收敛。最小修复 patch 见
`harness/mpk_workaround_only.patch`：
仅在 worker 轮询循环里加 `long _dbg_spins; ... ++_dbg_spins;`（打印分支不会触发），即可让官方用例
稳定收敛——这**反证了竞态就在 worker 侧"等 task"的抢跑/丢失唤醒**上。

### 7.6 局限与下一步

- 本轮的强证据是**可复现 + 可归因到 worker 轮询循环**；**未**把根因收敛到某一条具体的 load/store
  内存序（worker 的 `ld_acquire` 读计数、scheduler 的 `st_relaxed` 写 payload + `atom_add_release`
  写计数已按序）。下一步候选：① 用 `compute-sanitizer --tool racecheck` 跑官方用例抓数据竞争；
  ② 在 worker 侧把"每轮多一次自旋"换成等价语义的 `__nanosleep` 退避，确认是纯时序而非缺 fence；
  ③ 向 MPK 上游核对 `f9eb70c` 的 worker 抢跑路径是否有已知修复。**不构成对上游的 bug 判定**，
  仅为可复现的本机观察。
- 端到端**延迟**仍未测（只测了正确性与收敛/挂起）。收敛组正确性 `max|out-ref| ≤ 0.03125`（bf16）。

## 8. 端到端延迟实测（续做 P25-1：「收敛后测延迟」）

§7 把「不收敛」修成一类**可复现的 worker 轮询时序竞态**（加一处无害的自旋计数即收敛），但只测了
正确性与收敛/挂起，**没有端到端延迟数字**。本节补上：在**收敛配置**上测「一次 megakernel 调用」
从 host 提交到常驻 kernel 终止的墙钟，并扫 worker 数量、与 PyTorch「一算子一 kernel」参照对照。

### 8.1 方法

- **收敛配置**：把 §7.5 的最小修复固化成一个补丁 `harness/mpk_latency_run.patch`（对快照 `f9eb70c`
  的两处最小改动）：
  1. `include/mirage/persistent_kernel/persistent_kernel.cuh` 的 `execute_worker` 取 task 等待循环
     （`:1013` 一带）加一个 `MPK_DBG_WATCHDOG` 门控的自旋计数（与 §7 patch 同源，只是把
     `++_dbg_spins` 放到 `worker_id` 判断之外，让**每个 worker** 都受到同等时序扰动）；守门狗阈值
     放宽到 `2e8`（本机从未触发）。
  2. `python/mirage/mpk/persistent_kernel.py` 的 `target_cc==90` 编译命令（`:365-372`）追加
     `-DMPK_DBG_WATCHDOG`。
- **计时脚本**：`harness/mpk_latency.py`（`mpk_smoke.py` 的多 config 版）。每个 config：
  `pk.compile()`（nvcc JIT sm_90a）→ 20 次 warmup → 200 次计时，**每次 `pk()` 前后各记一对
  CUDA event**，循环后一次 `torch.cuda.synchronize()` 再读 `elapsed_time`。
- **计时口径**：`pk()` 内部把 default stream `cudaStreamWaitEvent` 到 worker/scheduler 的 done
  event（`persistent_kernel.cuh:2002-2008`），而 offline/test 模式下常驻 kernel 处理完一个 batch
  即终止；因此一对 event 测到的是**「host 提交 → 双常驻 kernel 执行 → 终止 → stream 就绪」的完整
  往返**，即一次 megakernel 调用的端到端墙钟。
- 环境：目标卡 H100/sm90，`CUDA_VISIBLE_DEVICES=7` 隔离空闲卡，`batch=16, hidden=4096`（bf16），
  快照 `f9eb70c`。原始输出 `harness/mpk_latency.out.txt`（config 扫描）与
  `harness/mpk_latency_repeat.out.txt`（重复稳定性 + plain 对照）。

### 8.2 结果：延迟 ~36–39 µs，且几乎与 worker 数无关

| config (workers/sched) | max\|out−ref\| | mean (µs) | median | min | max | sd |
|---|---|---|---|---|---|---|
| 128 / 16（= 论文 §5.1 / 官方 `get_configurations_from_gpu`） | 3.13e-02 | **39.05** | 38.99 | 36.22 | 56.19 | 2.14 |
| 64 / 16 | 3.13e-02 | **37.59** | 37.47 | 34.98 | 47.46 | 1.27 |
| 16 / 16 | 3.13e-02 | **36.40** | 36.38 | 33.95 | 47.39 | 1.34 |
| PyTorch 朴素 RMSNorm（kernel-per-operator 参照） | — | **53.99** | — | — | — | — |

- **正确性**：三档 `max|out-ref| = 0.03125`，在 bf16 容差（§7 用 0.05）内；与 PyTorch 参考逐元素一致。
- **重复稳定性**：`harness/mpk_latency_repeat.out.txt` 在 128/16 上连跑 3 次，**3/3 收敛且 PASSED**，
  单次 200-iter 均值 **37.64 / 37.78 / 38.08 µs**（run-to-run 抖动 <1.5%），与上表 39.05 µs 同量级。
- **plain 对照**：同一脚本、同一配置，仅删掉 `-DMPK_DBG_WATCHDOG`（回到快照原始头文件语义）后
  再次运行——`TIMEOUT_EXIT=124`，停在 `[run] launching persistent kernel ...` 之后、无
  `[check]/[timing]/[result]`，**复现 §7 的挂起**。二者合起来把「修复是否真的在起作用」钉死。

### 8.3 读法：这 36–39 µs 是什么

1. **延迟由「两个常驻 kernel 的启动/终止协议」主导，而非计算**。`batch=16, hidden=4096` 的 RMSNorm
   只有 16 个 task、约 6.5 万元素，FLOP 与字节都可忽略；而把 worker 数从 128 降到 16（常驻 grid 由
   132 block 降到 32 block）延迟只降 ~7%（39.05→36.40 µs）。**时间几乎全花在 launch + 设备内
   scheduler 冷启动 + 终止握手（done event）**，不是 worker 并行度。
2. **相对 kernel-per-operator 的朴素参照约 1.4–1.5×**（53.99 µs vs 36.40–39.05 µs）。PyTorch 的
   RMSNorm 在这台机器上是**多个小 kernel（pow/mean/rsqrt/mul/mul）串行 + 中间量落 HBM**，每次调用
   还叠加 host 侧逐 kernel 提交；MPK 侧是 **C1 单次 launch**（一条 `pk()` 提交双常驻 kernel），把
   这些摊平。这与 `docs/04` §17「损失三 = 中间张量 HBM 往返」和 §1「损失二 = launch 开销」在
   **真实 mirage 栈上**再次对上：**算子数与中间量越大，MPK 优势越大**；本用例只 1 个算子、张量极小，
   所以只是 1.4× 量级的差距。
3. **与论文口径的差异**：论文 §6.3（快照行 1501–1513）报的单 batch 端到端是**整个模型一次前向**
   （Qwen3-8B/A100 14.5→12.5 ms），这里测的是**单层 RMSNorm 的一次 megakernel 调用**，二者不可直接
   比较；本节的增量价值是把「MPK 栈在本机能**跑完并给出延迟**」这件事落地，给 C1（单次 launch）一个
   真实的本机数字。

### 8.4 局限

- **单层微内核、单次调用**：不能当作论文 §6 的端到端推理延迟；`pk()` 每次重新 launch+终止
  （offline/test 模式下一 batch 即终止），**不含**连续 serving 的稳态开销。
- **收敛依赖一处时序扰动**（`harness/mpk_latency_run.patch`）：虽然守门狗从未触发、插桩只在
  worker **空闲自旋**路径上多一个自增/分支，但**严格说这 36–39 µs 来自「打过时序修复的构建」**，
  可能与上游 clean 构建略有出入；鉴于差异只在 idle 循环，影响应在亚 µs 级。根因（哪条 load/store
  内存序）仍未钉死，见 §7.6。
- `batch=16` 恰好等于 task 数，`grid_dim=(16,1,1)`；未扫 batch/hidden、未测更大模型。
- **P25-1 状态**：构建 + JIT + launch + **收敛 + 正确性 + 端到端延迟**均已打通；仍未做的是
  §7.6 的根因内存序定位与「非 Hopper-TMA task 隔离 H2」，故 P25-1 继续记 `[~]`，但「端到端延迟」
  这一条已补齐。

## 9. 根因再定位（续做 P25-1）：挂起对「前置 dtype 转换 kernel」敏感，且不来自 worker 轮询循环的内存序

§7.5 把挂起归因到「`execute_worker` 取 task 轮询循环里加一处无害自旋计数即修复」，并据此判断
「竞态就在 worker 侧等 task 的抢跑/丢失唤醒上」。本轮把这条结论**降级**：同一个循环里的**内存序类**
改动（编译器屏障、`__threadfence`、`ld.relaxed`、改 `__nanosleep` 值）**都不修复**；修复与否只与
**「在这次 launch 之前，流上是否跑过一个 dtype 转换 kernel」**高度相关，而**与 worker:scheduler 的
grid 大小无关**。这更像 **launch/握手层面的时序敏感竞态**（宿主侧两条新建 stream 与 torch 默认流的
排序），而不是 device 侧某条 load/store 的内存序。

> 全部实验在**目标卡 H100/sm90**、快照 `mirage-project/mirage@f9eb70c`、`CUDA_VISIBLE_DEVICES=7`
> **隔离空闲卡**上完成（GPU 0 当时被其它任务占满，早期在 device 0 上跑的结果已作废）；每次
> `timeout` 单跑，退出码 `0=PASS / 1=WRONG / 124=HANG`。

### 9.1 判据一：只有「dtype 转换 kernel」能避开挂起

用 `harness/mpk_race_attr.sh` 驱动 `harness/mpk_race_probe.py`，在 `pk()` **之前**做不同的
「前置 GPU 工作」（`prewarm`），每次 `torch.cuda.synchronize()` 后再 launch（原始输出
`harness/mpk_race_attr.out.txt`）：

| 前置工作 | 类别 | 结果 |
|---|---|---|
| none / sync | 无 / 仅同步 | **HANG** |
| sleep（`_sleep` 300ms） | 纯时间延迟 | **HANG** |
| copy / mul / clamp / addrow | 同 dtype 逐元素 | **HANG** |
| matmul（cuBLAS bf16） | 纯计算 | **HANG** |
| alloc（`empty` 2× 大小） | 仅分配、无计算 | **HANG** |
| sumall / sumrows / mean / rmsnorm / softmax | 规约（内部含 cast） | **PASS** |
| **cast**（`x.to(float32)`） | dtype 转换 | **PASS** |
| **copyto32**（`y.copy_(x)`，bf16→fp32） | dtype 转换 | **PASS** |
| **to_half**（bf16→fp16，收窄） | dtype 转换 | **PASS** |
| **to64**（bf16→fp64，加宽） | dtype 转换 | **PASS** |
| to_same（`x.to(bf16)`，**不产生 kernel**） | 无操作 | **HANG** |

读法：

- **不是纯时间**：`sleep` 300 ms 不修复，`sync` 不修复 → 与「冷启动/预热时钟」无关。
- **不是分配**：`alloc` 造出与 `cast` 输出同尺寸的 buffer 但不修复 → 与 caching allocator 无关。
- **不是访存/计算**：`copy`/`mul`/`clamp`/`addrow`/`matmul` 全 HANG。
- **不是规约本身**：`sumall`/`sumrows`/`mean`/`softmax`/`rmsnorm` 全部**以 `x.to(float32)` 开头**，
  与 `cast`/`copyto32`/`to_half`/`to64` 同属**dtype 转换**这一类；而 `to_same`（同 dtype，PyTorch
  直接返回自身、不发射 kernel）HANG → **决定变量是「真的跑了一个 dtype 转换 kernel」**，与输出尺寸
  （`to_half` 与输入同宽）也无关。
- 把 `to_same` 作为阴性对照，排除了「`.to()` Python 包装 / TensorIterator 调度本身」。

**grid 无关**（判据二）：`harness/mpk_race_attr.sh` 的 B 段扫 `worker:scheduler = 4:4 / 16:16 /
64:16 / 128:16`，**四档全部 HANG** → 挂起与常驻 grid 大小、SM 占用量无关。（worker kernel 用
`MAX_DYNAMIC_SHARED_MEMORY_SIZE = 205824 B`，约 1 block/SM，但 4:4 也挂，排除 occupancy。）

### 9.2 判据三：worker 轮询循环的**内存序类**改动都不修复

§7.6 提出的候选是「缺 fence / 缺 `"memory"` clobber / 编译器 hoist」。快照里确有可疑的不对称：
`include/mirage/persistent_kernel/mpk_atoms.cuh` 的 `ld_acquire_gpu_u64`（`:50-54`）**没有**
`"memory"` clobber，而 `ld_acquire_sys_u64`（`:56-64`）有。为验证，`harness/mpk_race_sweep.py`
对 worker 取 task 轮询循环（`persistent_kernel.cuh:1013-1026`）做**仅循环内、语义等价**的最小改动
各跑 2 次（原始输出 `harness/mpk_race_sweep.out.txt`，device 7）：

| 变体 | 改动 | 结果 |
|---|---|---|
| plain | 无 | HANG ×2 |
| counter | 累加自旋计数（+`__trap`） | HANG ×2 |
| sleep0 | `__nanosleep(10)` → `__nanosleep(0)` | HANG ×2 |
| sleep1000 | `__nanosleep(10)` → `__nanosleep(1000)` | HANG ×2 |
| compiler_barrier | 循环内插 `asm volatile("" ::: "memory")` | HANG ×2 |
| threadfence | 循环内插 `__threadfence()` | HANG ×2 |
| relaxed | 循环内 `ld_acquire`→`ld_relaxed` | HANG ×2 |

- **`compiler_barrier` 不修复** → 排除「编译器把 `next_task_pos` 的 load 提升出循环」的 hoist 假设
  （否则加了编译器屏障就该好）；**`threadfence` 不修复** → 排除「缺硬件内存序/fence」；
  **`relaxed` 不修复** → 排除「acquire 语义过强/过弱」。**改 `__nanosleep` 值也不修复** → 不是循环
  自旋节奏。
- 这与 §9.1 的「dtype 转换 kernel」签名互相印证：**问题不在 worker 轮询循环的 load/store 内存序**。

### 9.3 判据四：worker 轮询循环**不是** worker[0] 长时间等待的地方

§7.5 的「自旋计数」是带 `printf`+`__trap` 的守门狗。`__trap()` 只会让 worker kernel 出错，但**另一条
常驻的 scheduler kernel 仍会自旋**，`torch.cuda.synchronize()` 因此仍不返回——所以「trap 变体 HANG」
无法区分。本轮改用 `harness/mpk_wlwatch.py`：在 worker 轮询循环内插一个**只 `printf` 不 `trap`** 的
守门狗（每 2×10⁷ 轮打印一次 `next/last/ready0`，原始输出 `harness/mpk_wlwatch.out.txt`）。结果：

- **该变体收敛（`RESULT=PASS`）**，且**没有任何 `[WL] worker0 STUCK` 打印** → worker[0] 从未在该循环
  里连续自旋到 2×10⁷ 轮（约 0.2 s）。**worker[0] 的长时间等待不在这个循环里**，挂起发生在别处
  （scheduler 侧或两条常驻 kernel 的握手）。
- 与 §9.2 合看：**能修复的插桩都显著改变了 worker kernel 的 codegen/调度**（`printf` 分支 vs 纯自增
  分支），**不改变语义**。这说明 §7.5 的「自旋计数」是一处**时序扰动式的偶然修复**，而非命中某条
  内存序 bug——把它升级为「竞态在 worker 等 task」的证据是不成立的。

### 9.4 与 §7 的关系：为什么 §7.5 会看起来「修好了」

| | §7.5 修复（`mpk_workaround_only.patch` / `mpk_latency_run.patch`） | 本轮同语义变体（`counter`） |
|---|---|---|
| worker 循环内 | `long _dbg_spins; ++_dbg_spins;`＋`printf`（阈值 2×10⁸，从未触发） | `++_dbg_spins; __trap()`（阈值 2×10⁷） |
| 结果 | 3/3 收敛 | 2/2 HANG |

两者都只是「循环里多一点代码」，差别在于 `printf` 分支让 worker kernel 的**寄存器/指令调度**变化更大，
并且 `__trap` 路径与「继续自旋」路径的 codegen 不同。**同一处逻辑的两种写法给出相反结果**，正说明
这是一个**对 codegen/调度敏感的时序竞态**，而不是「加了自旋计数就修好了某条内存序」。§7.5 的结论
应表述为：**「一处 worker 侧的无害扰动足以避开该竞态」**，而不是「竞态位于 worker 轮询循环」。

### 9.5 机制候选（标注为假设，非对上游的 bug 判定）

结合宿主侧源码（`persistent_kernel.cuh`）：

1. `worker_kernel` 与 `scheduler_kernel` 被 launch 在**两条每次调用新建的 `cudaStreamNonBlocking`
   stream** 上（`:1932-1935`）；与 torch 默认流（legacy default stream）的排序靠
   `prepare_kernel<<<…, default_stream>>>` + `cudaEventRecord(prepare_done_event, default_stream)`
   + `cudaStreamWaitEvent(worker/scheduler_stream, prepare_done_event)`（`:1963-1969`、`:1981-1986`）。
   `default_stream` 即 `torch.cuda.current_stream()`（`persistent_kernel.py:3574-3592`）。
2. **最可能的机制是 launch/握手时序**：两条 `NonBlocking` 新流与 legacy 默认流的**隐式同步关系**
   很弱，`prepare_done_event` 之外没有把常驻 kernel 与「默认流上刚发生的事」进一步绑定；一个**紧挨着
   的 dtype 转换 kernel**（读 A 写 B、访存受限、时长与 launch 同量级）恰好改变了驱动把这几条 launch
   下发/共驻的次序，从而避开竞态。这与「sleep/copy/matmul 都不行、只有转换 kernel 行」的签名一致
   （需要“短、同时进出两条流”的那种前置 kernel），但仍属**待验证假设**。
3. **可立即排除**：occupancy/网格（§9.1 B 段）、纯时间（`sleep`）、分配（`alloc`）、worker 循环内存序
   （§9.2）、worker 循环长时间自旋（§9.3）。

### 9.6 局限与下一步

- **未钉死根因**：本节把「worker 轮询循环内存序」这条 §7.6 候选**证伪**，并给出 dtype 转换 kernel 的
  强签名，但**没有**定位到具体那条 launch/握手路径。不构成对上游 `f9eb70c` 的 bug 判定。
- 下一步候选（按性价比）：① 用 `nsys` 抓 host 侧 launch/流时序（对照 prewarm=cast vs none 的
  D2D/CUDA API 序列）；② 把前置 cast 放到**独立非默认流**上，验证是否仍能避开（区分「设备全局态」
  与「默认流历史」）；③ 在 launcher 的 `prepare_kernel` 后显式加一次 `cudaStreamSynchronize(default)`
  或把 worker/scheduler 改成继承当前流，看是否消除；④ 向 MPK 上游核对 `f9eb70c` 的这条启动路径。
- 原始输出：`harness/mpk_race_attr.out.txt`（前置工作矩阵 + grid）、
  `harness/mpk_race_sweep.out.txt`（循环内变体消融）、`harness/mpk_wlwatch.out.txt`（printf 守门狗）；
  脚本 `harness/mpk_race_attr.sh`、`harness/mpk_race_probe.py`、`harness/mpk_race_sweep.py`、
  `harness/mpk_wlwatch.py`。
- **P25-1 仍记 `[~]`**：§8 的端到端延迟与本节的现象刻画已足够；根因（launch/握手 vs device 侧）尚需
  nsys 级别证据。

## 10. 根因再定位 2（续做 P25-1）：挂起是「设备侧两个常驻 kernel 同时自旋」的双向死锁，且只被前置 dtype 转换 kernel 避开

§9 把根因从「worker 轮询循环内存序」降到「launch/握手层面的时序敏感竞态」，并给出了 dtype 转换 kernel
的强签名，但**没有**区分「默认流历史 vs 设备全局态」，也没排除 host 侧排序。本节把 §9.6 的候选项
①②③ 全部做完，并把结论**再次改写**：挂起**不是 host 侧 launch/握手**（强制 host 排序无效），而是
**设备侧 worker 与 scheduler 两个常驻 kernel 同时卡住不终止**；而 dtype 转换 kernel 只是**阻止该
设备侧死锁发生**（而非「修好握手」）的一个充分条件。

> 全部实验同样在**目标卡 H100/sm90**、快照 `mirage-project/mirage@f9eb70c`、`CUDA_VISIBLE_DEVICES=7`
> **隔离空闲卡**上完成；每次 `timeout` 单跑，退出码 `0=PASS / 1=WRONG / 124=HANG`。除特别说明外
> worker:scheduler = 128:16（论文 §5.1 的 H100 划分）。

### 10.1 判据五：避开挂起与「默认流」「完成」「规模」都无关

用 `harness/mpk_race_stream.sh` 把 cast 改到**独立、非默认的 torch 流**上再测（原始输出
`harness/mpk_race_stream.out.txt`、`harness/mpk_race_stream2.out.txt`）：

| 前置工作（prewarm） | 关键区别 | 结果 |
|---|---|---|
| none | 什么都不做 | **HANG** |
| cast | bf16→fp32，默认流 | **PASS** |
| cast_s2 | 同上，但跑在**独立非默认流**、同步该流 | **PASS** |
| cast_s2_nosync | 同上，且**完全不 sync**（cast 可能仍在飞） | **PASS** |
| copyto32_s2 | bf16→fp32 走 `copy_`，独立流 | **PASS** |
| cast_s2_keepalloc | 独立流 + 保留输出张量 | **PASS** |
| **cast_1elem** | **只 1 个元素**的 dtype 转换（默认流） | **PASS** |
| cast_1elem_s2 | 1 元素 dtype 转换（独立流） | **PASS** |
| none_s2 | 仅创建+同步一个独立流、**不发 kernel** | **HANG** |

读法：

- **与默认流无关**：cast 从来不在 torch 默认流上执行也照样避开（`cast_s2`、`cast_1elem_s2`），
  故 §9.5 的「默认流历史」假设**证伪**；决定变量是**设备上真的跑过一个 dtype 转换 kernel**。
- **与完成无关**：`cast_s2_nosync` 连 `sync` 都没有，cast 可能与 `pk()` 并发，仍 PASS → 不需要
  「cast 先完成」。
- **与规模无关**：`cast_1elem` 只有 1 个元素、几乎零工作量，仍 PASS → 不是「预热了带宽/时钟」，
  也不是「改了 allocator/内存状态」的大小效应（§9.1 的 `alloc` 阴性对照已排除后者）。

### 10.2 判据六：**不是**「任意独立流 kernel」——只有 dtype 转换 kernel 能避开

`harness/mpk_race_stream3.out.txt` 把「同 dtype 逐元素 / 计算 / 纯时间」kernel 全部改到**独立流**上
再测，排除「只要有个 side-stream kernel 就行」：

| 前置工作（独立流，之后同步该流） | 结果 |
|---|---|
| none | **HANG** |
| **cast_s2**（dtype 转换） | **PASS** |
| mul_s2（bf16 逐元素乘，写 `out`） | **HANG** |
| copy_s2（bf16→bf16 `copy_`，走 `cudaMemcpyAsync`） | **HANG** |
| clamp_s2（bf16 逐元素 clamp） | **HANG** |
| sleep_s2（`_sleep` 300 ms 纯时间 kernel） | **HANG** |
| matmul_s2（cuBLAS bf16，50 次） | **HANG** |

→ 签名**精确地是「dtype 转换」这一类**，与执行流、完成、规模、时间、带宽都无关；同类的逐元素
kernel（`mul`/`clamp`）和 `cudaMemcpyAsync`（`copy_`）都**不能**避开。与 §9.1 合并：
`sumall`/`sumrows`/`mean`/`softmax`/`rmsnorm` 之所以 PASS，正是因为它们内部**以 `x.to(float32)`
开头**（判据一）。**这是一个非常窄的触发条件**。

### 10.3 判据七：host 侧强制排序**无效**——不是 launch/握手

§9.6 候选③（在 `prepare_kernel` 之后显式同步）用 `harness/mpk_launch_sync.patch` 实现：在
`persistent_kernel.cuh:1968` 的 `cudaEventRecord(prepare_done_event, …)` 之后插入两个**环境变量门控**
的宏（`MPK_EXTRA_DEFINES` 经 `persistent_kernel.py:371` 注入，语义等价、默认关闭）：

- `-DMPK_DBG_LAUNCH_SYNC` → `cudaStreamSynchronize(default_stream)`（host 等到 `prepare_kernel`
  真正做完再 launch worker/scheduler）；
- `-DMPK_DBG_DEVICE_SYNC` → `cudaDeviceSynchronize()`。

结果（`harness/mpk_launch_sync.out.txt`）：

| 配置 | 结果 |
|---|---|
| baseline(none) | **HANG** |
| **LAUNCH_SYNC**（host 等到 prepare 完成） | **HANG** |
| **DEVICE_SYNC** | **HANG** |
| cast(none) | PASS |
| cast + LAUNCH_SYNC | PASS |

→ 即使 host 已确保 `prepare_kernel` 完成、两个常驻 kernel 才启动，**仍然挂起**。故 §9.5 的
「launch/握手时序」假设也**被证伪**：这不是「worker 抢跑读了未写完的队列指针」之类的 host/launch 排序
问题（`cudaStreamSynchronize` 已把这条堵死），死锁在**常驻 kernel 启动之后**。

### 10.4 判据八：两个常驻 kernel 都在设备侧自旋（无 launch error，4:4 亦然）

把 host 侧诊断再加一个宏 `-DMPK_DBG_WAIT_TRACE`（`harness/mpk_launch_sync.patch`，在
`launch_persistent_kernel` 里对两条 stream 做非阻塞 `cudaStreamQuery` 轮询 ~2 s 并打印）；
`harness/mpk_wait_trace.sh` 对照 none vs cast_1elem（原始输出 `harness/mpk_wait_trace.out.txt`）：

| | HANG（prewarm=none） | PASS（prewarm=cast_1elem） |
|---|---|---|
| worker launch | `no error` | `no error` |
| scheduler launch | `no error` | `no error` |
| worker_stream | `device not ready` at it=0…16（**始终在跑**） | **DONE at it=1** |
| scheduler_stream | `device not ready` at it=0…16（**始终在跑**） | **DONE at it=1** |

→ **两个 kernel 都成功 launch（无错误），且都一直不结束**：不是某个 kernel 没起来，而是
worker 与 scheduler **互相等待**（`execute_worker` 等 task/事件、`execute_scheduler` 等事件），
即**设备侧双向死锁**。`harness/mpk_wait_trace_grid.out.txt` 进一步在 `4:4 / 16:16 / 64:16`
三档重复，**全部两个 stream 都自旋**：

- **排除 co-residency / SM 占用**：4 worker + 4 scheduler 只占 8 个 BLOCK（远小于 132 SM 的
  占用上限），两个 kernel 仍双双卡死 → 不是「worker 先占满 SM、scheduler 拿不到 SM」的启动顺序问题
  （这也解释了为什么 §9.1 的 4:4 会挂）。

### 10.5 排除项：不是频率/时序竞态

若挂起是「worker 与 scheduler 相对速度」触发的竞态，锁频应能改变结果。用
`nvidia-smi -i <gpu> -lgc 1980`（锁最高 SM 频率，`harness/mpk_clock_lock.out.txt`）：

| 配置（锁频 1980 MHz） | 结果 |
|---|---|
| none | **HANG** |
| cast_1elem | PASS |

→ 锁频**不改变**结果（HANG 依旧），故不是 GPU 频率/boost 造成的时序竞态。（`cast` 能避开也**不是**
因为它「唤醒了 GPU 时钟」——`sleep`/`matmul` 同样长时间占用 GPU 却不能避开，见 §10.2。）

### 10.6 机制候选（标注为假设，非对上游的 bug 判定）

把 §9 + 本节证据拼起来，事实是：

1. 挂起 = **设备侧 worker↔scheduler 双向死锁**（10.4），与 host 排序无关（10.3）、与 grid 无关（10.4）、
   与频率无关（10.5）。
2. 触发与否只取决于**在 `pk()` 之前设备是否执行过「dtype 转换 kernel」**（10.1、10.2）——流/完成/规模
   都不重要。
3. 这是一个**极窄的充分条件**：`.to(dtype)` / 跨 dtype `copy_`；同 dtype 的 `mul/clamp/copy`、
   cuBLAS `matmul`、`_sleep`、`cudaMemcpyAsync` 都不行。

「一个 1 元素、在别的流上、甚至还没跑完的 dtype 转换 kernel，能决定一个 128-block 常驻 kernel 是否
死锁」——这条签名的**机制仍未钉死**。可能的方向（均为**待验证假设**）：

- **H-a（最可能）设备级「首次 dtype 转换」初始化路径**：跨 dtype 逐元素在 PyTorch 里可能触发某条
  **懒初始化 / 一次性 driver 侧状态**（例如某个 `cudaFuncSetAttribute`、模块加载或内部事件/flag），
  它改变了随后两个 `NonBlocking` 流上常驻 kernel 的**共驻/调度**；而 `sleep/matmul/同 dtype 逐元素`
  不触发这条路径。要验证需抓 device 侧一次性状态（本轮 nsys 未成功，见下）。
- **H-b 与 `__nanosleep` 自旋的交互**：dtype 转换 kernel 的**收窄/加宽写**可能改变了 L2/内存子系统的
  **某些持久状态**，使 `execute_worker` 的自旋路径命中/不命中；但 §9.2 已证循环内改动（含
  `__nanosleep` 值）都不修复，此假设较弱。
- **H-c 快照自身的 test/offline 路径缺陷**：即 `MODE_OFFLINE` + `test_mode` 组合下的
  `prepare_next_batch`/终止事件协议本就有一处**依赖执行时序**的缺陷，dtype 转换只是恰好扰动到能收敛的
  分支。与「本用例只测了单个 operator、且是 offline/test 路径」一致。

**无论如何，这不构成对上游 `f9eb70c` 的 bug 判定**；只是把可复现条件收窄到「有无前置 dtype 转换
kernel」这一个可开关的旋钮。

### 10.7 局限与下一步

- **nsys 未成功**：`harness/mpk_nsys.sh` + `harness/mpk_nsys_parse.py` 尝试抓 host 侧 CUDA API/流
  序列（§9.6 候选①）。挂起用例经 `os._exit` 退出，CUPTI 缓冲未 flush，只导出了 3 条 API 记录；
  改用 `cudaProfilerStop()`、`--capture-range` 后 nsys 进程本身在挂起 kernel 上**不收敛/超时**
  （原始输出 `harness/mpk_nsys.out.txt` 仅有两次运行的标题）。故 host API 时间线证据本轮**缺失**，
  但 §10.3/§10.4 已从**行为**上排除了 host 排序。
- **机制未钉死**：给出「设备侧双向死锁 + 仅 dtype 转换 kernel 可避开」的强刻画与三个假设，未落到
  具体一条状态/指令。
- 下一步候选（按性价比）：① 在**收敛构建**（`harness/mpk_latency_run.patch`）上对 worker 与 scheduler
  各自的**事件等待点**做 `printf` 守门狗（不改语义、只观测），定位双方**各卡在哪个 event 计数**；
  ② 用 `compute-sanitizer --tool synccheck` / `racecheck` 抓设备侧竞争；③ 对照 MPK 上游 `f9eb70c`
  的 `MODE_OFFLINE` 终止协议核对事件计数。
- 原始输出：`harness/mpk_race_stream.out.txt` / `harness/mpk_race_stream2.out.txt`（流来源/规模/完成）、
  `harness/mpk_race_stream3.out.txt`（side-stream 非 cast 对照）、`harness/mpk_launch_sync.out.txt`
  （host 排序宏）、`harness/mpk_wait_trace.out.txt` / `harness/mpk_wait_trace_grid.out.txt`
  （两 stream 轮询）、`harness/mpk_clock_lock.out.txt`（锁频）、`harness/mpk_nsys.out.txt`（nsys 尝试）。
  脚本 `harness/mpk_race_stream.sh`、`harness/mpk_launch_sync.sh`、`harness/mpk_wait_trace.sh`、
  `harness/mpk_nsys.sh`、`harness/mpk_nsys_parse.py`；patch `harness/mpk_launch_sync.patch`。
- **P25-1 仍记 `[~]`**：本轮把 §9 的两条候选（默认流历史 / host 排序）**都证伪**，并把死锁**定位到
  设备侧 worker↔scheduler 双向自旋**、触发条件收窄到「前置 dtype 转换 kernel」；端到端延迟（§8）与
  现象刻画已足够，**具体机制仍需 device 侧事件级/竞争检测证据**。
