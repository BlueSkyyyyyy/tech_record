# 构建与实跑记录：在 H100 上把 MPK 跑起来

> 这是「Mirage 项目分析」的最后一篇。前几篇讲「它应该怎么工作」，这一篇讲「我们真的编译、真的运行，
> 看到了什么」。结论先说：**能编、能起 kernel；默认启动顺序会挂起；换成 scheduler 先启动就通过，
> 数值正确。**

## 0. 先记住一个画面

这像买回一台标注「一键启动」的整机。

通电（`pip install -e .`）很顺，指示灯亮（Python 能 import）、风扇转（kernel 起来了），
但屏幕一直停在开机画面——不是电源坏了，是**启动顺序**有一个出厂 bug：先点火的部件把
点火器挤出了调度队列。

> **类比的边界**：真实机器换个启动顺序只是慢一点；GPU 上这一步是**决定性**的——worker 一旦
> 占满 SM 并开始自旋，负责派活的 scheduler kernel 可能永远拿不到派发机会，整个系统就静默地
> 卡在那里，没有报错、没有超时，只有 100% 的利用率和一条不动的进度。

读完这一篇，你应该能回答：

1. 在 H100/CUDA 13.2 上从零构建 mirage 要多久？会踩哪些坑？
2. 最小的 MPK 程序长什么样？它跑出了什么？
3. 默认启动顺序为什么会挂？怎么验证、怎么绕开？

## 1. 环境

| 项 | 值 |
|---|---|
| GPU | NVIDIA H100 80GB HBM3（sm90，compute capability 9.0） |
| 容器 | `kernel_lab`（PyTorch 官方镜像系） |
| CUDA | 13.2（`nvcc` V13.2.78） |
| Python / PyTorch | 3.12 / 2.12.0a0 |
| CPU | 96 核（影响编译时间） |
| 快照 | `mirage-project/mirage` `mpk` 分支，commit `f9eb70c` |

## 2. 准备：三个 submodule 要手动补

官方步骤是：

```bash
git clone --recursive --branch mpk https://github.com/mirage-project/mirage
cd mirage
pip install -e . -v
export MIRAGE_HOME=$(pwd)
```

我们这里 `git clone` 走不通（TLS 连接被中断），于是改用 codeload 下 tar 包：

```bash
curl -L -o mirage.tar.gz https://codeload.github.com/mirage-project/mirage/tar.gz/f9eb70c
```

但 tar 包**不含 submodule**，三个依赖目录是空的，必须自己补：

| 目录 | 内容 | 我们用的版本 |
|---|---|---|
| `deps/cutlass` | NVIDIA CUTLASS | commit `f3fde583` |
| `deps/json` | nlohmann/json | v3.11.3 |
| `deps/z3` | Z3（形式化验证用） | 直接用 pip 的 `z3-solver`（4.16.0.0），`setup.py` 会按 `z3.__file__` 找头文件 |

> 这一步是本文最费时的部分：**忘了 submodule，报错会出现在 CMake 深处**，看起来像代码问题。

## 3. 构建：96 核上约 3 分钟

```bash
MIRAGE_HOME=$(pwd) pip install -e . -v
```

实测结果：

- **成功**（`EXIT=0`），装出 `mirage-project-0.2.4`；
- 96 核并行编译 **83 个 C++ 目标**，从开始到出 wheel 约 **3 分钟**（`01:44` → `01:46` UTC）；
- `build/` 目录 **337MB**；
- 顺带装了 `transformers 4.57.1`、`tg4perfetto 0.0.5`（profiling 用）；
- 中间会看到 pip 的 dependency resolver 警告（不影响）。

编译完成后：

```python
import mirage as mi
print(mi.__version__)        # 0.2.4
print(mi.PersistentKernel)   # <class 'mirage.mpk.persistent_kernel.PersistentKernel'>
```

到这里，「编译层」是通的。

## 4. 冒烟测试：最小的 MPK 程序

选官方自带的最小用例 `tests/runtime_python/test_mode/test_rmsnorm_testmode.py`：
16×4096 的 bf16 RMSNorm，用 `test_mode` 让常驻 kernel **跑一次就退出**。

```python
num_workers, num_schedulers = mirage.get_configurations_from_gpu(0)
params = PersistentKernel.get_default_init_parameters()
params["test_mode"] = True
params["num_workers"] = num_workers
params["num_local_schedulers"] = num_schedulers
pk = PersistentKernel(**params)

x_dt = pk.attach_input(x, name="x"); w_dt = pk.attach_input(w, name="w"); out_dt = pk.attach_input(out, name="out")
pk.rmsnorm_layer(input=x_dt, weight=w_dt, output=out_dt, grid_dim=(16,1,1), block_dim=(256,1,1))
pk.compile(); pk(); pk.wait()
```

第一次跑（**未打任何补丁**）的输出：

```
[MPK INIT] Total tasks: 18, Total events: 3
sm_cnt: 132
Compiling test kernel...
Finished megakernel compilation...
Running test kernel...
```

然后——**没有然后了**。进程不退出，GPU 利用率 100%，显存 643MiB，`nvidia-smi` 能看到一个
计算进程；`timeout` 到点后被杀，退出码 124。

也就是说：**kernel 确实起来了，但在设备上自旋，永远等不到该等的东西。**

## 5. 定位：不是我们的环境问题

这个现象在本仓库的 MegaKernel 专题里已经用 `cuda-gdb` 钉死过（`code/mega-kernel/docs/09` §11）：

- attach 到挂起的进程，设备上**只有 `worker_kernel` 常驻**，128 个 block 全部 `running`，
  PC 停在 `persistent_kernel.cuh:1016` 的「等 task」循环；
- `scheduler_kernel` **根本不在设备上**（`cuda kernel 1` 报 `Invalid coordinates requested`）；
- 把 worker 降到 4 个 block（只占 4 个 SM）也一样——所以不是 occupancy 不够，而是
  **scheduler kernel 已入队但从未被派发**。

一句话：`launch_persistent_kernel()` 默认先 launch worker、再 launch scheduler
（`persistent_kernel.cuh:1991-2006`）；worker 的常驻网格先把设备占住，scheduler 就饿死了。

## 6. 绕开：把 scheduler 先启动

仓库的 `code/mega-kernel/harness/mpk_sched_first.patch` 提供了只改启动顺序的补丁（编译期开关
`MPK_DBG_SCHED_FIRST`，不动任何 kernel 代码）。把它打到快照上，再用环境变量注入宏：

```bash
patch -p1 < code/mega-kernel/harness/mpk_sched_first.patch
MIRAGE_HOME=$(pwd) MPK_EXTRA_DEFINES=-DMPK_DBG_SCHED_FIRST \
  python tests/runtime_python/test_mode/test_rmsnorm_testmode.py
```

结果：

| 启动顺序 | 结果 | 说明 |
|---|---|---|
| 默认：worker → scheduler | **挂起**（timeout，GPU 100%） | 复现稳定 |
| 交换：scheduler → worker | **PASS** | `Max absolute difference: 0.0078125`（bf16 容差内） |

通过时的关键输出：

```
[MPK INIT] Total tasks: 18, Total events: 3
sm_cnt: 132
... -gencode=arch=compute_90a,code=sm_90a ... -DMODE_OFFLINE -DMPK_TEST_MODE -DMPK_DBG_SCHED_FIRST
Finished megakernel compilation...
PASSED: rmsnorm test_mode produces correct output
worker kernel & scheduler kernel
smem size: 205824
Finished Launching Persistent Kernel (Async)
```

注意几条信息：

- **`gencode=sm_90a`**：为 H100 的架构特性单独编译；
- **`smem size: 205824`**：约 201KB 动态共享内存，接近 H100 单 SM 上限（228KB）；
- **`Total tasks: 18, Total events: 3`**：一颗 16×4096 的 RMSNorm 被拆成 18 个 task；
- 编号 132 的 SM 里，128 个 worker + 4 个 scheduler SM。

跑完我们把补丁**回退**了，保证快照与文档里的行号一致。

## 7. 这件事说明什么

1. **构建门槛不高**：96 核 3 分钟，依赖只有 CUDA + 三个 submodule。真正麻烦的是网络。
2. **运行层比编译层脆**：一个 launch 顺序就能让整个系统静默挂起，没有错误码。
   这正好印证前面几篇反复强调的——megakernel 把「调度」搬进设备后，**调试面也搬进去了**。
3. **上游 bug 有明确边界**：只影响「worker/scheduler 分离 + 默认顺序」这条路径；
   换顺序即恢复，数值正确。这不是我们环境的偶发问题。
4. **工程建议**：如果你要在自己的卡上跑 MPK，第一件事就是确认启动顺序；第二件事是准备一个
   `timeout` + `nvidia-smi` 的看门狗，因为挂起时**没有任何日志**。

## 8. 复现清单

```bash
# 1) 取代码（git clone 不通时用 codeload）
curl -L -o mirage.tar.gz https://codeload.github.com/mirage-project/mirage/tar.gz/f9eb70c

# 2) 补 submodule：cutlass f3fde583 / json v3.11.3 / pip install z3-solver

# 3) 构建
MIRAGE_HOME=$(pwd) pip install -e . -v

# 4) 冒烟（预期：默认挂起）
timeout 900 python tests/runtime_python/test_mode/test_rmsnorm_testmode.py

# 5) 打补丁后再跑（预期：PASS）
patch -p1 < ~/proj/tech_record/code/mega-kernel/harness/mpk_sched_first.patch
MIRAGE_HOME=$(pwd) MPK_EXTRA_DEFINES=-DMPK_DBG_SCHED_FIRST \
  timeout 900 python tests/runtime_python/test_mode/test_rmsnorm_testmode.py
```

原始日志（构建、挂起、通过）留在 `code/mirage/harness/*.log.txt`；
按仓库策略 `*.txt` 不入库，只在本机留存。

## 9. 三句话小结

- 在 H100 + CUDA 13.2 上，`pip install -e .` 约 3 分钟成功，`import mirage` 正常。
- 最小的 test_mode RMSNorm 程序能编、能 launch，但默认启动顺序下 worker 饿死 scheduler，静默挂起。
- 只交换两个 kernel 的 launch 顺序即通过，数值在 bf16 容差内（max abs diff 0.0078125）。

至此这个专题结束。想继续深入，去 MegaKernel 专题按「编译栈 / 运行时 / task 实现」三条线读；
想动手改，就从 `demo/qwen3/demo.py` 换一个模型开始。
