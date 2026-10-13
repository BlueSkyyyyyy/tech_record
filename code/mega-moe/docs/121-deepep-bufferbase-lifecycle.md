# 121 · DeepEP 四个 Buffer 家族共享的生命周期契约：`BufferBase`

> 前面 120 篇把 DeepEP 的**通信原语**（第 7/48/51/53/58/61/63/65/71/73/75 篇 dispatch/combine、
> 第 54/62/64/112 篇 bucket 集合通信、第 58 篇 PP send/recv、第 59 篇 Engram）、**地基**
> （第 66 篇对称内存、第 76 篇 NCCL 通信域绑定、第 77 篇跨 SM/rank 同步、第 80 篇驱动符号懒加载）
> 与**测试/部署面**（第 85/91/100/101 篇验收测试、第 117 篇环境变量、第 118 篇进程引导）都拆遍了。
> 但有一条**四个 Buffer 家族都继承、却从没被单独读过**的公共基类：
> `deep_ep/buffers/base.py` 的 `BufferBase`（27 行）与 `csrc/buffers/base.hpp` 的 `BufferBase`
> （63 行）。`EPBuffer`/`EngramBuffer`/`PPBuffer`/`BucketBuffer` 全部从它派生，共享同一套
> 「**显式销毁还是析构自动销毁 / 销毁后泄漏告警 / 生命周期状态机**」契约。
>
> 本篇把它单独讲透：**为什么需要一个基类而不是四份复制**、**Python ABC 与 C++ pybind 为什么是
> 两条平行的继承链**、**`destroy_on_destruction` 的三分支状态机**、**析构函数为什么是
> `noexcept(false)`**、以及一个只在对照中才显现的**跨语言分歧**（Python 侧二次 `destroy()` 幂等、
> C++ 侧二次 `destroy()` 抛异常）。

本篇目标源码（`~/github/DeepEP@93eb6eb`，v1.2.1-41）：

- `deep_ep/buffers/base.py`（27 行，sha256[:16] `17e6d15585068396`）：Python ABC 契约。
- `csrc/buffers/base.hpp`（63 行，sha256[:16] `ba2c0cab18b449f4`）：C++ 契约 + pybind 注册。
- 四个派生家族（各含 C++ 头 + Python 包一层）：
  - `csrc/buffers/{ep,engram,pp,bucket}.hpp` + `deep_ep/buffers/{ep,engram,pp,bucket}.py`。
- 顶层 re-export：`deep_ep/__init__.py:63`。

**可跑边界**：`BufferBase` 的 Python 侧只 `import abc`，C++ 侧全是 host 代码（只用到 pybind11
注册宏与 `comm::Context` 的六个只读字段）。`import deep_ep` 会触发 `check_nccl_so()`（本机
NCCL 2.29.7 与编译期不符 → abort，见第 86/117 篇），但**契约本身与 GPU / 架构完全无关**，故
H100(sm90) 上：Python 侧按文件路径直接加载 `base.py`；C++ 侧在临时目录搭一棵「影子 include 树」
（真实 `base.hpp` 原样拷入，最小 stub 顶掉 `pybind11/pybind11.h` 与 `csrc/kernels/comm/api.hpp`）
后 g++ 编译运行，实测 **32 PASS / 0 FAIL**（`harness/exp_deepep_bufferbase_lifecycle.{sh,py,out.txt}`
+ `src/bufferbase_lifecycle.cpp`）。真正需要 NCCL GIN 的四个家族 C++ 构造函数本机不可用，故用一个
**逐字复刻真实逻辑**的 `DerivedBuffer` 承接 `base.hpp` 的 `destroy_on_destruction` 分支。

---

## 1. 为什么需要基类：四个家族、一条生命周期

四个 Buffer 家族的 **C++ 头**都长这样（以 EP 为例）：

```cpp
class EPBuffer: public BufferBase {              // ep.hpp:23
    ...
    explicit EPBuffer(..., const bool& explicitly_destroy):
        BufferBase(explicitly_destroy),           // :61  把显式销毁标志交给基类
        ...
    { ...; main_context = context; ... }          // :78  把自己的 Context 挂到基类字段

    ~EPBuffer() noexcept(false) override {        // :98
        destroy_on_destruction("EP");             // :99  析构体里调基类的「收尾」
    }
    void destroy() override {                     // :102
        EP_HOST_ASSERT(not destroyed);            // :103  基类标志：不可重复销毁
        ...; destroyed = true;                     // :119
    }
};
```

它们的共同点**只有生命周期**、其余（Context 数量、storage 形态、通信算子）完全不同：

| 家族 | C++ 头:类行 | Context | 析构体调 | Python 类 |
|---|---|---|---|---|
| `EPBuffer` | `ep.hpp:23` | 单个 | `destroy_on_destruction("EP")`（`:99`） | `ep.py:208` |
| `EngramBuffer` | `engram.hpp:26` | 单个（shared） | `"Engram"`（`:67`） | `engram.py:31` |
| `PPBuffer` | `pp.hpp:25` | 单个 | `"PP"`（`:76`） | `pp.py:12` |
| `BucketBuffer` | `bucket.hpp:36` | **多个**（`contexts`） | `"bucket"`（`:110`，注意小写） | `bucket.py:184` |

所以生命周期被抽进基类：**一个 `bool explicitly_destroy` + 一个 `bool destroyed` + 一个
`std::shared_ptr<comm::Context> main_context`**（`base.hpp:13/14/32`）就是全部状态。四个家族只需
在自己的析构体里调一次 `destroy_on_destruction(name)`、在自己的 `destroy()` 末尾置 `destroyed = true`。

---

## 2. 双层设计：Python ABC ↔ C++ pybind 是两条**平行**继承链

这一层最容易被误读。**Python 的 `BufferBase`（`base.py`）和 C++ 的 `BufferBase`（`base.hpp`）
是两个完全独立的类**，不是同一对象的绑定：

```
Python 侧（base.py）                     C++ 侧（base.hpp）
  BufferBase(ABC)                          deep_ep::BufferBase
     ▲                                          ▲
  EPBuffer(BufferBase)  --持有-->  self.runtime = _C.EPBuffer
                                                 └─ pybind: class_<EPBuffer, BufferBase>
```

- **C++ 侧**：`pybind11::class_<EPBuffer, BufferBase>`（`ep.hpp:1085`）把「EPBuffer 继承 BufferBase」
  这条 C++ 继承关系暴露给 pybind；同理 `engram.hpp:308`、`pp.hpp:127`、`bucket.hpp:344`。基类自己也
  注册了一个类型：`pybind11::class_<BufferBase>(m, "BufferBase")`（`base.hpp:57`）。
- **Python 侧**：`EPBuffer(BufferBase)`（`ep.py:208`）继承的是 Python ABC；构造时
  `super().__init__(explicitly_destroy)`（`ep.py:338`）只初始化 Python 字段，随后
  `self.runtime = _C.EPBuffer(...)`（`ep.py:339`）**另建**一个 C++ 对象存为属性。

也就是说：**Python 的 `BufferBase` 负责「用户可见的生命周期语义」，C++ 的 `BufferBase` 负责
「资源真正释放的时序」。** 二者靠 `self.runtime` 这根引线接起来——Python 的 `destroy()` 最终调
`self.runtime.destroy()`（`base.py:26`），由 C++ 实现真正干活。

Python 基类只有 27 行，逐行都有分工：

```python
class BufferBase(ABC):                                        # base.py:4
    def __init__(self, explicitly_destroy: bool = False):     # :7
        self.explicitly_destroy = explicitly_destroy          # :8
        self.runtime = None                                   # :9
    @property
    def main_context(self):                                   # :11-14
        return self.runtime.main_context if self.runtime is not None else None
    def print_memory_usage(self) -> None:                     # :16-19
        assert self.runtime is not None                       # :18
        self.runtime.print_memory_usage()
    @abstractmethod
    def destroy(self) -> None:                                # :21-27
        assert self.explicitly_destroy                        # :24
        if self.runtime is not None:
            self.runtime.destroy()                            # :26
            self.runtime = None                               # :27
```

两个非平凡点：

1. **`main_context` 是 property、不是字段**——它不缓存，每次都转发到 `self.runtime.main_context`；
   `runtime` 一旦被置 `None`（销毁后）就返回 `None`。所以「销毁后读 `main_context` 得 `None`」是
   自动的，不需要显式清理。
2. **`destroy` 是「带实现的 abstractmethod」**——`@abstractmethod`（`:21`）强迫**每个子类必须覆盖**
   （见实验 2.0：直接实例化 `BufferBase` 抛 `TypeError`），但函数体（`:24-27` 的 `assert` + 转发）
   可被子类用 `super().destroy()` 复用。这就解释了为什么四个家族**各自**都要写一个覆盖版
   `destroy()`，而每个覆盖版都以 `super().destroy()` 开头、再清自己的字段：

   ```python
   # ep.py:365-371
   def destroy(self) -> None:
       super().destroy()          # 断言 explicitly_destroy + 销毁 runtime + runtime=None
       self.context = None        # 再清 EPBuffer 自己的引用
       self.nccl_comm_handle = None
   ```

   四个覆盖版清理的字段不同（`ep.py:369-371` 清 `context`/`nccl_comm_handle`；`pp.py:69-72` 清
   `storage`/`nccl_comm_handle`；`bucket.py:394-398` 清 `contexts`/`storage`/`nccl_comm_handles`；
   `engram.py:106-110` 清 `context`/`nccl_comm_handle`）——这正是「基类管通用、派生管特有」的分工。

---

## 3. C++ 契约：`destroy_on_destruction` 的三分支状态机

C++ 侧的全部逻辑集中在 `base.hpp:18-29` 这 12 行：

```cpp
// Called from the derived destructor, while its members and destroy() implementation are still available.
void destroy_on_destruction(const char* name) {                  // :19
    if (not explicitly_destroy and not destroyed) {              // :20
        destroy();                                               // :21  自动销毁
        main_context = nullptr;                                  // :22
    }
    if (not destroyed) {                                         // :25
        std::printf("`destroy()` is not called before DeepEP %s buffer destruction, "
                    "which can leak resources.\n", name);        // :26
        std::fflush(stdout);                                     // :27
    }
}
```

配合「派生 `destroy()` 末尾置 `destroyed = true`」，组合出**三分支**：

| 场景 | `explicitly_destroy` | 用户是否显式 `destroy()` | 析构体行为 | 泄漏告警 |
|---|---|---|---|---|
| A | `false`（默认） | 否 | **自动**调 `destroy()` + 清 `main_context` | 无 |
| B | `true` | 否 | **不**自动销毁（资源泄漏） | **打印** |
| C | `true` | 是 | 什么都不做（`destroyed` 已为 `true`） | 无 |
| D | 任意 | 是（两次） | 第二次撞 `EP_HOST_ASSERT(not destroyed)` **抛异常** | — |

实验 A/B/C/D/E 逐条验证（C++ harness）。

**注释为什么强调「from the derived destructor」**（`base.hpp:18`）：`destroy()` 是纯虚
（`base.hpp:35`），**不能在基类析构体里通过虚派发调用**——进入 `~BufferBase()` 时对象的动态类型
已退化为 `BufferBase`，虚调用会落到纯虚函数上。所以基类**不自己**在析构里调 `destroy()`，而是
暴露 `destroy_on_destruction` 让**每个派生析构体**在「派生成员与 `destroy()` 实现都还活着」时调用。
这是一种「基类提供逻辑、派生提供时机」的模板方法（Template Method），也解释了为什么实验里
`destroy()` 内检查派生成员 `marker` 始终成立（C1/D1）。

**`name` 参数**只用于告警文案：`EP`/`Engram`/`PP`/`bucket`（`ep.hpp:99`/`engram.hpp:67`/`pp.hpp:76`/
`bucket.hpp:110`，注意 bucket 是小写）。实验坐实告警文本
`` `destroy()` is not called before DeepEP Derived buffer destruction `` 在整场只出现**一次**（仅
场景 B）。

> 实战约定：**DeepEP 全部测试都传 `explicitly_destroy=True`**（`tests/buffer/test_allocation.py:28`、
> `tests/bucket/test_all_gather.py:30`、`tests/pp/test_pp.py:64`、`tests/ep/test_ep.py:538`、
> `tests/engram/test_engram.py:38`…… 逐个数）。原因正是场景 B：测试要**确定性释放**、不想把
> 销毁时机交给 GC/析构，所以显式 `destroy()`。契约文档也直说（`ep.py:286-287`）：
> 「`explicitly_destroy=True` 时你必须自己调 `destroy()`；否则资源由析构函数释放」。

---

## 4. 为什么析构函数是 `noexcept(false)`

`base.hpp:34`：

```cpp
virtual ~BufferBase() noexcept(false) = default;
```

C++11 起**析构函数默认 `noexcept(true)`**。但 `destroy()` 里到处是会抛/会 abort 的检查——
`EP_HOST_ASSERT(not destroyed)`（`ep.hpp:103`）、`CUDA_RUNTIME_CHECK(cudaFreeHost(...))`（`ep.hpp:110`）、
`comm::barrier(...)`（`ep.hpp:106-107`）。若析构函数是 `noexcept(true)`，那么「默认模式下析构体自动
`destroy()` 时抛出的异常」会直接触发 `std::terminate`。把基类析构标为 `noexcept(false)` 是**为了让
自动销毁路径的异常能向上传播**（而不是静默终止）。四个派生家族都显式重复 `noexcept(false) override`
（`ep.hpp:98` 等），把整条链钉死为 `noexcept(false)`。

> 需要说明：即便声明了 `noexcept(false)`，若析构发生在**栈回滚**（另一异常未处理完）期间再抛，
> 仍会 `std::terminate`——这是 C++ 语言层面的限制，基类的 `noexcept(false)` 只覆盖「正常析构时
> `destroy()` 抛异常」这一种情况。所以这是**有意的、但有限的**逃生口。

---

## 5. 一个只在跨语言对照中显现的分歧：二次 `destroy()`

把 Python 基类（`base.py:21-27`）与 C++ 基类（`base.hpp:11-35`）并排看，会发现**同一个
「二次销毁」在两个语言里行为相反**：

- **Python 幂等**：`base.py:25` 是 `if self.runtime is not None:`。第一次 `destroy()` 后
  `self.runtime = None`（`:27`），第二次进入 `if` 判假、直接 no-op。**不抛**（实验 2.4）。
- **C++ 抛异常**：派生 `destroy()` 第一句是 `EP_HOST_ASSERT(not destroyed)`（`ep.hpp:103` 等），
  第二次撞断言抛 `EPException`（实验 D1）。

为什么会有这个不对称？因为两者守的**不是同一件事**：C++ 的 `destroyed` 守的是「**资源只能释放
一次**」（重复 `finalize()` NCCL Context 会 double-free），是硬约束；Python 的 `runtime is not None`
守的是「**别对已置空的句柄再操作**」，是防御式写法。正常用法下（用户写一次 `destroy()`，或干脆不写
交给析构）两者都正确；只有「手滑调两次」时才会暴露差异——Python 默默吞掉、C++ 直接报错。这条差异
没有对错，但**读代码时不能把一方的语义想当然搬到另一方**。

---

## 6. `print_memory_usage` 的三态与 ABI

`print_memory_usage` 是这套契约里唯一「只读、可重复调用」的接口，三态分明：

- **存活期**：正常打印一行（`base.hpp:37-49`）。实验 F0 实测输出：
  ```
  [DeepEP memory usage] rank_idx=3, num_ranks=8, GPU total 3.195 GiB (workspace 0.195 GiB,
  buffer 1.000 GiB, RDMA storage 2.000 GiB), CPU total 0.000 GiB
  ```
  格式细节：`use_cpu_rdma_storage` 为真时把 RDMA storage 从「GPU total」挪到「CPU total」
  （`base.hpp:41-42` 的三元）。
- **已销毁**：第一句 `EP_HOST_ASSERT(not destroyed and main_context != nullptr)`（`base.hpp:38`）
  抛异常（实验 F2）。
- **Python 层**：转发到 `self.runtime.print_memory_usage()`（`base.py:19`），但**先断言
  `self.runtime is not None`**（`base.py:18`）。所以 Python 侧「销毁后调用」抛的是 `AssertionError`
  而非 C++ 异常（实验 2.5）——又一处跨语言差异，但这次是**同类语义**（都拒绝）。

ABI：`sizeof(BufferBase)=32`、`alignof=8`（实验 G）——`vptr(8) + bool + bool + 6 字节对齐填充 +
shared_ptr<Context>(16) = 32`。这个值**与 GPU 架构无关**，`comm::Context` 是堆上的独立对象、
只被 shared_ptr 持有。

---

## 7. 与既有多篇的接缝

- **第 76 篇**拆了 `comm::Context` 的构造与 `finalize()`（`csrc/kernels/comm/context.cpp`）：本篇的
  `main_context` 正是那个 `Context` 的 owning `shared_ptr`（`base.hpp:32`），`destroy()` 里
  `context->finalize()`（`ep.hpp:113`）就是第 76 篇的逆序释放。
- **第 66 篇**拆了对称内存分配地基：`Context` 持有对称 buffer，而 `BufferBase` 是它在 Python 侧
  的「寿命控制器」。
- **第 83 篇**拆了 `BucketSession` 的生命周期守卫（`stream_control_epilogue`）：那是「**单次算子
  调用内**的临时张量生存期」；本篇的 `BufferBase` 是「**整个 buffer 对象**的生存期」。两个尺度。
- **第 110 篇（DeepGEMM）**拆了 `Runtime` 单例生命周期：与 DeepEP 的 `BufferBase` 是**两套不同
  的资源所有权模型**——DeepGEMM 用进程级单例 + `atexit`，DeepEP 用每对象 `explicitly_destroy`
  开关。对照读能看清「库级单例」与「实例级句柄」两种设计。

另注：`main_context` 与 `print_memory_usage` 在整仓**除 API 定义外没有别的消费者**（`grep
main_context` 只命中 `base.py` 与 `csrc/buffers/base.hpp`；`print_memory_usage` 同理）——它们是
**暴露给用户/调试的 introspection 面**，不是内部热路径。真正被测试依赖的是「构造 + `destroy()`」
这条主干。

---

## 8. H100 实测

`harness/exp_deepep_bufferbase_lifecycle.{sh,py,out.txt}`（H100 80GB HBM3 / sm90，torch 2.12，
g++ 13）+ `src/bufferbase_lifecycle.cpp`，**32 PASS / 0 FAIL**。三部分：

**Part 1（C++，真实 `base.hpp`）**——影子 include 树里跑真实的 `BufferBase` 类体：
- `1.0` 拷贝进影子树的 `base.hpp` 与真实文件 **sha256 逐字节相同**（`ba2c0cab18b449f4`）；
- `1.1` g++ `-std=c++17 -Wall` 编译 0 error（pybind11 用最小 stub 顶掉，仅需 `class_/def/def_readonly`）；
- `A0/A1/E1-E3` 默认模式：构造后 `main_context` 非空；析构自动 `destroy()` 恰好 1 次；`weak_ptr`
  证明 `destroy_on_destruction` 在自动分支把 `main_context` 置空（`:22`）；
- `B1` 显式模式不销毁 → `destroy()` 0 次；`1.2` 泄漏告警文本**恰好出现一次**；
- `C1-C3` 显式 `destroy()` 一次、且**显式路径不清基类 `main_context`**（只有自动分支清，`:22`
  与派生 `destroy()` 只清自己的字段形成分工）；
- `D1/D2` 二次显式 `destroy()` 抛异常、计数不再增；
- `F0-F2` `print_memory_usage` 三态；`G` `sizeof(BufferBase)=32 / alignof=8`。

**Part 2（Python，真实 `base.py`，按路径加载绕过 `check_nccl_so`）**：
- `2.0` 抽象基类不可直接实例化（`destroy` 的 `__isabstractmethod__` 为 `True`）；
- `2.1` `main_context` property 转发；`2.2` 默认模式显式 `destroy()` 撞 `assert explicitly_destroy`
  **且未销毁 runtime**（先断言、后动作）；`2.3` 显式模式委托 + 置 `None` + property 变 `None`；
- `2.4` 二次 `destroy()` **幂等**（与 C++ 抛异常对照）；`2.5/2.6` `print_memory_usage` 两态。

**Part 3（源解析）**：四个家族的 `class X : public BufferBase` + `pybind11::class_<X, BufferBase>`
+ `destroy_on_destruction(` + Python `class X(BufferBase)` + `super().destroy()` **五要素齐全**
（`3.1-3.5`）；`3.6` `deep_ep/__init__.py:63` 顶层 re-export；`3.7` C++ 注册了独立 `BufferBase`
类型（与 Python ABC 平行的第二条链）；`3.8` 派生 `destroy()` 的后置字段清理。

---

## 9. 小结

- `BufferBase` 是四个 DeepEP Buffer 家族的**唯一公共祖先**，把生命周期收敛成
  `explicitly_destroy` + `destroyed` + `main_context` 三个字段。
- **两条平行继承链**：Python ABC 管「用户可见语义」，C++ pybind 类管「资源释放时序」，靠
  `self.runtime` 引线相连。
- **C++ 三分支状态机** `destroy_on_destruction`：默认自动销毁；显式模式必须用户 `destroy()`，
  否则析构打印泄漏告警；`destroyed` 标志把重复销毁变成硬错误。
- **`noexcept(false)` 是有意的**：让默认模式析构时的资源释放异常能传播。
- **一个跨语言分歧**：二次 `destroy()` —— Python 幂等、C++ 抛异常；读代码不可互相想当然。
- 契约与 **GPU/架构无关**，H100(sm90) 真机 **32 PASS / 0 FAIL**。
