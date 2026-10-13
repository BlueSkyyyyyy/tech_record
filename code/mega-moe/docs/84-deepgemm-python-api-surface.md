# DeepGEMM 的 Python 绑定面：`csrc/python_api.cpp` 与 11 个 `register_apis`

> 前面 83 篇把 DeepGEMM / DeepEP 的**内核**与**测试**拆得很细，但一直缺一页把它们**在 Python 里
> 到底怎么被调到**讲清：`import deep_gemm` 之后手上那个 `_C` 扩展是怎么拼出来的、每个
> `deep_gemm.*` 对应 C++ 哪个函数、参数名与默认值从哪来、哪些名字是别名、哪些是 Python 包装、
> 哪些故意不对外。本篇把 `csrc/python_api.cpp`（49 行，模块装配）与 11 个 `register_apis`
> （`csrc/apis/*.hpp`，共 62 个 `m.def` + 10 个 `m.attr` + 2 个子模块）逐项对回真机。
>
> **整层是 pybind11 host 绑定，与 UMMA/SM100 无关**，本机 H100(sm90) 直接 `import deep_gemm`
> 就能拿到真实 `_C` 扩展：`harness/exp_deepgemm_python_api.{py,sh,out.txt}` **51/51 PASS**。
> 同源坐标：DeepGEMM `057ca59`（`deep_gemm 2.8.1`）。

## 1. 一句话总览

`deep_gemm` 是一个 **C++ 扩展 + 薄 Python 包**：

- **C++ 侧**：`csrc/python_api.cpp:24` 的 `PYBIND11_MODULE(_C, m)` 依次调用 **1 个 JIT 注册 +
  11 个 `register_apis`**，把「配置开关、epilogue 工厂、locality domain、以及全部 kernel 入口」
  平铺进一个 `_C` 模块对象。
- **Python 侧**：`deep_gemm/__init__.py` 只做四件事——re-export 想要的 `_C` 名字、从 `.mega`
  补上 5 个 **Python 包装函数**、从 `.utils` 补上量化/布局工具、最后 `_C.init(...)` 初始化 JIT，
  并 `atexit.register(shutdown)`。
- **约定**：`_C` 里能看到的**顶层**名字 = `62 m.def + 10 m.attr 别名 + {Runtime, get_jit} +
  {epilogue, locality_domain}` = **76 个**（真机对账 0 误差）。其中 `{Runtime, get_jit, init,
  get_symm_buffer_size_for_mega_moe, get_token_alignment_for_mega_moe}` **故意不 re-export**，
  只能从 `deep_gemm._C.*` 拿到。

## 2. 模块的装配：`python_api.cpp`

```
csrc/python_api.cpp
:24  PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {     // TORCH_EXTENSION_NAME 默认 _C（:20）
:25      m.doc() = "DeepGEMM C++ library";
:28      deep_jit::register_python_api(m, deep_gemm::jit);   // 注册 JIT 对象：Runtime / get_jit
:31      deep_gemm::config::register_apis(m);               // 11 个开关
:34      deep_gemm::locality_domain::register_apis(m);      // 子模块 locality_domain
:37      deep_gemm::epilogue_class::register_apis(m);       // 子模块 epilogue
:41      deep_gemm::attention::register_apis(m);            // 8 个
:42      deep_gemm::einsum::register_apis(m);               // 2 个
:43      deep_gemm::hyperconnection::register_apis(m);      // 1 个
:44      deep_gemm::gemm::register_apis(m);                 // 25 m.def + 10 m.attr
:45      deep_gemm::layout::register_apis(m);               // 8 个
:46      deep_gemm::mega::register_apis(m);                 // 5 个（mega moe）
:47      deep_gemm::mega_mhc::register_apis(m);             // 1 个
:48      deep_gemm::mega_gate::register_apis(m);            // 1 个
:49  }
```

两个「为什么」：

1. **为什么 `mega::register_apis` 排在 `gemm` 之后**：它注册的 `fp8_fp4_mega_moe`/`bf16_mega_moe`
   是**原始 C++ 入口**（参数是一长串已展开的裸张量 + `sym_buffer.buffer` + `buffer_ptrs`），
   Python 侧真正的用户 API 是 `deep_gemm/mega/__init__.py` 里的**同名包装**（`mega:168/193`），
   它把 `SymmBuffer` 对象拆成这些裸参数再转调 `_C`。C++ 先于 Python 注册，Python 包装才能
   用 `_C.fp8_fp4_mega_moe(...)` 转发。
2. **`deep_jit::register_python_api`** 注入的 `Runtime`（暴露 `get_num_sms`/`get_tc_util` 等
   只读镜像）与 `get_jit()` 是 DeepJIT 的调试面，`deep_gemm/__init__.py` **不 re-export**
   （见 §6）——它们属于「底层设施」，不属于算子 API。

行号坐标（`register_apis` 定义处）：

| 文件 | 行 | 贡献 |
|---|---|---|
| `csrc/apis/config.hpp` | `:8-42` | 11 `m.def`（`init`/`shutdown`/`set_*`/`get_*`） |
| `csrc/apis/epilogue_class.hpp` | `:13-25` | 子模块 `epilogue` + 4 个工厂类 |
| `csrc/apis/locality_domain.hpp` | `:149-160` | 子模块 `locality_domain` + 9 `submodule.def` |
| `csrc/apis/attention.hpp` | `:445-477` | 8 `m.def`（MQA logits 家族） |
| `csrc/apis/einsum.hpp` | `:299-309` | 2 `m.def` |
| `csrc/apis/hyperconnection.hpp` | `:58-62` | 1 `m.def` |
| `csrc/apis/gemm.hpp` | `:835-996` | 25 `m.def` + 10 `m.attr` + 4 `cublaslt_*` + 2 Muon |
| `csrc/apis/layout.hpp` | `:118-143` | 8 `m.def`（SF 布局 + mk-alignment） |
| `csrc/apis/mega_moe.hpp` | `:392-398` | 5 `m.def` |
| `csrc/apis/mega_mhc.hpp` | `:254-285` | 1 `m.def`（26 个参数） |
| `csrc/apis/mega_gate.hpp` | `:178-195` | 1 `m.def`（12 个参数） |

## 3. 别名：`m.attr` 是「同一个函数对象」，不是复制

第 46 篇讲过 FP8/FP4 在 host 层是**同一个实现**（一套模板通吃）。这一点在绑定层有一处非常
直白的体现：`csrc/apis/gemm.hpp:920-930` 用 `m.attr("新名") = m.attr("旧名")` 造了 **10 个别名**：

```
gemm.hpp:921  m.attr("fp4_gemm_nt") = m.attr("fp8_fp4_gemm_nt");
gemm.hpp:922  m.attr("fp8_gemm_nt") = m.attr("fp8_fp4_gemm_nt");
gemm.hpp:923-925  ... fp8_gemm_nn / fp8_gemm_tn / fp8_gemm_tt
gemm.hpp:926-930  ... m_grouped_fp4/fp8_gemm_*_contiguous/_masked
```

`m.attr` 是「把模块 `__dict__` 里的旧对象**再挂一个键**」，因此两名字指向**同一个 pybind
`builtin_function_or_method`**。真机验证（`harness` Part C）：

```
fp8_gemm_nt      is fp8_fp4_gemm_nt        -> True
fp4_gemm_nt      is fp8_fp4_gemm_nt        -> True
m_grouped_fp8_gemm_nt_contiguous is m_grouped_fp8_fp4_gemm_nt_contiguous -> True
...（10/10 True）
```

**为什么这么做而不是写 10 个 `m.def`**：`fp8_gemm_nt` 与 `fp8_fp4_gemm_nt` 若各自 `m.def`，
就是两个独立对象，JIT 缓存键/`__doc__`/类型签名都要维护两份，且容易出现「只改了一个」的漂移。
`m.attr` 保证「FP8-only 命名的旧代码」与「FP8/FP4 统一命名的新代码」在**运行时完全等价**——
`fp8_gemm_nt(a,b,d, ...)` 与 `fp8_fp4_gemm_nt(...)` 是同一段 C++、同一张缓存 entry。
真机冒烟 `bf16_gemm_nt`（relerr 1.77e-5）与 `alpha` 简写（见 §5）都直接跑通。

## 4. 默认值契约（一）：`compiled_dims` 随主序变

`register_apis` 里每个 `m.def` 的关键信息其实是那几个 `py::arg(...)=默认值`。第 46 篇讲过
`compiled_dims` 决定「哪些维进 JIT 缓存键（其余维变 0 一核多形状）」，而它在绑定层的默认值
**随输入主序变**（`csrc/apis/gemm.hpp:842/850/858/866` 等）：

| 入口族 | `nt` / `nn` | `tn` / `tt` |
|---|---|---|
| `fp8_fp4_gemm_*` | `'nk'` | `'mn'` |
| `bf16_gemm_*` | `'nk'` | `'mn'` |
| `m_grouped_*_contiguous` / `_masked` | `'nk'` | — |
| `k_grouped_*` | — | `'mn'` |

真机从 `__doc__` 解析 18 个入口的默认值，**18/18 与源码一致**（`harness` Part D）。为什么：
`nt`（A=[M,K]、B=[N,K]）里 K 是收缩维、M/N 是「主序」，编译维取 `nk` 意味着同一 kernel 能吃
不同 `m`（把 `m` 编成 0）；`tn`（A/B 转置）下 `mn` 才是「不随 batch 变的维」。这条默认值不是
随手写，而是与 `get_theoretical_mk_alignment_for_contiguous_layout` 那套对齐启发式一脉相承。

## 5. 默认值契约（二）：`recipe`、PSUM kwargs、`epilogue`/`alpha` 简写

`register_apis` 里另外三组「会咬人」的默认值：

**(a) `recipe` 默认值按 dtype 分**（`python_api.cpp:...` → `gemm.hpp:895-919`、`einsum.hpp:304-308`）：

```
k_grouped_fp8_gemm_tn_contiguous  recipe = (1, 1, 128)   # gemm.hpp:895-901
k_grouped_fp8_gemm_nt_contiguous  recipe = (1, 1, 128)
k_grouped_fp4_gemm_nt_contiguous  recipe = (1, 1, 32)    # FP4 的 gran_k=32
fp8_einsum                        recipe = (1, 128, 128)
```

而 `fp8_fp4_gemm_*` 的 `recipe = None`（dtype 可从张量自推）。这正对应第 10/12 篇讲过的
「FP8 收缩维 per-128 gran、FP4 per-32」。真机对账 4/4 一致（`harness` Part D2）。

**(b) PSUM 布局的四个 kwargs 只出现在 `m_grouped_*_contiguous`**（`gemm.hpp:870-880`）：

```
use_psum_layout = False
ensure_zero_padding = True
expected_m_for_psum_layout = None
```

第 11/13 篇讲过 PSUM 是「长度 G 的逻辑 K 前缀和」这一第三种布局。绑定层把它默认关掉
（`use_psum_layout=False`），且 `ensure_zero_padding=True` 默认要求 gap 行清零——这是一个
**正确性优先的默认**：宁可多写零也不要 kernel 读到脏 padding。

**(c) `epilogue = nullptr` 与 `alpha` 简写**。`gemm.hpp:844-845` 等每个 GEMM 都有
`py::arg("epilogue") = nullptr` 与 `py::arg("alpha") = std::nullopt`；`epilogue_class.hpp:12`
的注释点明两者关系：**`alpha` 与 FP8 的 `(d, sfd)` 输出对就是 `Alpha` 与 `FP8Quantization`
的简写**。真机验证（`harness` Part G）：

```
bf16_gemm_nt(a,b,d, alpha=2.0)                     # 简写
bf16_gemm_nt(a,b,d, epilogue=_C.epilogue.Alpha(2.0)) # 显式
-> 两者输出逐位一致（torch.equal True）
```

**(d) 没有 `py::arg` 的入口就退化成 `arg0..argN`**。例如 `mega_moe.hpp:393-394` 的
`get_token_alignment_for_mega_moe`/`get_block_m_for_mega_moe`、`layout.hpp:126-127` 的
`get_tma_aligned_size`/`get_mn_major_tma_aligned_tensor` 都是直接 `m.def("x", &x)`，于是
真机 `__doc__` 里签名是 `get_block_m_for_mega_moe(arg0, arg1, ..., arg5: str)`——**最后一个
位置参数其实是被调度的 `activation` 字符串**，但名字丢失了。这是「用位置参数调用没问题、
用关键字参数就会踩」的一类入口（`harness` Part D3）。

## 6. 两个子模块：`epilogue` 与 `locality_domain`

**(a) `epilogue`（`epilogue_class.hpp:13-25`）** 是一个**工厂类子模块**：

```
submodule "Epilogue" 基类（:15，不直接实例化）
  ├─ Identity              (init<>)                    :16-17
  ├─ Alpha                 (init<const float&>, arg "alpha")  :18-19
  ├─ FP8Quantization       (init<const Tensor&>, arg "sfd")   :20-21
  └─ BF16StochasticRounding(init<>)                    :22-24
```

第 31 篇讲过这四个 SM100 epilogue 算子的语义；这里补的是**它们怎么变成 Python 对象**。
真机（`harness` Part E）：四类均可实例化、`Alpha` 无参构造按预期抛 `TypeError`、
`isinstance(Alpha(1.0), Epilogue)` 成立——**「基类 + 4 子类」的 pybind 继承链是真的**。

**(b) `locality_domain`（`locality_domain.hpp:149-160`）** 是一个**函数子模块**：

```
def_submodule("locality_domain", ...)   :150
  get_num_locality_domains / get_granularity / is_localization_available
  empty / empty_per_domain / is_localized
  get_sm_locality_domains / get_balanced_sm_locality_domains / release_mlopart   # 共 9 个
```

注意它与 `deep_gemm/locality_domain.py` 的**同名子模块**不同：C++ 侧是纯 9 个函数；Python 侧
（`__init__.py:101-107`）在此之上再包了 `localize`/`destroy_localizer` 等便利函数。第 34/81 篇
拆过这套探针，此处只补「C++ 注册了哪 9 个」。真机对账：`submodule.def` 9 个全部出现在
`_C.locality_domain` 里（`harness` Part E）。

## 7. 公开面 vs 隐藏面：`deep_gemm/__init__.py` 的选择

`deep_gemm/__init__.py`（127 行）不是简单 `from ._C import *`，而是**显式挑选**：

```
:16   from . import _C                      # 整个模块对象
:17-27  from ._C import shutdown, set_num_sms, ..., use_deterministic_algorithms   # 配置面
:29-35  from ._C import cublaslt_*, batched_syrk, batched_symm
:38-86  from ._C import <全部算子：gemm/mega_gate/einsum/attention/hyperconnection/layout>
:89-95  from .mega import SymmBuffer, get_symm_buffer_for_mega_moe,
                     transform_weights_for_mega_moe, fp8_fp4_mega_moe, bf16_mega_moe
:98   from ._C import epilogue              # 子模块
:101-107 from . import locality_domain; from .locality_domain import ...
:113  from .utils import *                  # 量化/布局/dist 工具
:122  _C.init(os.path.dirname(os.path.abspath(__file__)))   # 初始化 JIT
:125  atexit.register(shutdown)
:127  __version__ = '2.8.1'
```

真机对账，**`_C` 顶层 76 个名字里有 5 个没有 re-export**（`harness` Part F）：

```
deep_gemm 顶层拿不到、只能 deep_gemm._C.* ：['Runtime', 'get_jit', 'init',
   'get_symm_buffer_size_for_mega_moe', 'get_token_alignment_for_mega_moe']
```

逐个「为什么」：

- `Runtime` / `get_jit`：DeepJIT 底层对象，直接暴露会诱导用户绕过配置面（应走 `set_num_sms`
  等），故隐藏。
- `init`：`_C.init(dirname(__file__))` 在 import 时已自动调过（`:122`），用户不需要也不应该
  再调（重复 init 会替换 `jit` singleton）。
- `get_symm_buffer_size_for_mega_moe` / `get_token_alignment_for_mega_moe`：Mega MoE 的**底层
  尺寸/对齐查询**。用户实际用 `deep_gemm.mega.get_symm_buffer_for_mega_moe`（Python，`:87`）
  或 `get_block_m_for_mega_moe`（已 re-export，纯启发式复用），不需要直接碰这两个裸查询。

反过来，**Python-only 的 5 个名字**（`SymmBuffer` / `get_symm_buffer_for_mega_moe` /
`transform_weights_for_mega_moe` / `fp8_fp4_mega_moe` / `bf16_mega_moe`）来自
`deep_gemm/mega/__init__.py`，**不在 `_C` 里**：它们是包装层——`fp8_fp4_mega_moe`（`mega:168`）
把 `SymmBuffer` 拆成 `sym_buffer.buffer` / `sym_buffer.handle.buffer_ptrs` / `group.rank()` /
`num_max_tokens_per_rank` / `num_experts` / `num_topk` 再转调 `_C.fp8_fp4_mega_moe`（其 pybind
签名是 `arg0..argN`，因为 `mega_moe.hpp:396` 没写 `py::arg`）。真机验证：`_C.fp8_fp4_mega_moe`
的 `__doc__` 首行是 `fp8_fp4_mega_moe(arg0: torch.Tensor, arg1: tuple[...])`（隐藏的裸入口），
而 `deep_gemm.fp8_fp4_mega_moe` 是**无 pybind doc 的 Python def**（`harness` Part F）。

**生命周期**：`:122` 的 `_C.init(...)` 是「用包目录当 library root」初始化 DeepJIT（第 33 篇）；
`:125` 的 `atexit.register(shutdown)` 保证在 Python / PyTorch / CUDA 尚存时释放 cuBLASLt
handle 与 workspace，规避「原生 cuBLASLt 退出回调先释放存储、再被 C++ 静态析构 `cublasLtDestroy`」
的 use-after-free（第 50 篇）。这两行是 `__init__.py` 里**唯一有副作用**的两行。

## 8. H100 实测（`harness/exp_deepgemm_python_api.{py,sh,out.txt}`，51/51 PASS）

| 组 | 内容 | 结果 |
|---|---|---|
| A | 解析 `csrc/apis/*.hpp`：62 `m.def` / 10 `m.attr` / 9 `submodule.def` / 2 子模块 | 与源码逐条列出 |
| B | `_C` 顶层 76 名 = 62 + 10 + {Runtime,get_jit} + {epilogue,locality_domain} | **0 多余、0 缺失** |
| C | 10 个别名 `m.attr` 与目标**同一对象** | 10/10 `is` True |
| D | 18 个入口 `compiled_dims` 默认值（`nk`/`mn` 随主序） | 18/18 |
| D2 | 4 个 `recipe` 默认值（`(1,1,128)` / `(1,1,32)` / `(1,128,128)`） | 4/4 |
| D3 | 无 `py::arg` 入口退化为 `arg0..`（`get_block_m_...` 末位是 `activation`） | 3/3 |
| E | `epilogue` 基类+4 子类、`locality_domain` 9 函数；工厂实例化/`TypeError`/继承 | 全 PASS |
| F | 5 个隐藏入口、5 个 Python-only mega 包装、`_C` 裸入口有 pybind 签名 | 全 PASS |
| G | `bf16_gemm_nt` 真实发射（relerr 1.77e-5）；`alpha=2.0` ≡ `epilogue.Alpha(2.0)` **逐位** | 全 PASS |

关键原始数字：

```
deep_gemm 2.8.1  from .../deep_gemm/__init__.py
torch 2.12.0a0+0291f960b6.nv26.04.48445190  cuda 13.2  cc (9, 0)  NVIDIA H100 80GB HBM3
m.def 条目 62、m.attr 别名 10、submodule.def 9、def_submodule ['epilogue', 'locality_domain']
_C 顶层共 76 个
bf16_gemm_nt relerr=1.771e-05
epilogue.Alpha(2.0) 与 alpha=2.0 逐位一致
TOTAL 51/51 PASS   FAIL=[]
```

## 9. 结论与边界

- **一句话**：`deep_gemm` 的 Python 面 = `python_api.cpp` 装配的 `_C`（76 名）+ `__init__.py`
  的显式 re-export + `.mega`/`.utils`/`.locality_domain` 的 Python 包装。绑定层本身架构无关，
  H100 可整份验证。
- **三条最容易踩的契约**：① `compiled_dims` 默认值随主序变（`nt/nn`=nk、`tn/tt`=mn）；
  ② `epilogue.Alpha`/`alpha=`、`FP8Quantization`/FP8 输出对是**同义简写**；③ 无 `py::arg`
  的入口只能按位置调用（`get_block_m_for_mega_moe(..., "swiglu")`）。
- **边界**：本篇只覆盖**绑定面**（名字、签名、默认值、装配顺序）；每个算子「算出什么、怎么实
  现」由前面各篇展开。跨模块的 pybind 细节（如 `deep_jit::register_python_api` 注册的
  `Runtime` 方法集）属于 DeepJIT，见第 33/47 篇。

## 10. 参考（真实核对）

- 源码：`~/github/DeepGEMM@057ca59`（`deep_gemm 2.8.1`）——
  `csrc/python_api.cpp:24-49`；`deep_gemm/__init__.py:16-127`；`deep_gemm/mega/__init__.py:18-215`；
  `csrc/apis/{gemm.hpp:835-996, attention.hpp:445-477, einsum.hpp:299-309, layout.hpp:118-143,
  config.hpp:8-42, epilogue_class.hpp:13-25, locality_domain.hpp:149-160, mega_moe.hpp:392-398,
  mega_gate.hpp:178-195, mega_mhc.hpp:254-285, hyperconnection.hpp:58-62}`。
- 实验：`harness/exp_deepgemm_python_api.{py,sh,out.txt}`（H100，51/51 PASS）。
- 关联篇目：第 46 篇（输入主序与 `compiled_dims`）、第 31 篇（epilogue 四算子）、第 34/81 篇
  （locality domain）、第 42 篇（Mega MoE host 编排/SymmBuffer）、第 33/47 篇（DeepJIT）、
  第 50 篇（runtime 生命周期与 `atexit(shutdown)`）。
