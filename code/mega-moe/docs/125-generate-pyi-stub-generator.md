# 125 · DeepGEMM 的 Python 类型桩生成器：`scripts/generate_pyi.py`

> 第 124 篇讲了「任何一行设备代码跑起来之前」的**构建 / 打包地基**，其中 `setup.py:136-146`
> 的 `generate_pyi_file` 只被一句话带过：它跑 `scripts/generate_pyi.py` 生成 `_C.pyi` 类型桩再拷进 wheel。
> 但这份 `.pyi` 是**用户唯一能看到的 Python 侧接口契约**（IDE 补全、`pyright`/`mypy` 类型检查、
> 文档站抓取签名都靠它），而生成它的那 890 行纯 Python 却从没被单独读过。
>
> 本篇把 `scripts/generate_pyi.py` 逐段拆开：它怎么用正则扫 C++ 源码建「函数名→签名」索引、
> 怎么用**四种括号的嵌套计数器**（`BracketTracker`）把 `m.def(...)` 的顶层实参切开、
> 怎么把 C++ 类型/默认值翻译成 Python 注解，以及其中**几处会被真实 C++ 语法击穿的边界**。
> 纯 stdlib、**架构无关**，H100(sm90) 真机用 importlib 按路径加载**真实**脚本跑 **45 PASS / 0 FAIL**。
> 目标源码：`~/github/DeepGEMM@057ca59`（`deep_gemm.__version__ 2.8.1`，
> `scripts/generate_pyi.py` sha256[:20] `f0d0e7e9da88cf28b580`）。

## 1. 它在整条链上的位置

DeepGEMM 是 **JIT** 的：`pip install` 只编一个很薄的 host `_C`（第 33 篇 DeepJIT），设备 kernel 首次
调用才现编。`_C` 的 pybind11 绑定不是写在一个 `.cpp` 里，而是分散在 `csrc/apis/*.hpp` 各自的
`register_apis(m)` 函数中——`csrc/python_api.cpp:31-48` 逐个调用 11 个 `register_apis`（第 84 篇）。
每个 `register_apis` 里用 `m.def("名字", &cpp_fn, py::arg(...) = 默认值, ...)` 注册公开函数，用
`m.attr("别名") = m.attr("原名")` 造别名。真机统计：**`m.def` 62 条、`m.attr` 20 处调用（10 个别名）**，
分布在 9 个 `apis/*.hpp`（`gemm.hpp` 25、`config.hpp` 11、`attention.hpp` 8、`layout.hpp` 8、
`mega_moe.hpp` 5、`einsum.hpp` 2、`mega_mhc.hpp`/`hyperconnection.hpp`/`mega_gate.hpp` 各 1）。

`generate_pyi.py` 要做的就是**不编译、不 import `_C`**，只靠文本分析把上面这些 `m.def` 变成一张
`.pyi` 类型桩。调用点在 `setup.py:19/120/136-146`：

```python
def generate_pyi_file(self):
    generate_pyi_file(name='_C', root='./csrc', output_dir='./stubs')
    pyi_source = os.path.join(current_dir, 'stubs', '_C.pyi')
    pyi_target = os.path.join(self.build_lib, 'deep_gemm', '_C.pyi')
    ...  # 拷进 wheel
```

注意 `stubs/` 被 `.gitignore:20-21` 忽略——**`.pyi` 是构建产物、不进仓库**（但它一定随 wheel 发布）。
所以本文把「跑两遍逐位一致」和「与该构建产物 `stubs/_C.pyi` 逐位一致」都当判据（§8）。

## 2. 主流程：五段式（`generate_pyi_file:872-890`）

```python
def generate_pyi_file(name, root, output_dir='.'):
    func_index = build_cpp_function_index(root)      # ① 扫 csrc/** 建「函数名 → C++ 签名」
    results = extract_m_def_statements(root)         # ② 扫所有 m.def(...) 完整语句
    cpp_results = []
    for item in results:
        enhanced_item = parse_mdef_and_attach_cpp_signatures(item, func_index)  # ③ 拆实参 + 挂签名
        cpp_item = extract_cpp_signature_details(enhanced_item)                 # ④ 拆返回/参数类型
        cpp_results.append(cpp_item)
    pyi_content = generate_pyi_file_content(cpp_results, module_name=name)      # ⑤ 渲染 .pyi
```

这条链有一个**反直觉的设计**：`.pyi` 里的类型和参数名**主要来自 C++ 侧**，不是 `py::arg`。
`py::arg(...)` 只贡献两样东西——**参数个数**（`num_args`）和**默认值**（按序位）。参数名和类型都从
`build_cpp_function_index` 抓来的 C++ 函数签名里解析。下面逐段看为什么。

## 3. `extract_m_def_statements`：逐行括号配平取语句（`:151-222`）

它不是用正则一次匹配，而是**逐行扫描 + 手写括号配平**，因为 `m.def(...)` 经常跨十几行（真机
`fp8_fp4_gemm_nt` 的注册就跨 8 行）。核心：

- `:179` 用 `'m.def(' in line` 做**子串**命中（不是锚定行首）；
- `:183-186` 只跳过**整行**注释：`stripped.startswith('//')` 或 `stripped.startswith('/*')`；
- `:192-211` 从命中行起累加 `(`/`)` 计数直到归零，`''.join(lines[i:j+1])` 拼成完整语句；
- `:196` 用 `re.search(r'm\.def\s*\(', current_line[:k+1])` 确认 `(` 确实属于 `m.def`。

**为什么用子串而不是锚定**：`m.def` 有时前面有缩进或写在其它表达式后。代价是**行内尾注释会被误采**
（§6.4）。真机 `extract_m_def_statements` 得到 **62 条**，与 `m.def` 总数一致。

## 4. `build_cpp_function_index`：`csrc/**` 的函数名索引（`:5-94`）

`:9-14` 的正则把「返回类型 + 函数名 + `(`」切出来：

```python
pattern = re.compile(r'([\w:\s*<&>,\[\]\(\)]+?)\s+([a-zA-Z_][a-zA-Z0-9_:]*)\s*\(')
```

`:30-32` 先**按行过滤**掉 `#`（预处理指令）与 `//`（注释）开头的行；`:41-43` 再用关键字黑名单
（`return`/`if`/`for`/`while`/`switch`/`case`/`throw`/`catch`/`auto`…）排除控制流语句；
`:46-49` 把 `ns::Cls::foo` 归一成末段 `foo`；`:76-80` 检查 match 前 50 字符——若紧邻字母/数字/
下划线（说明它是某个更长表达式的中段，而非语句边界）就丢弃；`:82-92` 只收**函数定义**（后面有 `{`）
或**头文件声明**（`.hpp/.h` 且 `;` 在 `{` 之前）。`:91-92` **首个定义胜**：`if base_name not in func_index`
——同名重载/多次声明只保留第一次出现的签名。真机 `csrc` 建出 **290 条**索引。

## 5. `parse_m_def_statement` + `BracketTracker`：拆顶层实参

### 5.1 `BracketTracker`（`:97-148`）——四种括号的嵌套计数

```python
def update(self, char):
    if char == '(': self.paren += 1
    elif char == ')': self.paren -= 1
    elif char == '[': self.bracket += 1
    elif char == ']': self.bracket -= 1
    elif char == '{': self.brace += 1
    elif char == '}': self.brace -= 1
    elif char == '<' and self._in_top_level_of_other_brackets(): self.angle += 1
    elif char == '>' and self.angle > 0 and self._in_top_level_of_other_brackets(): self.angle -= 1
```

三个关键设计：

1. **`<`/`>` 只有在 `()[]{}` 全为 0 时才算模板括号**（`_in_top_level_of_other_brackets:135-139`）。
   这样 `foo(a<b)` 里括号内的比较 `<` 不被误当模板——`(` 已经把 `paren` 抬到 1，`<` 被忽略。
2. **`>` 必须 `angle > 0` 才减**（`:132`）。这让 `a->b`（`-` 被忽略、`angle==0`）和比较 `a > b`
   都不影响计数;嵌套 `std::vector<std::pair<int,int>>` 的 `>>` 连减两次正确闭环。
3. **`is_top_level:141-148` 要求四个计数器全为 0**，`parse_m_def_statement:264` 用它判定
   「这个 `,` 是不是顶层实参分隔符」。

**已知边界**：顶层移位 `a << b` 会让 `angle` 变成 2（`:130` 对两个 `<` 都加），使 `is_top_level()` 假。
参数列表里不会出现 `<<`（模板实参也是单 `<`），所以实践中没触发——但这是「用 `<`/`>` 计数」的固有代价。

### 5.2 拆实参（`:225-342`）

- `:241-252` 先手写括号配平定位 `m.def(` 的内容区间；
- `:256-271` 用 `BracketTracker` 在**顶层逗号**处切分参数列表；
- `:276-282` 第 0 个实参必须是字符串字面量（Python 函数名），否则抛错；
- `:284-301` 第 1 个实参：以 `[` 开头 → **lambda**（`is_lambda=True`，如 `config.hpp` 的 `init`
  `[&](const std::string& ...)`）；否则去掉 `&`、取 `::` 末段得到 C++ 函数名；
- `:303-341` 其余实参是 `py::arg(...)`。**关键**：它只统计个数、并记录**带默认值的那些的下标**
  （`:331` 找顶层 `=`），默认值字符串（如 `std::nullopt`、`"nk"`、`std::make_tuple(1,1,128)`）。

真机 `fp8_fp4_gemm_nt`：`num_args=11`、`default_args={3:nullopt, 4:nullopt, 5:nullopt, 6:nullopt,
7:"nk", 8:false, 9:nullopt, 10:nullptr}`。而 lambda 型（`init`、`set_block_size_multiple_of` 等 14 个）
只会产出 `def 名字(*args, **kwargs) -> Any: ...`（`:785-786`）——**配置类 API 没有精确签名**。

## 6. 类型 / 默认值翻译与三处边界

### 6.1 `parse_cpp_signature`（`:417-462`）——从签名拆返回类型与形参

函数名取 `(` 前**最后一个 token**（`:439`），返回类型是「其余全部」（`:440`），所以
`static void fp8_fp4_mega_moe(` → 返回类型 `static void`。参数用 `split_cpp_parameters:465-491`
（同一个 `BracketTracker` 顶层逗号切分）再逐个 `parse_parameter_declaration:494-578`。

`parse_parameter_declaration` 的巧妙处：**从右往左、在顶层找形参名**（`:536-560`）。它先剥掉顶层
`= 默认值`（`:504-514`），再从字符串末尾回扫，遇到「顶层标识符字符」就回退找标识符起点，再用
**类型关键字黑名单**（`int`/`bool`/`char`/`const`/`unsigned`… `:551-553`）排除「这个标识符其实是类型名」，
剩下的就是形参名，前面是类型。真机 `fp8_fp4_mega_moe` 拆出 **16 个形参**，名字与源码逐字一致
（`y, l1_weights_tuple, l2_weights_tuple, shared_l1_weights_tuple_opt, ...`）。

### 6.2 `cpp_type_to_python_type`（`:608-696`）——注释里写的 ORDER MATTERS

`:617-619` 先删 `static/inline/const/volatile` 等限定符和 `&`/`*`，再**按顺序**匹配：

| C++ | Python | 行 |
|---|---|---|
| `void` | `None` | `:621-622` |
| `std::pair<T1,T2>` | `tuple[T1, T2]` | `:627-637` |
| `std::tuple<...>` | `tuple[...]` | `:639-644` |
| `std::vector<T>` | `list[T]` | `:646-655` |
| `std::optional<T>` | `Optional[T]` | `:657-666` |
| `std::string` / `char*` | `str` | `:668-674` |
| `bool` | `bool` | `:676-678` |
| `int/long/size_t/intN_t/uintN_t` | `int` | `:680-684` |
| `float/double` | `float` | `:686-688` |
| `torch::Tensor` | `torch.Tensor` | `:690-692` |
| 其它 | `Any`（并告警） | `:694-696` |

模板分支**必须排在标量分支前**，否则 `std::optional<torch::Tensor>` 会先撞上「含 `bool`/`int` 的子串」。
嵌套也能递归：`std::optional<std::tuple<torch::Tensor, torch::Tensor>>` →
`Optional[tuple[torch.Tensor, torch.Tensor]]`，靠 `split_template_args:699-720` 逐层切。

### 6.3 三处真实边界（都已被 H100 实验复现）

1. **别名缺口**（最重要）：`m.attr("fp8_gemm_nt") = m.attr("fp8_fp4_gemm_nt")` 造的别名**不会进 `.pyi`**，
   因为整条链只认 `m.def`。真机 `gemm.hpp:921-930` 有 **10 个别名**（`fp8_gemm_nt`/`fp4_gemm_nt`/
   `fp8_gemm_nn`/`fp8_gemm_tn`/`fp8_gemm_tt` 及 5 个 grouped 别名），**10 个全部没有 stub**；
   而它们正是文档和用户最常用的名字。`.pyi` 里只有被别名的原名 `fp8_fp4_gemm_nt`。
2. **`std::variant` 被过宽正则击穿**：`cpp_type_to_python_type('std::variant<int, std::tuple<int,int>>')`
   返回 `'int'`——因为 `:681-683` 的 `\b(int|...)\b` 在子串里匹配到了 `int`。也就是说带 `int` 的
   未知模板会被误标成 `int` 而不是 `Any`（好在这类签名目前大多走 lambda 分支）。
3. **`default_args` 的「序位」语义**：`:307-340` 记录的默认值下标是**相对 `py::arg` 列表**的，而消费端
   `generate_pyi_function:800-817` 是按**C++ 形参序位** `i` 去查。当前所有函数 `py::arg` 都按 C++ 形参
   顺序、数量一致（`fp8_fp4_gemm_nt` 11==11），所以巧合对齐；一旦某函数少写几个 `py::arg`，默认值就会
   **错位挂到前面的 C++ 形参上**（实验 J2 用合成反例坐实）。

另外两处小瑕疵：`BracketTracker` 对顶层 `<<` 的误计（§5.1）；`extract_m_def_statements` 只跳过整行注释，
**行内尾注释 `int x=0; // m.def("fake", ...)` 仍会被当成语句采集**（实验 B4）。

### 6.4 `cpp_default_to_python_default`（`:723-778`）

字符串字面量原样保留（`:734-736`）；`true/false → True/False`（`:738-742`）；`nullptr/NULL/nullopt → None`
（`:744-746`）；`std::tuple<...>({a,b}) → (a, b)`（`:748-753`）；`std::make_tuple(...) → (...)`（`:755-761`）；
`std::vector<T>({...}) → [...]`（`:763-768`）；整数/浮点字面量原样（`:770-774`）；其余告警回退 `None`
（`:776-778`）。真机 `recipe` 默认 `std::make_tuple(1, 1, 128)` → `(1, 1, 128)`、`compiled_dims` 默认
`"nk"` → `"nk"`、`epilogue` 默认 `nullptr` → `None`。

### 6.5 死代码

`:345-386` 的 `extract_cpp_signature_from_content(cpp_func_name, content)` 定义了「在给定内容里找某函数
签名」的能力，但**全仓库没有任何调用点**（它被 `build_cpp_function_index` 那条更早的统一索引取代了）。
实验 K1 确认它只出现 1 次（定义处）。留着无害，但属于重构残留。

## 7. 渲染（`generate_pyi_function:781-825` / `:828-869`）

`generate_pyi_function` 把一条 `m.def` 渲染成 `def 名(形参: 类型 = 默认, ...) -> 返回类型: ...`；lambda 直接
`(*args, **kwargs) -> Any`；形参名若撞 Python 关键字（`def`/`class`/`None`… `:806-807`）加下划线后缀。
`generate_pyi_file_content:828-869` 收集所有 decl，并按是否出现 `Optional[`/`torch.Tensor`/`numpy.ndarray`
动态决定 import 行——真机输出第二行是 `from typing import Any, Optional`、有 `import torch`、**无 numpy**。

一个副作用：因为参数名/类型来自 **C++ 签名**而非 `py::arg`，而没有 `py::arg` 的函数（如
`fp8_fp4_mega_moe`、`bf16_mega_moe`，`mega_moe.hpp:393-397`）会直接暴露**C++ 内部形参名**
（`y`、`l1_weights_tuple`、`shared_l1_weights_tuple_opt`…），且**没有任何默认值**——它们与官方文档里
用的 kwarg 名不一定一致。

## 8. H100 实测

- `harness/exp_generate_pyi.py` / `.sh` → `harness/exp_generate_pyi.out.txt`。
- 纯 stdlib、**架构无关**；用 `importlib.util.spec_from_file_location` 按路径加载**真实**
  `scripts/generate_pyi.py`，跑真机 `csrc` 与合成 C++ 片段。
- H100(sm90) 真机 **45 PASS / 0 FAIL**：
  - **A 端到端**：两次生成逐位一致（sha256[:16] `57a06883344d96b5`，612 行 / 62 函数）；与仓库既有
    `stubs/_C.pyi`（构建产物）**逐位一致**；import 行、只 torch 不 numpy。
  - **B 扫 m.def**：真机 62 条 / 9 文件；整行注释跳过；**行内尾注释仍被采集**（边界）；`m.attr` 20 处不计入。
  - **C 拆实参**：函数指针 vs lambda、`&ns::cls::foo` 归一化、11 个 `py::arg` 与 `{3..10}` 默认值索引、
    `mega_moe` 无 `py::arg` → `num_args=0`。
  - **D 建索引**：290 条；`static void` 返回、单行声明、**首定义胜**。
  - **E BracketTracker**：`>>` 闭环、`->`/括号内 `<`/比较 `>` 安全、顶层 `<<` 误计（边界）。
  - **F/G 类型与默认值映射**：18 + 11 组全对；`variant`→`int` 过宽、未识别→`Any`/`None`。
  - **H 拆签名**：`fp8_fp4_mega_moe` 16 形参、名字逐字一致。
  - **I 别名缺口**：10 个别名 0 个有 stub。
  - **J 序位语义**：当前对齐；合成反例触发错位。
  - **K 死代码**：`extract_cpp_signature_from_content` 无调用点。

## 9. 与其它篇的关系、边界

- 与**第 84 篇（Python 绑定面 `python_api.cpp`）**互补：那篇讲 `register_apis` 的 11 个模块如何被
  `PYBIND11_MODULE` 串起来（运行时视图）；本篇讲**同一批绑定如何被文本分析成类型桩**（静态视图）。
- 与**第 124 篇（构建/打包地基）**接续：那篇到 `setup.py:136-146` 调用点为止，本篇展开被调用者。
- 与**第 33/47 篇（DeepJIT）**互补：JIT 缓存键也一样靠「源码文本摘要」，是同一种「不编译、只读源码」
  的思路。
- 与 13 个 SM100 kernel 的关系：**无关**——`.pyi` 只描述 host `_C` 的 Python 接口，不描述设备 kernel；
  但它把第 10/12/13 篇反复出现的 `k_grouped_*`/`m_grouped_*`/`fp8_fp4_mega_moe` 等名字的
  **参数名与默认值**固化成用户可见契约。

一句话：`.pyi` 生成器是**两库唯一一处「用 Python 解析 C++」的静态工具**，它复用了 DeepJIT 的
「文本即真相」哲学；其正确性完全建立在几个手写正则 + `BracketTracker` 的边界约定上，我们今天把
这些边界连同三处会被真实语法击穿的角落一起钉住了。

## 10. 引用

- 源码：`~/github/DeepGEMM@057ca59`（`v2.1.1.post3-24-g057ca59`，`deep_gemm 2.8.1`）
  - `scripts/generate_pyi.py`（890 行，sha256[:20] `f0d0e7e9da88cf28b580`）
  - `setup.py:19/120/136-146`、`csrc/python_api.cpp:31-48`、`csrc/apis/gemm.hpp:838-930`、
    `csrc/apis/mega_moe.hpp:155-397`、`.gitignore:20-21`
- 线上：DeepGEMM 仓库 https://github.com/deepseek-ai/DeepGEMM
- 本专题：第 33 篇（DeepJIT）、第 47 篇（post_hook/后端）、第 84 篇（Python 绑定面）、
  第 124 篇（构建/打包地基）。
