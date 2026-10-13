# 124 · 两库的「构建 / 打包 / 安装期地基」：`find_pkgs.py` 与两个 `setup.py`

> 前面 123 篇把 DeepGEMM / DeepEP 的 kernel、host 编排、测试基座、JIT 缓存键、环境变量面、
> 生命周期契约都拆遍了，但有一条**在任何一行设备代码跑起来之前就已定型**的线索一直没被单独讲过：
> **构建 / 打包 / 安装期**。它决定了「这份 wheel 链到哪份 NCCL」「哪些环境变量被烤进包里」
> 「本机为什么能 `pip install` 成功」「`DG_SKIP_CUDA_BUILD` 到底跳过了什么」「editable 安装为什么
> 找不到 `cute/cutlass`」——也正是本机 **DeepEP 运行期阻塞**（路线图「阻塞」）的确切落点之一。
>
> 本篇把 `deep_ep/utils/find_pkgs.py`（78 行）与两库的 `setup.py`（DeepEP 140 行 / DeepGEMM 222 行）
> 逐段讲透，并用 H100 真机（纯 host / Python、**架构无关**）把每一条语义都跑成可复现数字。
> 目标源码：`~/github/DeepEP@93eb6eb`（`__version__ 2.5.0`，`find_pkgs.py` sha256[:16] `5464c1a307f36b5b`）、
> `~/github/DeepGEMM@057ca59`（`deep_gemm.__version__ 2.8.1`，`setup.py` sha256[:16] `7f67d3ec663b8fc8`）。

## 1. `find_pkg_root`：安装期定位 NCCL 的唯一入口

`find_pkgs.py:9-55` 的 `find_pkg_root(name, lib_name=None, optional=False)` 是整条构建链的起点——
`setup.py:78` 第一句就是 `nccl_root_dir = find_pkgs.find_nccl_root()`。它的查找顺序是**四层**：

```python
def find_pkg_root(name, lib_name=None, optional=False):
    upper = name.upper()
    for env_name in (f'EP_{upper}_ROOT_DIR', f'{upper}_DIR'):   # ① 环境变量覆盖
        if env_name in os.environ:
            return os.environ[env_name]

    path_priority = {p: i for i, p in enumerate(sys.path)}       # ② sys.path 优先级表
    best, best_priority = None, len(sys.path)

    for dist in distributions():                                 # ③ 扫全部安装元数据
        dist_name = (dist.metadata['Name'] or '').lower()
        if f'nvidia-{name}' not in dist_name and f'nvidia_{name}' not in dist_name:
            continue
        dist_site = str(dist._path.parent)
        priority = path_priority.get(dist_site, len(sys.path))
        if priority > best_priority:                             # ④ 只接受更小 priority
            continue
        if lib_name is not None:
            for f in (dist.files or []):
                if lib_name in str(f):
                    lib_dir = os.path.dirname(str(f.locate()))
                    root = os.path.dirname(lib_dir) if os.path.basename(lib_dir) == 'lib' else lib_dir
                    best, best_priority = root, priority
                    break
        else:
            pkg_dir = os.path.join(dist_site, 'nvidia', name)
            if os.path.isdir(pkg_dir):
                best, best_priority = pkg_dir, priority

    if not optional:
        assert best is not None, f'Cannot find package: {name}'
    return best
```

**① 环境变量覆盖**（`:22-25`）：先看 `EP_NCCL_ROOT_DIR`、再看 `NCCL_DIR`，命中即**原样返回、不校验路径是否存在**。
H100 实测 A1a/A1b/A1c：两个都设时 `EP_NCCL_ROOT_DIR` 胜；去掉前缀后 `NCCL_DIR` 顶上；传一个不存在的
路径也照返回（这解释了为什么 `EP_NCCL_ROOT_DIR=/tmp/nccl229` 这种「指哪编哪」的用法有效——它同时是
`check_nccl_so` 的期望库来源，见第 117 篇）。

**②/④ `sys.path` 优先级**（`:27-38`）：`priority` 是「该包所在 site-packages 在 `sys.path` 里的下标」，
不在 `sys.path` 里的记为 `len(sys.path)`（最低）。判据是 `if priority > best_priority: continue`——**严格大于**，
所以「priority 更小者胜；priority 相等时**枚举在后者覆盖前者**」。H100 实测 A4a/A4b：把 `s9` 插到
`sys.path[0]`（priority 0）、`s0` 不在 path（priority 大），无论枚举顺序都取 `s9`；A4c/A4d 让两者都不在
`sys.path`（priority 相等），反转枚举结果跟着反转——坐实「同优先级后者胜」这条用 `>` 而非 `>=` 的细节。
含义：机器上同时装 `nvidia-nccl-cu12` 与 `nvidia-nccl-cu13` 时，**谁更靠前在 `sys.path` 谁说了算**，
跟 `site-packages` 之外的同名包（priority 更低）一律让位。

**③ 元数据名字匹配**（`:31-33`）：`f'nvidia-{name}' not in dist_name and f'nvidia_{name}' not in dist_name`
是**子串匹配**，同时兼容 `nvidia-nccl` 与 `nvidia_nccl` 两种写法。H100 实测 A2a/A2b/A2c：
`nvidia-nccl-cu13` 与 `nvidia_nccl` 都命中、`nvidia-cudnn-cu13` 不算 `nccl`。

**lib_name 分支与「根目录裁剪」**（`:40-46`）：当给了 `lib_name`，就在该 dist 的文件清单里找第一个
**文件名含 `lib_name`** 的项；其所在目录若名为 `lib`，root 退到**上一级**（即包根），否则 root 就是该目录。
H100 实测 A3a/A3b：命中 `<pkg>/lib/libnccl.so.2` → root = `<pkg>`；命中 `<pkg>/deep/libnccl.so.2` →
root = `<pkg>/deep`。这就是为什么最终能稳定地拼出 `{root}/include` 与 `{root}/lib`（`setup.py:91-93`）。
不给 `lib_name` 时走 `:47-50` 的 else：只认 `{site}/nvidia/{name}` 这个目录存在，root 即该目录。

**optional**（`:52-55`）：`optional=True` 找不到返回 `None`，否则 `assert`。H100 实测 A5。
**真机**（A6a）：本机 `find_nccl_root()` 解析到 `/usr/local/lib/python3.12/dist-packages/nvidia/nccl`
（pip 包 `nvidia-nccl-cu13`），其 `lib/` 下确有 `libnccl.so*`——注意它**不是** PyTorch 实际加载的系统
`/usr/lib/x86_64-linux-gnu/libnccl.so.2.29.7`，两者版本不同正是本机阻塞的根（见 §4）。

## 2. `get_nccl_lib_name`：pip wheel 是 SONAME-only

`find_pkgs.py:72-78` 只做一件小事但很关键——决定 `setup.py:93` 里 `-l:{什么名}`：

```python
def get_nccl_lib_name(root: str) -> str:
    lib_dir = Path(root) / 'lib'
    for path in (lib_dir / 'libnccl.so', *sorted(lib_dir.glob('libnccl.so.*'))):
        if path.is_file():
            return path.name
    raise ModuleNotFoundError(f'libnccl.so not found under {lib_dir}')
```

顺序是「**无版本名的 `libnccl.so` 优先，否则把 `libnccl.so.*` 按字典序排序取第一个**」。为什么要这条：
很多 pip wheel 只发 **SONAME-only** 的库（`libnccl.so.2`，没有开发用的 `libnccl.so` 软链），
直接用 `-lnccl` 会让链接器找不到。H100 实测（B1a-B1d）：同时有 `libnccl.so`/`libnccl.so.2` → 取前者；
只剩版本名 → 取 `libnccl.so.2`；同时有 `.1`/`.2` → 字典序 `libnccl.so.1` 胜（注意是**字典序**不是语义版本）；
一个都没有 → `ModuleNotFoundError`。**真机**（B2）：本机返回 `libnccl.so.2`（pip 包确实没发无版本名软链），
`setup.py:93` 于是生成 `-l:libnccl.so.2 -Wl,-rpath,{root}/lib`——`-l:` 前缀让 ld 按**精确文件名**链接、rpath
把运行时查找钉在**同一个 wheel 目录**，这正是「编到哪份 NCCL」被写死的地方，也是第 117 篇
`check_nccl_so` 能逐字节比对的前提。

## 3. 持久化环境变量：安装期烤进包、运行时**只补不覆盖**

两库的 `setup.py` 都有一段**代码生成**，把「构建时已设」的白名单环境变量写进包内的 `envs.py`：

- DeepEP `setup.py:14-15` 的白名单恰 **7** 条（H100 实测 C0 从 AST 抽取核对）：
  `EP_JIT_CACHE_DIR`、`EP_JIT_PRINT_COMPILER_COMMAND`、`EP_JIT_CPP_STANDARD`、`EP_NUM_TOPK_IDX_BITS`、
  `EP_NCCL_ROOT_DIR`、`EP_DEFAULT_RDMA_SL`、`EP_OVERRIDE_RDMA_SL`；
- DeepGEMM `setup.py:148-155` 只有 **3** 条：`DG_JIT_CACHE_DIR`、`DG_JIT_PRINT_COMPILER_COMMAND`、
  `DG_JIT_CPP_STANDARD`。两库白名单前缀不重叠（实测 C0b）。

生成逻辑（`setup.py:63-74` / `:148-155`）等价于：

```python
code = '# Pre-installed environment variables\n'
code += 'persistent_envs = dict()\n'
for name in persistent_env_names:
    code += f"persistent_envs['{name}'] = '{os.environ[name]}'\n" if name in os.environ else ''
```

H100 实测 C1/C2/C3：只有**构建时确实存在**的变量才进 dict、值原样内联为字符串字面量、一个都没设时写空 dict。
运行时消费（DeepEP `__init__.py:11-14`、DeepGEMM `__init__.py:8-11`）是同一个模板：

```python
from .envs import persistent_envs
for key, value in persistent_envs.items():
    if key not in os.environ:          # ← 只补不覆盖
        os.environ[key] = value
```

H100 实测 C4/C5：包内 baked 值只在**用户没设**时生效；用户显式设的值一律胜出。**安装现场**（C6）：
容器里 `deep_ep` 的 `envs.py` 恰为 `persistent_envs = dict()`（这份是 `+local` 安装、构建时没设任何白名单变量），
而 `deep_gemm` 是 editable 安装、源码目录里根本没有 `envs.py`（`from .envs import ...` 被 `except ImportError` 吞掉）。
这就是第 117 篇说的「persistent envs 只补不覆盖」的**生成侧**：**部署机在哪里、用什么 NCCL，
在 `pip install` 那一刻就被烤进了 wheel**，换机要重装或显式覆盖。

## 4. `spec_from_file_location`：为什么不直接 `import deep_ep`

`setup.py:17-20` 有一段看似多余的加载方式：

```python
find_pkgs_spec = importlib.util.spec_from_file_location('find_pkgs', os.path.join(current_dir, 'deep_ep', 'utils', 'find_pkgs.py'))
find_pkgs = importlib.util.module_from_spec(find_pkgs_spec)
find_pkgs_spec.loader.exec_module(find_pkgs)
```

它**刻意绕开** `import deep_ep`。原因：`deep_ep/__init__.py:19-56` 顶部会立即执行
`check_nccl_so()`（读 `/proc/self/maps`、逐字节比对运行时与链接的 `libnccl`）和 `init_jit()`
（`import deep_ep._C`，此刻 C 扩展**还没编译出来**）。在安装/构建阶段这些副作用要么不可能成功、
要么直接断言退出；而 `setup.py` 只需要 `find_nccl_root` 这一个纯 stdlib 函数。H100 实测 G1/G2：
按路径加载 `find_pkgs` 成功，且过程中 `sys.modules` 里**没有** `deep_ep`。G3/G4 给出反证：直接
`import deep_ep` 在本机 `rc=1`、stderr 末行正是那条
`Invalid NCCL versions: /usr/lib/x86_64-linux-gnu/libnccl.so.2.29.7 (loaded) v.s. /usr/local/.../nvidia/nccl/lib/libnccl.so.2 (expected)`
——**这就是路线图「阻塞」的确切落点**：构建期 `find_pkg_root` 解析到 pip 的 cu13 NCCL，
运行期 PyTorch 却加载了系统 NCCL 2.29.7，`check_nccl_so` 逐字节比对不通过。（第 105/112/115 篇
的 DeepEP 设备实验同样受此阻塞，都靠「按路径加载真实 Python 源」绕过。）

## 5. 版本号：git 干净断言、`+hash` / `+local`

两库的 `get_package_version`（DeepEP `setup.py:23-40`、DeepGEMM `:51-70`）都是「公开版本 + git 短哈希」：

```python
public_version = ast.literal_eval(re.search(r'^__version__\s*=\s*(.*)$', txt, re.MULTILINE).group(1))
try:
    status = subprocess.check_output(['git', 'status', '--porcelain']).decode().strip()
    if status:                                         # ← 脏工作区直接断言
        assert False, 'Git working directory is not clean'
    revision = '+' + subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode().rstrip()
except Exception:                                      # ← 非 git / git 不可用
    revision = '+local'
```

**「脏工作区断言」是写出来给发布者看的，但被 broad `except` 吞掉了**：`:32-34`（DeepEP）/`:64`
（DeepGEMM）的 `assert False, 'Git working directory is not clean'` 落在 `try` **内**，外层
`:38` 是裸 `except:`、DeepGEMM `:68` 是 `except Exception:`——`AssertionError` 也是 `Exception` 子类，
于是**脏工作区最终也退成 `+local`，并不报错**。H100 实测 D3/D4：临时干净仓库得 `9.9.9+878c8bf`；
改一行后得 `9.9.9+local`（**不是**断言失败）；D4b 把 broad except 去掉后同一次调用确实抛
`AssertionError`，证明那条断言只是被吞、并非没触发。D2b 从另一角度坐实 `except` 的吞并：
本容器里 `git -C <repo>` 因
**dubious ownership** 拒绝执行，函数**静默**返回 `2.5.0+local` / `2.8.1+local`（本该是 `+93eb6eb` / `+057ca59`）。
D5 展示 `DG_USE_LOCAL_VERSION=False`（`setup.py:57`）时完全不碰 git、只返回纯公开版本；D6 非 git 目录
稳稳 `+local`。

> 含义：**`+local` 不是「本地版本」的可靠标记，而是「git 没成功得出结论」的垃圾桶**——脏工作区、
> 无 git、git 因权限/所有权拒绝，都落到这里。看到 `deep_ep 2.5.0+local`（本机 metadata 实测）
> 只能说明「不是带哈希的正式构建」，不能反推源码干净。

## 6. DeepGEMM 的 wheel 下载方案与 `DG_SKIP_CUDA_BUILD`

DeepGEMM `setup.py:80-96` 的 `get_wheel_url` 把「平台 / torch / CUDA 大版本 / C++11 ABI / Python」
编码进 wheel 文件名，`CachedWheelsCommand:176-200` 会在 `bdist_wheel` 时**先尝试按 URL 下载预编译 wheel**
（`urllib.request.urlopen(wheel_url, timeout=1)`，`:185`），失败才本地源码构建：

```
deep_gemm-{version}+cu{cuda_major}-torch{torch_maj.min}-cxx11abi{0|1}-cp{xy}-linux_{arch}.whl
```

H100 实测 E：本机 `torch=2.12.0a0+... cuda=13.2 cxx11abi=1` → 文件名
`deep_gemm-2.8.1+cu13-torch2.12-cxx11abi1-cp312-linux_x86_64.whl`，URL 前缀为
`https://github.com/DeepSeek-AI/DeepGEMM/releases/download/`（真实官方 release，非杜撰）。
E4 揭示一个边角：`get_package_version` 若带 `+hash`（本地构建），文件名会出现**两个 `+`**
（`deep_gemm-1.2.3+abc+cu13-...`）——所以下载分支实际只在 `DG_USE_LOCAL_VERSION=False`
（`:178` 直接 `return super().run()`）时才有意义。

`get_ext_modules`（`:99-108`）是**本机能纯 Python 安装**的关键：

```python
def get_ext_modules():
    if DG_SKIP_CUDA_BUILD:          # setup.py:22 DG_SKIP_CUDA_BUILD = os.getenv('DG_SKIP_CUDA_BUILD','0')=='1'
        return []
    return [CUDAExtension(name='deep_gemm._C', sources=['csrc/python_api.cpp'], ...)]
```

H100 实测 F：`DG_SKIP_CUDA_BUILD=1` → `ext_modules=[]`。这就是第 0/8 篇里本机安装 DeepGEMM 时
用的 `DG_SKIP_CUDA_BUILD`——它跳过的是 **`deep_gemm._C` 这个 host C++ 扩展**（`csrc/python_api.cpp`
+ `-lcudart`），而**所有真正的 device kernel 都是运行时 JIT 的**（第 33 篇 DeepJIT），所以跳过它
只影响 pybind 入口能否编译、不影响 JIT 出的 kernel。DeepEP 没有这个开关（`setup.py:131-138` 无条件
编译 `deep_ep._C`），因此本机的 DeepEP 装得上、但 `import` 仍卡在 `check_nccl_so`。

## 7. `build_py` 的三个拷贝，以及 editable 安装的已知坑

两库都自定义了 `build_py`。DeepGEMM `CustomBuildPy.run`（`:112-126`）依次做四件事：

1. `prepare_includes`（`:157-173`）：把 `third-party/cutlass/include/{cute,cutlass}`（`:42-45`）**拷进构建目录**
   `deep_gemm/include/`。H100 实测 H1/H2/H3：目录名恰为 `cute`/`cutlass`、源在 `third-party/cutlass/include/` 下、
   目标 `deep_gemm/include/{cute,cutlass}` 存在。
2. `generate_default_envs`（`:148-155`）：写 `envs.py`（§3）。
3. `generate_pyi_file`（`:136-146`）：跑 `scripts/generate_pyi.py`（正则扫 `csrc/**`，`build_cpp_function_index`
   + `BracketTracker` 配平括号取函数签名）生成 `_C.pyi` 类型桩再拷进 wheel。
4. `prepare_agent_files`（`:128-134`）：把 `csrc` **和 `docs`** 一起拷进 wheel，让**装了 wheel 的机器也能在
   报错时按源码行号查错**（agent-side error lookup）。DeepEP `setup.py:54-61` 同样拷 `csrc`，但用
   `shutil.ignore_patterns('cmake-build-*')` 排除 CMake 调试产物（H100 实测 H4 坐实过滤语义）。

**editable 安装的坑（路线图「已知坑」）**：`pip install -e`（或 `setup.py develop`，见 `develop.sh`）
**只跑 `build_ext`、不跑 `build_py`**，于是 `prepare_includes` **从不执行**，`deep_gemm/include/` 下不会出现
`cute`/`cutlass`；而 JIT 编译每个 kernel 都要 `#include <cute/...>`（第 33 篇）。第 0/8 篇给出的解法
就是手动 `cp -r third-party/cutlass/include/{cute,cutlass} deep_gemm/include/`——H3 显示本仓库现在
两个目录都在。反过来说，`build.sh`（`python setup.py bdist_wheel`）与 `develop.sh`（`python setup.py build`
+ 手动把 `build/` 里的 `.so` 软链回 `deep_ep/`）是**两条不同的安装路径**：正式 wheel 走 `build_py`
（含 envs.py / cute / csrc 拷贝），开发模式走 `build_ext`（只出 `.so`，其余靠源码目录本身）。

## 8. `CMakeLists.txt`：只是调试用

DeepEP `CMakeLists.txt:1` 首行注释直说 `# NOTES: this CMake is only for debugging; for setup, please use
Torch extension`，并要求 `NCCL_ROOT_DIR` 必填（`:27-29`，找不到 `FATAL_ERROR`），默认 `CUDA_ARCH_LIST 9.0`
（`:15`）。也就是说 **`CMakeLists.txt` 不参与 `pip install`**，真正的打包/编译入口永远是 `setup.py` 里的
`CUDAExtension`。这对本专题的意义：想「只编某一部分 kernel」不能改 CMakeLists（它会被忽略），
只能改 `setup.py` 的 `sources`（`DeepEP:81-83` / `DeepGEMM:32`）或走 `DG_SKIP_CUDA_BUILD`。

## 9. 与专题的联系 / 结论

- **通信原语的地基**：`find_pkgs.py` 是 DeepEP 与「哪份 NCCL」之间的唯一开关；`EP_NCCL_ROOT_DIR` /
  `NCCL_DIR` 的优先级、`get_nccl_lib_name` 的 SONAME 兜底，共同决定第 86 篇「NCCL 通信域绑定」与
  第 117 篇 `check_nccl_so` 三把尺子里的期望库从哪来。
- **Mega MoE 依赖的安装前提**：DeepGEMM 的 `DG_SKIP_CUDA_BUILD` + JIT（第 33 篇）让「host 扩展」与
> 「设备 kernel」解耦——这也是本机能装 DeepGEMM 却编不了 SM100 kernel（第 8 篇）的原因。
- **阻塞的确切落点**：`import deep_ep` 失败不是设备问题，而是**构建期 `find_pkg_root` 选的 NCCL**
> 与**运行期 PyTorch 加载的 NCCL** 不是同一份（§1、§4）。修复需要链接 NCCL ≥ 2.32.3 的 PyTorch 环境
> （路线图「阻塞」），或在构建时用 `EP_NCCL_ROOT_DIR` 指向一致的一份并保证 PyTorch 也用它。

一句话：**行号、kernel、通信、JIT 都是「运行期」；`find_pkgs.py` + `setup.py` 是「运行前」**。
这两份文件把「装在哪、链到哪、烤进了什么、哪些目录被拷进 wheel」一次定死，后续一切精读都建立在它们之上。

## 10. 实验与证据

- `harness/exp_build_packaging.py` / `.sh` → `harness/exp_build_packaging.out.txt`。
- 纯 host / Python、**架构无关**；按文件路径加载**真实** `find_pkgs.py`，用 AST 抽取两个 `setup.py`
  的常量与逻辑（不执行其构建）。
- H100(sm90) 真机 **54 PASS / 0 FAIL**：Part A `find_pkg_root` 四层语义 14 项（env 覆盖 / 子串匹配 /
  根裁剪 / path 优先级含「同优先级后者胜」）、Part B `get_nccl_lib_name` 5 项（真机 `libnccl.so.2`）、
  Part C 持久化环境变量生成 8 项（DeepEP 7 条 / DeepGEMM 3 条）、Part D 版本号 10 项
  （`2.5.0+93eb6eb` / `2.8.1+057ca59` / 脏工作区被吞成 `+local` / `+local`）、Part E wheel 文件名 4 项
  （`deep_gemm-2.8.1+cu13-torch2.12-cxx11abi1-cp312-linux_x86_64.whl`）、Part F
  `DG_SKIP_CUDA_BUILD` 3 项、Part G `spec_from_file_location` 4 项（含本机 `import deep_ep` 的阻塞复现）、
  Part H 拷贝/过滤 6 项。
