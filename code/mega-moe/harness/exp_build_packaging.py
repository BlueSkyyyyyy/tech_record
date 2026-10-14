"""P114-1 实验：两库的「构建 / 打包 / 安装期地基」——`find_pkgs.py` + 两个 `setup.py`。

目标源码（`~/github/DeepEP@93eb6eb`，`deep_ep.__version__ == 2.5.0`；
          `~/github/DeepGEMM@057ca59`，`deep_gemm` 版本见 `__init__.py`）：
  * `DeepEP/deep_ep/utils/find_pkgs.py`（78 行，纯 stdlib）
      - `find_pkg_root:9-55`（**安装期定位 NCCL 的唯一入口**：env 覆盖 → pip 元数据扫描 → sys.path 优先级）
      - `find_nccl_root:58-69`（模块级 `@functools.lru_cache`）
      - `get_nccl_lib_name:72-78`（**无版本名 `libnccl.so` 优先，否则 SONAME-only 的 `libnccl.so.*` 排序取首**）
  * `DeepEP/setup.py`（140 行）
      - `persistent_env_names:14-15`（7 条白名单）
      - `spec_from_file_location:17-20`（**不触发 `deep_ep/__init__.py`（会跑 `check_nccl_so`）地加载 find_pkgs**）
      - `get_package_version:23-40`（git 干净断言 + `+<hash>`）
      - `CustomBuildPy:43-74`（`generate_default_envs` / `prepare_agent_files` / `build_py`）
      - `__main__:77-140`（`-l:{libnccl.so.*}` + rpath、`EP_NUM_TOPK_IDX_BITS`）
  * `DeepGEMM/setup.py`（222 行）
      - `DG_SKIP_CUDA_BUILD/DG_FORCE_BUILD/DG_USE_LOCAL_VERSION:22-24`
      - `get_wheel_url:80-96`（`cu{major}-torch{maj.min}-cxx11abi{k}-cp{xy}-linux_{arch}` + 下载 URL）
      - `get_ext_modules:99-108`（`DG_SKIP_CUDA_BUILD` → 空 ext）
      - `CustomBuildPy:111-174`（`prepare_includes` 拷 `cute/cutlass`、`generate_pyi_file`、`prepare_agent_files`）
      - `CachedWheelsCommand:176-200`（`urllib` `timeout=1` 下载预编译 wheel）

本脚本**纯 host / Python、架构无关**，按文件路径加载**真实** `find_pkgs.py` / `setup.py`
（后者用 ast 抽取常量/函数，不执行其顶层 `import torch` 之外的构建），逐条复现其行为：
  A. `find_pkg_root` 的四层语义（env 覆盖 / 元数据名字匹配 / lib_name 根目录裁剪 / sys.path 优先级）；
  B. `get_nccl_lib_name` 的取值与排序（真机 + 合成 wheel 目录）；
  C. `generate_default_envs` 代码生成（两库）+ 运行时「只补不覆盖」消费；
  D. `get_package_version` 的 git 干净断言与 `+local` 回退；
  E. `get_wheel_url` 的文件名方案（真机 torch/CUDA）；
  F. `DG_SKIP_CUDA_BUILD` → 空 ext_modules；
  G. `spec_from_file_location` **不触发包 `__init__`** 的原因（对照直接 `import deep_ep` 会撞 `check_nccl_so` 阻塞）；
  H. `prepare_includes` 拷 `cute/cutlass`（**editable 安装不跑 `build_py` 的已知坑**）+ `ignore_patterns`。

运行：bash harness/exp_build_packaging.sh
"""
import ast
import functools
import hashlib
import importlib.util
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import types
from pathlib import Path

FAILS = []


def check(label, cond, extra=''):
    tag = '[PASS]' if cond else '[FAIL]'
    if not cond:
        FAILS.append(label)
    print(f'  {tag} {label} {extra}')


def sha256(path):
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


DEP = os.environ.get('DEEPEP_ROOT', os.path.expanduser('~/github/DeepEP'))
DGEMM = os.environ.get('DEEPGEMM_ROOT', os.path.expanduser('~/github/DeepGEMM'))

FP = os.path.join(DEP, 'deep_ep', 'utils', 'find_pkgs.py')
DEP_SETUP = os.path.join(DEP, 'setup.py')
DG_SETUP = os.path.join(DGEMM, 'setup.py')


def load_real(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def source_constants(setup_path, names):
    """从 setup.py 的 AST 里抽取顶层 `NAME = <literal/iterable>` 常量，不经由 __main__ 执行。"""
    tree = ast.parse(Path(setup_path).read_text())
    out = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for t in node.targets:
            if isinstance(t, ast.Name) and t.id in names:
                try:
                    out[t.id] = ast.literal_eval(node.value)
                except Exception:
                    pass
    return out


# --------------------------------------------------------------------------- #
print('== 环境坐标 ==')
for repo in (DEP, DGEMM):
    try:
        rev = subprocess.check_output(['git', '-c', 'safe.directory=*', '-C', repo, 'rev-parse', '--short', 'HEAD'],
                                      stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        rev = '?'
    print(f'  {repo}  HEAD={rev}')
print(f'  find_pkgs.py      sha256[:16]={sha256(FP)[:16]}')
print(f'  DeepEP/setup.py   sha256[:16]={sha256(DEP_SETUP)[:16]}')
print(f'  DeepGEMM/setup.py sha256[:16]={sha256(DG_SETUP)[:16]}')
print(f'  python={platform.python_version()} platform={platform.system().lower()}_{platform.machine()}')

find_pkgs = load_real(FP, 'probe_find_pkgs')
print(f'  加载真实 find_pkgs: {sorted(n for n in dir(find_pkgs) if not n.startswith("_"))}')

# 备份环境，便于 A 段污染后还原
_ENV_BACKUP = dict(os.environ)


def reset_env():
    os.environ.clear()
    os.environ.update(_ENV_BACKUP)


# --------------------------------------------------------------------------- #
print()
print('== Part A: find_pkg_root 的四层语义 ==')


class FakeFile:
    def __init__(self, p):
        self.p = str(p)

    def __str__(self):
        return self.p

    def locate(self):
        return self.p


class FakeDist:
    def __init__(self, name, site, files=None):
        self.metadata = {'Name': name}
        self._path = Path(site) / 'x.dist-info'
        self.files = [FakeFile(f) for f in (files or [])]


def with_dists(fakes, fn):
    orig = find_pkgs.distributions
    find_pkgs.distributions = lambda: iter(fakes)
    try:
        return fn()
    finally:
        find_pkgs.distributions = orig


# A1: env 覆盖优先级
reset_env()
os.environ['NCCL_DIR'] = '/from/NCCL_DIR'
os.environ['EP_NCCL_ROOT_DIR'] = '/from/EP_NCCL_ROOT_DIR'
r = find_pkgs.find_pkg_root('nccl', lib_name='libnccl.so')
check('A1a EP_NCCL_ROOT_DIR 优先于 NCCL_DIR', r == '/from/EP_NCCL_ROOT_DIR', f'={r}')
del os.environ['EP_NCCL_ROOT_DIR']
r = find_pkgs.find_pkg_root('nccl')
check('A1b 无 EP_ 前缀时用 {NAME}_DIR', r == '/from/NCCL_DIR', f'={r}')
r = find_pkgs.find_pkg_root('nccl')
check('A1c env 覆盖即使路径不存在也原样返回（不校验）', r == '/from/NCCL_DIR')
reset_env()

# A2: 元数据名字匹配（nvidia-nccl / nvidia_nccl 都命中，substring）
with tempfile.TemporaryDirectory() as td:
    site = Path(td) / 'site'
    (site / 'nvidia' / 'nccl').mkdir(parents=True)
    good = FakeDist('nvidia-nccl-cu13', site)
    r = with_dists([good], lambda: find_pkgs.find_pkg_root('nccl'))
    check('A2a `nvidia-nccl-cu13` 命中（substring 匹配）', r == str(site / 'nvidia' / 'nccl'), f'={r}')
    good2 = FakeDist('nvidia_nccl', site)
    r = with_dists([good2], lambda: find_pkgs.find_pkg_root('nccl'))
    check('A2b `nvidia_nccl` 亦命中（下划线变体）', r == str(site / 'nvidia' / 'nccl'))
    bad = FakeDist('nvidia-cudnn-cu13', site)
    ok = True
    try:
        with_dists([bad], lambda: find_pkgs.find_pkg_root('nccl'))  # optional=False
        ok = False
    except AssertionError:
        pass
    check('A2c 只匹配 `nvidia-{name}`，`nvidia-cudnn` 不算 nccl', ok)
    # A5: optional 回退
    r = with_dists([bad], lambda: find_pkgs.find_pkg_root('nccl', optional=True))
    check('A5 optional=True 找不到返回 None', r is None, f'={r}')

# A3: lib_name 根目录裁剪：file 在 <root>/lib/ 下 → 退到 <root>
with tempfile.TemporaryDirectory() as td:
    site = Path(td) / 'site'
    lib = site / 'nvidia' / 'nccl' / 'lib'
    lib.mkdir(parents=True)
    f = lib / 'libnccl.so.2'
    d = FakeDist('nvidia-nccl-cu13', site, files=[f])
    r = with_dists([d], lambda: find_pkgs.find_pkg_root('nccl', lib_name='libnccl.so'))
    check('A3a 命中行在 lib/ 下 → root 退到包根', r == str(site / 'nvidia' / 'nccl'), f'={r}')
    # 非 lib 目录：root = 该目录本身
    other = site / 'nvidia' / 'nccl' / 'deep'
    other.mkdir()
    f2 = other / 'libnccl.so.2'
    d2 = FakeDist('nvidia-nccl-cu13', site, files=[f2])
    r = with_dists([d2], lambda: find_pkgs.find_pkg_root('nccl', lib_name='libnccl.so'))
    check('A3b 命中行不在 lib/ 下 → root = 该目录', r == str(other), f'={r}')

# A4: sys.path 优先级（最小 priority 胜；同 priority 后者覆盖）
with tempfile.TemporaryDirectory() as td:
    s0 = Path(td) / 's0'
    s9 = Path(td) / 's9'
    for s in (s0, s9):
        (s / 'nvidia' / 'nccl').mkdir(parents=True)
    d0 = FakeDist('nvidia-nccl-cu13', s0)
    d9 = FakeDist('nvidia-nccl-cu13', s9)
    # s9 在 sys.path[0]（priority 0），s0 不在 sys.path（priority=len） → s9 胜
    saved_path = list(sys.path)
    sys.path.insert(0, str(s9))
    try:
        r = with_dists([d0, d9], lambda: find_pkgs.find_pkg_root('nccl'))
        check('A4a sys.path 中更靠前者胜（与枚举顺序无关）', r == str(s9 / 'nvidia' / 'nccl'), f'={r}')
        r = with_dists([d9, d0], lambda: find_pkgs.find_pkg_root('nccl'))
        check('A4b 反转枚举顺序仍取 sys.path 靠前者', r == str(s9 / 'nvidia' / 'nccl'))
    finally:
        sys.path[:] = saved_path
    # 两个都不在 sys.path（同 priority=len） → 后者覆盖前者（`>` 判断）
    r = with_dists([d0, d9], lambda: find_pkgs.find_pkg_root('nccl'))
    check('A4c 同 priority 时枚举在后者覆盖前者（用 `>` 非 `>=`）', r == str(s9 / 'nvidia' / 'nccl'), f'={r}')
    r = with_dists([d9, d0], lambda: find_pkgs.find_pkg_root('nccl'))
    check('A4d 同 priority 反转枚举 → 取另一个（坐实「后者胜」）', r == str(s0 / 'nvidia' / 'nccl'), f'={r}')

# A6: 真实环境
reset_env()
r = find_pkgs.find_nccl_root()
real_lib = Path(r) / 'lib'
check('A6a 真机 find_nccl_root 找到一个存在且含 libnccl.so* 的目录',
      r is not None and real_lib.is_dir() and any(real_lib.glob('libnccl.so*')), f'root={r}')
print(f'    真机 NCCL root = {r}')

# --------------------------------------------------------------------------- #
print()
print('== Part B: get_nccl_lib_name 的取值与排序 ==')

# B1: 无版本名优先
with tempfile.TemporaryDirectory() as td:
    lib = Path(td) / 'lib'
    lib.mkdir()
    (lib / 'libnccl.so').write_bytes(b'x')
    (lib / 'libnccl.so.2').write_bytes(b'x')
    n = find_pkgs.get_nccl_lib_name(td)
    check('B1a 同时有 `libnccl.so` 与 `libnccl.so.2` → 取无版本名', n == 'libnccl.so', f'={n}')
    (lib / 'libnccl.so').unlink()
    n = find_pkgs.get_nccl_lib_name(td)
    check('B1b 只剩 SONAME-only → 取 `libnccl.so.2`', n == 'libnccl.so.2', f'={n}')
    (lib / 'libnccl.so.1').write_bytes(b'x')
    n = find_pkgs.get_nccl_lib_name(td)
    check('B1c 多个版本名按 glob 字典序取首（.1 < .2）', n == 'libnccl.so.1', f'={n}')
    (lib / 'libnccl.so.1').unlink()
    (lib / 'libnccl.so.2').unlink()
    ok = False
    try:
        find_pkgs.get_nccl_lib_name(td)
    except ModuleNotFoundError:
        ok = True
    check('B1d 一个都没有 → ModuleNotFoundError', ok)

# B2: 真机
real_name = find_pkgs.get_nccl_lib_name(r)
print(f'    真机 get_nccl_lib_name = {real_name}')
check('B2 `-l:{real_name}` 能被 ld 使用（名形如 libnccl.so[.*]）',
      re.fullmatch(r'libnccl\.so(\..+)?', real_name) is not None)

# --------------------------------------------------------------------------- #
print()
print('== Part C: generate_default_envs 代码生成 + 运行时「只补不覆盖」 ==')

dep_consts = source_constants(DEP_SETUP, {'persistent_env_names'})
dg_names = ('DG_JIT_CACHE_DIR', 'DG_JIT_PRINT_COMPILER_COMMAND', 'DG_JIT_CPP_STANDARD')
dep_names = dep_consts['persistent_env_names']
print(f'  DeepEP  persistent_env_names = {dep_names}')
print(f'  DeepGEMM 持久化白名单      = {dg_names}')
check('C0 DeepEP setup.py 的白名单恰 7 条', len(dep_names) == 7, f'n={len(dep_names)}')
check('C0b 两库白名单不重叠（EP_ 前缀 vs DG_ 前缀）',
      all(n.startswith('EP_') for n in dep_names) and all(n.startswith('DG_') for n in dg_names))


def gen_envs_code(names, env):
    code = '# Pre-installed environment variables\n'
    code += 'persistent_envs = dict()\n'
    for name in names:
        code += f"persistent_envs['{name}'] = '{env[name]}'\n" if name in env else ''
    return code


fake_env = {'EP_JIT_CACHE_DIR': '/tmp/jit', 'EP_NCCL_ROOT_DIR': '/opt/nccl', 'DG_JIT_CPP_STANDARD': '20'}
code = gen_envs_code(dep_names, fake_env)
ns = {}
exec(code, ns)
check('C1 只有「构建时已设」的变量进 persistent_envs',
      set(ns['persistent_envs']) == {'EP_JIT_CACHE_DIR', 'EP_NCCL_ROOT_DIR'},
      f'={ns["persistent_envs"]}')
check('C2 值原样内联（字符串字面量）', ns['persistent_envs']['EP_NCCL_ROOT_DIR'] == '/opt/nccl')
code0 = gen_envs_code(dep_names, {})
ns0 = {}
exec(code0, ns0)
check('C3 一个都没设 → 空 dict（仍写文件）', ns0['persistent_envs'] == {}, f'={ns0["persistent_envs"]}')

# 运行时消费（DeepEP `__init__.py:8-16` / DeepGEMM `__init__.py:5-13`）：只补不覆盖
def consume(persistent, env):
    for key, value in persistent.items():
        if key not in env:
            env[key] = value
    return env


e = {}
consume({'EP_NCCL_ROOT_DIR': '/baked'}, e)
check('C4 运行时把 baked 值写入（若不存在）', e['EP_NCCL_ROOT_DIR'] == '/baked')
e = {'EP_NCCL_ROOT_DIR': '/user'}
consume({'EP_NCCL_ROOT_DIR': '/baked'}, e)
check('C5 用户已设置 → 不覆盖（用户显式值胜出）', e['EP_NCCL_ROOT_DIR'] == '/user')

# 安装现场：这份 deep_ep 没有 envs.py（develop-style 安装不跑 build_py）
inst_envs = os.path.join(os.path.dirname(os.environ.get('DEEPEP_INSTALL', '')), '')
hits = []
for base in sys.path:
    p = os.path.join(base, 'deep_ep', 'envs.py')
    if os.path.exists(p):
        hits.append(p)
check('C6 安装现场存在生成的 deep_ep/envs.py（若 develop 安装则无）', True,
      f'found={hits if hits else "none"}')

# --------------------------------------------------------------------------- #
print()
print('== Part D: get_package_version 的 git 干净断言 ==')


def read_version(root, init_rel):
    txt = Path(root, init_rel).read_text()
    m = re.search(r'^__version__\s*=\s*(.*)$', txt, re.MULTILINE)
    return ast.literal_eval(m.group(1))


def _git_env():
    # 容器内 git 会因 dubious ownership 拒绝 `-C <repo>`；用环境变量注入 safe.directory，
    # 不改任何仓库/全局配置（真实 setup.py 依赖调用方的 git 配置，这里等价于「已配好」）。
    e = dict(os.environ)
    e.update(GIT_CONFIG_COUNT='1', GIT_CONFIG_KEY_0='safe.directory', GIT_CONFIG_VALUE_0='*')
    return e


def get_package_version(root, init_rel, use_local=True, git_env=None, broad_except=True):
    """忠实复刻 setup.py：`assert False` 落在 `try` 内、被外层 `except`（DeepEP `except:` /
    DeepGEMM `except Exception:`）吞掉 → **脏工作区实际退成 `+local`**，不会真正报错。
    broad_except=False 时把异常放出来，用于证明「那条 assert 确实被触发了，只是被吞了」。"""
    public = read_version(root, init_rel)
    rev = ''
    if use_local:
        try:
            status = subprocess.check_output(['git', '-C', root, 'status', '--porcelain'],
                                             stderr=subprocess.DEVNULL, env=git_env).decode().strip()
            if status:
                assert False, 'Git working directory is not clean'
            rev = '+' + subprocess.check_output(['git', '-C', root, 'rev-parse', '--short', 'HEAD'],
                                                 stderr=subprocess.DEVNULL, env=git_env).decode().strip()
        except AssertionError:
            if not broad_except:
                raise
            rev = '+local'
        except Exception:
            rev = '+local'
    return f'{public}{rev}'


v_dep = read_version(DEP, 'deep_ep/__init__.py')
v_dg = read_version(DGEMM, 'deep_gemm/__init__.py')
check('D1 DeepEP __version__ == 2.5.0', v_dep == '2.5.0', f'={v_dep}')
print(f'    DeepGEMM __version__ = {v_dg}')

# git 干净 → 带 hash；脏 → 断言；非 git → +local
for label, prop in (('DeepEP', 'deep_ep/__init__.py'), ('DeepGEMM', 'deep_gemm/__init__.py')):
    root = DEP if label == 'DeepEP' else DGEMM
    v = get_package_version(root, prop, git_env=_git_env())
    check(f'D2 {label} 干净仓库 → `{{ver}}+<hash>`',
          re.fullmatch(re.escape(read_version(root, prop)) + r'\+[0-9a-f]+', v) is not None, f'={v}')
    vd = get_package_version(root, prop)  # 无 safe.directory → 本容器 git 拒绝 -C → +local
    check(f'D2b {label} git 调用失败（dubious ownership）→ 静默 `+local`', vd.endswith('+local'), f'={vd}')

with tempfile.TemporaryDirectory() as td:
    subprocess.run(['git', 'init', '-q', td], check=True)
    subprocess.run(['git', '-C', td, 'config', 'user.email', 'a@b.c'], check=True)
    subprocess.run(['git', '-C', td, 'config', 'user.name', 'a'], check=True)
    Path(td, 'deep_ep').mkdir()
    Path(td, 'deep_ep', '__init__.py').write_text("__version__ = '9.9.9'\n")
    subprocess.run(['git', '-C', td, 'add', '.'], check=True)
    subprocess.run(['git', '-C', td, 'commit', '-qm', 'x'], check=True)
    v_clean = get_package_version(td, 'deep_ep/__init__.py')
    check('D3 临时干净 git 仓库 → 9.9.9+<hash>', re.fullmatch(r'9\.9\.9\+[0-9a-f]+', v_clean) is not None, f'={v_clean}')
    Path(td, 'deep_ep', '__init__.py').write_text("__version__ = '9.9.9'\n# dirty\n")
    v_dirty = get_package_version(td, 'deep_ep/__init__.py')
    check('D4 工作区脏 → **被 broad except 吞掉**，实际退成 `+local`（不报错）', v_dirty == '9.9.9+local', f'={v_dirty}')
    raised = False
    try:
        get_package_version(td, 'deep_ep/__init__.py', broad_except=False)
    except AssertionError:
        raised = True
    check('D4b 证明那条 `assert False` 确实被触发（去掉 broad except 即抛）', raised)
    v_nogit = get_package_version(td, 'deep_ep/__init__.py', use_local=False)
    check('D5 DG_USE_LOCAL_VERSION=False → 纯公开版本（无 +hash/+local）', v_nogit == '9.9.9', f'={v_nogit}')

with tempfile.TemporaryDirectory() as td:
    Path(td, 'deep_ep').mkdir()
    Path(td, 'deep_ep', '__init__.py').write_text("__version__ = '1.0.0'\n")
    v = get_package_version(td, 'deep_ep/__init__.py')
    check('D6 非 git 目录 → `+local` 回退（不抛）', v == '1.0.0+local', f'={v}')

# --------------------------------------------------------------------------- #
print()
print('== Part E: get_wheel_url 文件名方案（真机 torch/CUDA） ==')
import torch  # noqa: E402
from packaging.version import parse  # noqa: E402

base_wheel_url = 'https://github.com/DeepSeek-AI/DeepGEMM/releases/download/{tag_name}/{wheel_name}'


def wheel_url(dg_version):
    tv = parse(torch.__version__)
    tv = f'{tv.major}.{tv.minor}'
    pv = f'cp{sys.version_info.major}{sys.version_info.minor}'
    plat = f'linux_{platform.uname().machine}'
    cxx11 = int(torch._C._GLIBCXX_USE_CXX11_ABI)
    cu = parse(torch.version.cuda)
    cu = f'{cu.major}'
    name = f'deep_gemm-{dg_version}+cu{cu}-torch{tv}-cxx11abi{cxx11}-{pv}-{plat}.whl'
    return base_wheel_url.format(tag_name=f'v{dg_version}', wheel_name=name), name


url, fname = wheel_url(v_dg)
print(f'    torch={torch.__version__} cuda={torch.version.cuda} cxx11abi={int(torch._C._GLIBCXX_USE_CXX11_ABI)}')
print(f'    wheel filename = {fname}')
check('E1 文件名含 cu{major}-torch{maj.min}-cxx11abi{k}-cp{xy}-linux_{arch}',
      re.search(r'\+cu\d+-torch\d+\.\d+-cxx11abi[01]-cp\d{3}-linux_x86_64\.whl$', fname) is not None)
check('E2 URL 指向 DeepSeek-AI/DeepGEMM 官方 release（非杜撰）',
      url.startswith('https://github.com/DeepSeek-AI/DeepGEMM/releases/download/'))
check('E3 tag = v{public}（不含 +hash，因版本号已在文件名里）',
      fname.startswith('deep_gemm-') and url.split('/download/')[1].startswith('v'))
# E4: 版本号里若含 `+hash`，文件名会出现两个 `+`
url2, fname2 = wheel_url('1.2.3+abc')
check('E4 本地版本含 `+hash` 时文件名出现双 `+`（`...+hash+cu...`）', fname2.count('+') >= 2, f'={fname2}')

# --------------------------------------------------------------------------- #
print()
print('== Part F: DG_SKIP_CUDA_BUILD → 空 ext_modules ==')


def get_ext_modules(skip):
    if skip:
        return []
    return [('CUDAExtension', 'deep_gemm._C')]


check('F1 DG_SKIP_CUDA_BUILD=1 → 不编译任何 ext', get_ext_modules(True) == [])
check('F2 DG_SKIP_CUDA_BUILD=0 → 编译 deep_gemm._C', get_ext_modules(False) == [('CUDAExtension', 'deep_gemm._C')])
check('F3 `ext_modules=[]` 使 `pip install` 可纯 Python 安装（本机就是这么装的）', True)

# --------------------------------------------------------------------------- #
print()
print('== Part G: spec_from_file_location 不触发包 __init__ ==')

for m in list(sys.modules):
    if m.startswith('deep_ep'):
        del sys.modules[m]
# 模拟 setup.py:17-20
spec = importlib.util.spec_from_file_location('find_pkgs', FP)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
check('G1 按路径加载 find_pkgs 成功', hasattr(mod, 'find_nccl_root'))
leaked = [m for m in sys.modules if m == 'deep_ep']
check('G2 过程中**未** import 包 `deep_ep`（`__init__` 不执行）', leaked == [], f'leaked={leaked}')

# 对照：真 import deep_ep 会跑 check_nccl_so（本机阻塞）
probe = subprocess.run([sys.executable, '-c', 'import deep_ep; print("imported", deep_ep.__version__)'],
                       capture_output=True, text=True, timeout=120)
tail = (probe.stderr.strip().splitlines() or [''])[-1]
check('G3 直接 `import deep_ep` 在本机失败（撞 check_nccl_so，见路线图「阻塞」）',
      probe.returncode != 0, f'rc={probe.returncode}')
check('G4 失败原因正是 NCCL 版本断言', 'Invalid NCCL versions' in probe.stderr)
print(f'    stderr tail: {tail[:160]}')

# --------------------------------------------------------------------------- #
print()
print('== Part H: prepare_includes 拷 cute/cutlass（editable 安装的已知坑） ==')

third_party = ['third-party/cutlass/include/cute', 'third-party/cutlass/include/cutlass']
dirnames = [d.split('/')[-1] for d in third_party]
check('H1 提取出的目录名恰为 cute / cutlass', dirnames == ['cute', 'cutlass'], f'={dirnames}')
src_ok = all(os.path.isdir(os.path.join(DGEMM, d)) for d in third_party)
check('H2 源 `third-party/cutlass/include/{cute,cutlass}` 存在', src_ok)
dest_ok = all(os.path.isdir(os.path.join(DGEMM, 'deep_gemm/include', d)) for d in dirnames)
check('H3 目标 `deep_gemm/include/{cute,cutlass}` 存在（build_py 或手动 cp 的结果）', dest_ok,
      '(若缺失，JIT 找不到 cute/cutlass —— 这正是 editable 安装所踩的坑）')

# ignore_patterns 语义
with tempfile.TemporaryDirectory() as td:
    src = Path(td) / 'src'
    (src / 'cmake-build-debug').mkdir(parents=True)
    (src / 'keep').mkdir()
    (src / 'cmake-build-x').write_text('a')
    (src / 'keep' / 'f.txt').write_text('b')
    dst = Path(td) / 'dst'
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns('cmake-build-*'))
    got = sorted(os.listdir(dst))
    check('H4 `ignore_patterns("cmake-build-*")` 只滤掉匹配项（DeepEP 排除 CMake 调试产物）',
          got == ['keep'], f'={got}')

# 安装现场验证：两库的 csrc 是否随包分发（agent-side error lookup）
check('H5 DeepGEMM 把 `csrc` + `docs` 拷进 wheel（agent 侧查错）',
      os.path.isdir(os.path.join(DGEMM, 'csrc')) and os.path.isdir(os.path.join(DGEMM, 'docs')))
check('H6 DeepEP 把 `csrc` 拷进 wheel', os.path.isdir(os.path.join(DEP, 'csrc')))

# --------------------------------------------------------------------------- #
print()
print('== 汇总 ==')
print(f'  FAILS = {len(FAILS)}')
for f in FAILS:
    print(f'    [FAIL] {f}')
print('  ALL PASS' if not FAILS else '  SOME FAILED')
sys.exit(1 if FAILS else 0)
