"""P74-1 实验：DeepGEMM 的 Python 绑定面（`csrc/python_api.cpp` + 11 个 `register_apis`）。

目标：把「Python 里能调到的 `deep_gemm.*`」逐项对回 C++ 的 `m.def` / `m.attr` / `submodule.def`，
并核对 4 个非平凡契约点：
  1. `m.attr` 别名——`fp8_gemm_nt`/`fp4_gemm_nt` 与 `fp8_fp4_gemm_nt` 是**同一个 pybind 函数对象**；
  2. `compiled_dims` 默认值随入口主序变（`nt/nn`→`nk`、`tn/tt`→`mn`），且与第 46 篇一致；
  3. `k_grouped_*` / `fp8_einsum` 的 `recipe` 默认值（`(1,1,128)` / `(1,1,32)` / `(1,128,128)`）；
  4. 两个 Python 子模块 `epilogue`（工厂类）与 `locality_domain`，以及 `_C` 里「不对外 re-export」
     的隐藏入口（`init`/`shutdown`/`Runtime`/`get_jit`/`get_symm_buffer_size_for_mega_moe`/...）。

为什么 H100(sm90) 可跑：整层是 pybind11 host 绑定，系统里装的就是 `~/github/DeepGEMM` 的 editable
包，`import deep_gemm` 直接拿到真实 `_C` 扩展；不涉及任何 SM100 kernel。

运行：bash harness/exp_deepgemm_python_api.sh > harness/exp_deepgemm_python_api.out.txt
"""
import os
import re
import glob

import torch
import deep_gemm
from deep_gemm import _C


DG = os.environ.get('DEEPGEMM_ROOT', os.path.expanduser('~/github/DeepGEMM'))
if not os.path.isdir(DG):
    DG = os.path.expanduser('~/github/DeepGEMM')

RESULTS = []


def check(name, ok, detail=''):
    RESULTS.append((name, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ''))


def names(mod):
    return sorted(n for n in dir(mod) if not n.startswith('__'))


# ---------------------------------------------------------------------------
# A. 从 C++ 源码解析绑定面
# ---------------------------------------------------------------------------
def parse_source():
    src_dir = os.path.join(DG, 'csrc', 'apis')
    mdefs, attrs, subdefs, submods = {}, {}, {}, {}
    for f in sorted(glob.glob(os.path.join(src_dir, '*.hpp'))):
        s = open(f).read()
        base = os.path.basename(f)
        for n in re.findall(r'\bm\.def\(\s*"([^"]+)"', s):
            mdefs[n] = base
        for tgt, src in re.findall(r'\bm\.attr\(\s*"([^"]+)"\s*\)\s*=\s*m\.attr\(\s*"([^"]+)"\s*\)', s):
            attrs[tgt] = src
        for n in re.findall(r'submodule\.def\(\s*"([^"]+)"', s):
            subdefs[n] = base
        for n in re.findall(r'def_submodule\(\s*"([^"]+)"', s):
            submods[n] = base
    return mdefs, attrs, subdefs, submods


print("=" * 78)
print("DeepGEMM Python 绑定面 —— H100(sm90) 真机")
print("=" * 78)
print(f"deep_gemm {deep_gemm.__version__}  from {deep_gemm.__file__}")
print(f"torch {torch.__version__}  cuda {torch.version.cuda}  "
      f"cc {torch.cuda.get_device_capability(0)}  {torch.cuda.get_device_name(0)}")
print(f"DeepGEMM 源码 {DG}")
print()

mdefs, attrs, subdefs, submods = parse_source()
print(f"--- A. 源码解析（csrc/apis/*.hpp）---")
print(f"  m.def 条目 {len(mdefs)}、m.attr 别名 {len(attrs)}、"
      f"submodule.def {len(subdefs)}、def_submodule {sorted(submods)}")
for k, v in sorted(mdefs.items()):
    print(f"    def  {k:48s}  {v}")
for k, v in sorted(attrs.items()):
    print(f"    attr {k:48s}  = {v}")
print()

# ---------------------------------------------------------------------------
# B. `_C` 顶层名字与源码对账
# ---------------------------------------------------------------------------
print("--- B. `_C` 顶层名字 vs 源码 ---")
c_top = names(_C)
# deep_jit 的 register_python_api 额外注册
DJIT = {'Runtime', 'get_jit'}
submodule_names = set(submods)            # epilogue, locality_domain（顶层可见）
expected = set(mdefs) | set(attrs) | DJIT | submodule_names
actual = set(c_top)
check("m.def 全部出现在 _C", set(mdefs) <= actual,
      f"缺 {sorted(set(mdefs) - actual)}")
check("m.attr 别名全部出现在 _C", set(attrs) <= actual)
check("def_submodule 顶层可见", submodule_names <= actual, sorted(submodule_names))
check("_C 顶层无多余项", actual == expected, f"多 {sorted(actual - expected)}")
print(f"  _C 顶层共 {len(c_top)} 个：{62} m.def + {len(attrs)} attr + "
      f"{sorted(DJIT)} + {sorted(submodule_names)} = {len(expected)}")
print()

# ---------------------------------------------------------------------------
# C. 别名同一性（pybind m.attr，不是复制）
# ---------------------------------------------------------------------------
print("--- C. 别名 `m.attr` 是同一函数对象 ---")
for tgt, src in sorted(attrs.items()):
    check(f"{tgt} is {src}", getattr(_C, tgt) is getattr(_C, src))
print()

# ---------------------------------------------------------------------------
# D. docstring 签名解析：compiled_dims / recipe 默认值
# ---------------------------------------------------------------------------
def sig_defaults(fn_name, arg):
    fn = getattr(_C, fn_name, None)
    if fn is None:
        return None
    doc = fn.__doc__ or ''
    m = re.search(rf'{re.escape(arg)}: [^,)]*? = ([^,)]+)', doc)
    return m.group(1).strip() if m else None


print("--- D. `compiled_dims` 默认值（随主序变）---")
cd_expect = {
    'fp8_fp4_gemm_nt': 'nk', 'fp8_fp4_gemm_nn': 'nk',
    'fp8_fp4_gemm_tn': 'mn', 'fp8_fp4_gemm_tt': 'mn',
    'm_grouped_fp8_fp4_gemm_nt_contiguous': 'nk',
    'm_grouped_fp8_fp4_gemm_nn_contiguous': 'nk',
    'm_grouped_fp8_fp4_gemm_nt_masked': 'nk',
    'k_grouped_fp8_gemm_tn_contiguous': 'mn',
    'k_grouped_fp8_gemm_nt_contiguous': 'mn',
    'k_grouped_fp4_gemm_nt_contiguous': 'mn',
    'bf16_gemm_nt': 'nk', 'bf16_gemm_nn': 'nk',
    'bf16_gemm_tn': 'mn', 'bf16_gemm_tt': 'mn',
    'm_grouped_bf16_gemm_nt_contiguous': 'nk',
    'm_grouped_bf16_gemm_nn_contiguous': 'nk',
    'm_grouped_bf16_gemm_nt_masked': 'nk',
    'k_grouped_bf16_gemm_tn_contiguous': 'mn',
}
for fn, exp in cd_expect.items():
    got = sig_defaults(fn, 'compiled_dims')
    got = got.strip("'\"") if got else got
    check(f"{fn}.compiled_dims = {exp}", got == exp, f"got={got}")

print()
print("--- D2. `recipe` 默认值 ---")
recipe_expect = {
    'k_grouped_fp8_gemm_tn_contiguous': '(1, 1, 128)',
    'k_grouped_fp8_gemm_nt_contiguous': '(1, 1, 128)',
    'k_grouped_fp4_gemm_nt_contiguous': '(1, 1, 32)',
    'fp8_einsum': '(1, 128, 128)',
}
for fn, exp in recipe_expect.items():
    d = getattr(_C, fn).__doc__ or ''
    m = re.search(r'recipe:.*?= (\([^)]*\))', d)
    got = m.group(1) if m else None
    check(f"{fn}.recipe = {exp}", got == exp, f"got={got}")

print()
print("--- D3. 无 py::arg 的入口 → 位置参数名为 arg0.. ---")
for fn in ['get_block_m_for_mega_moe', 'get_tma_aligned_size',
           'get_mn_major_tma_aligned_tensor']:
    d = getattr(_C, fn).__doc__ or ''
    args = re.findall(rf'{re.escape(fn)}\(([^)]*)\)', d)
    check(f"{fn} 用 argN 占位", bool(args) and 'arg0' in args[0],
          (args[0] if args else '(no sig)'))
print()

# ---------------------------------------------------------------------------
# E. 两个 Python 子模块
# ---------------------------------------------------------------------------
print("--- E. 子模块 ---")
epi = names(_C.epilogue)
check("epilogue 基类 + 4 算子",
      set(epi) == {'Epilogue', 'Identity', 'Alpha', 'FP8Quantization', 'BF16StochasticRounding'},
      str(epi))
check("locality_domain 9 个", len(names(_C.locality_domain)) == 9,
      f"{len(subdefs)} submodule.def 对账: {set(subdefs) <= set(names(_C.locality_domain))}")
# 工厂实例化
try:
    a = _C.epilogue.Alpha(0.5)
    q = _C.epilogue.FP8Quantization(torch.zeros(4, dtype=torch.float32, device='cuda'))
    i = _C.epilogue.Identity()
    s = _C.epilogue.BF16StochasticRounding()
    check("epilogue 四类可实例化", all(x is not None for x in (a, q, i, s)))
except Exception as e:  # noqa
    check("epilogue 四类可实例化", False, repr(e))
try:
    _C.epilogue.Alpha()
    check("Alpha 无参被拒", False, "(未抛异常)")
except TypeError:
    check("Alpha 无参被拒", True, "(TypeError)")
check("Alpha 是 Epilogue 子类", isinstance(_C.epilogue.Alpha(1.0), _C.epilogue.Epilogue))
print()

# ---------------------------------------------------------------------------
# F. 公开面：`__init__` re-export 了什么、隐藏了什么
# ---------------------------------------------------------------------------
print("--- F. `deep_gemm` 公开面 vs `_C` ---")
pub = set(names(deep_gemm))
c_only = set(c_top) - pub
print(f"  在 _C 但未在 deep_gemm 顶层 re-export：{sorted(c_only)}")
check("隐藏入口符合预期",
      c_only == {'Runtime', 'get_jit', 'init', 'get_symm_buffer_size_for_mega_moe',
                 'get_token_alignment_for_mega_moe'},
      str(sorted(c_only)))

# Python 侧包装（来自 .mega / .utils，非 _C）
mega_names = {'SymmBuffer', 'get_symm_buffer_for_mega_moe',
              'transform_weights_for_mega_moe', 'fp8_fp4_mega_moe', 'bf16_mega_moe'}
check("Mega MoE 公开入口来自 .mega（Python 包装）",
      mega_names <= pub and mega_names.isdisjoint(set(c_top) - {'fp8_fp4_mega_moe', 'bf16_mega_moe'}),
      str(sorted(mega_names)))
check("get_block_m_for_mega_moe 是 _C 直通（辅助），非 .mega",
      'get_block_m_for_mega_moe' in set(c_top) and 'get_block_m_for_mega_moe' not in
      {n for n in pub if n in mega_names})
# mega moe 包装函数无 pybind doc（Python def），_C 版才有
check("_C.fp8_fp4_mega_moe 有 pybind 签名", 'arg0' in (getattr(_C, 'fp8_fp4_mega_moe').__doc__ or '')
      or 'l1_weights' in (getattr(_C, 'fp8_fp4_mega_moe').__doc__ or ''),
      (getattr(_C, 'fp8_fp4_mega_moe').__doc__ or '').strip().splitlines()[0][:70])
print()

# ---------------------------------------------------------------------------
# G. 端到端冒烟：epilogue 对象真的被 GEMM 接受
# ---------------------------------------------------------------------------
print("--- G. 冒烟：真实调用（arch 无关）---")
torch.manual_seed(0)


def per_token_cast(t):
    amax = t.abs().amax(dim=-1, keepdim=True).clamp_min(1e-4)
    sf = torch.pow(2.0, torch.ceil(torch.log2(amax / 448.0)))
    q = (t / sf).to(torch.float8_e4m3fn)
    return q, sf


M, N, K = 256, 256, 512
a = (torch.randn(M, K, device='cuda') * 0.1).to(torch.bfloat16)
b = (torch.randn(N, K, device='cuda') * 0.1).to(torch.bfloat16)
d = torch.empty(M, N, device='cuda', dtype=torch.bfloat16)
_C.bf16_gemm_nt(a, b, d)
ref = (a.float() @ b.float().T).to(torch.bfloat16)
rel = ((d.float() - ref.float()).norm() / ref.float().norm()).item()
check("bf16_gemm_nt 真实发射 + 数值合理", rel < 2e-2, f"relerr={rel:.3e}")

# alpha 简写 == epilogue.Alpha（同一 epilogue 对象）；数值上 d2 应约等于 2·(A@B^T)
d2 = torch.empty(M, N, device='cuda', dtype=torch.bfloat16)
_C.bf16_gemm_nt(a, b, d2, alpha=2.0)
ref2 = (2.0 * (a.float() @ b.float().T)).to(torch.bfloat16)
rel2 = ((d2.float() - ref2.float()).norm() / ref2.float().norm()).item()
check("alpha=2.0 简写 ≈ 输出乘 2", rel2 < 1e-2, f"relerr={rel2:.3e}")

d3 = torch.empty(M, N, device='cuda', dtype=torch.bfloat16)
_C.bf16_gemm_nt(a, b, d3, epilogue=_C.epilogue.Alpha(2.0))
check("epilogue.Alpha(2.0) 与 alpha=2.0 逐位一致", torch.equal(d2, d3))
print()

# ---------------------------------------------------------------------------
# 汇总
# ---------------------------------------------------------------------------
print("=" * 78)
passes = sum(1 for _, ok in RESULTS if ok)
fails = [n for n, ok in RESULTS if not ok]
print(f"TOTAL {passes}/{len(RESULTS)} PASS   FAIL={fails}")
print("=" * 78)
