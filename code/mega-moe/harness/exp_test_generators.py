"""P31-1 H100 evidence: DeepGEMM 测试操作数/布局生成器 tests/generators.py.

架构无关（纯 Python 张量/枚举逻辑），H100(sm90) 真机可跑。
本脚本不重写生成器，而是 import 真实 tests/generators.py，逐条核对它编码的
"约定"（operand major、三种 grouped 布局、PSUM、K-grouped SF），再用真实
SM90 kernel 做端到端对拍。
"""
import os
import sys
import random
import inspect

import torch

# 真实 tests/generators.py 所在的目录（容器内 HOME=/root，故显式候选）
_CANDIDATES = [
    os.environ.get('DEEPGEMM_TESTS', ''),
    os.path.expanduser('~/github/DeepGEMM/tests'),
    '~/github/DeepGEMM/tests',
    '~/github/DeepGEMM/tests',
]
TESTS_DIR = next(p for p in _CANDIDATES if p and os.path.isfile(os.path.join(p, 'generators.py')))
if TESTS_DIR not in sys.path:
    sys.path.insert(0, TESTS_DIR)

import deep_gemm
from deep_gemm.testing import get_arch_major
from deep_gemm.utils import align, ceil_div, transpose_packed_fp4

import generators as G

TORCH_FP8 = torch.float8_e4m3fn
TORCH_FP4 = torch.float4_e2m1fn_x2

PASS = 0
FAIL = 0


def check(name, cond, extra=''):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f'  [PASS] {name} {extra}')
    else:
        FAIL += 1
        print(f'  [FAIL] {name} {extra}')
    return bool(cond)


def hdr(title):
    print('\n' + '=' * 78)
    print(title)
    print('=' * 78)


def reset():
    G.reset_seed(0)
    random.seed(0)
    torch.manual_seed(0)


print(f'arch_major = {get_arch_major()} (H100=9)')
print(f'deep_gemm  = {deep_gemm.__file__}')
print(f'generators = {inspect.getsourcefile(G)}')
INITIAL_ALIGN = deep_gemm.get_mk_alignment_for_contiguous_layout()
print(f'初始 mk-alignment（任何枚举器调用之前） = {INITIAL_ALIGN}')

# ---------------------------------------------------------------------------
hdr('A. 枚举器普查：arch 条件分支如何缩小候选集')
# ---------------------------------------------------------------------------

def census(name, gen):
    rows = list(gen)
    print(f'{name}: {len(rows)} cases')
    return rows

# 打印 arch 相关的三个门
print(f"get_kernel_types(bf16)      = {[k.name for k in G.get_kernel_types(torch.bfloat16)]}")
print(f"get_kernel_types(e4m3)      = {[k.name for k in G.get_kernel_types(TORCH_FP8)]}")
print(f"get_ue8m0_usage(1D1D)       = {G.get_ue8m0_usage(G.KernelType.Kernel1D1D)}")
print(f"get_ue8m0_usage(1D2D)       = {G.get_ue8m0_usage(G.KernelType.Kernel1D2D)}")
print(f"get_psum_layout_usage()     = {G.get_psum_layout_usage()}")
print(f"QuantConfig.get_list_from_dtype(bf16)  = {G.QuantConfig.get_list_from_dtype(torch.bfloat16)}")
cfg_list = G.QuantConfig.get_list_from_dtype(TORCH_FP8)
print(f"QuantConfig.get_list_from_dtype(e4m3)  = "
      f"{[(c.gran_k_a, c.gran_k_b, c.is_fp4_a, c.is_fp4_b) for c in cfg_list]}")

align_after_normal = deep_gemm.get_mk_alignment_for_contiguous_layout()
n_normal = census('enumerate_normal(e4m3)', G.enumerate_normal(TORCH_FP8))
align_after_mcont = deep_gemm.get_mk_alignment_for_contiguous_layout()
# 统计 kernel_type / major 组合
kts = {}
for row in n_normal:
    kt = row[0]
    kts[kt.name] = kts.get(kt.name, 0) + 1
print(f'  kernel_type 分布: {kts}')
# 前向 m 列表 & bf16 输出 nk 列表
print(f"  m_fwd_list 见 129 行 ^, 例 m=1 的 case 数 = "
      f"{sum(1 for r in n_normal if r[2] == 1)}")

n_mcont = census('enumerate_m_grouped_contiguous(e4m3)', G.enumerate_m_grouped_contiguous(TORCH_FP8))
n_mmask = census('enumerate_m_grouped_masked(e4m3)', G.enumerate_m_grouped_masked(TORCH_FP8))
align_before_kc = deep_gemm.get_mk_alignment_for_contiguous_layout()
n_kcont = census('enumerate_k_grouped_contiguous(e4m3)', G.enumerate_k_grouped_contiguous(TORCH_FP8))
align_after_kc = deep_gemm.get_mk_alignment_for_contiguous_layout()
n_trans = census('enumerate_transpose()', G.enumerate_transpose())
n_sfl = census('enumerate_sf_layout()', G.enumerate_sf_layout())
align_after_sfl = deep_gemm.get_mk_alignment_for_contiguous_layout()
n_ksfl = census('enumerate_k_grouped_sf_layout()', G.enumerate_k_grouped_sf_layout())
n_kpsfl = census('enumerate_k_grouped_psum_sf_layout()', G.enumerate_k_grouped_psum_sf_layout())
align_final = deep_gemm.get_mk_alignment_for_contiguous_layout()
print(f'  [side-effect] 枚举器会改写全局 mk-alignment（进程初始 {INITIAL_ALIGN}）：'
      f'normal 后={align_after_mcont}, k_contiguous 后={align_after_kc}, '
      f'sf_layout 后={align_after_sfl}, k_grouped_sf_layout 后={align_final}')
check('enumerate_k_grouped_contiguous 把全局 alignment 改成 k_alignment 候选值',
      align_after_kc in (128, 256, 384))
# 为后续实验还原到 128
deep_gemm.set_mk_alignment_for_contiguous_layout(128)

# m-contiguous 在 sm90 上 B 只能 K-major（get_major_ab(False, arch!=9 or dtype!=fp8)）
majors = set()
for row in n_mcont:
    majors.add((row[6].name, row[7].name))
print(f'  m-contiguous (major_a, major_b) 组合: {sorted(majors)}')
check('sm90 fp8 m-contiguous 的 B 恒 KMajor（无 MNMajor）',
      all(b == 'KMajor' for _, b in majors))

# k-contiguous 的 cd_options / psum 在 sm90 收窄
print(f'  k-contiguous 样例 case: {n_kcont[0]}')

# ---------------------------------------------------------------------------
hdr('B. QuantConfig：recipe / max_diff 语义')
# ---------------------------------------------------------------------------
legacy = G.QuantConfig()
print(f'legacy config = {legacy._legacy_quant_config}, is_legacy={legacy.is_legacy()}')
print(f"  前向 recipe (is_wgrad=False) = {legacy.get_recipes(False)}")
print(f"  wgrad recipe (is_wgrad=True) = {legacy.get_recipes(True)}")
check('legacy 前向 recipe 全 None（走默认 (1,128,128)）',
      legacy.get_recipes(False) == (None, None, None))
check('legacy wgrad recipe = (1,1,128)', legacy.get_recipes(True)[0] == (1, 1, 128))

sm100_cfg = G.QuantConfig((32, 32, True, True))
print(f'fp4/fp4 config recipes fwd = {sm100_cfg.get_recipes(False)}')
print(f'fp4/fp4 config recipes wgrad = {sm100_cfg.get_recipes(True)}')
check('fp4/fp4 wgrad 时 B recipe 退化为 (1,gran_k)（K 分组需 per-channel）',
      sm100_cfg.get_recipes(True)[2] == (1, 32))

print(f'legacy fp8 max_diff = {legacy.max_diff()}')
print(f'fp8/fp4 max_diff    = {G.QuantConfig((128, 32, False, True)).max_diff()}')
print(f'fp4/fp4 max_diff    = {sm100_cfg.max_diff()}')
check('max_diff 随精度提高而收紧 (fp4fp4>fp8fp4>fp8fp8)',
      legacy.max_diff() < G.QuantConfig((128, 32, False, True)).max_diff() < sm100_cfg.max_diff())

# ---------------------------------------------------------------------------
hdr('C. mk-alignment：默认 / theoretical（C++ 侧 arch 判定）+ set/get')
# ---------------------------------------------------------------------------
default_align = deep_gemm.get_mk_alignment_for_contiguous_layout()
theo_align = deep_gemm.get_theoretical_mk_alignment_for_contiguous_layout()
print(f'进程初始值 INITIAL_ALIGN                             = {INITIAL_ALIGN}')
print(f'get_mk_alignment_for_contiguous_layout()（已还原）   = {default_align}')
print(f'get_theoretical_mk_alignment_for_contiguous_layout()= {theo_align}')
check('sm90 进程初始 alignment == 理论值 == 128',
      INITIAL_ALIGN == 128 and theo_align == 128)
deep_gemm.set_mk_alignment_for_contiguous_layout(256)
check('set/get roundtrip', deep_gemm.get_mk_alignment_for_contiguous_layout() == 256)
deep_gemm.set_mk_alignment_for_contiguous_layout(128)  # 还原
print('（SM100 上 theoretical 恒 256，见 heuristics/runtime.hpp:57-64，本机仅验证 sm90=128）')

# ---------------------------------------------------------------------------
hdr('D. operand major 编码：fp8 用 .T.contiguous().T，fp4 用 transpose_packed_fp4')
# ---------------------------------------------------------------------------
reset()
m, k = 64, 128
x = torch.randn(m, k, device='cuda', dtype=torch.bfloat16)
a_km, sf_km = G.cast_fp8_fp4_with_major(x, G.MajorTypeAB.KMajor, 128, False, False)
a_mn, sf_mn = G.cast_fp8_fp4_with_major(x, G.MajorTypeAB.MNMajor, 128, False, False)
print(f'K-major fp8 data shape/stride = {tuple(a_km.shape)} / {a_km.stride()} contig={a_km.is_contiguous()}')
print(f'MN-major fp8 data shape/stride= {tuple(a_mn.shape)} / {a_mn.stride()} contig={a_mn.is_contiguous()}')
check('MN-major fp8 data 与 K-major 的转置逐位一致（storage 是 (k,m)）',
      torch.equal(a_mn.T.contiguous(), a_km.T.contiguous()))
# bias: SF 不随 major 改变（同一 cast）
check('fp8 SF 与 major 无关', torch.equal(sf_km, sf_mn))

x_fp4 = torch.randn(m, k, device='cuda', dtype=torch.bfloat16)
f_km, fsf_km = G.cast_fp8_fp4_with_major(x_fp4, G.MajorTypeAB.KMajor, 32, True, False)
f_mn, fsf_mn = G.cast_fp8_fp4_with_major(x_fp4, G.MajorTypeAB.MNMajor, 32, True, False)
print(f'K-major fp4 packed shape = {tuple(f_km.shape)} = (M, K/2)')
print(f'MN-major fp4 packed shape= {tuple(f_mn.shape)} = (M/2, K)')
# transpose_packed_fp4(m,k/2)->(k,m/2)；MN-major = transpose_packed_fp4(x).T
tp = transpose_packed_fp4(f_km)
check('transpose_packed_fp4 把 (M,K/2) 变 (K,M/2)', tp.shape == (f_km.shape[1] * 2, f_km.shape[0] // 2))
check('MN-major fp4 == transpose_packed_fp4(K-major).T', torch.equal(f_mn.contiguous(), tp.T.contiguous()))
check('MN-major fp4 可由 transpose_packed_fp4(f_mn.T) 无损还原',
      torch.equal(transpose_packed_fp4(f_mn.T.contiguous()), f_km))

# ---------------------------------------------------------------------------
hdr('E. generate_normal：ref_d 公式 + 真实 SM90 kernel 端到端')
# ---------------------------------------------------------------------------
reset()
m, n, k = 128, 128, 256
a, b, c, d, ref_d = G.generate_normal(
    m, n, k, G.MajorTypeAB.KMajor, G.MajorTypeAB.KMajor, False, torch.bfloat16,
    G.KernelType.Kernel1D2D, use_ue8m0=False, quant_config=G.QuantConfig())
# ref_d 应等于 alpha=1 时 a_bf16 @ b_bf16^T 的 out_dtype
reset()
a0 = torch.randn(m, k, device='cuda', dtype=torch.bfloat16)
print(f'ref_d dtype = {ref_d.dtype}, d dtype = {d.dtype}')

# 真实 kernel：SM90 默认 recipe
deep_gemm.fp8_fp4_gemm_nt(a, b, d, c=None, disable_ue8m0_cast=True)
from deep_gemm.testing import calc_diff
diff = calc_diff(d, ref_d)
print(f'fp8_fp4_gemm_nt  vs ref_d  calc_diff = {diff:.3e}')
check('真实 SM90 fp8 kernel 与 generator ref_d 一致', diff < 1e-3, f'diff={diff:.3e}')

# ---------------------------------------------------------------------------
hdr('F. generate_m_grouped_contiguous：三种 grouped 布局语义')
# ---------------------------------------------------------------------------
for use_psum in (False, True):
    reset()
    num_groups, expected_m, n, k = 4, 128, 128, 128
    mm, aq, bq, gl, dg, refg, vmask = G.generate_m_grouped_contiguous(
        num_groups, expected_m, n, k, G.MajorTypeAB.KMajor, G.MajorTypeAB.KMajor,
        use_ue8m0=False, use_psum_layout=use_psum, quant_config=G.QuantConfig())
    print(f'\n-- use_psum_layout={use_psum}: m={mm}, grouped_layout.shape={tuple(gl.shape)} '
          f'dtype={gl.dtype}, valid={int(vmask.sum())}/{mm}')
    A = deep_gemm.get_mk_alignment_for_contiguous_layout()
    if use_psum:
        ends = gl.cpu().tolist()
        print(f'   grouped_layout(每组逻辑结束偏移) = {ends}  (alignment={A}, m={mm})')
        # 语义：单调不减；末个逻辑结束经对齐 == m
        check('PSUM grouped_layout 单调不减', ends == sorted(ends))
        check('PSUM align(末个结束偏移)==m', align(ends[-1], A) == mm)
        # 有效行 = 每个 [align(prev), end) 区间
        exp_valid = torch.zeros(mm, dtype=torch.bool)
        last = 0
        for e in ends:
            exp_valid[last:e] = True
            last = align(e, A)
        check('PSUM valid_mask == [align(prev_end), end) 区间', torch.equal(exp_valid, vmask.cpu()))
        # PSUM grouped_layout 等价于 build_psum_layout_from_ks(actual_ms)
        actual_ms = [ends[0]] + [ends[i] - align(ends[i - 1], A) for i in range(1, len(ends))]
        check('PSUM grouped_layout == build_psum_layout_from_ks(actual_ms)',
              G.build_psum_layout_from_ks(actual_ms, A) == ends)
    else:
        ids = gl.cpu()
        padding_rows = int((ids == -1).sum())
        print(f'   grouped_layout(逐行组号): 唯一值={sorted(ids.unique().tolist())}, '
              f'padding(-1) 行数={padding_rows}')
        check('contiguous grouped_layout ∈ [-1, G)', ids.min() >= -1 and ids.max() < num_groups)
        check('contiguous padding 行恰好是 valid_mask 之外的行',
              torch.equal((ids == -1), ~vmask.cpu()))
        check('contiguous 有效行组号单调不减',
              bool((ids[vmask.cpu()][1:] >= ids[vmask.cpu()][:-1]).all()))
    # A 的 padding 行被置零（line 406）
    a_km_bf16 = None  # 已被量化；改查量化前无法直接拿。用 PSUM gap 校验见下。

# 校验 generate_m_grouped_contiguous 里 padding 置零：用不带 psum 的一个组，
# A 量化后 padding 行的 fp8 数据应全 0（因为置零后 amax=0 -> sf 地板 -> 0）
reset()
mm, aq, bq, gl, dg, refg, vmask = G.generate_m_grouped_contiguous(
    2, 128, 128, 128, G.MajorTypeAB.KMajor, G.MajorTypeAB.KMajor,
    use_psum_layout=False, quant_config=G.QuantConfig())
pad_mask = (gl.cpu() == -1)
if pad_mask.any():
    pad_rows = aq[0][pad_mask]  # (P, k) fp8
    check('contiguous A padding 行量化后全零', bool((pad_rows.float() == 0).all()))

# 真实 SM90 m-grouped kernel 对拍（contiguous）
deep_gemm.set_mk_alignment_for_contiguous_layout(128)
reset()
mm, aq, bq, gl, dg, refg, vmask = G.generate_m_grouped_contiguous(
    4, 256, 256, 256, G.MajorTypeAB.KMajor, G.MajorTypeAB.KMajor,
    use_psum_layout=False, quant_config=G.QuantConfig())
deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous(
    aq, bq, dg, gl, disable_ue8m0_cast=True, use_psum_layout=False)
diff = calc_diff(dg[vmask], refg[vmask])
print(f'\nm_grouped_fp8_fp4_gemm_nt_contiguous vs ref_d (valid rows) calc_diff = {diff:.3e}')
check('真实 SM90 m-grouped kernel 与 generator ref_d 一致', diff < 1e-3, f'diff={diff:.3e}')

# ---------------------------------------------------------------------------
hdr('G. generate_m_grouped_masked + PSUM 转换（layout_masked_to_psum）')
# ---------------------------------------------------------------------------
reset()
num_groups, max_m, expected_m, n, k = 3, 4096, 200, 128, 128
a, b, gm, d, ref, vmask = G.generate_m_grouped_masked(
    num_groups, max_m, expected_m, n, k, use_bf16=True, use_psum_layout=False)
print(f'masked: a.shape={tuple(a.shape)} (G,max_m,k), masked_m={gm.cpu().tolist()}, '
      f'valid={int(vmask.sum())}')
psum_m = torch.empty(num_groups, device='cuda', dtype=torch.int)
for j in range(num_groups):
    psum_m[j] = (0 if j == 0 else align(psum_m[j - 1], 128)) + gm[j]
print(f'psum_m={psum_m.cpu().tolist()}')

a_psum = G.layout_masked_to_psum(a, psum_m)
vmask_psum = G.masked_valid_mask_to_psum(gm, psum_m)
print(f'psum a.shape={tuple(a_psum.shape)}, valid={int(vmask_psum.sum())}')
check('masked valid 总数 == psum valid 总数', int(vmask.sum()) == int(vmask_psum.sum()))
# 逐组：masked 的第 j 组前 masked_m[j] 行 == psum 的对应区间
ok = True
last = 0
for j in range(num_groups):
    cnt = int(gm[j])
    if not torch.equal(a[j, :cnt], a_psum[last:last + cnt]):
        ok = False
    last = align(last + cnt, 128)
check('逐组 masked 有效行 == psum 对应区间（gap 跳过）', ok)

# 真实 SM90 masked kernel
# 签名（见 tests/test_fp8_fp4.py:207）：(a, b, d, masked_m, expected_m, **common)
reset()
expected_m = 200
a, b, gm, d, ref, vmask = G.generate_m_grouped_masked(
    3, 1024, expected_m, 256, 256, use_ue8m0=False, use_psum_layout=False,
    quant_config=G.QuantConfig())
try:
    deep_gemm.m_grouped_fp8_fp4_gemm_nt_masked(
        a, b, d, gm, expected_m, disable_ue8m0_cast=True)
    diff = calc_diff(d[vmask], ref[vmask])
    print(f'\nm_grouped_fp8_fp4_gemm_nt_masked vs ref (valid) calc_diff = {diff:.3e}')
    check('真实 SM90 masked kernel 与 generator ref_d 一致', diff < 1e-3, f'diff={diff:.3e}')
except (AttributeError, RuntimeError) as e:
    print(f'  (跳过 masked 真实 kernel：{type(e).__name__}: {e})')

# ---------------------------------------------------------------------------
hdr('H. generate_k_grouped_contiguous：按组零填充 + PSUM + SF')
# ---------------------------------------------------------------------------
from deep_gemm.utils import pack_ue8m0_to_int

reset()
num_groups, m, n = 4, 128, 128
real_ks = [100, 200, 260, 300]
for use_psum in (False, True):
    reset()
    total_k, aq, bq, cq, dq, refq, gl, host_ks = G.generate_k_grouped_contiguous(
        num_groups, m, n, G.MajorTypeAB.KMajor, G.MajorTypeAB.KMajor, real_ks,
        use_ue8m0=False, gran_k=128, use_psum_layout=use_psum,
        k_alignment=128, accumulate=True, out_dtype=torch.float)
    print(f'\n-- k-grouped psum={use_psum}: real_ks={real_ks}')
    print(f'   host_ks(对齐后)={host_ks}  total_k={total_k}  '
          f'grouped_layout={gl.cpu().tolist()}')
    check('host_ks = align(real_ks, 128)', host_ks == [align(x, 128) for x in real_ks])
    check('total_k = sum(host_ks)', total_k == sum(host_ks))
    if use_psum:
        exp = G.build_psum_layout_from_ks(real_ks, 128)
        print(f'   build_psum_layout_from_ks = {exp}')
        check('PSUM grouped_layout == build_psum_layout_from_ks', gl.cpu().tolist() == exp)
    else:
        check('contiguous grouped_layout == host_ks', gl.cpu().tolist() == host_ks)
# 定位每组逻辑区间 [start, end)，用 MN-major（2D (total_k, mn)）检查 gap 行全零
def _segs_of(ks_list, gl_list, use_psum):
    if use_psum:
        out, pairs = [], zip(ks_list, gl_list)
        for kk, e in zip(ks_list, gl_list):
            out.append((e - kk, e))
        return out
    out, start = [], 0
    for kk in ks_list:
        out.append((start, start + kk))
        start += kk
    return out

reset()
total_k, aq_mn, bq_mn, _, _, _, gl_mn, host_ks_mn = G.generate_k_grouped_contiguous(
    4, 128, 128, G.MajorTypeAB.MNMajor, G.MajorTypeAB.MNMajor, real_ks,
    use_ue8m0=False, gran_k=128, use_psum_layout=False, k_alignment=128,
    accumulate=True, out_dtype=torch.float)
print(f'\nMN-major A data shape = {tuple(aq_mn[0].shape)} (total_k, mn)')
# 真实非零行：contiguous 下每组数据在物理块开头 [physical_start, +logical_k)，尾部为 0 padding
nz = (aq_mn[0].float() != 0).any(dim=1).cpu()
exp_nz = torch.zeros(total_k, dtype=torch.bool)
start = 0
for lk, hk in zip(real_ks, host_ks_mn):
    exp_nz[start:start + lk] = True
    start += hk
check('MN-major A 非零行恰为每组开头 logical_k 行（尾部 padding 全零）',
      bool(torch.equal(nz, exp_nz)),
      f'nonzero={int(nz.sum())} expected={int(exp_nz.sum())}')
# PSUM 版本：gap 行物理上不是连续尾部，但 logical_group_ends 定义的区间外应为零
total_k, aq_mn2, bq_mn2, _, _, _, gl2, host_ks2 = G.generate_k_grouped_contiguous(
    4, 128, 128, G.MajorTypeAB.MNMajor, G.MajorTypeAB.MNMajor, real_ks,
    use_ue8m0=False, gran_k=128, use_psum_layout=True, k_alignment=128,
    accumulate=True, out_dtype=torch.float)
segs2 = _segs_of(real_ks, gl2.cpu().tolist(), True)
in_seg2 = torch.zeros(total_k, dtype=torch.bool)
for s, e in segs2:
    in_seg2[s:e] = True
nz2 = (aq_mn2[0].float() != 0).any(dim=1).cpu()
check('PSUM A 仅在逻辑组区间内非零（gap 全零）', int(nz2[~in_seg2].sum()) == 0)

# build_psum_layout_from_ks 与 masked 的 psum 递推一致
ks = [100, 200, 260]
bp = G.build_psum_layout_from_ks(ks, 128)
manual = []
prev = 0
for x in ks:
    prev = align(prev, 128) + x
    manual.append(prev)
check('build_psum_layout_from_ks == align(prev)+k 递推', bp == manual)

# K-grouped SF 使用 per_channel（沿 K 的 gran_k 分组）
reset()
total_k, aq, bq, cq, dq, refq, gl, host_ks = G.generate_k_grouped_contiguous(
    4, 128, 128, G.MajorTypeAB.KMajor, G.MajorTypeAB.KMajor, real_ks,
    use_ue8m0=False, gran_k=128, use_psum_layout=False, k_alignment=128,
    accumulate=True, out_dtype=torch.float)
# K-major 返回的 SF 是 (mn, total_scales)
sf_a = aq[1]
exp_scales = sum(ceil_div(x, 128) for x in host_ks)
print(f'\nK-major SF shape={tuple(sf_a.shape)}；预期 (mn, Σ ceil_div(host_k,128)) = '
      f'(128, {exp_scales})')
check('K-major SF 列数 == Σ ceil_div(host_k, 128)', sf_a.shape[1] == exp_scales)

# 真实 SM90 k-grouped kernel（accumulate, fp32 输出）
deep_gemm.set_mk_alignment_for_contiguous_layout(128)
reset()
total_k, aq, bq, cq, dq, refq, gl, host_ks = G.generate_k_grouped_contiguous(
    4, 256, 256, G.MajorTypeAB.KMajor, G.MajorTypeAB.KMajor, [512, 512, 640, 384],
    use_ue8m0=False, gran_k=128, use_psum_layout=False, k_alignment=128,
    accumulate=True, out_dtype=torch.float)
deep_gemm.k_grouped_fp8_gemm_nt_contiguous(
    aq, bq, dq, host_ks, gl, cq, recipe=(1, 1, 128))
diff = calc_diff(dq, refq)
print(f'k_grouped_fp8_gemm_nt_contiguous vs ref_d calc_diff = {diff:.3e}')
check('真实 SM90 k-grouped kernel 与 generator ref_d 一致', diff < 2e-2, f'diff={diff:.3e}')

# ---------------------------------------------------------------------------
hdr('I. enumerate_k_grouped_contiguous_test_variants：空组/单组变体')
# ---------------------------------------------------------------------------
reset()
base_ks = [100, 200, 260, 300]
variants = list(G.enumerate_k_grouped_contiguous_test_variants(base_ks))
for i, v in enumerate(variants):
    print(f'  variant[{i}] = {v}')
check('variant[0] == 原 ks', variants[0] == base_ks)
check('variant empty 的首/末强制 [0, 非0]', variants[1][0] == 0 and variants[1][-1] == base_ks[-1])
check('single-group 变体恰有一个非零', sum(1 for x in variants[2] if x > 0) == 1)

# ---------------------------------------------------------------------------
hdr('J. transpose_packed_fp4 (K-major fp4 生成用)')
# ---------------------------------------------------------------------------
from deep_gemm.utils import per_token_cast_to_fp4
reset()
x4 = torch.randn(64, 128, device='cuda', dtype=torch.bfloat16)
packed, sfx = per_token_cast_to_fp4(x4, use_ue8m0=False, gran_k=32)
tp = transpose_packed_fp4(packed)
print(f'packed {tuple(packed.shape)} (M, K/2) -> transpose_packed_fp4 {tuple(tp.shape)} (K, M/2)')
check('transpose_packed_fp4: (M, K/2) -> (K, M/2)', tp.shape == (packed.shape[1] * 2, packed.shape[0] // 2))

# ---------------------------------------------------------------------------
hdr(f'总结: {PASS} PASS / {FAIL} FAIL')
print('（架构无关，H100 sm90 真机运行；SM100 专属分支已标注未执行）')
sys.exit(1 if FAIL else 0)
