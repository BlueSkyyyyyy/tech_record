"""P57-1 H100 evidence: Mega mHC 的端到端验收测试与正确性 oracle
(`tests/test_mega_mhc.py`)。

Mega mHC 主体 kernel (`sm100_mega_mhc`) 是 SM100 专属
(`csrc/apis/mega_mhc.hpp:223` 断言 `arch_major == 10`)，H100(sm90) 跑不了。
但它的**验收测试**里那套「算什么对」的规范全是纯 torch / host 逻辑，架构无关：

  * `mhc_reference` (tests/test_mega_mhc.py:123-210)：残差多流混合 → mix 投影 →
    sigmoid/Sinkhorn → RMSNorm → FP8 量化，纯 torch。
  * SF 页布局 `extra_sf_rows` / `_allocate_extra_sf` (:96-111)：把逐 token 的
    SF 写进 Mega MoE 共享专家的 4×32 交错页布局，**逐字等价于 kernel**
    `get_shared_sf_row_idx` (epilogue/sm100_mega_mhc.cuh:116-122)。
  * host 契约 `csrc/apis/mega_mhc.hpp:116-221`：**在 H100 上真实执行**——
    非法组合在 arch 闸门 (:223) 之前被逐条拒绝，合法调用才撞到 arch 闸门。

本脚本直接 import 真实 `tests/test_mega_mhc.py` 复用它的 helper
（`make_inputs`/`make_outputs`/`mhc_reference`/`extra_sf_rows`/`check_correctness`/
`logical_outputs`/`comparable`/`enumerate_cases`），做独立 fp64 对拍、页布局
公式核验、判据反例、以及真实 host 契约拒绝。
"""
import os
import sys
import importlib.util

import torch

_CANDIDATES = [
    os.environ.get('DEEPGEMM_TESTS', ''),
    os.path.expanduser('~/github/DeepGEMM/tests'),
    '~/github/DeepGEMM/tests',
    '~/github/DeepGEMM/tests',
]
TESTS_DIR = next(p for p in _CANDIDATES if p and os.path.isfile(os.path.join(p, 'test_mega_mhc.py')))
if TESTS_DIR not in sys.path:
    sys.path.insert(0, TESTS_DIR)

import deep_gemm

PASS = 0
FAIL = 0


def check(name, cond, extra=''):
    global PASS, FAIL
    cond = bool(cond)
    if cond:
        PASS += 1
        print(f'  [PASS] {name} {extra}')
    else:
        FAIL += 1
        print(f'  [FAIL] {name} {extra}')
    return cond


def hdr(title):
    print('\n' + '=' * 78)
    print(title)
    print('=' * 78)


def relerr(a, b):
    a = a.double().flatten()
    b = b.double().flatten()
    denom = b.norm().clamp_min(1e-30)
    return float((a - b).norm() / denom)


# --- 直接加载真实 test_mega_mhc.py 作为模块，复用它的 helper ---
spec = importlib.util.spec_from_file_location('test_mega_mhc', os.path.join(TESTS_DIR, 'test_mega_mhc.py'))
T = importlib.util.module_from_spec(spec)
spec.loader.exec_module(T)


# =====================================================================
hdr('A. 验收测试结构：用例矩阵 / 输入构造 / 输出布局')
# =====================================================================
cases = list(T.enumerate_cases())
check('enumerate_cases 数量 = 2 sf_layout x 2 hidden x 7 tokens = 28',
      len(cases) == 28, f'got {len(cases)}')
check('sf_layout 取值 {col, extra}',
      {c[0] for c in cases} == {'col', 'extra'})
check('hidden 取值 {4096, 7168}', {c[2] for c in cases} == {4096, 7168})
check('num_tokens 取值含 1/65/1025/32768',
      {c[1] for c in cases} == {1, 64, 65, 200, 1025, 4096, 32768})

# make_inputs 的「重复 token」装置：把 63/64/511/512/T-1 行克隆成第 0 行，
# 用来暴露 split-K 边界（BLOCK_M=64、每 128/512 行一个边界）上的 bug。
for num_tokens in (200, 1025, 4096):
    inputs, dups = T.make_inputs(num_tokens, 4096, 2026, False)
    ok = all(torch.equal(inputs['x'][d], inputs['x'][0]) for d in dups)
    check(f'make_inputs 重复行 T={num_tokens} dups={dups}', ok, f'dups={dups}')
check('hc_mult=4, kNumHCOutputs=hc_mult*(hc_mult+2)=24',
      inputs['fn'].shape[0] == 24 and inputs['fn'].shape[1] == 4 * 4096)
crm = inputs['comb_res_mix']
row_dev = float((crm.sum(-1) - 1).abs().max())
col_dev = float((crm.sum(-2) - 1).abs().max())
# make_inputs 做三遍「行归一化→列归一化」，最后一次是列归一化，故列和精确=1；
# 行和只是近似（Sinkhorn 才把它逼到双随机，这里给的是个接近均衡的起点）。
check('comb_res_mix 三遍归约后 col 和=1（row 和仅近似）',
      col_dev < 1e-5, f'col_dev={col_dev:.3e} row_dev={row_dev:.3e}')

# 输出布局：col -> y_gemm_sf stride (1, align(T,4))；extra -> y_routed_sf + y_shared_sf
inp, _ = T.make_inputs(200, 4096, 1, False)
oc = T.make_outputs(inp, 'col')
check('col: y_gemm_sf stride=(1, align(T,4))', tuple(oc['y_gemm_sf'].stride()) == (1, (200 + 3) // 4 * 4))
ox = T.make_outputs(inp, 'extra')
check('extra: y_routed_sf contiguous', ox['y_routed_sf'].is_contiguous())
check('extra: y_shared_sf stride=(1, num_rows)',
      tuple(ox['y_shared_sf'].stride()) == (1, ox['y_shared_sf_storage'].shape[0]))
nrows_expected = (200 + 224 - 1) // 224 * (((224 + 127) // 128) * 128)
check('extra: storage 行数 = ceil(T/block_m)*align(block_m,128)',
      ox['y_shared_sf_storage'].shape[0] == nrows_expected, f'{ox["y_shared_sf_storage"].shape[0]} vs {nrows_expected}')


# =====================================================================
hdr('B. SF 页布局 `extra_sf_rows` 是「kernel 行索引」的精确复刻')
# =====================================================================
def kernel_row_idx(token_idx, block_m):
    # 逐字来自 epilogue/sm100_mega_mhc.cuh:116-122
    idx = token_idx % block_m
    return (token_idx // block_m * ((block_m + 127) & ~127)
            + (idx & ~127) + (idx & 31) * 4 + ((idx >> 5) & 3))


for block_m in (64, 128, 224, 240, 512):
    for num_tokens in (1, 65, 200, 1025, 32768):
        tok = torch.arange(num_tokens)
        rows_test = T.extra_sf_rows(torch.empty(num_tokens, 1), block_m)
        rows_kern = torch.tensor([kernel_row_idx(t, block_m) for t in tok.tolist()])
        if not torch.equal(rows_test, rows_kern):
            check(f'extra_sf_rows==kernel block_m={block_m} T={num_tokens}', False)
            break
    else:
        check(f'extra_sf_rows==kernel 全 token 相同 block_m={block_m}', True)

# 每个 block 页内是单射（4×32 交错把 128 连续 token 行打散到 4 个 32 行组）
ok = True
bmm = 224
aligned = (bmm + 127) // 128 * 128
for base in range(0, 4096, bmm):
    cnt = min(bmm, 4096 - base)
    rows = [kernel_row_idx(base + i, bmm) - base // bmm * aligned for i in range(cnt)]
    ok = ok and (len(set(rows)) == cnt) and all(0 <= r < aligned for r in rows)
check('页内单射且落在 [0, align(block_m,128))', ok)

# 用真实量化器造 SF 并按页布局 scatter，再按页布局 gather 应逐位还原
inp, _ = T.make_inputs(1025, 4096, 7, False)
ref = T.mhc_reference(**{**inp, 'sf_layout': 'extra', 'shared_sf_block_m': 224})
y_fp8_sf = ref['y_routed_sf']
storage = ref['y_shared_sf_storage']
gathered = storage[T.extra_sf_rows(torch.empty(1025, 1), 224)]
check('页布局 scatter->gather 逐位还原 SF', torch.equal(gathered, y_fp8_sf))
# padding 行保持全零
rows_used = set(T.extra_sf_rows(torch.empty(1025, 1), 224).tolist())
pad_rows = [r for r in range(storage.shape[0]) if r not in rows_used]
check('未使用行全零', bool((storage[pad_rows] == 0).all()))


# =====================================================================
hdr('C. `mhc_reference` 独立 fp64 对拍 + Sinkhorn 双随机')
# =====================================================================
@torch.no_grad()
def mhc_reference_fp64(x, residual, post_mix, comb_res_mix, fn, mix_scales, mix_bases,
                       rmsnorm_weight, hc_mult, hc_norm_eps, hc_pre_eps, hc_post_scale,
                       sinkhorn_eps, num_sinkhorn_iters, rmsnorm_eps, rmsnorm_scale,
                       shifted_prev_mix=None):
    x = x.double(); residual = residual.double(); post_mix = post_mix.double()
    comb = comb_res_mix.double(); fn = fn.double(); scales = mix_scales.double()
    bases = mix_bases.double(); w = rmsnorm_weight.double()
    Tn, H = x.shape
    new_residual = x.unsqueeze(1) * post_mix + torch.einsum('tij,tih->tjh', comb, residual)
    new_residual_bf16 = new_residual.bfloat16().double()
    norm_input = None
    if shifted_prev_mix is not None:
        norm_input = (new_residual_bf16 * shifted_prev_mix.double()).sum(1)
    mixes = new_residual_bf16.flatten(1) @ fn.mT.float().double()
    hc_sqr = new_residual_bf16.square().sum((1, 2))
    mixes = mixes * torch.rsqrt(hc_sqr / (hc_mult * H) + hc_norm_eps).unsqueeze(1)
    sc = torch.cat((scales[0].expand(hc_mult), scales[1].expand(hc_mult), scales[2].expand(hc_mult * hc_mult)))
    mixes = mixes * sc + bases
    new_prev_mix = mixes[:, :hc_mult].sigmoid().unsqueeze(2) + hc_pre_eps
    new_post_mix = (mixes[:, hc_mult:2 * hc_mult].sigmoid() * hc_post_scale).unsqueeze(2)
    v = mixes[:, 2 * hc_mult:].view(-1, hc_mult, hc_mult).softmax(-1) + sinkhorn_eps
    v = v / (v.sum(-2, keepdim=True) + sinkhorn_eps)
    for _ in range(num_sinkhorn_iters - 1):
        v = v / (v.sum(-1, keepdim=True) + sinkhorn_eps)
        v = v / (v.sum(-2, keepdim=True) + sinkhorn_eps)
    new_comb = v
    if norm_input is None:
        norm_input = (new_residual_bf16 * new_prev_mix).sum(1)
    y = (norm_input * torch.rsqrt(norm_input.square().mean(1) + rmsnorm_eps).unsqueeze(1)
         * (w * rmsnorm_scale)).bfloat16()
    return dict(new_residual=new_residual_bf16.bfloat16(), new_post_mix=new_post_mix.float(),
                new_comb_res_mix=new_comb.float(), y_bf16=y, new_prev_mix=new_prev_mix.float())


for is_shifted in (False, True):
    inp, _ = T.make_inputs(200, 7168, 555, is_shifted)
    ref = T.mhc_reference(**{**inp, 'sf_layout': 'bf16'})
    alt = mhc_reference_fp64(**inp)
    tag = 'shifted' if is_shifted else 'normal'
    e = relerr(ref['new_residual'].float(), alt['new_residual'].float())
    check(f'{tag}: new_residual fp64 relerr<3e-3', e < 3e-3, f'{e:.2e}')
    e = relerr(ref['y_bf16'].float(), alt['y_bf16'].float())
    check(f'{tag}: y_bf16 fp64 relerr<3e-3', e < 3e-3, f'{e:.2e}')
    e = relerr(ref['new_comb_res_mix'], alt['new_comb_res_mix'])
    check(f'{tag}: new_comb_res_mix fp64 relerr<3e-3', e < 3e-3, f'{e:.2e}')
    if is_shifted:
        e = relerr(ref['new_prev_mix'], alt['new_prev_mix'])
        check(f'{tag}: new_prev_mix fp64 relerr<3e-3', e < 3e-3, f'{e:.2e}')

# Sinkhorn 输出双随机：S^T 的行/列和都应 = 1
S = ref['new_comb_res_mix']
check('Sinkhorn 行和=1', bool(torch.allclose(S.sum(-1), torch.ones_like(S.sum(-1)), atol=1e-4)))
check('Sinkhorn 列和=1', bool(torch.allclose(S.sum(-2), torch.ones_like(S.sum(-2)), atol=1e-4)))


# =====================================================================
hdr('D. 判据 `check_correctness` / `comparable` 能抓错')
# =====================================================================
a = torch.randn(64, 4096, dtype=torch.bfloat16, device='cuda')
b = a.clone()
d = T.check_correctness({'y_bf16': a}, {'y_bf16': b}, ('t',))
check('相同张量 diff=0', d[0] == 0.0, f'{d[0]:.1e}')
b = b + 1.0  # 整体偏移 1.0（归一化平方误差 ~0.33 >> 5e-5）
caught = False
try:
    T.check_correctness({'y_bf16': a}, {'y_bf16': b}, ('t',))
except AssertionError:
    caught = True
check('整体扰动被 assert 抓到', caught)
# comparable：int32 -> uint8 视图，逐字节比较
ia = torch.tensor([[0x01020304]], dtype=torch.int32, device='cuda')
c = T.comparable(ia)
check('comparable(int32)->uint8 view', c.dtype == torch.uint8 and c.numel() == ia.numel() * 4)
# y_fp8 阈值更松 (2e-4) 而其它 5e-5：造一个 calc_diff 落在两者之间的对（一个元素 1.0->0.5，
# N=1024 → calc_diff ≈ 0.25/(2*1024) ≈ 1.2e-4），应「作为 y_fp8 放行、作为 y_bf16 被拒」。
p = torch.ones(1024, dtype=torch.float8_e4m3fn, device='cuda')
q = p.clone(); q[0] = torch.tensor(0.5, dtype=torch.float8_e4m3fn, device='cuda')
diff = float(T.calc_diff(q, p))
fp8_ok = True
try:
    T.check_correctness({'y_fp8': p}, {'y_fp8': q}, ('t',))
except AssertionError:
    fp8_ok = False
bf16_rejected = False
try:
    T.check_correctness({'y_bf16': p}, {'y_bf16': q}, ('t',))
except AssertionError:
    bf16_rejected = True
check('calc_diff 介于 5e-5 与 2e-4 之间', 5e-5 < diff < 2e-4, f'calc_diff={diff:.3e}')
check('同一对：y_fp8 放行 / y_bf16 拒绝', fp8_ok and bf16_rejected)
check('logical_outputs 顺序', list(T.logical_outputs({'new_residual': 1, 'new_post_mix': 2,
      'new_comb_res_mix': 3, 'y_bf16': 4}).keys()) == ['new_residual', 'new_post_mix', 'new_comb_res_mix', 'y_bf16'])


# =====================================================================
hdr('E. 真实 host 契约（csrc/apis/mega_mhc.hpp）：H100 上逐条拒绝')
# =====================================================================
def call(inputs, outputs, **ov):
    kw = {**inputs, **outputs}
    kw.pop('y_shared_sf_storage', None)
    kw.update(ov)
    deep_gemm.mega_mhc(**kw)


def expect_reject(label, line, fn):
    try:
        fn()
    except RuntimeError as e:
        msg = str(e).split('\n')[0]
        ok = f'mega_mhc.hpp:{line}' in msg
        check(label, ok, f'-> {msg[:90]}')
        return
    check(label, False, 'no error raised')


inp, _ = T.make_inputs(64, 4096, 2026, False)
inp_shifted, _ = T.make_inputs(64, 4096, 2026, True)

expect_reject('shifted 全有或全无 (:118)', 118,
              lambda: call(inp_shifted, T.make_outputs(inp_shifted, 'bf16'), new_prev_mix=None))
dual = T.make_outputs(inp, 'col'); dual.update(T.make_outputs(inp, 'extra'))
expect_reject('y_gemm_sf 与 y_routed_sf 互斥 (:120)', 120,
              lambda: call(inp, dual))
no_out = T.make_outputs(inp, 'bf16'); no_out['y_bf16'] = None
expect_reject('至少一个 norm 输出 y_bf16/y_fp8 (:121)', 121,
              lambda: call(inp, no_out))
fp8_no_sf = T.make_outputs(inp, 'bf16')
fp8_no_sf['y_fp8'] = torch.empty((64, 4096), dtype=torch.float8_e4m3fn, device='cuda')
expect_reject('y_fp8 必须配 SF (:122)', 122,
              lambda: call(inp, fp8_no_sf))
shared_no_block = T.make_outputs(inp, 'extra'); shared_no_block['shared_sf_block_m'] = 0
expect_reject('y_shared_sf 需要 shared_sf_block_m>0 (:123)', 123,
              lambda: call(inp, shared_no_block))
inp_bad_h, _ = T.make_inputs(64, 3008, 2026, False)  # 3008 % (16*64=1024) != 0
expect_reject('hidden % (kDefaultNumSplits*BLOCK_K) (:141)', 141,
              lambda: call(inp_bad_h, T.make_outputs(inp_bad_h, 'bf16')))
# 合法调用到达 arch 闸门 (:223)
expect_reject('合法调用撞到 arch 闸门 (:223)', 223,
              lambda: call(inp, T.make_outputs(inp, 'bf16')))

# y_gemm_sf 必须列主序 + TMA 对齐列距 (:204-206)
bad_stride = T.make_outputs(inp, 'col')
bad_stride['y_gemm_sf'] = bad_stride['y_gemm_sf'].contiguous()
expect_reject('y_gemm_sf stride(0)==1 (:204)', 204, lambda: call(inp, bad_stride))
# y_routed_sf 必须连续 (:210)
bad_contig = T.make_outputs(inp, 'extra')
sf = bad_contig['y_routed_sf']
bad_contig['y_routed_sf'] = torch.empty_strided(sf.shape, (1, sf.shape[0]), dtype=torch.int32, device='cuda')
expect_reject('y_routed_sf 必须连续 (:210)', 210, lambda: call(inp, bad_contig))


print('\n' + '=' * 78)
print(f'TOTAL: PASS={PASS} FAIL={FAIL}')
print('=' * 78)
sys.exit(1 if FAIL else 0)
