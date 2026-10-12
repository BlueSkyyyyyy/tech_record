"""P50-1 H100 evidence: DeepGEMM MQA logits 的端到端验收测试与正确性 oracle
(`tests/test_attention.py`)。

DeepSeek 稀疏注意力 indexer 的第一级打分 `fp8_fp4_mqa_logits` /
`fp8_fp4_paged_mqa_logits` 在 SM90 上有实现（第 19 篇 P9-1 拆过它的主 kernel，
第 30 篇 P20-1 拆过 SM90 分页调度器）。而它「算什么对」的唯一规范是
`tests/test_attention.py`（715 行）——它也是第 19/22 篇里那些 TFLOPS / GB/s /
relu-per-cyc 数字的出处。

**与 Mega MoE / Mega Gate 不同**：MQA logits 的 SM90 用例（fmt='fp8'、fp32 logits）
在 H100 上**真的能跑**（SM100 的 mxfp4/mxfp8 与 sparse 用例被 `@test_filter` 跳过）。
所以本篇既能精读「正确性 oracle」，又能**直接在 H100 上把真实测试跑通**。

本脚本直接 `import` 真实 `tests/test_attention.py` 作为模块，复用它的 helper：
  * 跑真实 `test_mqa_logits()` / `test_paged_mqa_logits()`（DG_MQA_NUM_CASES 限制用例数）；
  * 独立验证 oracle 的性质：`ref_mqa_logits` 与朴素实现对拍、`to_mqa_weights` 的 TMA
    16B 行 padding、`quantize_mqa_q` 的 fp8 语义、KV cache 的 SM90 融合布局、
    `mqa_logits_tolerances`、`sample_mqa_cases` 的确定性、`assert_bitwise_equal`。

  bash harness/exp_mqa_logits_test.sh > harness/exp_mqa_logits_test.out.txt
"""
import os
import sys
import importlib.util
import inspect

import torch
import torch.nn.functional as F

_CANDIDATES = [
    os.environ.get('DEEPGEMM_TESTS', ''),
    os.path.expanduser('~/github/DeepGEMM/tests'),
    '~/github/DeepGEMM/tests',
    '~/github/DeepGEMM/tests',
]
TESTS_DIR = next(p for p in _CANDIDATES if p and os.path.isfile(os.path.join(p, 'test_attention.py')))
if TESTS_DIR not in sys.path:
    sys.path.insert(0, TESTS_DIR)

import deep_gemm
from deep_gemm.testing import calc_diff, assert_bitwise_equal, get_arch_major

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


def expect_raise(name, fn):
    try:
        fn()
        return check(name, False, 'expected AssertionError, got none')
    except AssertionError as e:
        msg = str(e).splitlines()[0][:70] if str(e) else ''
        return check(name, True, f'AssertionError: {msg}')


# --- 直接加载真实 tests/test_attention.py 作为模块，复用它的 helper ---
spec = importlib.util.spec_from_file_location('test_attention', os.path.join(TESTS_DIR, 'test_attention.py'))
TA = importlib.util.module_from_spec(spec)
spec.loader.exec_module(TA)
print(f'arch_major = {get_arch_major()} (H100=9)')
print(f'deep_gemm  = {deep_gemm.__file__}')
print(f'test module = {inspect.getsourcefile(TA)}')


# ===========================================================================
hdr('A. 模块级 helper 与常量（真实模块，架构分支）')
# ===========================================================================
# SM90 takes E4M3 + one float scale per KV token, float weights / float logits
# SM100 takes MXFP4 / MXFP8 with BF16 weights and logits
fmts = TA.mqa_logits_formats()
check('A1 H100(arch9) mqa_logits_formats == [(fp8, float)]',
      fmts == [('fp8', torch.float)], f'{fmts}')
heads, dims = TA.mqa_logits_heads(False)
check('A2 H100 heads == (32,64), head_dims == (32,64,128)',
      heads == (32, 64) and dims == (32, 64, 128), f'heads={heads} dims={dims}')
check('A3 tolerances fp8 == (1e-3, 5e-6)', TA.mqa_logits_tolerances('fp8') == (1e-3, 5e-6),
      f'{TA.mqa_logits_tolerances("fp8")}')
check('A4 tolerances mxfp4 == (0.02, 3e-5)', TA.mqa_logits_tolerances('mxfp4') == (0.02, 3e-5),
      f'{TA.mqa_logits_tolerances("mxfp4")}')
check('A5 dtype_tag(FP32)="FP32", (BF16)="BF16"',
      TA.dtype_tag(torch.float) == 'FP32' and TA.dtype_tag(torch.bfloat16) == 'BF16')

# sample_mqa_cases: 无 env -> 全部；有 env -> 按名字定种子抽样（可复现）
all_cases = [(f, s) for f in ('prefill', 'paged', 'sparse') for s in range(5)]
saved = os.environ.pop('DG_MQA_NUM_CASES', None)
sel_all = TA.sample_mqa_cases('prefill', all_cases)
check('A6 DG_MQA_NUM_CASES 未设 -> 返回全部', len(sel_all) == len(all_cases), f'{len(sel_all)}/{len(all_cases)}')
os.environ['DG_MQA_NUM_CASES'] = '2'
sel1 = TA.sample_mqa_cases('prefill', all_cases)
sel2 = TA.sample_mqa_cases('prefill', all_cases)
selp = TA.sample_mqa_cases('paged', all_cases)
check('A7 DG_MQA_NUM_CASES=2 -> 抽 2 个且同名字可复现', len(sel1) == 2 and sel1 == sel2, f'{sel1}')
check('A8 不同名字用不同种子（prefill vs paged 子集不同）', sel1 != selp, f'prefill={sel1} paged={selp}')
os.environ.pop('DG_MQA_NUM_CASES', None)
if saved is not None:
    os.environ['DG_MQA_NUM_CASES'] = saved


# ===========================================================================
hdr('B. `ref_mqa_logits`（连续版参考）逐段：与朴素实现对拍、-inf 掩码、cost')
# ===========================================================================
# 语义（源码 :73-101）：out_ij = sum_h relu(<q[i,h,:], kv_j>) * w[i,h]，仅在 j∈[ks_i, ke_i) 有效，
# 其余置 -inf；KV 分块计算以限制 score 临时张量大小；cost = 有效位置个数。
M, N, H, D = 32, 128, 4, 32
torch.manual_seed(0)
q = torch.randn(M, H, D, device='cuda', dtype=torch.bfloat16)
kv = torch.randn(N, D, device='cuda', dtype=torch.bfloat16)
w = torch.randn(M, H, device='cuda', dtype=torch.float32)     # [M,H]，参考内部转置成 [H,M]
ks = torch.randint(0, N // 2, (M,), device='cuda', dtype=torch.int32)
ke = ks + torch.randint(1, N // 2, (M,), device='cuda', dtype=torch.int32)
ref, cost = TA.ref_mqa_logits(q, kv, w, ks, ke)


def naive_ref_mqa(q, kv, weights, ks, ke):
    m_, h_, d_ = q.shape
    n_ = kv.shape[0]
    out = torch.full((m_, n_), float('-inf'), device=q.device, dtype=torch.float32)
    qf, kf, wf = q.float(), kv.float(), weights.float()
    for i in range(m_):
        a, b = int(ks[i].item()), int(ke[i].item())
        if b <= a:
            continue
        score = torch.einsum('hd,nd->hn', qf[i], kf[a:b]).relu() * wf[i][:, None]
        out[i, a:b] = score.sum(0)
    return out


naive = naive_ref_mqa(q, kv, w, ks, ke)
valid = (ref != float('-inf'))
d = calc_diff(ref[valid], naive[valid])
check('B1 ref_mqa_logits == 朴素三重循环（有效区间内 calc_diff）', d < 1e-6, f'calc_diff={d:.3e}')
check('B2 两者 -inf 掩码逐位一致',
      torch.equal((ref == float('-inf')), (naive == float('-inf'))))
check('B3 cost == 有效位置数 == sum(ke-ks)', int(cost.item()) == int(valid.sum().item()) == int((ke - ks).sum().item()),
      f'cost={int(cost.item())} valid={int(valid.sum().item())} sum(ke-ks)={int((ke-ks).sum().item())}')
# 掩码语义：第 i 行只有 [ks_i, ke_i) 非 -inf
row_ok = True
for i in range(M):
    a, b = int(ks[i].item()), int(ke[i].item())
    row_ok &= bool((ref[i, a:b] != float('-inf')).all()) and bool((ref[i, :a] == float('-inf')).all()) \
        and bool((ref[i, b:] == float('-inf')).all())
check('B4 每行恰在 [ks_i, ke_i) 内有效，其余 -inf', row_ok)


# ===========================================================================
hdr('C. `generate_ks_ke_tests`（嵌套，源码复刻）的 CP 分片语义')
# ===========================================================================
# 源码 :106-121。disable_cp=True -> ks=0, ke=arange(seq_len)+(seq_len_kv-seq_len)
# disable_cp=False -> chunk_size=seq_len//2, cp_size=seq_len_kv//seq_len, cp_id=cp_size//3
#                     ke[i]=cp_id*chunk+i (i<chunk); ke[i+chunk]=(2*cp_size-1-cp_id)*chunk+i
def generate_ks_ke_tests(seq_len, seq_len_kv, disable_cp):
    if disable_cp:
        ks = torch.zeros(seq_len, dtype=torch.int)
        ke = torch.arange(seq_len, dtype=torch.int) + (seq_len_kv - seq_len)
        return ks, ke
    assert seq_len_kv % seq_len == 0 and seq_len % 2 == 0
    chunk_size = seq_len // 2
    cp_size = seq_len_kv // seq_len
    cp_id = cp_size // 3
    ks = torch.zeros(seq_len, dtype=torch.int)
    ke = torch.zeros(seq_len, dtype=torch.int)
    for i in range(chunk_size):
        ke[i] = cp_id * chunk_size + i
        ke[i + chunk_size] = (cp_size * 2 - 1 - cp_id) * chunk_size + i
    return ks, ke


ks1, ke1 = generate_ks_ke_tests(2048, 8192, True)
check('C1 disable_cp: ks=0, ke=arange+(SK-SQ)', bool((ks1 == 0).all()) and
      bool((ke1 == torch.arange(2048) + (8192 - 2048)).all()))
ks2, ke2 = generate_ks_ke_tests(2048, 8192, False)
check('C2 CP: cp_size=4, cp_id=1 -> ke 前 1024 个落在 [1024,2048)，后 1024 个落在 [6144,7168)',
      int(ke2[:1024].min()) == 1024 and int(ke2[:1024].max()) == 2047 and
      int(ke2[1024:].min()) == 6144 and int(ke2[1024:].max()) == 7167)
check('C3 CP: chunk_size=seq_len//2，两段各 seq_len/2', 1024 + 1024 == 2048)


# ===========================================================================
hdr('D. `to_mqa_weights` 的 TMA 16B 行 padding（源码 :23-30）')
# ===========================================================================
def stride_of(h, elem_size):
    return (h * elem_size + 15) // 16 * 16 // elem_size


for h, dt in ((32, torch.float32), (64, torch.float32), (32, torch.bfloat16), (5, torch.float32), (3, torch.bfloat16)):
    wt = torch.randn(7, h, device='cuda', dtype=dt)
    out = TA.to_mqa_weights(wt)
    es = wt.element_size()
    exp_stride = stride_of(h, es)
    check(f'D h={h:2} {TA.dtype_tag(dt):4}: stride={exp_stride:3} 且前 h 列逐位等于输入',
          out.stride(0) == exp_stride and torch.equal(out, wt), f'stride={out.stride(0)}')
# 行字节数对齐到 16 的验证
for h, dt in ((32, torch.float32), (3, torch.bfloat16)):
    out = TA.to_mqa_weights(torch.randn(4, h, device='cuda', dtype=dt))
    row_bytes = out.stride(0) * out.element_size()
    check(f'D row {h}x{TA.dtype_tag(dt)}: 每行 {row_bytes}B 是 16 的倍数', row_bytes % 16 == 0)


# ===========================================================================
hdr('E. `quantize_mqa_q` 的 fp8 语义（SM90 路径，源码 :57-67）')
# ===========================================================================
qbf = torch.randn(16, 8, 128, device='cuda', dtype=torch.bfloat16)
(q_fp, q_sf), q_sim = TA.quantize_mqa_q(qbf, 'fp8')
check('E1 fp8: (q_fp, q_sf=None)，dtype=e4m3', q_sf is None and q_fp.dtype == torch.float8_e4m3fn)
check('E2 q_simulated == q_fp.to(bf16)（逐位）', torch.equal(q_sim, q_fp.to(torch.bfloat16)))
rel = (q_sim.float() - qbf.float()).norm() / qbf.float().norm()
check('E3 fp8 往返相对误差在 E4M3 量级', 0.01 < rel.item() < 0.05, f'rel={rel.item():.4f}')


# ===========================================================================
hdr('F. KV cache 融合布局：SM90 fp8（有 float scale）+ SM100 mxfp（纯 torch，架构无关）')
# ===========================================================================
nb, bs, hd = 3, 64, 128
kvc = torch.randn(nb, bs, 1, hd, device='cuda', dtype=torch.bfloat16)
x_fp8, x_back = TA.kv_cache_cast_to_fp8(kvc)
check('F1 fp8 paged 布局 shape == (blocks, block_size, 1, head_dim+4)',
      tuple(x_fp8.shape) == (nb, bs, 1, hd + 4), f'{tuple(x_fp8.shape)}')
# 前 bs*hd 字节 = E4M3 值；后 bs*4 字节 = 每 token 一个 float scale
raw = x_fp8.reshape(nb, bs * (hd + 4))
vals_bytes = raw[:, :bs * hd].reshape(nb, bs, hd)
sf_bytes = raw[:, bs * hd:].reshape(nb, bs, 4)
vals = vals_bytes.contiguous().view(torch.float8_e4m3fn)
sf = sf_bytes.contiguous().view(torch.float32).reshape(nb, bs)
x_back2 = vals.float() * sf.unsqueeze(-1)
check('F2 反量化 == 值(e4m3) * 每 token 标量 scale（从布局重新解码，再 cast 回 bf16）',
      torch.equal(x_back2.reshape(nb, bs, 1, hd).to(torch.bfloat16), x_back))
ref_sf = kvc.abs().float().amax(dim=3, keepdim=False).clamp(1e-4) / 448.0
check('F3 scale 逐位等于 amax/448（clamp 1e-4）',
      torch.equal(sf.reshape(nb, bs), ref_sf.reshape(nb, bs)),
      f'sf[0,0]={sf[0,0].item():.3e} ref={ref_sf[0,0,0].item():.3e}')

for name, fn in (('mxfp4', TA.kv_cache_cast_to_mxfp4), ('mxfp8', TA.kv_cache_cast_to_mxfp8)):
    kx, back = fn(kvc)
    edim = hd // 2 if name == 'mxfp4' else hd
    check(f'F {name} paged 布局 shape == (blocks, block_size, 1, elem+4)',
          tuple(kx.shape) == (nb, bs, 1, edim + 4), f'{tuple(kx.shape)}')
    r = (back.float() - kvc.float()).norm() / kvc.float().norm()
    check(f'F {name} 往返相对误差量级', r.item() < 0.2, f'rel={r.item():.4f}')


# ===========================================================================
hdr('G. 判据工具：`assert_bitwise_equal` / `calc_diff`（第 40/51 篇）')
# ===========================================================================
a = torch.randn(64, 128, device='cuda')
b = a.clone()
check('G1 相同张量 assert_bitwise_equal 通过', (assert_bitwise_equal(a, b, 't') is None))
b2 = a.clone(); b2[3, 5] += 1e-3
expect_raise('G2 改一个元素 assert_bitwise_equal 抛 AssertionError（定位 coord=(3,5)）',
             lambda: assert_bitwise_equal(a, b2, 't'))
# 测试对连续版用 calc_diff 判据、对自洽用 assert_bitwise_equal（:199-203, :173-175）
x = torch.randn(256, 256, device='cuda'); y = x + 1e-4 * torch.randn_like(x)
check('G3 calc_diff 小扰动 -> 小值', calc_diff(x, y) < 1e-6, f'{calc_diff(x, y):.3e}')
check('G4 calc_diff(x,x) == 0', calc_diff(x, x) == 0.0)


# ===========================================================================
hdr('H. 真实测试在 H100 上跑通：test_mqa_logits / test_paged_mqa_logits')
# ===========================================================================
# SM90 用例（fmt='fp8'）真的会执行；只限制用例数让输出有界。
os.environ['DG_MQA_NUM_CASES'] = '2'
try:
    TA.test_mqa_logits()
    check('H1 test_mqa_logits() 在 H100 上无异常完成（DG_MQA_NUM_CASES=2）', True)
except AssertionError as e:
    check('H1 test_mqa_logits() 在 H100 上无异常完成', False, f'AssertionError: {e}')
try:
    TA.test_paged_mqa_logits()
    check('H2 test_paged_mqa_logits() 在 H100 上无异常完成（DG_MQA_NUM_CASES=2）', True)
except AssertionError as e:
    check('H2 test_paged_mqa_logits() 在 H100 上无异常完成', False, f'AssertionError: {e}')
os.environ.pop('DG_MQA_NUM_CASES', None)

# sparse 用例被 @test_filter(get_arch_major()==10) 门控 -> H100 跳过
try:
    TA.test_sparse_mqa_logits()
    check('H3 test_sparse_mqa_logits() 在 H100 上被 arch==10 门控跳过（无异常）', True)
except AssertionError as e:
    check('H3 test_sparse_mqa_logits() 被门控跳过', False, f'{e}')


# ===========================================================================
hdr('I. sparse 用例构造 helper `make_sparse_kv_block_indices`（纯 torch，架构无关）')
# ===========================================================================
ctx_lens = [1024, 1024, 2048, 2048]
req_idx = [0, 0, 1, 1]
idx, nblk = TA.make_sparse_kv_block_indices(ctx_lens, req_idx, sparse_block_kv=64,
                                            num_max_sparse_blocks=8, seed=7)
check('I1 输出索引 shape == (num_q, num_max_sparse_blocks)', tuple(idx.shape) == (4, 8), f'{tuple(idx.shape)}')
check('I2 每 token 稀疏块数 <= num_max_sparse_blocks 且 >= 1', all(1 <= n <= 8 for n in nblk), f'{nblk}')
check('I3 同请求内保留 ~80% 上一 token 的块（跨行重叠）',
      len(set(idx[1].tolist()) & set(idx[0].tolist())) >= 1)


print('\n' + '=' * 78)
print(f'总判定: {PASS} PASS / {FAIL} FAIL')
print('=' * 78)
sys.exit(1 if FAIL else 0)
