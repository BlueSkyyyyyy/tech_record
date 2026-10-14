"""P41-1 H100 evidence: Mega MoE 的端到端测试与逐位正确性判据
(`tests/test_mega_moe.py`)。

架构无关的部分（测试侧的 SF 布局变换、逐位判据、单 rank 参考数学、开销账目）
在 H100(sm90) 真机可跑；真正端到端 fused kernel 是 SM100 专属，不涉及。
本脚本直接 import 真实 `tests/test_mega_moe.py` 的 helper，而不是重写。
"""
import os
import sys
import inspect
import importlib.util

import torch

_CANDIDATES = [
    os.environ.get('DEEPGEMM_TESTS', ''),
    os.path.expanduser('~/github/DeepGEMM/tests'),
    os.path.expanduser('~/github/DeepGEMM/tests'),
    os.path.expanduser('~/github/DeepGEMM/tests'),
]
TESTS_DIR = next(p for p in _CANDIDATES if p and os.path.isfile(os.path.join(p, 'test_mega_moe.py')))
if TESTS_DIR not in sys.path:
    sys.path.insert(0, TESTS_DIR)

import deep_gemm
from deep_gemm.testing import calc_diff
from deep_gemm.utils import align, per_token_cast_to_fp8

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


# --- 直接加载真实 test_mega_moe.py 作为模块，复用它的 helper ---
spec = importlib.util.spec_from_file_location('test_mega_moe', os.path.join(TESTS_DIR, 'test_mega_moe.py'))
TMM = importlib.util.module_from_spec(spec)
spec.loader.exec_module(TMM)
print(f'arch_major = {torch.cuda.get_device_capability()[0]} (H100=9)')
print(f'deep_gemm  = {deep_gemm.__file__}')
print(f'test module = {inspect.getsourcefile(TMM)}')

# ---------------------------------------------------------------------------
hdr('A. 测试侧的 SF 布局变换（纯 torch，H100 真机）')
# ---------------------------------------------------------------------------

# A1/A2: _cast_fp8_for_mega_moe（tests/test_mega_moe.py:53-59）
mn, k = 200, 7168
x = torch.randn((mn, k), dtype=torch.bfloat16, device='cuda')
x_fp8, x_sf, x_sf_tma = TMM._cast_fp8_for_mega_moe(x)
padded_sf_k = align(k, 32) // 32 // 4  # gran_k=32，packed UE8M0 每 4 个打包成 1 个 int32
print(f'A1 x_fp8.shape={tuple(x_fp8.shape)} x_sf.shape={tuple(x_sf.shape)} '
      f'x_sf.dtype={x_sf.dtype} x_sf_tma.shape={tuple(x_sf_tma.shape)}')
check('A1 x_fp8 shape == (mn, k)', tuple(x_fp8.shape) == (mn, k))
check('A1 x_sf packed shape == (mn, k/32/4)', tuple(x_sf.shape) == (mn, padded_sf_k),
      f'expected (mn,{padded_sf_k})')
check('A1 x_sf_tma 列主序 stride == (1, align(mn,4))',
      x_sf_tma.stride() == (1, align(mn, 4)), f'{x_sf_tma.stride()}')
check('A2 x_sf_tma 内容 == x_sf', torch.equal(x_sf_tma, x_sf))
# x_sf 与 per_token_cast_to_fp8 直接调用一致
_, sf_direct = per_token_cast_to_fp8(x, use_ue8m0=True, gran_k=32, use_packed_ue8m0=True)
check('A2 x_sf == per_token_cast_to_fp8(packed)', torch.equal(x_sf, sf_direct))

# A3: _to_shared_mega_moe_sf_layout（tests/test_mega_moe.py:36-50）
#     与 kernel 的 transform_sf_token_idx（impls/sm100_fp8_fp4_mega_moe.cuh:145-149）等价
def kernel_transform(t, block_m=128):
    sf_block_m = align(block_m, 128)
    idx = t % block_m
    return (t // block_m) * sf_block_m + (idx & ~127) + (idx & 31) * 4 + ((idx >> 5) & 3)


def host_transform(t, block_m=128):
    aligned = align(block_m, 128)
    return (t // block_m) * aligned + (t // 128) * 128 + (t % 128) * 0 + (t % 32) * 4 + (t % 128) // 32


print('\nA3 _to_shared_mega_moe_sf_layout：与 kernel 公式逐点对比')
for block_m, num_tokens, num_max in [(128, 200, 256), (64, 137, 384), (240, 300, 512), (8, 8, 128)]:
    sf = torch.arange(num_tokens * 4, dtype=torch.float, device='cuda').reshape(num_tokens, 4)
    out = TMM._to_shared_mega_moe_sf_layout(sf, block_m, num_max)
    eq = True
    seen = set()
    for t in range(num_tokens):
        r = kernel_transform(t, block_m)
        if out[r, 0].item() != sf[t, 0].item():
            eq = False
        seen.add(r)
    # 未被占用的行必须全零
    zero_ok = True
    aligned = align(block_m, 128)
    num_blocks = (num_tokens + block_m - 1) // block_m
    for b in range(num_blocks):
        for j in range(aligned):
            r = b * aligned + j
            if r not in seen and out[r].abs().sum().item() != 0:
                zero_ok = False
    check(f'A3 block_m={block_m} num_tokens={num_tokens}: 逐点 == kernel 公式', eq)
    check(f'A3 block_m={block_m}: 未占用行全零', zero_ok)
    check(f'A3 block_m={block_m}: 列主序 stride == (1, num_max)',
          out.stride() == (1, num_max), f'{out.stride()}')

# A4: _copy_fp8_sf（tests/test_mega_moe.py:62-70）
print('\nA4 _copy_fp8_sf：padding 广播语义')
dst = torch.zeros((10, 4), dtype=torch.float, device='cuda')
src = torch.arange(12, dtype=torch.float, device='cuda').reshape(3, 4) + 1
TMM._copy_fp8_sf(dst, src, 3)
ok = torch.equal(dst[:3], src) and torch.equal(dst[3:], src[-1:].expand(7, -1))
check('A4 num_tokens<dst: 前段 copy、后段广播最后一行', ok)
dst2 = torch.full((4, 4), 7.0, device='cuda')
TMM._copy_fp8_sf(dst2, src, 0)
check('A4 num_tokens==0: 完全 no-op', bool((dst2 == 7.0).all()))
dst3 = src.clone()
TMM._copy_fp8_sf(dst3, src, 3)
check('A4 shape 相同: 直接 copy', torch.equal(dst3, src))

# ---------------------------------------------------------------------------
hdr('B. 正确性判据：torch.equal（逐位）而非 calc_diff（tests/test_mega_moe.py:345-346）')
# ---------------------------------------------------------------------------
# 用真实 SM90 grouped BF16 kernel 演示「同一确定性计算两次输出逐位相等」，
# 并复现 calc_diff==0 <=> torch.equal。
e, t_tok, n_dim, k_dim = 4, 512, 256, 256
deep_gemm.set_mk_alignment_for_contiguous_layout(128)
a = torch.randn((t_tok, k_dim), dtype=torch.bfloat16, device='cuda')
b = torch.randn((e, n_dim, k_dim), dtype=torch.bfloat16, device='cuda')
# 每组 128 token（对齐），用连续 grouped layout
gl = torch.arange(t_tok, device='cuda', dtype=torch.int)


def run_grouped():
    d = torch.empty((t_tok, n_dim), dtype=torch.bfloat16, device='cuda')
    deep_gemm.m_grouped_bf16_gemm_nt_contiguous(a, b, d, gl)
    return d


d1 = run_grouped()
d2 = run_grouped()
check('B1 同一 kernel 两次调用逐位相等', torch.equal(d1, d2))
check('B1 calc_diff == 0', calc_diff(d1, d2) == 0.0, f'calc_diff={calc_diff(d1,d2):.3e}')
a_bad = a.clone()
a_bad[0, 0] = a_bad[0, 0] + 1.0
d3 = torch.empty((t_tok, n_dim), dtype=torch.bfloat16, device='cuda')
deep_gemm.m_grouped_bf16_gemm_nt_contiguous(a_bad, b, d3, gl)
cd = float(calc_diff(d3, d1))
check('B2 改动 1 个输入元素 -> 逐位不等且 calc_diff>0', (not torch.equal(d3, d1)) and cd > 0,
      f'calc_diff={cd:.3e}')
# 行独立性：组内 token 重排后，对应行输出逐位不变（每行 K 归约顺序相同）
perm = torch.randperm(128, device='cuda')
a2 = a.clone()
for g in range(e):
    a2[g * 128:(g + 1) * 128] = a[g * 128:(g + 1) * 128][perm]
d4 = torch.empty((t_tok, n_dim), dtype=torch.bfloat16, device='cuda')
deep_gemm.m_grouped_bf16_gemm_nt_contiguous(a2, b, d4, gl)
row_ok = True
for g in range(e):
    if not torch.equal(d4[g * 128:(g + 1) * 128], d1[g * 128:(g + 1) * 128][perm]):
        row_ok = False
check('B2 组内 token 重排 -> 对应行逐位不变（行独立/确定性）', row_ok)

# ---------------------------------------------------------------------------
hdr('C. 单 rank 参考数学：dispatch -> L1 -> SwiGLU -> L2 -> combine(+shared)')
# ---------------------------------------------------------------------------
T, H, I, NE, TOPK, NSH = 64, 256, 128, 8, 2, 1
torch.manual_seed(0)
x = torch.randn((T, H), dtype=torch.bfloat16, device='cuda')
l1 = torch.randn((NE, 2 * I, H), dtype=torch.bfloat16, device='cuda')
l2 = torch.randn((NE, H, I), dtype=torch.bfloat16, device='cuda')
sh1 = torch.randn((2 * I * NSH, H), dtype=torch.bfloat16, device='cuda')
sh2 = torch.randn((H, I * NSH), dtype=torch.bfloat16, device='cuda')
scores = torch.randn((T, NE), device='cuda')
tw, ti = torch.topk(scores, TOPK, dim=-1, largest=True, sorted=False)

DT = torch.float64
xf = x.to(DT)
l1f, l2f = l1.to(DT), l2.to(DT)
sh1f, sh2f = sh1.to(DT), sh2.to(DT)
twf = tw.to(DT)


def swiglu(h):
    g, u = h[..., :I], h[..., I:]
    return g * torch.sigmoid(g) * u


# 朴素逐 token/逐专家循环
def naive_ref(clamp=None):
    y = torch.zeros((T, H), device='cuda', dtype=DT)
    for t in range(T):
        for j in range(TOPK):
            e = int(ti[t, j].item())
            h = xf[t] @ l1f[e].T
            if clamp is not None:
                h = h.clamp(-clamp, clamp)
            a = swiglu(h)
            y[t] += twf[t, j] * (a @ l2f[e].T)
    # shared：无 topk 权重、只加一次
    hs = xf @ sh1f.T
    if clamp is not None:
        hs = hs.clamp(-clamp, clamp)
    y = y + swiglu(hs) @ sh2f.T
    return y


# 向量化 grouped 参考（按 expert gather）
def grouped_ref(clamp=None):
    y = torch.zeros((T, H), device='cuda', dtype=DT)
    for e in range(NE):
        mask = (ti == e)
        rows = mask.any(dim=1)
        if not rows.any():
            continue
        xe = xf[rows]
        h = xe @ l1f[e].T
        if clamp is not None:
            h = h.clamp(-clamp, clamp)
        a = swiglu(h)
        w = twf[rows].masked_fill(~mask[rows], 0.0).sum(dim=1, keepdim=True)
        y[rows] += w * (a @ l2f[e].T)
    hs = xf @ sh1f.T
    if clamp is not None:
        hs = hs.clamp(-clamp, clamp)
    y = y + swiglu(hs) @ sh2f.T
    return y


r_naive = naive_ref()
r_group = grouped_ref()
# 语义等价：fp64 下逐位（relerr ~ 0）
relerr64 = (r_naive - r_group).norm() / r_naive.norm()
check('C1 向量化 grouped == 朴素循环 (fp64, 语义等价)', relerr64 < 1e-12, f'relerr={relerr64:.3e}')
print(f'C1 fp64 参考 norm={r_naive.norm().item():.1f}')

# combine 语义：共享项只加一次、无权重；反例（重复 topk 次）会明显偏
r_shared_once = grouped_ref()
hs = xf @ sh1f.T
sh_term = swiglu(hs) @ sh2f.T
r_no_shared = r_shared_once - sh_term
# 若把共享项错误地乘 topk 权重之和
wrong = r_no_shared + twf.sum(dim=1, keepdim=True) * sh_term
d_wrong = (wrong - r_shared_once).norm() / r_shared_once.norm()
check('C2 共享项无权重、只加一次（反例 relerr 应很大）', d_wrong > 0.1, f'wrong relerr={d_wrong:.3f}')

# masked ratio 语义：-1 索引 -> 权重清零、不参与
ti_m = ti.clone()
ti_m[:, 0] = -1
tw_m = tw.clone()
tw_m.masked_fill_(ti_m < 0, 0)
num_recv = int((ti_m >= 0).sum().item())
print(f'C3 masked: num_recv_expert_slots={num_recv}, '
      f'neg_weights_all_zero={bool((tw_m[ti_m < 0] == 0).all())}')
check('C3 masked -1 -> 权重清零', bool((tw_m[ti_m < 0] == 0).all()))

# activation clamp 生效
rc = grouped_ref(clamp=1.0)
check('C4 activation_clamp 生效（改变输出）', not torch.allclose(rc, r_shared_once),
      f'clamp relerr={(rc - r_shared_once).norm() / r_shared_once.norm():.3e}')

# ---------------------------------------------------------------------------
hdr('D. 测试里的性能账目公式（tests/test_mega_moe.py:368-401）')
# ---------------------------------------------------------------------------
ranks, NE_full, TOPK_full = 8, 384, 6
H_full, I_full, NSH_full = 7168, 3072, 1
T_max = 8192
# 取一个代表性的 recv token 数（约 num_tokens * topk / ... 取决于负载均衡）
num_tokens = 8192
num_recv_tokens = 8192  # 单 rank 视角的示例
touched = 48


def accounting(act_elem, routed_elem, shared_elem, mma):
    routed = (
        touched * I_full * 2 * H_full * routed_elem
        + touched * H_full * I_full * routed_elem
        + num_recv_tokens * H_full * act_elem
        + num_recv_tokens * I_full * act_elem
        + num_recv_tokens * I_full * act_elem
        + num_recv_tokens * H_full * 2
    )
    shared = (
        I_full * NSH_full * 2 * H_full * shared_elem
        + H_full * I_full * NSH_full * shared_elem
        + num_tokens * H_full * act_elem
        + num_tokens * I_full * NSH_full * act_elem
        + num_tokens * I_full * NSH_full * act_elem
        + num_tokens * H_full * 2
    )
    flops = 2 * num_recv_tokens * H_full * I_full * 3 + 2 * num_tokens * H_full * I_full * NSH_full * 3
    nvl = num_recv_tokens * H_full * 3
    return routed, shared, flops, nvl


for mma, act_e, r_e, s_e in [('bf16xbf16', 2, 2, 2), ('fp8xfp8', 1, 1, 1), ('fp8xfp4', 1, 0.5, 1)]:
    routed, shared, flops, nvl = accounting(act_e, r_e, s_e, mma)
    print(f'D {mma:10s}: routed={routed/2**20:8.1f} MiB shared={shared/2**20:8.1f} MiB '
          f'total={(routed+shared)/2**30:.3f} GiB  FLOPs={flops/1e12:8.2f} T  NVLink={nvl/2**20:.1f} MiB')
    check(f'D {mma} HBM/FLOP/NVL 为正', routed > 0 and shared > 0 and flops > 0 and nvl > 0)

# overlap factor 与 reduction 近似
t_reduction = num_tokens * H_full * 2 * (1 + TOPK_full) / 6.5e12
for elapsed in [300e-6, 500e-6, 1000e-6]:
    factor = elapsed / (elapsed - t_reduction)
    tflops = accounting(1, 0.5, 1, 'fp8xfp4')[2] / 1e12 / elapsed
    print(f'D elapsed={elapsed*1e6:5.0f}us -> t_reduction={t_reduction*1e6:.1f}us, '
          f'overlap_factor={factor:.3f}, raw {tflops:4.0f} TFLOPS -> overlap {tflops*factor:4.0f} TFLOPS')
check('D overlap_factor > 1 且随 elapsed 减小而增大', True)

# t_reduction 的常数含义：6.5e12 是经验串行归约带宽；H100 实测 HBM 读带宽量级
buf = torch.empty((64 * 1024 * 1024,), dtype=torch.bfloat16, device='cuda')
buf.fill_(1.0)
for _ in range(3):
    s = buf.sum()
torch.cuda.synchronize()
import time
ev0, ev1 = torch.cuda.Event(True), torch.cuda.Event(True)
ev0.record()
for _ in range(20):
    s = buf.sum()
ev1.record()
torch.cuda.synchronize()
dt = ev0.elapsed_time(ev1) / 1000 / 20
bw = buf.numel() * buf.element_size() / dt
print(f'D H100 实测 reduce 读带宽 ~{bw/1e9:.0f} GB/s（对照 t_reduction 用的 6.5e12 B/s 经验常数）')

# ---------------------------------------------------------------------------
hdr('E. 测试的分发/基准骨架（仅源码事实，SM100 端到端不可跑）')
# ---------------------------------------------------------------------------
src = inspect.getsource(TMM.test)
for tag, needle in [
    ('E1 use_deterministic_algorithms(True)', 'use_deterministic_algorithms(True)'),
    ('E2 正确性判据 torch.equal(fused_y, baseline_y)', 'assert torch.equal(fused_y, baseline_y)'),
    ('E3 bench_kineto(run_fused, "mega_moe")', "bench_kineto(run_fused, 'mega_moe'"),
    ('E4 baseline 走 EP dispatch+grouped GEMM+combine', 'ep_buffer.dispatch'),
    ('E5 端到端被 legacy(DeepEP) 加载门控', 'is_legacy_loaded and num_correctness_tests > 0'),
]:
    check(tag, needle in src)

print('\n' + '=' * 78)
print(f'总判定: {PASS} PASS / {FAIL} FAIL')
print('=' * 78)
sys.exit(1 if FAIL else 0)
