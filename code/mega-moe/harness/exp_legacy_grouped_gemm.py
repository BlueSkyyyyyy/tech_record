"""P29-1 实验：DeepGEMM 的 **legacy Triton MoE grouped GEMM 家族**（`deep_gemm/legacy/`）
的「显式索引 gather」语义 + H100 实测。

背景：前面第 12/13 篇拆的是**现代 C++** 的 `k_grouped_*`（wgrad，K 轴分组）/`m_grouped_*`
（dgrad，M 轴分组），它们靠 `grouped_layout` + **BLOCK 对齐的连续段**取址，并要求
`M%align==0`（或 `k%align==0`）。但 DeepGEMM 还留了一整套 **A100 时代的 Triton 兜底 kernel**
（`deep_gemm/legacy/`，`tests/test_legacy.py` 是它的官方测试）——它们用**逐元素索引张量**
（`m_row_indices` / `k_indices`）实现**任意 gather**，比现代 API 更灵活。这套 kernel 是纯 Triton，
**架构无关**，H100 可跑（官方测试也能跑，只是注释说「只在 Ampere 上被采用」）。

本脚本用真实数字回答几个现代 API 文档里没有的问题：

  A) `m_grouped_bf16_gemm_nt_contiguous_tl`（dgrad 骨架）：`m_indices` 的 **block 首行取组号** 约定、
     padding 行（`m_indices=-1`）是否**严格写零**；现代 API 只给 `grouped_layout`，这里精确到行。
  B) `a_fused_m_grouped_...`（A 逐行 gather 的扩展版）：`m_row_indices` 如何把「输出行」映回
     「源激活行」；identity 时是否**逐位等于** `m_grouped`；置换成 gather 时能否复现任意散布。
  C) `a_fused_k_grouped_...` / `b_fused_k_grouped_...`（wgrad 的 K 轴分组，`tn` 布局）：
     `handle=(k_indices,k_start,k_end)` 的语义；identity 时两种「gather A / gather B」是否给出
     **同一数学**；非 identity 时到底等价于对哪一侧做行置换（真正坐实 `a_fused`/`b_fused` 命名）。
  D) 空组（`k_start>=k_end`）：`ACC=True` 时 D 原样保留、`ACC=False` 时 D 写零。
  E) 契约：这些 Triton kernel 硬性要求 `M%align==0`（m 轴）/ `非gather侧K%align==0`，与
     现代 C++ API 的段对齐是**同一个数**（`get_mk_alignment_for_contiguous_layout()`）。
  F) 性能：H100 上跑（Triton sm90 后端）两个方向的 TFLOPS。

来源核对（~/github/DeepGEMM，公开版 commit 057ca59，版本 2.8.1）：
  - deep_gemm/__init__.py:117-120          legacy 导入（try/except，注释 "Legacy Triton kernels for A100"）
  - deep_gemm/legacy/__init__.py:1-5       五个模块的 * 导出
  - deep_gemm/legacy/tune_options.py:12-23 _gemm_configs（A100 smem 166912 过滤 + <=align 过滤）
  - deep_gemm/legacy/m_grouped_gemm.py:14-38  block 首行取组号 + 空 token 写零
  - deep_gemm/legacy/m_grouped_gemm.py:56-70  主循环（b 按 batch_id 取组，IS_B_K_MAJOR 走址）
  - deep_gemm/legacy/m_grouped_gemm.py:86-92  契约断言（M%align==0）
  - deep_gemm/legacy/a_fused_m_grouped_gemm.py:35-44  空 token 写零；:46-48 m_row_indices 逐行 gather
  - deep_gemm/legacy/a_fused_m_grouped_gemm.py:74-83  契约断言（m_indices.numel()%align==0）
  - deep_gemm/legacy/a_fused_k_grouped_gemm.py:45-56  主循环（A 用 rows=k_indices gather）
  - deep_gemm/legacy/a_fused_k_grouped_gemm.py:73-83  契约断言（b.size(0)%align==0）
  - deep_gemm/legacy/b_fused_k_grouped_gemm.py:52-63  主循环（B 用 rows=k_indices gather）
  - tests/test_legacy.py:14-49                      官方 m-grouped 用法（identity k_indices）
  - tests/test_legacy.py:52-84                      官方 k-grouped 用法（handle 构造）
  - tests/generators.py:378-412                     generate_m_grouped_contiguous（-1 padding 语义）
  - tests/generators.py:551-591                     generate_k_grouped_contiguous（host_ks 对齐）
"""
import random
import sys
import torch

import deep_gemm
from deep_gemm.legacy import (
    m_grouped_bf16_gemm_nt_contiguous_tl,
    a_fused_m_grouped_bf16_gemm_nt_contiguous_tl,
    a_fused_k_grouped_bf16_gemm_tn_contiguous_tl,
    b_fused_k_grouped_bf16_gemm_tn_contiguous_tl,
    get_mk_alignment_for_contiguous_layout,
)
from deep_gemm.testing import calc_diff, bench_kineto

torch.manual_seed(0)
random.seed(0)
ALIGN = get_mk_alignment_for_contiguous_layout()
DEV = 'cuda'

_checks = []


def align(x, a=ALIGN):
    return (x + a - 1) // a * a


def check(name, ok, detail=''):
    _checks.append((name, ok, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}  {detail}")


def bitexact(x, y):
    return torch.equal(x.contiguous().view(torch.uint8), y.contiguous().view(torch.uint8))


def max_abs_diff(x, y):
    return (x.float() - y.float()).abs().max().item()


def raises(fn):
    try:
        fn()
        return False, 'no exception'
    except AssertionError as e:
        s = str(e).splitlines()
        return True, (s[0][:70] if s and s[0] else 'AssertionError (无消息)')
    except Exception as e:  # noqa
        return True, type(e).__name__


print('=' * 78)
print('DeepGEMM legacy Triton MoE grouped GEMM — semantics + H100')
print(f'mk_alignment_for_contiguous_layout = {ALIGN}')
print('=' * 78)

# ---------------------------------------------------------------------------
# A) m-grouped dgrad 骨架：block 首行取组号 + padding 严格写零
# ---------------------------------------------------------------------------
print('\n[A] m_grouped_bf16_gemm_nt_contiguous_tl —— dgrad 骨架')
G, n, k = 4, 512, 256
actual_ms = [200, 130, 400, 65]
aligned_ms = [align(m) for m in actual_ms]
M = sum(aligned_ms)
print(f'    G={G} actual_ms={actual_ms} -> aligned={aligned_ms} M={M} n={n} k={k}')

a = torch.randn((M, k), device=DEV, dtype=torch.bfloat16)
b = torch.randn((G, n, k), device=DEV, dtype=torch.bfloat16)
m_indices = torch.full((M,), -1, device=DEV, dtype=torch.int32)
d = torch.empty((M, n), device=DEV, dtype=torch.bfloat16)
ref = torch.zeros((M, n), device=DEV, dtype=torch.bfloat16)
start = 0
for i, (am, alm) in enumerate(zip(actual_ms, aligned_ms)):
    ae, ale = start + am, start + alm
    m_indices[start:ae] = i          # 有效行 -> 组号
    a[ae:ale] = 0                    # 与官方 generator 一致：padding 激活清零（generators.py:406）
    ref[start:ale] = a[start:ale] @ b[i].t()
    start = ale

m_grouped_bf16_gemm_nt_contiguous_tl(a, b, d, m_indices)
pad_mask = m_indices < 0
check('m-grouped vs torch 参考 (calc_diff 判据)',
      calc_diff(d, ref) < 1e-5, f'calc_diff={calc_diff(d, ref):.3e}')
check('padding 行严格写零', bool((d[pad_mask] == 0).all().item()),
      f'pad rows={int(pad_mask.sum())}, nonzeros={int((d[pad_mask] != 0).sum())}')
check('有效行逐位等于参考', bitexact(d[~pad_mask], ref[~pad_mask]))

# 反例：M 非 align 倍数被断言拒绝（与现代 C++ 的段对齐是同一个数）
a_bad = torch.randn((1000, k), device=DEV, dtype=torch.bfloat16)
d_bad = torch.empty((1000, n), device=DEV, dtype=torch.bfloat16)
mi_bad = torch.zeros((1000,), device=DEV, dtype=torch.int32)
ok, msg = raises(lambda: m_grouped_bf16_gemm_nt_contiguous_tl(a_bad, b, d_bad, mi_bad))
check('M%align!=0 被拒 (m_grouped_gemm.py:92)', ok, msg)

# ---------------------------------------------------------------------------
# B) a_fused_m：m_row_indices 逐行 gather 源激活行
# ---------------------------------------------------------------------------
print('\n[B] a_fused_m_grouped_... —— A 逐行 gather（dgrad 的 scatter 版本）')
# identity -> 应逐位等于 m_grouped
m_row_identity = torch.arange(0, M, device=DEV, dtype=torch.int32)
d_id = torch.empty((M, n), device=DEV, dtype=torch.bfloat16)
a_fused_m_grouped_bf16_gemm_nt_contiguous_tl(a, b, d_id, (m_indices, m_row_identity))
check('identity m_row_indices 逐位等于 m_grouped', bitexact(d_id, d),
      f'max_abs_diff={max_abs_diff(d_id, d):.3e}')

# permutation: 输出行 row 读源激活行 perm[row]（跨组混合），坐实「逐行 gather」。
# 注意：kernel 用 **block 首行** 的 m_indices 决定 batch_id，首行为 -1 的整块会被写零
# （m_grouped_gemm.py:38-41）；故这里给整段都标组号（mi_full），避免整块被判空。
mi_full = torch.empty((M,), device=DEV, dtype=torch.int32)
start = 0
for i, alm in enumerate(aligned_ms):
    mi_full[start:start + alm] = i
    start += alm
perm = torch.randperm(M, device=DEV, dtype=torch.int64)
m_row_perm = perm.to(torch.int32)
d_perm = torch.empty((M, n), device=DEV, dtype=torch.bfloat16)
ref_perm = torch.zeros((M, n), device=DEV, dtype=torch.bfloat16)
start = 0
for i, alm in enumerate(aligned_ms):
    ale = start + alm
    ref_perm[start:ale] = (a[perm[start:ale]].float() @ b[i].t().float()).to(torch.bfloat16)
    start = ale
a_fused_m_grouped_bf16_gemm_nt_contiguous_tl(a, b, d_perm, (mi_full, m_row_perm))
check('置换 m_row_indices 复现任意逐行 gather', calc_diff(d_perm, ref_perm) < 1e-5,
      f'calc_diff={calc_diff(d_perm, ref_perm):.3e}, max_abs={max_abs_diff(d_perm, ref_perm):.3e}')

# block 首行规则：首行 m_indices=-1 的整块被写零（与 m_grouped 同规则）
mi_padblock = mi_full.clone()
mi_padblock[128:256] = -1      # 第二个 block 整块标 -1
d_pb = torch.full((M, n), 7.0, device=DEV, dtype=torch.bfloat16)
a_fused_m_grouped_bf16_gemm_nt_contiguous_tl(a, b, d_pb, (mi_padblock, m_row_identity))
check('block 首行 m_indices=-1 => 整块写零（首行定组号规则）',
      bool((d_pb[128:256] == 0).all().item()),
      f'nonzero={int((d_pb[128:256] != 0).sum())}')

ok, msg = raises(lambda: a_fused_m_grouped_bf16_gemm_nt_contiguous_tl(
    a, b, torch.empty((1000, n), device=DEV, dtype=torch.bfloat16),
    (torch.zeros((1000,), device=DEV, dtype=torch.int32),
     torch.zeros((1000,), device=DEV, dtype=torch.int32))))
check('m_indices.numel()%align!=0 被拒 (a_fused_m:83)', ok, msg)

# ---------------------------------------------------------------------------
# C) k-grouped wgrad（tn）：a_fused（gather A）vs b_fused（gather B）
# ---------------------------------------------------------------------------
print('\n[C] a_fused_k / b_fused_k_grouped_... —— wgrad K 轴分组（tn）')
Gk = 4
logical_ks = [256, 128, 384, 256]         # 每组真实 K，均为 ALIGN 倍数
host_ks = [align(x) for x in logical_ks]
total_k = sum(host_ks)
Mk, Nk = 256, 512
a_k = torch.randn((total_k, Mk), device=DEV, dtype=torch.bfloat16)   # [K_total, M]
b_k = torch.randn((total_k, Nk), device=DEV, dtype=torch.bfloat16)   # [K_total, N]
c_k = torch.randn((Gk, Mk, Nk), device=DEV, dtype=torch.float) * 4   # accumulate 初值 fp32
k_start = torch.zeros((Gk,), device=DEV, dtype=torch.int32)
k_end = torch.zeros((Gk,), device=DEV, dtype=torch.int32)
p = 0
for i, gk in enumerate(host_ks):
    k_start[i] = p
    k_end[i] = p + gk
    p += gk
k_indices_id = torch.arange(0, total_k, device=DEV, dtype=torch.int32)

ref_c = torch.empty((Gk, Mk, Nk), device=DEV, dtype=torch.float)
for i, gk in enumerate(host_ks):
    s, e = int(k_start[i]), int(k_end[i])
    ref_c[i] = a_k[s:e].float().T @ b_k[s:e].float() + c_k[i]

d_a = c_k.clone()
a_fused_k_grouped_bf16_gemm_tn_contiguous_tl(a_k, b_k, d_a, (k_indices_id, k_start, k_end), True)
check('a_fused_k(identity) 累积 vs A^T B + C', calc_diff(d_a, ref_c) < 1e-5,
      f'calc_diff={calc_diff(d_a, ref_c):.3e}')

d_b = c_k.clone()
b_fused_k_grouped_bf16_gemm_tn_contiguous_tl(a_k, b_k, d_b, (k_indices_id, k_start, k_end), True)
check('b_fused_k(identity) == a_fused_k（同一数学）', bitexact(d_a, d_b),
      f'max_abs={max_abs_diff(d_a, d_b):.3e}')

# 非 identity：坐实命名。a_fused 用 rows=k_indices 索引 **A**；b_fused 索引 **B**。
perm_k = torch.roll(torch.arange(0, total_k, device=DEV, dtype=torch.int64), 37).to(torch.int32)
d_ag = c_k.clone()
a_fused_k_grouped_bf16_gemm_tn_contiguous_tl(a_k, b_k, d_ag, (perm_k, k_start, k_end), True)
ref_ag = torch.empty((Gk, Mk, Nk), device=DEV, dtype=torch.float)
ref_bg = torch.empty((Gk, Mk, Nk), device=DEV, dtype=torch.float)
for i, gk in enumerate(host_ks):
    s, e = int(k_start[i]), int(k_end[i])
    rows = perm_k[s:e].long()
    ref_ag[i] = a_k[rows].float().T @ b_k[s:e].float() + c_k[i]   # A 被 perm 置换
    ref_bg[i] = a_k[s:e].float().T @ b_k[rows].float() + c_k[i]   # B 被 perm 置换
check('a_fused_k: 索引作用在 A（rows=k_indices[A]）', calc_diff(d_ag, ref_ag) < 1e-5,
      f'calc_diff={calc_diff(d_ag, ref_ag):.3e}')
d_bg = c_k.clone()
b_fused_k_grouped_bf16_gemm_tn_contiguous_tl(a_k, b_k, d_bg, (perm_k, k_start, k_end), True)
check('b_fused_k: 索引作用在 B（rows=k_indices[B]）', calc_diff(d_bg, ref_bg) < 1e-5,
      f'calc_diff={calc_diff(d_bg, ref_bg):.3e}')

# -1 索引 = 掩码该行贡献为 0（a_fused 掩 A 行，b_fused 掩 B 行）
k_masked = k_indices_id.clone()
k_masked[0:32] = -1
d_am = c_k.clone()
a_fused_k_grouped_bf16_gemm_tn_contiguous_tl(a_k, b_k, d_am, (k_masked, k_start, k_end), True)
ref_am = ref_c.clone()
ref_am[0] -= a_k[0:32].float().T @ b_k[0:32].float()
check('a_fused_k 的 -1 索引掩掉对应 A 行贡献', calc_diff(d_am, ref_am) < 1e-5,
      f'calc_diff={calc_diff(d_am, ref_am):.3e}')

# 契约：非 gather 侧 K 必须 %align==0
a_bad_k = torch.randn((1000, Mk), device=DEV, dtype=torch.bfloat16)  # 1000 % 128 != 0
b_bad_k = torch.randn((1000, Nk), device=DEV, dtype=torch.bfloat16)
d_badk = torch.empty((1, Mk, Nk), device=DEV, dtype=torch.float)
ok, msg = raises(lambda: a_fused_k_grouped_bf16_gemm_tn_contiguous_tl(
    a_bad_k, b_bad_k, d_badk,
    (torch.arange(0, 1000, device=DEV, dtype=torch.int32),
     torch.tensor([0], device=DEV, dtype=torch.int32),
     torch.tensor([1000], device=DEV, dtype=torch.int32)), True))
check('非 gather 侧 K%align!=0 被拒 (a_fused_k:82)', ok, msg)

# ---------------------------------------------------------------------------
# D) 空组：k_start >= k_end
# ---------------------------------------------------------------------------
print('\n[D] 空组（k_start>=k_end）行为')
ks_with_empty = [256, 0, 384, 256]          # 第 1 组空
hk = [align(x) for x in ks_with_empty]
tk = sum(hk)
a_e = torch.randn((tk, Mk), device=DEV, dtype=torch.bfloat16)
b_e = torch.randn((tk, Nk), device=DEV, dtype=torch.bfloat16)
ks_e = torch.zeros((4,), device=DEV, dtype=torch.int32)
ke_e = torch.zeros((4,), device=DEV, dtype=torch.int32)
p = 0
for i, g in enumerate(hk):
    ks_e[i] = p
    ke_e[i] = p + g
    p += g
c_e = torch.randn((4, Mk, Nk), device=DEV, dtype=torch.float)
ki_e = torch.arange(0, tk, device=DEV, dtype=torch.int32)
d_acc = c_e.clone()
a_fused_k_grouped_bf16_gemm_tn_contiguous_tl(a_e, b_e, d_acc, (ki_e, ks_e, ke_e), True)
check('ACC=True：空组 D 原样保留', bitexact(d_acc[1], c_e[1]),
      f'max_abs={max_abs_diff(d_acc[1], c_e[1]):.3e}')
d_noacc = c_e.clone()
a_fused_k_grouped_bf16_gemm_tn_contiguous_tl(a_e, b_e, d_noacc, (ki_e, ks_e, ke_e), False)
check('ACC=False：空组 D 写零', bool((d_noacc[1] == 0).all().item()),
      f'nonzeros={int((d_noacc[1] != 0).sum())}')

# ---------------------------------------------------------------------------
# E) 性能（Triton sm90 后端，H100）—— 与现代 C++ kernel 的直觉对照
# ---------------------------------------------------------------------------
print('\n[E] H100 性能（legacy Triton，Triton 在 sm90 上编译）')
Gp, mp, np_, kp = 8, 4096, 4096, 4096
actual = [mp // Gp] * Gp
al = [align(x) for x in actual]
Mp = sum(al)
ap = torch.randn((Mp, kp), device=DEV, dtype=torch.bfloat16)
bp = torch.randn((Gp, np_, kp), device=DEV, dtype=torch.bfloat16)
dp = torch.empty((Mp, np_), device=DEV, dtype=torch.bfloat16)
mip = torch.full((Mp,), -1, device=DEV, dtype=torch.int32)
s = 0
for i, (am, alm) in enumerate(zip(actual, al)):
    mip[s:s + am] = i
    s += alm
t = bench_kineto(lambda: m_grouped_bf16_gemm_nt_contiguous_tl(ap, bp, dp, mip),
                 'm_grouped_bf16_gemm_contiguous_tl_impl', suppress_kineto_output=True)
flops = 2 * sum(actual) * np_ * kp
print(f'    m-grouped: {Mp}x{np_}x{kp} (G={Gp})  {t * 1e6:7.1f} us  {flops / t / 1e12:6.1f} TFLOPS')

# k-grouped 性能
Gs, ms, ns = 4, 4096, 7168
ks_s = [4096] * Gs
hks = [align(x) for x in ks_s]
tks = sum(hks)
aps = torch.randn((tks, ms), device=DEV, dtype=torch.bfloat16)
bps = torch.randn((tks, ns), device=DEV, dtype=torch.bfloat16)
cs = torch.randn((Gs, ms, ns), device=DEV, dtype=torch.float)
kss = torch.zeros((Gs,), device=DEV, dtype=torch.int32)
kes = torch.zeros((Gs,), device=DEV, dtype=torch.int32)
p = 0
for i, g in enumerate(hks):
    kss[i] = p
    kes[i] = p + g
    p += g
kis = torch.arange(0, tks, device=DEV, dtype=torch.int32)
t = bench_kineto(lambda: b_fused_k_grouped_bf16_gemm_tn_contiguous_tl(aps, bps, cs, (kis, kss, kes), True),
                 'b_fused_k_grouped_bf16_gemm_contiguous_tl_impl', suppress_kineto_output=True)
flops = 2 * sum(hks) * ms * ns
print(f'    k-grouped: K={tks} {ms}x{ns} (G={Gs})  {t * 1e6:7.1f} us  {flops / t / 1e12:6.1f} TFLOPS')

# ---------------------------------------------------------------------------
print('\n' + '=' * 78)
n_pass = sum(1 for _, ok, _ in _checks if ok)
print(f'RESULT: {n_pass}/{len(_checks)} PASS')
for name, ok, _ in _checks:
    if not ok:
        print(f'  FAILED: {name}')
print('=' * 78)
sys.exit(0 if n_pass == len(_checks) else 1)
