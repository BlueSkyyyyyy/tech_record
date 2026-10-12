#!/usr/bin/env python3
"""DeepGEMM GEMM API 的输入主序（nt/nn/tn/tt + grouped nn）与输出累积（early_return）契约（P36-1，H100 真机）。

参考源码（~/github/DeepGEMM，commit 057ca59）：
  csrc/apis/gemm.hpp:17-53    early_return：m/n==0 空转、k==0 退化 copy/zero、C==D 原地、C!=D 先拷、量化 D 不累积
  csrc/apis/gemm.hpp:78-133   fp8_fp4_gemm_nt（基准 nt 入口）
  csrc/apis/gemm.hpp:135-148  fp8_fp4_gemm_nn（B 转置）
  csrc/apis/gemm.hpp:150-164  fp8_fp4_gemm_tn（A、B 都转置）
  csrc/apis/gemm.hpp:166-179  fp8_fp4_gemm_tt（A 转置）
  csrc/apis/gemm.hpp:252-267  m_grouped_fp8_fp4_gemm_nn_contiguous（B 沿 (1,2) 转置）
  csrc/apis/gemm.hpp:537-565  bf16_gemm_nn / tn / tt
  csrc/apis/gemm.hpp:625-633  m_grouped_bf16_gemm_nn_contiguous
  csrc/utils/layout.hpp:17-45 major_check / get_major_type_ab（K-major=stride(-1)==1）/ check_major_type_cd
  csrc/utils/layout.hpp:42-45 fp8_fp4_requires_k_major（SM90 恒真；SM100 仅 FP4×FP4）
  csrc/utils/layout.hpp:66-73 check_ab_fp8_fp4（packed FP4 逻辑形状随 major 翻倍）

一句话结论：
  * **nt 是唯一“基准”实现**，nn/tn/tt 全是「把操作数转置后转调 nt」的薄壳（shape 变、数学不变）。
  * major 只由 `stride(-1)==1 ? K : MN` 决定；输出 D 强制 N-major（行主序）。
  * `early_return` 统一了 C/D 累积语义：m/n==0 空转、k==0 退化、C==D 原地、C!=D 先拷、量化 D 不累积。
  * SM90 的 FP8/FP4 要求 A/B 都 K-major（WGMMA 约束），所以 `fp8_fp4_gemm_nn` 在 H100 上必然被拒。

运行：bash harness/exp_gemm_major_accum.sh > harness/exp_gemm_major_accum.out.txt
"""
import sys

import torch
import torch.profiler as profiler

try:
    import deep_gemm
except Exception as e:  # pragma: no cover
    print("FATAL: cannot import deep_gemm:", e)
    sys.exit(1)

from deep_gemm.testing import calc_diff, get_arch_major

DEV = "cuda"
torch.manual_seed(0)

print(f"torch {torch.__version__} | {torch.cuda.get_device_name(0)} | "
      f"cc={torch.cuda.get_device_capability(0)} | arch_major={get_arch_major()} | "
      f"device_num_sms={deep_gemm.get_num_sms()}")
print()


def sec(title):
    print("=" * 78)
    print(title)
    print("=" * 78)


def rel(x, ref):
    x = x.float().flatten().double()
    ref = ref.float().flatten().double()
    return ((x - ref).norm() / (ref.norm() + 1e-30)).item()


def deterministic(on=True):
    deep_gemm.use_deterministic_algorithms(on)


# --------------------------------------------------------------------- A) bf16 nt/nn/tn/tt
sec("A) bf16 nt / nn / tn / tt：四个入口算同一数学 D = A @ B^T（转置薄壳）")

m, n, k = 512, 384, 256
A = torch.randn(m, k, device=DEV, dtype=torch.bfloat16)
B = torch.randn(n, k, device=DEV, dtype=torch.bfloat16)
ref = (A.float() @ B.float().mT).to(torch.bfloat16)

# 每种入口用「K-major(major=N)」与「MN-major(major=T，连续存储后转置)」各造一份，坐实两主序都可。
variants = {
    "nt": (A, B),
    "nn": (A, B.mT.contiguous()),                              # 用户给 [k,n]
    "tn": (A.mT.contiguous(), B.mT.contiguous()),              # 用户给 [k,m] 与 [k,n]
    "tt": (A.mT.contiguous(), B),                              # 用户给 [k,m]，B 不变
}
fn = {
    "nt": deep_gemm.bf16_gemm_nt,
    "nn": deep_gemm.bf16_gemm_nn,
    "tn": deep_gemm.bf16_gemm_tn,
    "tt": deep_gemm.bf16_gemm_tt,
}

deterministic(True)
outs = {}
# 传给内部 nt 的「生效主序」：转置只改 shape/stride，转置后新鲜连续张量变成 MN-major。
eff_major = {"nt": ("K", "K"), "nn": ("K", "MN"), "tn": ("MN", "MN"), "tt": ("MN", "K")}
print(f"{'entry':>4} | {'eff.a':>5} {'eff.b':>5} | {'passed a':>11} {'passed b':>11} | "
      f"{'calc_diff':>10} {'relerr':>10}")
for name, (a, b) in variants.items():
    d = torch.empty(m, n, device=DEV, dtype=torch.bfloat16)
    fn[name](a, b, d)
    torch.cuda.synchronize()
    outs[name] = d
    ea, eb = eff_major[name]
    print(f"{name:>4} | {ea:>5} {eb:>5} | {str(a.stride()):>11} {str(b.stride()):>11} | "
          f"{calc_diff(d, ref).item():>10.3e} {rel(d, ref):>10.3e}")
# nt 与其它三者逐位相等？
for name in ("nn", "tn", "tt"):
    eq = torch.equal(outs["nt"].view(torch.int16), outs[name].view(torch.int16))
    print(f"  nt vs {name:<3} 逐位相等 = {eq}")

# nt 自己也能吃 MN-major A/B（用 .T.contiguous().T 造「同 shape、换 stride」的视图形）
A_mn = A.T.contiguous().T      # shape [m,k]，stride (1,m) → MN-major
B_mn = B.T.contiguous().T      # shape [n,k]，stride (1,n) → MN-major
for tag, a_, b_ in (("nt(MN-A,K-B)", A_mn, B), ("nt(K-A,MN-B)", A, B_mn),
                    ("nt(MN-A,MN-B)", A_mn, B_mn)):
    d = torch.empty(m, n, device=DEV, dtype=torch.bfloat16)
    deep_gemm.bf16_gemm_nt(a_, b_, d)
    torch.cuda.synchronize()
    print(f"  {tag:<16} calc_diff={calc_diff(d, ref).item():.3e}  "
          f"与 nt 逐位相等={torch.equal(d.view(torch.int16), outs['nt'].view(torch.int16))}")
print("  → 四个入口数学一致（转置只改 shape/stride，不改数值）；nt 本身也支持两种主序。")
print()

# --------------------------------------------------------------------- B) major 检测与 K-major 契约
sec("B) major 检测（stride(-1)==1 → K-major）与 SM90 的 K-major 硬约束")

print("  get_major_type_ab 只看最后一维 stride（csrc/utils/layout.hpp:30-33）：")
for tag, t in (("A   [m,k] 连续", A),
               ("A   [m,k] MN-major(.T.contig().T)", A.T.contiguous().T),
               ("A.mT[k,m] 连续", A.mT.contiguous()),
               ("B   [n,k] 连续", B),
               ("B   [n,k] MN-major(.T.contig().T)", B.T.contiguous().T),
               ("B.mT[k,n] 连续", B.mT.contiguous())):
    print(f"    {tag:<16} stride={str(t.stride()):>12}  → "
          f"{'K-major' if t.stride(-1) == 1 else 'MN-major'}")

# fp8：SM90 要求 A/B 都 K-major；nn 会把 B 转成 MN-major，故必然被拒。
print()
print("  fp8_fp4_gemm_* 在 SM90 要求 A/B 都 K-major（fp8_fp4_requires_k_major，layout.hpp:42-45）：")
a8 = (torch.randn(m, k, device=DEV, dtype=torch.bfloat16).to(torch.float8_e4m3fn),
      torch.ones(m, k // 128, device=DEV, dtype=torch.float))
b8 = (torch.randn(n, k, device=DEV, dtype=torch.bfloat16).to(torch.float8_e4m3fn),
      torch.ones(n // 128, k // 128, device=DEV, dtype=torch.float))

d = torch.empty(m, n, device=DEV, dtype=torch.bfloat16)
try:
    deep_gemm.fp8_fp4_gemm_nt(a8, b8, d, disable_ue8m0_cast=True)
    torch.cuda.synchronize()
    print(f"    fp8 nt (A/B 都 K-major)      → 通过  relerr={rel(d, (a8[0].float() @ b8[0].float().mT).to(torch.bfloat16)):.3e}"
          f"  （SF 全 1，隔离出纯 FP8 舍入）")
except Exception as e:
    print(f"    fp8 nt                       → 异常 {type(e).__name__}: {str(e).strip().splitlines()[-1][:90]}")
try:
    b8_mn = (b8[0].mT.contiguous(), b8[1])  # [k,n] 连续 → nn 转置后 MN-major
    deep_gemm.fp8_fp4_gemm_nn(a8, b8_mn, d, disable_ue8m0_cast=True)
    torch.cuda.synchronize()
    print("    fp8 nn (B→MN-major)          → 未拒绝（不符合预期）")
except Exception as e:
    print(f"    fp8 nn (B→MN-major)          → 被拒 {type(e).__name__}: "
          f"{str(e).strip().splitlines()[-1][:90]}")
print()

# --------------------------------------------------------------------- C) bf16 grouped nn
sec("C) m_grouped bf16 nt_contiguous vs nn_contiguous（B 沿 (1,2) 转置）")

deep_gemm.set_mk_alignment_for_contiguous_layout(
    deep_gemm.get_theoretical_mk_alignment_for_contiguous_layout())
al = deep_gemm.get_mk_alignment_for_contiguous_layout()
G, exp_m, gn, gk = 2, al, 256, 512          # m = G*exp_m，组对齐 128
gm = G * exp_m
Ag = torch.randn(gm, gk, device=DEV, dtype=torch.bfloat16)
Bg = torch.randn(G, gn, gk, device=DEV, dtype=torch.bfloat16)
layout = torch.repeat_interleave(torch.arange(G, device=DEV, dtype=torch.int32), exp_m)
refg = torch.stack([Ag[i * exp_m:(i + 1) * exp_m].float() @ Bg[i].float().mT for i in range(G)]).reshape(gm, gn)
refg = refg.to(torch.bfloat16)

d_nt = torch.empty(gm, gn, device=DEV, dtype=torch.bfloat16)
deep_gemm.m_grouped_bf16_gemm_nt_contiguous(Ag, Bg, d_nt, layout)
d_nn = torch.empty(gm, gn, device=DEV, dtype=torch.bfloat16)
deep_gemm.m_grouped_bf16_gemm_nn_contiguous(Ag, Bg.transpose(1, 2).contiguous(), d_nn, layout)
torch.cuda.synchronize()
print(f"  alignment={al}  G={G} m={gm} n={gn} k={gk}")
print(f"  nt: major_b={'K' if Bg.stride(-1) == 1 else 'MN'}  calc_diff={calc_diff(d_nt, refg).item():.3e}  relerr={rel(d_nt, refg):.3e}")
print(f"  nn: major_b={'K' if Bg.transpose(1, 2).stride(-1) == 1 else 'MN'}  calc_diff={calc_diff(d_nn, refg).item():.3e}  relerr={rel(d_nn, refg):.3e}")
print(f"  nt vs nn 逐位相等 = {torch.equal(d_nt.view(torch.int16), d_nn.view(torch.int16))}")
print()

# --------------------------------------------------------------------- D) early_return / 累积
sec("D) 输出累积契约（early_return，csrc/apis/gemm.hpp:17-53）")

m2, n2, k2 = 256, 256, 128
A2 = torch.randn(m2, k2, device=DEV, dtype=torch.bfloat16)
B2 = torch.randn(n2, k2, device=DEV, dtype=torch.bfloat16)
base = (A2.float() @ B2.float().mT)  # fp32 参考乘积

# D1: 无 C → D = A@B^T
d1 = torch.empty(m2, n2, device=DEV, dtype=torch.bfloat16)
deep_gemm.bf16_gemm_nt(A2, B2, d1)
torch.cuda.synchronize()
print(f"  D1 无 C            : relerr={rel(d1, base):.3e}（纯乘积）")

# D2: 分离 C → D = C + A@B^T（early_return 先把 C 拷进 D）
C2 = torch.randn(m2, n2, device=DEV, dtype=torch.bfloat16)
d2 = torch.full((m2, n2), float("nan"), device=DEV, dtype=torch.bfloat16)
deep_gemm.bf16_gemm_nt(A2, B2, d2, c=C2)
torch.cuda.synchronize()
print(f"  D2 分离 C          : relerr={rel(d2, C2.float() + base):.3e}  "
      f"(C+AB)  vs  纯 AB relerr={rel(d2, base):.3e}")

# D3: 原地 C==D → D = D_old + A@B^T
d3 = C2.clone()
deep_gemm.bf16_gemm_nt(A2, B2, d3, c=d3)
torch.cuda.synchronize()
print(f"  D3 原地 C==D       : relerr={rel(d3, C2.float() + base):.3e}  "
      f"与 D2(分离C) 逐位相等 = {torch.equal(d2.view(torch.int16), d3.view(torch.int16))}")

# D4: k==0 & c=None → D 清零
d4 = torch.full((m2, n2), 3.14, device=DEV, dtype=torch.bfloat16)
deep_gemm.bf16_gemm_nt(A2[:, :0].contiguous(), B2[:, :0].contiguous(), d4)
torch.cuda.synchronize()
print(f"  D4 k==0, c=None    : D 全零 = {bool((d4 == 0).all())}")

# D5: k==0 & c=C → D = C
d5 = torch.full((m2, n2), float("nan"), device=DEV, dtype=torch.bfloat16)
deep_gemm.bf16_gemm_nt(A2[:, :0].contiguous(), B2[:, :0].contiguous(), d5, c=C2)
torch.cuda.synchronize()
print(f"  D5 k==0, c=C       : D==C 逐位 = {torch.equal(d5.view(torch.int16), C2.view(torch.int16))}")

# D6: m==0 → 整个 launch 空转（a/d 的 m 必须一致，否则先在 gemm.hpp:506 断言）
d6 = torch.empty(0, n2, device=DEV, dtype=torch.bfloat16)
deep_gemm.bf16_gemm_nt(A2[:0].contiguous(), B2, d6)
torch.cuda.synchronize()
print(f"  D6 m==0            : launch 空转、无异常（D 形状 [0,{n2}]）")

# D7: n==0 → 空转
d7 = torch.empty(m2, 0, device=DEV, dtype=torch.bfloat16)
deep_gemm.bf16_gemm_nt(A2, B2[:0].contiguous(), d7)
torch.cuda.synchronize()
print(f"  D7 n==0            : launch 空转、无异常（D 形状 [{m2},0]）")

# D8: C 与 D dtype 不符 → 拒（gemm.hpp:36）
try:
    deep_gemm.bf16_gemm_nt(A2, B2, torch.empty(m2, n2, device=DEV, dtype=torch.bfloat16),
                           c=torch.randn(m2, n2, device=DEV, dtype=torch.float))
    print("  D8 C dtype 不符     : 未拒绝（不符合预期）")
except Exception as e:
    print(f"  D8 C dtype 不符     : 被拒 {type(e).__name__}")

# D9: C 非行主序（.mT，stride(-1)!=1）→ 拒（check_major_type_cd，gemm.hpp:35）
try:
    Cbad = torch.randn(n2, m2, device=DEV, dtype=torch.bfloat16).mT
    deep_gemm.bf16_gemm_nt(A2, B2, torch.empty(m2, n2, device=DEV, dtype=torch.bfloat16), c=Cbad)
    print("  D9 C 非行主序       : 未拒绝（不符合预期）")
except Exception as e:
    print(f"  D9 C 非行主序       : 被拒 {type(e).__name__}")

# D10: 量化 D 不累积（源码断言 not sfd or not c）——fp8_fp4_gemm_nt 不接受 FP8 D（gemm.hpp:106）
try:
    dfp8 = torch.empty(m2, n2, device=DEV, dtype=torch.float8_e4m3fn)
    deep_gemm.fp8_fp4_gemm_nt(a8, b8, dfp8, disable_ue8m0_cast=True)
    print("  D10 bf16 入口给 FP8 D : 未拒绝（不符合预期）")
except Exception as e:
    print(f"  D10 bf16 入口给 FP8 D : 被拒 {type(e).__name__}（fp8_fp4_gemm_nt 只收 bf16/fp32 D）")
print()

# --------------------------------------------------------------------- E) 真实 kernel 名
sec("E) 抓真实发射的 kernel 名（torch.profiler）")


def kernel_names(fn_):
    fn_()
    torch.cuda.synchronize()
    with profiler.profile(activities=[profiler.ProfilerActivity.CUDA]) as prof:
        fn_()
        torch.cuda.synchronize()
    names = set()
    for ev in prof.events():
        if ev.device_type == profiler.DeviceType.CUDA and ev.name:
            if any(t in ev.name for t in ("nvjet", "gemv", "gemm", "sm90_bf16", "cutlass")):
                names.add(ev.name)
    return sorted(names)


dd = torch.empty(m, n, device=DEV, dtype=torch.bfloat16)
deterministic(False)
print("  deterministic=False（默认，bf16 稠密→cuBLASLt）:")
for nm in kernel_names(lambda: deep_gemm.bf16_gemm_nt(A, B, dd)):
    print(f"    {nm}")
deterministic(True)
print("  deterministic=True（强制 DeepGEMM）:")
for entry in ("nt", "nn", "tn", "tt"):
    a_, b_ = variants[entry]
    dd = torch.empty(m, n, device=DEV, dtype=torch.bfloat16)
    names = kernel_names(lambda a_=a_, b_=b_, dd=dd, entry=entry: fn[entry](a_, b_, dd))
    print(f"    {entry:<3}: {names[0] if names else '(none)'}")
deterministic(False)

print()
print("全域完成。")
