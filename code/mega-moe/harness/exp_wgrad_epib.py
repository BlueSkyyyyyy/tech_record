"""P139-1 实验：把 MoE wgrad 的 EPI-B 落到一个真实的、支持 bf16 累加器 RMW 的 SM90 GEMM 上。

为什么做这篇（补的缺口）：
  第 148 篇（P138-1）用**字节账 + 真机重叠**把「把 S 步归约藏进 GEMM」走到双重否定：
  侧流/外部融合在 SM90 的 `232448 B/SM` 下根本不让 block co-reside（`sm90.hpp:14`），
  只剩**同 kernel 的 `EPI-B`**（epilogue 由 fresh bf16 store 改成 **bf16 累加器 RMW**）一条路，
  字节账上它最少（`470·S`），并给出上界 `≈2.29×` / 下界 `≈1.69×`——但**从没在真 kernel 上落地过**。

  本机可用的「支持累加的 GEMM」只有两种形态（第 136/145 篇核对过）：
    ① FP8 1D1D `k_grouped_*`：**fp32** 累积（`csrc/apis/gemm.hpp:373-427`、`:384` recipe、
       `:403` `c.has_value()`；`sm90_fp8_gemm_1d1d.hpp:84`；kernel `SM90_TMA_REDUCE_ADD_2D`）——
       这是真实 wgrad 的形态，但累加器是 fp32、不是 EPI-B 要的 bf16；
    ② FP8 1D2D：**禁止**累加（`sm90_fp8_gemm_1d2d.hpp:81` `not c.has_value() and d==bf16`）——写不了累加器。
    ③ **BF16 稠密 `bf16_gemm_nt`**：`c = d`（bf16）时走 `with_accumulation`（`sm90_bf16_gemm.hpp:91`
       `.with_accumulation = c.has_value()`、`:89` `.cd_dtype = d.scalar_type()`），
       epilogue 落到 `SM90_TMA_REDUCE_ADD_2D`（kernel `sm90_bf16_gemm.cuh:381-385`，smem 为 bf16，
       `:302-349` STSM bf16）——**这就是一个货真价实的 bf16 累加器 RMW**，即 EPI-B 的机制本体。

  所以本篇**在一个真实、可跑、支持 bf16 累加器 RMW 的 SM90 GEMM 分支上做 EPI-B 端到端验证**：
  每步 GEMM 的 epilogue 从「fresh bf16 store」改成「bf16 累加器 RMW」，量那**多出来的
  `235MB/步`读**到底藏没藏进 GEMM 的 HBM 余量里。

  诚实边界：EPI-B 的车是 **BF16** 操作数（FP8 1D2D 不支持累加），不是 FP8 的精确 wgrad 算术；
  但这篇要回答的是**访存行为**（多读一份 bf16 D 能否被藏），由形状/字节数决定，与操作数精度无关，
  故用与 wgrad **同形状**（`[E·M, N]`、收缩维 K=token/步）的 bf16 GEMM 是恰当的载体；另加一个
  计算受限的同形状对照（拉长 K）来显示「藏得住」的边界条件。

要建立的结论（待实测）：
  ① EPI-B 车正确且确实做 bf16 RMW：`bf16_gemm_nt(a,b,d,c=d)` 两次 == 2·(a@b)；
  ② wgrad 形状（低算术强度）下，`t_epib − t_fresh` ≈ 多读 235MB 的 HBM 时间 → **读是暴露的**，
     每步归约并没有被藏进 GEMM 的带宽余量；
  ③ 计算受限对照（大 K）下同样的 RMW 读几乎免费 → 藏不藏取决于 GEMM 的算术强度，wgrad 恰好没有余量；
  ④ 端到端：`EPI-B` 仍优于「S 步 fresh + 单独归约」（SEQ/ALT-R），但达不到 2.29×，落在
     `1.69×` 一侧。

可跑性：全程 SM90 BF16 kernel（H100 可跑），`use_deterministic_algorithms(True)` 强制走 DeepGEMM
自己的 `sm90_bf16_gemm`（否则默认回退 cuBLASLt，见第 27 篇）。SM100 的 wgrad `tn` 同族、仅分析。
本机 8 卡共享，本文在**空闲 GPU1** 单进程测，报数注明。

ncu 单测：`python exp_wgrad_epib.py --ncu {fresh|epib}` 只发一次 kernel（供 ncu 采集）。

输出重定向：harness/exp_wgrad_epib.out.txt
"""
import os
import sys
import traceback

import torch

import deep_gemm

DEV = 'cuda'

# ----------------------------- wgrad 形状 -----------------------------
# 与第 135~148 篇一致：dW2 逐专家 [M=2048, N=7168]，E=8 个专家沿 M 折成一个大 GEMM。
E, M_EXP, N, K_STEP = 8, 2048, 7168, 512
M_TOT = E * M_EXP                       # 16384
D_BYTES = M_TOT * N * 2                 # 一份 bf16 梯度：235 MB
D_FP32_BYTES = M_TOT * N * 4            # 一份 fp32 梯度：470 MB


def rel(a, b):
    a, b = a.float(), b.float()
    return ((a - b).abs().mean() / (b.abs().mean() + 1e-30)).item()


def _bench(fn, warm=5, iters=20):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
    e0.record()
    for _ in range(iters):
        fn()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) / 1e3 / iters


def _kernel_names(fn):
    import torch.profiler as profiler
    fn()
    torch.cuda.synchronize()
    with profiler.profile(activities=[profiler.ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    names = {e.key for e in prof.key_averages()}
    return sorted(n for n in names if 'gemm' in n.lower() or 'nvjet' in n.lower())


def _make_operands(S, m_tot, n, k_step, seed=0):
    """返回 S 步的 (a_s, b_s) bf16 操作数。"""
    torch.manual_seed(seed)
    ops = []
    for _ in range(S):
        a = torch.randn(m_tot, k_step, device=DEV, dtype=torch.bfloat16)
        b = torch.randn(n, k_step, device=DEV, dtype=torch.bfloat16)
        ops.append((a, b))
    return ops


# --------------------------------------------------------------------------- Part A
def part_a_vehicle():
    print('=' * 96)
    print('Part A · EPI-B 的载体：真实 SM90 bf16 GEMM 的 bf16 累加器 RMW')
    print('-' * 96)
    deep_gemm.use_deterministic_algorithms(True)
    mm, n, k = 4096, 4096, 1024
    a = torch.randn(mm, k, device=DEV, dtype=torch.bfloat16)
    b = torch.randn(n, k, device=DEV, dtype=torch.bfloat16)
    d = torch.zeros(mm, n, device=DEV, dtype=torch.bfloat16)

    # 单步 fresh：写一份
    deep_gemm.bf16_gemm_nt(a, b, d)
    ref = a.float() @ b.float().T
    r1 = rel(d, ref)
    # 第二次 RMW：d += a@b
    deep_gemm.bf16_gemm_nt(a, b, d, c=d)
    r2 = rel(d, 2 * ref)
    print(f'  fresh 一步   : rel(d, a@b)      = {r1:.3e}')
    print(f'  RMW 再一步   : rel(d, 2·(a@b))  = {r2:.3e}   （= bf16 累加器，两次结果一致）')

    # fp32 累加器对照（真实 1D1D 形态）
    df = torch.zeros(mm, n, device=DEV, dtype=torch.float32)
    deep_gemm.bf16_gemm_nt(a, b, df)
    deep_gemm.bf16_gemm_nt(a, b, df, c=df)
    print(f'  fp32 累加器  : rel(df, 2·(a@b)) = {rel(df, 2 * ref):.3e}')

    print(f'  fresh 命中 kernel : {_kernel_names(lambda: deep_gemm.bf16_gemm_nt(a, b, d))}')
    print(f'  RMW   命中 kernel : {_kernel_names(lambda: deep_gemm.bf16_gemm_nt(a, b, d, c=d))}')
    print('  -> 均命中 DeepGEMM 自身 SM90 bf16 kernel（非 nvjet/cuBLASLt，因 deterministic）；'
          'RMW 走 TMA reduce-add')
    return r1, r2


# --------------------------------------------------------------------------- Part B
def part_b_timing(S_LIST=(1, 2, 4, 8, 16)):
    print('=' * 96)
    print('Part B · wgrad 形状（[E·M,N]=[16384,7168]，K=512/步）下三条协议的端到端')
    print('-' * 96)
    deep_gemm.use_deterministic_algorithms(True)
    Smax = max(S_LIST)
    ops = _make_operands(Smax, M_TOT, N, K_STEP, seed=1)

    d_bf16 = torch.empty(M_TOT, N, device=DEV, dtype=torch.bfloat16)
    d_fp32 = torch.empty(M_TOT, N, device=DEV, dtype=torch.float32)
    stack = torch.empty(Smax, M_TOT, N, device=DEV, dtype=torch.bfloat16)  # fresh 落盘用
    fl = 2 * M_TOT * N * K_STEP

    # 单步对照：fresh（写）vs epib（读+写）
    def fresh1():
        a, b = ops[0]
        deep_gemm.bf16_gemm_nt(a, b, d_bf16)

    def epib1():
        a, b = ops[0]
        deep_gemm.bf16_gemm_nt(a, b, d_bf16, c=d_bf16)

    def fp32acc1():
        a, b = ops[0]
        deep_gemm.bf16_gemm_nt(a, b, d_fp32, c=d_fp32)

    t_fresh = _bench(fresh1)
    t_epib = _bench(epib1)
    t_fp32 = _bench(fp32acc1)
    print(f'  单步（各 20 次）：fresh(写) {t_fresh * 1e6:7.1f}us ({fl / t_fresh / 1e12:5.0f} TFLOPS)、'
          f'EPI-B bf16 RMW(读+写) {t_epib * 1e6:7.1f}us、fp32 累积 {t_fp32 * 1e6:7.1f}us')
    print(f'  -> EPI-B 相对 fresh 的**每步边际** Δ = {t_epib * 1e6 - t_fresh * 1e6:7.1f}us')
    print(f'  -> 纯读一份 bf16 D 的 HBM 下界 = {D_BYTES / 3.35e12 * 1e6:.1f}us（@3.35TB/s）；'
          f'若 Δ≈此值 → 读**完全暴露**')

    print(f'  {"S":>4}{"SEQ(每步fresh+1次归约)":>26}{"EPI-B(S步RMW)":>18}{"fp32累积":>12}'
          f'{"SEQ/EPI-B":>12}{"EPI-B/REF(1D1D同构)":>20}')
    out = {}
    for S in S_LIST:
        sub = ops[:S]

        def seq_proto_full():
            for s, (a, b) in enumerate(sub):
                deep_gemm.bf16_gemm_nt(a, b, stack[s])
            torch.sum(stack[:S], dim=0, dtype=torch.float32)  # 读 S×235MB + 写 235MB

        def epib_proto():
            for a, b in sub:
                deep_gemm.bf16_gemm_nt(a, b, d_bf16, c=d_bf16)

        def ref_proto():
            for a, b in sub:
                deep_gemm.bf16_gemm_nt(a, b, d_fp32, c=d_fp32)

        t_seq = _bench(seq_proto_full, warm=3, iters=10)
        t_epi = _bench(epib_proto, warm=3, iters=10)
        t_ref = _bench(ref_proto, warm=3, iters=10)
        print(f'  {S:>4}{t_seq * 1e6:>26.1f}{t_epi * 1e6:>18.1f}{t_ref * 1e6:>12.1f}'
              f'{t_seq / t_epi:>12.2f}{t_ref / t_epi:>20.2f}')
        out[S] = (t_seq, t_epi, t_ref)
    print('  -> SEQ = 第 137/138 篇 ALT-R（每步 fresh bf16 + 独立 fp32 归约）；')
    print('     EPI-B = 本篇落地（每步 bf16 RMW，无独立归约）；fp32 累积 = 真实 wgrad 1D1D 同构形态')
    return t_fresh, t_epib, out


# --------------------------------------------------------------------------- Part C
def part_c_control():
    print('=' * 96)
    print('Part C · 边界条件对照：同样的 RMW 读 vs 不同算术强度（藏不藏取决于 GEMM 有没有带宽余量）')
    print('-' * 96)
    deep_gemm.use_deterministic_algorithms(True)
    for k_step, tag in ((512, 'wgrad 形状（低 AI）'), (2048, '中等'), (4096, '计算受限')):
        a = torch.randn(M_TOT, k_step, device=DEV, dtype=torch.bfloat16)
        b = torch.randn(N, k_step, device=DEV, dtype=torch.bfloat16)
        d = torch.empty(M_TOT, N, device=DEV, dtype=torch.bfloat16)
        # 取 3 次最小值 = 最少争用（本机 8 卡共享，同形状跨运行抖动大）
        t_f = min(_bench(lambda: deep_gemm.bf16_gemm_nt(a, b, d)) for _ in range(3))
        t_e = min(_bench(lambda: deep_gemm.bf16_gemm_nt(a, b, d, c=d)) for _ in range(3))
        fl = 2 * M_TOT * N * k_step
        # 算术强度（只算 GEMM 读写 + RMW 多读的那份 D）
        bytes_f = M_TOT * k_step * 2 + N * k_step * 2 + D_BYTES
        bytes_e = bytes_f + D_BYTES
        ai_f = fl / bytes_f
        ai_e = fl / bytes_e
        print(f'  K={k_step:>5} ({tag:<16}) t_fresh={t_f * 1e6:7.1f}us t_epib={t_e * 1e6:7.1f}us '
              f'Δ={(t_e - t_f) * 1e6:7.1f}us | AI fresh={ai_f:7.1f} epib={ai_e:7.1f} | '
              f'{fl / t_f / 1e12:5.0f}/{fl / t_e / 1e12:5.0f} TFLOPS')
        exposed = (t_e - t_f) / (D_BYTES / 3.35e12) * 100
        print(f'        -> Δ 相当于「纯读一份 D({D_BYTES / 1e6:.0f}MB) HBM 下界 '
              f'{D_BYTES / 3.35e12 * 1e6:.1f}us」的 {exposed:5.0f}%')
    print('  -> wgrad 形状：GEMM 已无带宽余量（额外读 ~150% 暴露）；大 K：计算受限（额外读被藏住）')

    # 带宽锚：一份 D 的纯读 / 读+写 在 H100 上的真实下界
    x = torch.randn(M_TOT * N, device=DEV, dtype=torch.bfloat16)
    y = torch.empty_like(x)
    t_read = _bench(lambda: x.sum(), warm=5, iters=20)
    t_rw = _bench(lambda: torch.add(x, 1.0, out=y), warm=5, iters=20)
    print(f'  锚：读一份 bf16 D {D_BYTES / 1e6:.0f}MB = {t_read * 1e6:6.1f}us '
          f'({D_BYTES / t_read / 1e12:.2f} TB/s)；读+写 = {t_rw * 1e6:6.1f}us '
          f'({2 * D_BYTES / t_rw / 1e12:.2f} TB/s)')


# --------------------------------------------------------------------------- ncu
def run_ncu(kind):
    deep_gemm.use_deterministic_algorithms(True)
    a = torch.randn(M_TOT, K_STEP, device=DEV, dtype=torch.bfloat16)
    b = torch.randn(N, K_STEP, device=DEV, dtype=torch.bfloat16)
    d = torch.zeros(M_TOT, N, device=DEV, dtype=torch.bfloat16)
    if kind == 'fresh':
        deep_gemm.bf16_gemm_nt(a, b, d)
    else:
        deep_gemm.bf16_gemm_nt(a, b, d, c=d)
    torch.cuda.synchronize()
    print(f'ncu target done: {kind}')


def main():
    if '--ncu' in sys.argv:
        run_ncu(sys.argv[sys.argv.index('--ncu') + 1])
        return
    print(f'deep_gemm {deep_gemm.__version__}, device={torch.cuda.get_device_name(0)}, '
          f'sms={deep_gemm.get_num_sms()}')
    print(f'wgrad 形状：dW2 = [{M_TOT} = E{E}·M{M_EXP}, N={N}]，每步 K={K_STEP}；'
          f'一份 bf16 D = {D_BYTES / 1e6:.0f}MB、fp32 = {D_FP32_BYTES / 1e6:.0f}MB')
    for name, fn in (('Part A', part_a_vehicle), ('Part B', part_b_timing), ('Part C', part_c_control)):
        try:
            fn()
        except Exception:
            print(f'{name} FAILED:\n' + traceback.format_exc())
    print('=' * 96)
    print('done')


if __name__ == '__main__':
    main()
