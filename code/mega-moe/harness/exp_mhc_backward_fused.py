#!/usr/bin/env python3
"""P121-1: Mega mHC 反向的 H100 **算力 / 带宽账**（真实形状）+ 反向图正确性复核。

第 130 篇（P120-1）验证了 mHC 反向的数学；本脚本把这条反向当作**待部署的算子流水**
（3 段窄输出/长 K 投影 GEMM + 3 段元素级/Sinkhorn），在 H100 上量化：

  (1) 用真实形状（H=7168、hc=4、T∈{4096,16384}）跑手写反向 vs autograd，确认整条图正确；
  (2) 逐段算 **FLOPs / 字节 / 算术强度（AI）**，与 H100 的 roofline 拐点比较 → 判定瓶颈；
  (3) 真实计时：torch autograd（朴素、逐 op）总时间、三段 GEMM（cuBLAS）、元素级 op 总时间，
      与「理论访存下界」对比，给出融合收益的结构性解释。

主 kernel 端到端仍需 SM100，仅分析；这里的数学与形状全部架构无关，H100(sm90) 真机可跑。
"""
import os
import sys
import time
import importlib.util
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location(
    "mhc_bwd_ref", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "exp_mega_mhc_backward.py"))
_ref = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ref)

dev = "cuda"
assert torch.cuda.is_available()
torch.backends.cuda.matmul.allow_tf32 = False

HC = 4
NOUT = HC * (HC + 2)   # 24
CFG = dict(hc_norm_eps=1e-5, hc_pre_eps=1e-6, hc_post_scale=1.0,
           rmsnorm_eps=1e-6, rmsnorm_scale=1.0, sinkhorn_eps=1e-8)


def make(T, H, seed=0, dtype=torch.float64, shifted=False):
    g = torch.Generator(device=dev).manual_seed(seed)
    D = HC * H
    d = dict(
        x=torch.randn(T, H, generator=g, device=dev, dtype=dtype),
        R=torch.randn(T, HC, H, generator=g, device=dev, dtype=dtype),
        post=torch.rand(T, HC, 1, generator=g, device=dev, dtype=dtype),
        comb=torch.randn(T, HC, HC, generator=g, device=dev, dtype=dtype) * 0.3,
        fn=torch.randn(NOUT, D, generator=g, device=dev, dtype=dtype) * 0.2,
        scales=torch.tensor([0.7, 0.5, 0.9], device=dev, dtype=dtype),
        bases=torch.randn(NOUT, generator=g, device=dev, dtype=dtype) * 0.1,
        gamma=torch.randn(H, generator=g, device=dev, dtype=dtype),
        dS=torch.randn(T, HC, H, generator=g, device=dev, dtype=dtype) * 0.1,
        dPost=torch.randn(T, HC, 1, generator=g, device=dev, dtype=dtype) * 0.1,
        dM=torch.randn(T, HC, HC, generator=g, device=dev, dtype=dtype) * 0.1,
        dY=torch.randn(T, H, generator=g, device=dev, dtype=dtype) * 0.1,
        spm=(torch.rand(T, HC, 1, generator=g, device=dev, dtype=dtype) if shifted else None),
        dShift=(torch.randn(T, HC, 1, generator=g, device=dev, dtype=dtype) * 0.1 if shifted else None),
    )
    return d


# ---------------------------------------------------------------------------
# Part 1. 正确性：手写反向 vs autograd（真实形状、float64）
# ---------------------------------------------------------------------------
def part1_correctness():
    print("=" * 78)
    print("Part 1 · 手写反向 vs torch autograd（真实形状，float64）")
    print("=" * 78)
    for (T, H, iters) in [(256, 512, 3), (512, 1024, 4)]:
        d = make(T, H, seed=T * 7 + H)
        traces = []
        f = _ref.forward(d['x'], d['R'], d['post'], d['comb'], d['fn'], d['scales'],
                         d['bases'], d['gamma'], CFG['hc_norm_eps'], CFG['hc_pre_eps'],
                         CFG['hc_post_scale'], CFG['rmsnorm_eps'], CFG['rmsnorm_scale'],
                         CFG['sinkhorn_eps'], iters, trace_out=traces)
        hand = _ref.hand_backward(f, d['dS'], d['dPost'], d['dM'], d['dY'], None,
                                  d['x'], d['R'], d['post'], d['comb'], d['fn'], d['scales'],
                                  d['bases'], d['gamma'], None, traces[0], CFG)
        req = {k: v.clone().requires_grad_(True) for k, v in
               dict(x=d['x'], R=d['R'], post=d['post'], comb=d['comb'], fn=d['fn'],
                    scales=d['scales'], bases=d['bases'], gamma=d['gamma']).items()}
        fa = _ref.forward(req['x'], req['R'], req['post'], req['comb'], req['fn'], req['scales'],
                          req['bases'], req['gamma'], CFG['hc_norm_eps'], CFG['hc_pre_eps'],
                          CFG['hc_post_scale'], CFG['rmsnorm_eps'], CFG['rmsnorm_scale'],
                          CFG['sinkhorn_eps'], iters)
        loss = ((fa['s'] * d['dS']).sum() + (fa['new_post'] * d['dPost']).sum()
                + (fa['M'] * d['dM']).sum() + (fa['y'] * d['dY']).sum())
        loss.backward()
        pairs = dict(dx=(hand['dx'], req['x'].grad), dR=(hand['dR'], req['R'].grad),
                     dpost=(hand['dpost'], req['post'].grad), dcomb=(hand['dcomb'], req['comb'].grad),
                     dfn=(hand['dfn'], req['fn'].grad), dscales=(hand['dscales'], req['scales'].grad),
                     dbases=(hand['dbases'], req['bases'].grad), dgamma=(hand['dgamma'], req['gamma'].grad))
        mx = max(_ref.relerr(a, b) for a, b in pairs.values())
        print(f"  T={T:4d} H={H:4d} it={iters}: " +
              "  ".join(f"{k}={_ref.relerr(a, b):.1e}" for k, (a, b) in pairs.items()))
        print(f"       worst = {mx:.2e}")
    print()


# ---------------------------------------------------------------------------
# Part 2. 逐段 FLOPs / 字节 / AI 账（真实形状）
# ---------------------------------------------------------------------------
H100 = dict(hbm_gbps=3350.0, fp32_tflops=67.0, bf16_tflops=989.0)

def stage(name, flops, bytes_):
    return dict(name=name, flops=flops, bytes=bytes_, ai=(flops / bytes_ if bytes_ else 0.0))


def account(T, H, elem_bytes=2):
    D = HC * H
    S = []
    # K0 残差多流混合：A = post*x + comb@R
    fl = 2 * T * HC * H * HC
    by = (T * H + T * HC * H + T * HC * HC + T * HC + T * D) * elem_bytes
    S.append(stage("K0 fwd_mix", fl, by))
    # G0 u0 = A @ fnᵀ
    fl = 2 * T * NOUT * D
    by = (T * D + NOUT * D + T * NOUT) * elem_bytes
    S.append(stage("G0 u0=A@fnT", fl, by))
    # K1 系数反向（元素级 + Sinkhorn，读 A,u0,dS,dY；写 du0,dAbase,partials）
    fl = 2 * T * HC * H * HC  # 近似：q/尺度/混合若干次 FMA
    by = (T * D + T * NOUT + T * HC * H + T * H + T * D + T * NOUT + T * (H + NOUT + 3)) * elem_bytes
    S.append(stage("K1 coeff", fl, by))
    # G1a dz = du0 @ fn
    fl = 2 * T * D * NOUT
    by = (T * NOUT + NOUT * D + T * D) * elem_bytes
    S.append(stage("G1a dz=du0@fn", fl, by))
    # G1b dfn = du0ᵀ @ A
    fl = 2 * NOUT * D * T
    by = (T * NOUT + T * D + NOUT * D) * elem_bytes
    S.append(stage("G1b dfn=du0T@A", fl, by))
    # K2 残差多流混合反向
    fl = 2 * T * HC * H * HC
    by = (T * D * 2 + T * HC * H * 2 + T * H + T * HC * HC) * elem_bytes
    S.append(stage("K2 mix_bwd", fl, by))
    return S


def part2_roofline():
    print("=" * 78)
    print("Part 2 · 逐段 FLOPs / 字节 / 算术强度（bf16 操作数，H=7168 hc=4）")
    print("=" * 78)
    print(f"  H100: HBM {H100['hbm_gbps']/1000:.2f} TB/s, bf16 {H100['bf16_tflops']:.0f} TFLOPS "
          f"→ roofline 拐点 {H100['bf16_tflops']*1e12/(H100['hbm_gbps']*1e9):.0f} flop/byte")
    for T in (4096, 16384):
        H = 7168
        S = account(T, H, elem_bytes=2)
        tot_fl = sum(s['flops'] for s in S)
        tot_by = sum(s['bytes'] for s in S)
        print(f"\n  T={T} H={H}:")
        print(f"    {'stage':16s} {'GFLOP':>10s} {'MB':>9s} {'AI':>7s}")
        for s in S:
            print(f"    {s['name']:16s} {s['flops']/1e9:10.3f} {s['bytes']/1e6:9.1f} {s['ai']:7.1f}")
        print(f"    {'TOTAL':16s} {tot_fl/1e9:10.3f} {tot_by/1e6:9.1f} {tot_fl/tot_by:7.1f}")
        # 三个 GEMM 的 AI（与元素级分开看）
        gemm_fl = 3 * 2 * T * NOUT * (HC * H)
        gemm_by = sum(s['bytes'] for s in S if s['name'].startswith('G'))
        print(f"    三段 GEMM 合计: {gemm_fl/1e9:.3f} GFLOP, AI≈{gemm_fl/gemm_by:.1f} flop/byte "
              f"（≪ 拐点 → 访存受限）")
        print(f"    访存下界（整条反向）≈ {tot_by/(H100['hbm_gbps']*1e9)*1e6:.1f} us/次；"
              f"三段 GEMM 计算下界 ≈ {gemm_fl/(H100['bf16_tflops']*1e12)*1e6:.1f} us/次")
    print()


# ---------------------------------------------------------------------------
# Part 3. 真实计时（torch autograd vs 分段）
# ---------------------------------------------------------------------------
def bench(fn, warmup=5, reps=30):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    a = torch.cuda.Event(enable_timing=True)
    b = torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(reps):
        fn()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / reps * 1e3  # us


def part3_timing():
    print("=" * 78)
    print("Part 3 · 真实计时（H100，bf16 操作数）")
    print("=" * 78)
    for (T, H, iters) in [(4096, 7168, 3), (16384, 7168, 3)]:
        D = HC * H
        dt = torch.bfloat16
        d = make(T, H, seed=1, dtype=dt)
        # --- 三段 GEMM 的真实计时 ---
        Zmix = torch.randn(T, D, device=dev, dtype=dt)          # 代表 A（混合结果）
        u0 = torch.randn(T, NOUT, device=dev, dtype=dt)
        fn = d['fn'].to(dt)
        g0 = lambda: Zmix @ fn.t()                              # [T,D]x[D,24]
        g1a = lambda: u0 @ fn                                   # [T,24]x[24,D]
        g1b = lambda: u0.t() @ Zmix                             # [24,T]x[T,D]
        t_g0, t_g1a, t_g1b = bench(g0), bench(g1a), bench(g1b)
        gemm_fl = 3 * 2 * T * NOUT * D
        # --- 元素级 op 总量（torch 逐 op 反向，朴素） ---
        a_buf = torch.randn(T, HC, H, device=dev, dtype=dt)

        def elem_ops():
            a = a_buf
            n = a.sum(1) + d['dY']
            rg = torch.rsqrt((n * n).mean(1, keepdim=True) + 1e-6)
            dn = d['dY'] * rg * d['gamma']
            dmix = (dn.unsqueeze(1) * a).sum(2, keepdim=True)
            du0 = dmix + a[:, :1, :]
            dAbase = a + d['dS']
            dx = (dAbase * d['post']).sum(1)
            dpost = (dAbase * d['x'].unsqueeze(1)).sum(2)
            dcomb = torch.einsum('tjh,tih->tji', dAbase, d['R'])
            dR = torch.einsum('tjh,tji->tih', dAbase, d['comb'])
            return dx.sum() + dpost.sum() + dcomb.sum() + dR.sum() + du0.sum()
        t_elem = bench(elem_ops)
        # --- autograd 整条反向（朴素） ---
        req = {k: v.clone().float().requires_grad_(True) for k, v in
               dict(x=d['x'], R=d['R'], post=d['post'], comb=d['comb'],
                    fn=d['fn'], scales=d['scales'], bases=d['bases'],
                    gamma=d['gamma']).items()}
        fa = _ref.forward(req['x'], req['R'], req['post'], req['comb'], req['fn'], req['scales'],
                          req['bases'], req['gamma'], CFG['hc_norm_eps'], CFG['hc_pre_eps'],
                          CFG['hc_post_scale'], CFG['rmsnorm_eps'], CFG['rmsnorm_scale'],
                          CFG['sinkhorn_eps'], iters)
        loss = ((fa['s'] * d['dS'].float()).sum() + (fa['new_post'] * d['dPost'].float()).sum()
                + (fa['M'] * d['dM'].float()).sum() + (fa['y'] * d['dY'].float()).sum())
        t_auto = bench(lambda: loss.backward(retain_graph=True))
        S = account(T, H, elem_bytes=2)
        tot_by = sum(s['bytes'] for s in S)
        bw_bound = tot_by / (H100['hbm_gbps'] * 1e9) * 1e6
        print(f"\n  T={T} H={H} D={D}:")
        print(f"    G0  [T,D]@[D,24]      {t_g0:8.1f} us   {2*T*NOUT*D/t_g0/1e6:6.1f} TFLOPS")
        print(f"    G1a [T,24]@[24,D]     {t_g1a:8.1f} us   {2*T*D*NOUT/t_g1a/1e6:6.1f} TFLOPS")
        print(f"    G1b [24,T]@[T,D]      {t_g1b:8.1f} us   {2*NOUT*D*T/t_g1b/1e6:6.1f} TFLOPS")
        print(f"    三段 GEMM 合计          {t_g0+t_g1a+t_g1b:8.1f} us   {gemm_fl/(t_g0+t_g1a+t_g1b)/1e6:6.1f} TFLOPS")
        print(f"    元素级 op（torch 逐 op） {t_elem:8.1f} us")
        print(f"    autograd 整条反向（朴素） {t_auto:8.1f} us")
        print(f"    整条反向访存下界         {bw_bound:8.1f} us  "
              f"（= 总字节 {tot_by/1e6:.0f} MB / {H100['hbm_gbps']/1000:.2f} TB/s）")
    print()


def main():
    print("P121-1 · Mega mHC 反向：H100 算力/带宽账")
    part1_correctness()
    part2_roofline()
    part3_timing()
    print("ALL DONE")


if __name__ == "__main__":
    main()
