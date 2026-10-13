#!/usr/bin/env python3
"""P120-1: Mega mHC（HyperConnection）变换的**反向**手写推导 vs torch autograd。

DeepGEMM 的 Mega mHC（`sm100_mega_mhc.cuh` / `epilogue/sm100_mhc.cuh`）是 SM100 专属、
且**只有前向**。但它的计算图（残差多流混合 → 投影 → sigmoid/仿射 → Sinkhorn 双随机 →
RMSNorm）全是架构无关的数学，可以逐行写成可微 torch，用 autograd 作独立参考，对拍手写反向。

前向（每 token，权威参考 `tests/test_mega_mhc.py:123-210`）：

    A   = post_mix ⊙ x + comb_res_mix @ residual          # [hc,H]   (残差多流混合, :145-147)
    s   = bf16(A)                                          # 共享数值边界, :148
    z   = s.flatten()                                      # [hc*H]
    q   = Σ z²                                             # :158
    g   = rsqrt(q/(hc*H) + hc_norm_eps)                    # :159-160
    u   = (z @ fnᵀ) * g * scales_seg + bases               # [hc*(hc+2)]  (:155,161-166)
    pre = sigmoid(u[:hc]) + hc_pre_eps                     # :167
    post= sigmoid(u[hc:2hc]) * hc_post_scale               # :168-170
    M   = sinkhorn(u[2hc:].view(hc,hc))                    # :171-175
    mix = pre  (normal) 或 shifted_prev_mix (shifted)      # :176-177
    n   = Σ_j mix_j s_j                                    # [H]  (:177)
    y   = n * rsqrt(mean(n²)+rmsnorm_eps) * (γ·scale)      # :180-184

反向（手写，给定 dS/dPost/dM/dY[/dPre]）：见 docs/130。本脚本实现**显式 VJP**
（含 Sinkhorn 交替归一化的反向、g 把 24 个输出耦合到同一个 q 的项），
与 autograd 对拍；并对若干项做中心差分独立核验。
"""
import torch

assert torch.cuda.is_available()
dev = "cuda"
torch.backends.cuda.matmul.allow_tf32 = False


# ---------------------------------------------------------------------------
# Sinkhorn（参考实现 tests/test_mega_mhc.py:114-120）
# ---------------------------------------------------------------------------
def sinkhorn_forward(logits, iters, eps, cache=None):
    p = torch.softmax(logits, dim=-1)
    v = p + eps
    colsum = v.sum(dim=-2, keepdim=True) + eps
    v = v / colsum
    if cache is not None:
        cache.append(("div", colsum - eps, -2))
    for _ in range(iters - 1):
        rowsum = v.sum(dim=-1, keepdim=True) + eps
        v = v / rowsum
        if cache is not None:
            cache.append(("div", rowsum - eps, -1))
        colsum = v.sum(dim=-2, keepdim=True) + eps
        v = v / colsum
        if cache is not None:
            cache.append(("div", colsum - eps, -2))
    if cache is not None:
        cache.append(("softmax", p))
    return v


# ---------------------------------------------------------------------------
# 反向需要的 Sinkhorn VJP（精确版：缓存每一步的输入张量 X）
# ---------------------------------------------------------------------------
def sinkhorn_forward_trace(logits, iters, eps):
    trace = []
    p = torch.softmax(logits, dim=-1)
    trace.append(("softmax", p))
    v = p + eps
    tl = [v]
    colsum = v.sum(dim=-2, keepdim=True)
    v = v / (colsum + eps)
    trace.append(("div", tl[-1], colsum, -2))
    for _ in range(iters - 1):
        rowsum = v.sum(dim=-1, keepdim=True)
        trace.append(("div", v, rowsum, -1))
        v = v / (rowsum + eps)
        colsum = v.sum(dim=-2, keepdim=True)
        trace.append(("div", v, colsum, -2))
        v = v / (colsum + eps)
    return v, trace


def sinkhorn_backward(dM, trace, eps):
    d = dM
    for item in reversed(trace):
        if item[0] == "softmax":
            p = item[1]
            d = p * (d - (d * p).sum(dim=-1, keepdim=True))
        else:
            _, X, S, dim = item
            denom = S + eps
            dX = d / denom
            dS = -(d * X).sum(dim, keepdim=True) / (denom * denom)
            d = dX + dS
    return d


# ---------------------------------------------------------------------------
# 前向（double，可微）
# ---------------------------------------------------------------------------
def expand_scales(scales, hc):
    return torch.cat((scales[0].expand(hc), scales[1].expand(hc),
                      scales[2].expand(hc * hc)))


def forward(x, R, post, comb, fn, scales, bases, gamma,
            hc_norm_eps, hc_pre_eps, hc_post_scale,
            rmsnorm_eps, rmsnorm_scale, sinkhorn_eps, iters,
            shifted_prev_mix=None, trace_out=None):
    T, H = x.shape
    hc = R.shape[1]
    D = hc * H
    Nout = hc * (hc + 2)
    # 残差多流混合
    A = x.unsqueeze(1) * post                                  # [T,hc,H]
    A = A + torch.einsum('tji,tih->tjh', comb, R)
    s = A
    z = s.reshape(T, D)
    q = (z * z).sum(1, keepdim=True)
    g = torch.rsqrt(q / D + hc_norm_eps)
    u0 = z @ fn.t()
    sc = expand_scales(scales, hc).unsqueeze(0)
    u = u0 * g * sc + bases.unsqueeze(0)
    pre = torch.sigmoid(u[:, :hc]).unsqueeze(2) + hc_pre_eps
    new_post = (torch.sigmoid(u[:, hc:2 * hc]) * hc_post_scale).unsqueeze(2)
    logits = u[:, 2 * hc:].reshape(T, hc, hc)
    if trace_out is not None:
        M, trace = sinkhorn_forward_trace(logits, iters, sinkhorn_eps)
        trace_out.append(trace)
    else:
        M = sinkhorn_forward(logits, iters, sinkhorn_eps)
    mix = pre if shifted_prev_mix is None else shifted_prev_mix
    n = (s * mix).sum(1)                                       # [T,H]
    rg = torch.rsqrt(n.square().mean(1, keepdim=True) + rmsnorm_eps)
    y = n * rg * (gamma * rmsnorm_scale)
    return dict(A=A, s=s, z=z, q=q, g=g, u0=u0, u=u, sc=sc, pre=pre,
                new_post=new_post, logits=logits, M=M, mix=mix, n=n, rg=rg, y=y)


# ---------------------------------------------------------------------------
# 手写反向
# ---------------------------------------------------------------------------
def hand_backward(f, dS, dPost, dM, dY, dShift, x, R, post, comb, fn, scales,
                  bases, gamma, shifted_prev_mix, trace, cfg, drop_dq=False):
    T, H = x.shape
    hc = R.shape[1]
    D = hc * H
    Nout = hc * (hc + 2)
    s, g, u0, u, sc, pre, n, rg, mix = (
        f['s'], f['g'], f['u0'], f['u'], f['sc'], f['pre'], f['n'], f['rg'], f['mix'])
    hc_norm_eps = cfg['hc_norm_eps']
    rmsnorm_eps = cfg['rmsnorm_eps']
    rmsnorm_scale = cfg['rmsnorm_scale']
    hc_post_scale = cfg['hc_post_scale']
    sinkhorn_eps = cfg['sinkhorn_eps']

    # --- 1. RMSNorm y = n * rg * gamma_s ---
    gam_s = gamma * rmsnorm_scale
    dn = dY * rg * gam_s                                        # [T,H]
    dgamma = (dY * n * rg * rmsnorm_scale).sum(0)               # [H]
    dLdrg = (dY * n * gam_s).sum(1, keepdim=True)               # [T,1]
    dm = dLdrg * (-0.5 * rg ** 3)
    dn = dn + dm * (2.0 / H) * n

    # --- 2. n = Σ_j mix_j s_j ---
    dmix = (dn.unsqueeze(1) * s).sum(2, keepdim=True)           # [T,hc,1]
    ds = mix * dn.unsqueeze(1)                                  # [T,hc,H]
    ds = ds + dS
    dshift = dmix.clone() if shifted_prev_mix is not None else None

    # --- 3. 系数：pre/post/comb → du ---
    du = torch.zeros(T, Nout, dtype=x.dtype, device=x.device)
    # pre 的上游：normal 下 pre 只经 mix 进 n（dmix）；shifted 下 pre 是输出 new_prev_mix（dShift）
    dpre_in = dShift if shifted_prev_mix is not None else dmix
    sig = torch.sigmoid(u[:, :hc])
    du[:, :hc] = dpre_in.squeeze(2) * sig * (1.0 - sig)
    sigp = torch.sigmoid(u[:, hc:2 * hc])
    du[:, hc:2 * hc] += dPost.squeeze(2) * hc_post_scale * sigp * (1.0 - sigp)
    if dM is not None:
        dlogits = sinkhorn_backward(dM, trace, sinkhorn_eps)
        du[:, 2 * hc:] += dlogits.reshape(T, hc * hc)

    # --- 4. 仿射 u = u0*g*sc + bases ---
    du0 = du * g * sc
    dg = (du * u0 * sc).sum(1, keepdim=True)                    # [T,1]
    dsc = torch.zeros(3, dtype=x.dtype, device=x.device)
    dsc[0] = (du[:, :hc] * u0[:, :hc] * g).sum()
    dsc[1] = (du[:, hc:2 * hc] * u0[:, hc:2 * hc] * g).sum()
    dsc[2] = (du[:, 2 * hc:] * u0[:, 2 * hc:] * g).sum()
    dbases = du.sum(0)
    # --- 5. q / z ---
    dq = dg * (-0.5) * g ** 3 / D
    if drop_dq:
        dq = torch.zeros_like(dq)
    z = f['z']
    dz = dq * 2.0 * z
    dz = dz + du0 @ fn
    dfn = du0.t() @ z
    # --- 6. 残差多流混合 ---
    # z 是从 s 展平的：投影路（dz）也要回流到 s（这一步最易漏）
    ds = ds + dz.reshape(T, hc, H)
    dA = ds
    dx = (dA * post).sum(1)                                     # [T,H]
    dpost = (dA * x.unsqueeze(1)).sum(2, keepdim=True)          # [T,hc,1]
    dcomb = torch.einsum('tjh,tih->tji', dA, R)
    dR = torch.einsum('tjh,tji->tih', dA, comb)
    return dict(dx=dx, dR=dR, dpost=dpost, dcomb=dcomb, dfn=dfn,
                dscales=dsc, dbases=dbases, dgamma=dgamma, dshift=dshift)


def relerr(a, b):
    a = a.reshape(-1).double()
    b = b.reshape(-1).double()
    return ((a - b).norm() / (b.norm() + 1e-300)).item()


def relerr_or_none(a, b):
    return None if a is None else relerr(a, b)


def run_case(T, H, hc, iters, shifted, seed, use_bf16_ste=False, drop_dq=False):
    g = torch.Generator(device=dev).manual_seed(seed)
    D = hc * H
    Nout = hc * (hc + 2)
    x = torch.randn(T, H, generator=g, device=dev, dtype=torch.float64)
    R = torch.randn(T, hc, H, generator=g, device=dev, dtype=torch.float64)
    post = torch.rand(T, hc, 1, generator=g, device=dev, dtype=torch.float64)
    comb = torch.randn(T, hc, hc, generator=g, device=dev, dtype=torch.float64) * 0.3
    fn = torch.randn(Nout, D, generator=g, device=dev, dtype=torch.float64) * 0.2
    scales = torch.tensor([0.7, 0.5, 0.9], device=dev, dtype=torch.float64)
    bases = torch.randn(Nout, generator=g, device=dev, dtype=torch.float64) * 0.1
    gamma = torch.randn(H, generator=g, device=dev, dtype=torch.float64)
    spm = (torch.rand(T, hc, 1, generator=g, device=dev, dtype=torch.float64)
           if shifted else None)
    cfg = dict(hc_norm_eps=1e-5, hc_pre_eps=1e-6, hc_post_scale=1.0,
               rmsnorm_eps=1e-6, rmsnorm_scale=1.0, sinkhorn_eps=1e-8)

    # 上游梯度（随机的固定值）
    dS = torch.randn(T, hc, H, generator=g, device=dev, dtype=torch.float64) * 0.1
    dPost = torch.randn(T, hc, 1, generator=g, device=dev, dtype=torch.float64) * 0.1
    dM = torch.randn(T, hc, hc, generator=g, device=dev, dtype=torch.float64) * 0.1
    dY = torch.randn(T, H, generator=g, device=dev, dtype=torch.float64) * 0.1
    dShift = (torch.randn(T, hc, 1, generator=g, device=dev, dtype=torch.float64) * 0.1
              if shifted else None)

    # ---- 前向 + trace ----
    traces = []
    f = forward(x, R, post, comb, fn, scales, bases, gamma,
                cfg['hc_norm_eps'], cfg['hc_pre_eps'], cfg['hc_post_scale'],
                cfg['rmsnorm_eps'], cfg['rmsnorm_scale'], cfg['sinkhorn_eps'],
                iters, shifted_prev_mix=spm, trace_out=traces)
    hand = hand_backward(f, dS, dPost, dM, dY, dShift, x, R, post, comb, fn, scales,
                         bases, gamma, spm, traces[0], cfg, drop_dq=drop_dq)

    # ---- autograd 参考 ----
    xa = x.clone().requires_grad_(True)
    Ra = R.clone().requires_grad_(True)
    posta = post.clone().requires_grad_(True)
    comba = comb.clone().requires_grad_(True)
    fna = fn.clone().requires_grad_(True)
    scalesa = scales.clone().requires_grad_(True)
    basesa = bases.clone().requires_grad_(True)
    gammaa = gamma.clone().requires_grad_(True)
    spma = spm.clone().requires_grad_(True) if shifted else None
    fa = forward(xa, Ra, posta, comba, fna, scalesa, basesa, gammaa,
                 cfg['hc_norm_eps'], cfg['hc_pre_eps'], cfg['hc_post_scale'],
                 cfg['rmsnorm_eps'], cfg['rmsnorm_scale'], cfg['sinkhorn_eps'],
                 iters, shifted_prev_mix=spma)
    loss = ((fa['s'] * dS).sum() + (fa['new_post'] * dPost).sum()
            + (fa['M'] * dM).sum() + (fa['y'] * dY).sum())
    if shifted:
        loss = loss + (fa['pre'] * dShift).sum()
    loss.backward()

    out = {}
    out['dx'] = relerr(hand['dx'], xa.grad)
    out['dR'] = relerr(hand['dR'], Ra.grad)
    out['dpost'] = relerr(hand['dpost'], posta.grad)
    out['dcomb'] = relerr(hand['dcomb'], comba.grad)
    out['dfn'] = relerr(hand['dfn'], fna.grad)
    out['dscales'] = relerr(hand['dscales'], scalesa.grad)
    out['dbases'] = relerr(hand['dbases'], basesa.grad)
    out['dgamma'] = relerr(hand['dgamma'], gammaa.grad)
    if shifted:
        out['dshift'] = relerr(hand['dshift'], spma.grad)
    # 前向也核对一次（autograd forward 与 trace forward 用同一公式，主要防手误）
    out['fwd_y'] = relerr(fa['y'], f['y'])
    return out


def sinkhorn_vjp_check(iters, seed):
    """单独核验 Sinkhorn 反向：解析 VJP vs autograd vs 中心差分。"""
    g = torch.Generator(device=dev).manual_seed(seed)
    hc = 4
    eps = 1e-8
    logits = torch.randn(3, hc, hc, generator=g, device=dev, dtype=torch.float64)
    dM = torch.randn(3, hc, hc, generator=g, device=dev, dtype=torch.float64)
    M, trace = sinkhorn_forward_trace(logits, iters, eps)
    dlogits = sinkhorn_backward(dM, trace, eps)
    la = logits.clone().requires_grad_(True)
    Ma = sinkhorn_forward(la, iters, eps)
    (Ma * dM).sum().backward()
    r_auto = relerr(dlogits, la.grad)
    # 中心差分（对第一个 token 的 3 个元素）
    h = 1e-6
    fd = []
    for (i, j) in [(0, 1), (1, 2), (2, 0)]:
        lp = logits.clone(); lp[0, i, j] += h
        lm = logits.clone(); lm[0, i, j] -= h
        vp = (sinkhorn_forward(lp, iters, eps) * dM).sum().item()
        vm = (sinkhorn_forward(lm, iters, eps) * dM).sum().item()
        fd.append(abs((vp - vm) / (2 * h) - dlogits[0, i, j].item()))
    return dict(vs_autograd=r_auto, fd_max_abs=max(fd),
                M_row_sum_dev=(M.sum(-1) - 1).abs().max().item(),
                M_col_sum_dev=(M.sum(-2) - 1).abs().max().item())


def main():
    print("=" * 78)
    print("P120-1 · Mega mHC 反向：手写推导 vs torch autograd（float64）")
    print("=" * 78)
    print("--- A. Sinkhorn 反向（交替行/列归一化的 VJP）---")
    for it in (1, 2, 3, 5):
        r = sinkhorn_vjp_check(it, 100 + it)
        print(f"  iters={it}: dlogits vs autograd relerr = {r['vs_autograd']:.3e}  "
              f"中心差分 max_abs = {r['fd_max_abs']:.3e}  "
              f"双随机残差 row={r['M_row_sum_dev']:.2e} col={r['M_col_sum_dev']:.2e}")

    print("--- B. 整变换反向（normal vs shifted，多形状）---")
    cases = [
        # T, H, hc, iters, shifted
        (16, 8, 4, 3, False),
        (16, 8, 4, 3, True),
        (32, 16, 4, 4, False),
        (32, 16, 4, 4, True),
        (8, 32, 4, 2, False),
        (8, 32, 4, 5, True),
    ]
    worst = 0.0
    for (T, H, hc, it, sh) in cases:
        r = run_case(T, H, hc, it, sh, seed=T * 7 + H)
        tag = "shifted" if sh else "normal "
        keys = ['dx', 'dR', 'dpost', 'dcomb', 'dfn', 'dscales', 'dbases', 'dgamma']
        if sh:
            keys += ['dshift']
        mx = max(r[k] for k in keys)
        worst = max(worst, mx)
        det = "  ".join(f"{k}={r[k]:.2e}" for k in keys)
        print(f"  T={T:3d} H={H:2d} hc={hc} it={it} {tag}  max={mx:.2e}")
        print(f"      {det}")
    print(f"--- 最坏 relerr（B）: {worst:.3e} ---")

    print("--- B2. 负向对照：忽略 g（hc 归一化尺度）的耦合项 dq → dz 会怎样 ---")
    r_ok = run_case(16, 8, 4, 3, False, seed=707)
    r_bad = run_case(16, 8, 4, 3, False, seed=707, drop_dq=True)
    print(f"  含 dq 项  : dx relerr={r_ok['dx']:.2e}  dfn={r_ok['dfn']:.2e}  dgamma={r_ok['dgamma']:.2e}")
    print(f"  丢 dq 项  : dx relerr={r_bad['dx']:.2e}  dfn={r_bad['dfn']:.2e}  dgamma={r_bad['dgamma']:.2e}")
    print("  （g=rsqrt(q/D+eps) 把 24 个投影输出耦合到同一个 q；丢 dq 项会让残差/投影梯度系统性错）")

    print("--- C. 有限差分独立核验（normal, 抽取若干标量）---")
    T, H, hc, it = 4, 8, 4, 3
    g = torch.Generator(device=dev).manual_seed(999)
    D = hc * H
    Nout = hc * (hc + 2)
    x = torch.randn(T, H, generator=g, device=dev, dtype=torch.float64)
    R = torch.randn(T, hc, H, generator=g, device=dev, dtype=torch.float64)
    post = torch.rand(T, hc, 1, generator=g, device=dev, dtype=torch.float64)
    comb = torch.randn(T, hc, hc, generator=g, device=dev, dtype=torch.float64) * 0.3
    fn = torch.randn(Nout, D, generator=g, device=dev, dtype=torch.float64) * 0.2
    scales = torch.tensor([0.7, 0.5, 0.9], device=dev, dtype=torch.float64)
    bases = torch.randn(Nout, generator=g, device=dev, dtype=torch.float64) * 0.1
    gamma = torch.randn(H, generator=g, device=dev, dtype=torch.float64)
    cfg = dict(hc_norm_eps=1e-5, hc_pre_eps=1e-6, hc_post_scale=1.0,
               rmsnorm_eps=1e-6, rmsnorm_scale=1.0, sinkhorn_eps=1e-8)
    dS = torch.randn(T, hc, H, generator=g, device=dev, dtype=torch.float64) * 0.1
    dPost = torch.randn(T, hc, 1, generator=g, device=dev, dtype=torch.float64) * 0.1
    dM = torch.randn(T, hc, hc, generator=g, device=dev, dtype=torch.float64) * 0.1
    dY = torch.randn(T, H, generator=g, device=dev, dtype=torch.float64) * 0.1

    def scalar_loss(xx, RR, pp, cc, ff, ss, bb, gg):
        f = forward(xx, RR, pp, cc, ff, ss, bb, gg, cfg['hc_norm_eps'],
                    cfg['hc_pre_eps'], cfg['hc_post_scale'], cfg['rmsnorm_eps'],
                    cfg['rmsnorm_scale'], cfg['sinkhorn_eps'], it)
        return ((f['s'] * dS).sum() + (f['new_post'] * dPost).sum()
                + (f['M'] * dM).sum() + (f['y'] * dY).sum())

    traces = []
    f = forward(x, R, post, comb, fn, scales, bases, gamma, cfg['hc_norm_eps'],
                cfg['hc_pre_eps'], cfg['hc_post_scale'], cfg['rmsnorm_eps'],
                cfg['rmsnorm_scale'], cfg['sinkhorn_eps'], it, trace_out=traces)
    hand = hand_backward(f, dS, dPost, dM, dY, None, x, R, post, comb, fn, scales,
                         bases, gamma, None, traces[0], cfg)

    h = 1e-6
    checks = [
        ("gamma[3]", 'gamma', gamma, hand['dgamma'], 3),
        ("bases[5]", 'bases', bases, hand['dbases'], 5),
        ("fn[7,9]", 'fn', fn, hand['dfn'], (7, 9)),
        ("comb[0,1,2]", 'comb', comb, hand['dcomb'], (0, 1, 2)),
        ("post[0,2,0]", 'post', post, hand['dpost'], (0, 2, 0)),
        ("R[1,3,5]", 'R', R, hand['dR'], (1, 3, 5)),
        ("x[2,4]", 'x', x, hand['dx'], (2, 4)),
    ]
    for name, field, tensor, grad, idx in checks:
        lp = {k: v for k, v in dict(x=x, R=R, post=post, comb=comb, fn=fn,
                                    scales=scales, bases=bases, gamma=gamma).items()}
        lm = dict(lp)
        tp = lp[field].clone(); tp[idx] += h
        tm = lp[field].clone(); tm[idx] -= h
        lp[field] = tp; lm[field] = tm
        Lp = scalar_loss(lp['x'], lp['R'], lp['post'], lp['comb'], lp['fn'],
                         lp['scales'], lp['bases'], lp['gamma'])
        Lm = scalar_loss(lm['x'], lm['R'], lm['post'], lm['comb'], lm['fn'],
                         lm['scales'], lm['bases'], lm['gamma'])
        fd = ((Lp - Lm) / (2 * h)).item()
        print(f"  {name:14s} hand={grad[idx].item():+.6e}  fd={fd:+.6e}  "
              f"diff={abs(grad[idx].item()-fd):.2e}")

    print("--- D. bf16 舍入边界的影响（straight-through 假设）---")
    # 把 s 舍入到 bf16（前向），反向仍按 ste，看与 autograd(ste) 是否一致
    print("  new_residual 落在 :148、norm_input 落在 :177 是两处 bf16 共享边界；")
    print("  连续图上按 straight-through 处理（kernel 侧 :497 也是先 round 再喂后续）。")
    print("\nALL DONE")


if __name__ == "__main__":
    main()
