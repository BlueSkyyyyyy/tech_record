"""P6-6: B0 门控重算最小可运行 kernel（前向 g_cum/β + 反向前缀和 + 门控反向）。

FlashKDA 前向 K1 只把 ``kd/qd/kr/gt/INV/Mqk`` 六个字段落进 workspace（``csrc/smxx/utils.cuh:70-76``），
**原始门 logits 与 beta logits 没有落盘**（K1 只在 smem 里用过就丢了，`fwd_kernel1.cuh:306-331`）。
所以目标卡 CUDA 反向需要一个轻量 **B0 重算 kernel**：读原始 ``g``(logits)/``beta``/``A_log``/``dt_bias``，
重算 ``beta=σ(beta_raw)`` 与 ``g_cum``（块内 log2 累计衰减），以及反向的门控链
（反向前缀和 + ``dg→dg_logits/dA/dbias``）。本脚本用 Triton 写出这个最小 kernel 并与
torch 参考 / FLA 对应 kernel / autograd / FlashKDA K1 workspace 对拍。

四个部分：
  Part A  B0 前向重算 ``g_cum``/``beta`` vs 纯 torch 参考（公式直算），多 shape/seed/非整除 T。
  Part B  B0 前向重算 vs FLA ``kda_gate_chunk_cumsum``（g_cum）+ ``fused_beta_sigmoid``（β）。
  Part C  B0 反向前缀和（APPLY_GATE=False, REVERSE=True）vs FLA ``chunk_local_cumsum(reverse=True)``。
  Part D  B0 门控反向 vs FLA ``kda_gate_bwd`` vs ``torch.autograd``（同一上游 ``d g_cum``）。
  Part E  交叉验证：把 B0 的 ``exp2(g_cum[-1])`` 对上 **FlashKDA K1 workspace 的 kGTotal（E^tot）**
          （唯一能直接证明 B0 复现了前向 kernel 当场算的门控的地方）。

约定（与 FlashKDA / FLA 一致）：
  * ``g``: [B,T,H,D] 门 logits（bf16 输入）；``beta``: [B,T,H] logits；
  * ``A_log``: [H] fp32；``dt_bias``: [H,D] fp32；``lower_bound`` 典型 -5；
  * ``inner = exp(A_log) * (g + dt_bias)``；``gate = lower_bound * sigmoid(inner)``；
  * ``g_cum = local_cumsum_{chunk}(gate) * log2e``（**块内**累计，单位 log2，<=0）。

跑法（kernel_lab 容器内，绝对路径）：
  python ~/proj/tech_record/code/flash-kda/harness/check_bwd_b0_gate.py
"""

import math

import torch
import triton
import triton.language as tl

LOG2E = 1.4426950408889634
RCP_LN2 = 1.0 / math.log(2.0)  # == LOG2E, FLA 用的常量名


# ---------------------------------------------------------------------------
# B0 前向重算 kernel：gate 激活 + 块内（反）前缀和 + 可选 beta sigmoid
# ---------------------------------------------------------------------------
@triton.jit
def _b0_gate_fwd_kernel(
    s, A_log, dt_bias, beta, o, gbeta, scale, lower_bound, T,
    H: tl.constexpr, S: tl.constexpr, BT: tl.constexpr,
    APPLY_GATE: tl.constexpr, REVERSE: tl.constexpr,
    HAS_BIAS: tl.constexpr, HAS_BETA: tl.constexpr,
):
    i_t = tl.program_id(0).to(tl.int64)
    i_bh = tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H

    o_t = i_t * BT + tl.arange(0, BT)
    o_s = tl.arange(0, S)
    m = o_t[:, None] < T

    base = (i_b * T * H + i_h) * S
    p_s = s + base + o_t[:, None] * (H * S) + o_s[None, :]
    b = tl.load(p_s, mask=m, other=0.0).to(tl.float32)

    if HAS_BIAS:
        b_bias = tl.load(dt_bias + i_h * S + o_s).to(tl.float32)
        b = b + b_bias[None, :]

    if APPLY_GATE:
        b_A = tl.load(A_log + i_h).to(tl.float32)
        gate = lower_bound * tl.sigmoid(tl.exp(b_A) * b)
    else:
        gate = b

    if REVERSE:
        acc = tl.cumsum(gate, axis=0, reverse=True)
    else:
        acc = tl.cumsum(gate, axis=0)
    acc = acc * scale

    p_o = o + base + o_t[:, None] * (H * S) + o_s[None, :]
    tl.store(p_o, acc, mask=m)

    if HAS_BETA:
        bbase = i_b * T * H + i_h
        p_b = beta + bbase + o_t * H
        m_t = o_t < T
        b_b = tl.load(p_b, mask=m_t, other=0.0).to(tl.float32)
        p_gb = gbeta + bbase + o_t * H
        tl.store(p_gb, tl.sigmoid(b_b), mask=m_t)


# ---------------------------------------------------------------------------
# B0 门控反向 kernel：给定 dyg = dL/dgate（已反向前缀和），回传 dg_logits/dA
# 逐行对应 FLA `kda_gate_bwd_kernel`（`fla/ops/kda/gate.py`，USE_LOWER_BOUND 分支）。
# ---------------------------------------------------------------------------
@triton.jit
def _b0_gate_bwd_kernel(
    g, A_log, dt_bias, dyg, dg, dA, lower_bound, T,
    H: tl.constexpr, D: tl.constexpr, BT: tl.constexpr, BD: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    i_t = tl.program_id(0).to(tl.int64)
    i_h = tl.program_id(1)

    o_t = i_t * BT + tl.arange(0, BT)
    o_d = tl.arange(0, BD)
    m_t = o_t < T
    m = m_t[:, None] & (o_d[None, :] < D)

    p_g = g + i_h * D + o_t[:, None] * (H * D) + o_d[None, :]
    p_dyg = dyg + i_h * D + o_t[:, None] * (H * D) + o_d[None, :]
    p_dg = dg + i_h * D + o_t[:, None] * (H * D) + o_d[None, :]

    b = tl.load(p_g, mask=m, other=0.0).to(tl.float32)
    b_dyg = tl.load(p_dyg, mask=m, other=0.0).to(tl.float32)
    if HAS_BIAS:
        o_b = i_h * D + tl.arange(0, BD)
        b = b + tl.load(dt_bias + o_b, mask=o_b < H * D, other=0.0).to(tl.float32)

    b_A = tl.exp(tl.load(A_log + i_h).to(tl.float32))
    inner = b_A * b
    sig = tl.sigmoid(inner)
    dsig = sig * (1.0 - sig)
    b_dg = b_dyg * (lower_bound * dsig) * b_A
    b_dA = tl.sum(tl.sum(b_dg * b, 1), 0)

    tl.store(p_dg, b_dg, mask=m)
    tl.store(dA + i_t * H + i_h, b_dA)


def b0_gate_fwd(g, A_log, dt_bias, beta, lower_bound, chunk, scale=LOG2E,
                reverse=False, apply_gate=True, bs=None):
    """B0 前向重算：返回 (g_cum_or_prefix, gbeta)。layout [B,T,H,D]/[B,T,H]。"""
    B, T, H, D = g.shape
    g = g.contiguous()
    o = torch.empty_like(g, dtype=torch.float32)
    gbeta = None
    if beta is not None:
        gbeta = torch.empty_like(beta, dtype=torch.float32)
    S = D if bs is None else bs
    grid = (triton.cdiv(T, chunk), B * H)
    _b0_gate_fwd_kernel[grid](
        g, A_log, dt_bias, beta, o, gbeta, scale, lower_bound, T,
        H=H, S=D, BT=chunk, APPLY_GATE=apply_gate, REVERSE=reverse,
        HAS_BIAS=dt_bias is not None, HAS_BETA=beta is not None,
    )
    return o, gbeta


def b0_gate_bwd(g, A_log, dt_bias, dyg, lower_bound, BT=32):
    """B0 门控反向：dyg 已反向前缀和（dL/dgate），返回 (dg_logits, dA, dbias)。"""
    B, T, H, D = g.shape
    g2 = g.contiguous().reshape(B * T, H, D)
    dyg2 = dyg.contiguous().reshape(B * T, H, D)
    TT = B * T
    NT = triton.cdiv(TT, BT)
    dg = torch.empty_like(g2, dtype=torch.float32)
    dA_partial = torch.empty(NT, H, dtype=torch.float32, device=g.device)
    _b0_gate_bwd_kernel[(NT, H)](
        g2, A_log, dt_bias, dyg2, dg, dA_partial, lower_bound, TT,
        H=H, D=D, BT=BT, BD=triton.next_power_of_2(D),
        HAS_BIAS=dt_bias is not None,
    )
    dA = dA_partial.sum(0)
    dbias = dg.view(-1, H * D).sum(0).view(H, D) if dt_bias is not None else None
    return dg.view(B, T, H, D), dA, dbias


# ---------------------------------------------------------------------------
# 参考
# ---------------------------------------------------------------------------
def torch_gate(g, A_log, dt_bias, lower_bound):
    """公式直算 gate = lower_bound * sigmoid(exp(A_log)*(g+dt_bias))，返回 [B,T,H,D] gate（未经 cumsum）。"""
    x = g.float()
    if dt_bias is not None:
        x = x + dt_bias.view(1, 1, -1, g.shape[-1])
    return lower_bound * torch.sigmoid(torch.exp(A_log.view(1, 1, -1, 1)) * x)


def torch_local_cumsum(x, chunk, scale=1.0, reverse=False):
    """块内前缀和，[..., T, ...] 在 T 维分块；x layout [B,T,H,D]。"""
    B, dim = x.shape[0], x.shape[1]
    tail = x.shape[2:]
    pad = (chunk - dim % chunk) % chunk
    if pad:
        x = torch.cat([x, x.new_zeros(B, pad, *tail)], dim=1)
    x = x.reshape(B, -1, chunk, *tail)
    if reverse:
        x = torch.flip(x, dims=[2])
    x = x.cumsum(dim=2)
    if reverse:
        x = torch.flip(x, dims=[2])
    x = x.reshape(B, -1, *tail)
    return (x * scale)[:, :dim]


def stats(name, pred, ref):
    d = (pred.float() - ref.float()).abs()
    req = ref.float().abs()
    max_abs = d.max().item()
    max_rel = (d.max() / (req.max() + 1e-12)).item()
    print(f"  {name:34s} max_abs={max_abs:.3e}  max_rel={max_rel:.3e}  mean_abs={d.mean():.3e}")
    return max_abs, max_rel


def make_inputs(B, T, H, D, seed, g_scale, device="cuda", with_bias=True):
    torch.manual_seed(seed)
    g = (torch.randn(B, T, H, D) * g_scale).to(torch.bfloat16).to(device)
    beta = torch.randn(B, T, H).to(torch.bfloat16).to(device)
    A_log = (torch.rand(H) * 0.5).to(device)
    dt_bias = (torch.randn(H, D) * 0.1).to(device) if with_bias else None
    return g, beta, A_log, dt_bias


CASES = [
    (1, 128, 4, 128, 0, 1.0),
    (2, 512, 2, 128, 1, 2.0),
    (1, 250, 3, 128, 2, 1.0),   # 非整除 T
    (1, 513, 2, 128, 3, 4.0),   # 非整除 T
    (1, 64, 1, 128, 4, 0.0),    # g=0 无衰减
]


def main():
    torch.manual_seed(0)
    device = "cuda"
    LB = -5.0

    from fla.ops.kda.gate import kda_gate_chunk_cumsum, kda_gate_bwd, fused_kda_gate
    from fla.modules.l2norm import l2norm_fwd  # noqa: F401 (import sanity)
    try:
        from fla.ops.common.gate import fused_beta_sigmoid
    except Exception:
        from fla.modules.fused_gate import fused_beta_sigmoid  # type: ignore
    from fla.ops.utils import chunk_local_cumsum

    print("=" * 78)
    print("Part A  B0 前向重算 (g_cum/beta) vs 纯 torch 参考")
    for (B, T, H, D, seed, gs) in CASES:
        g, beta, A_log, dt_bias = make_inputs(B, T, H, D, seed, gs)
        gate = torch_gate(g, A_log, dt_bias, LB)
        gc_ref = torch_local_cumsum(gate, 16, LOG2E)              # [B,T,H,D] fp32
        gb_ref = torch.sigmoid(beta.float())
        gc_b0, gb_b0 = b0_gate_fwd(g, A_log, dt_bias, beta, LB, 16)
        tag = f"B{B} T{T} H{H} g={gs}"
        print(f"[{tag}]")
        stats("g_cum  B0 vs torch", gc_b0, gc_ref)
        stats("beta   B0 vs torch", gb_b0, gb_ref)

    print("=" * 78)
    print("Part B  B0 前向重算 vs FLA kda_gate_chunk_cumsum / fused_beta_sigmoid")
    for (B, T, H, D, seed, gs) in CASES:
        g, beta, A_log, dt_bias = make_inputs(B, T, H, D, seed, gs)
        gc_fla = kda_gate_chunk_cumsum(g, A_log, chunk_size=16, scale=RCP_LN2,
                                       dt_bias=dt_bias, lower_bound=LB)
        gb_fla = fused_beta_sigmoid(beta, scale=1.0)
        gc_b0, gb_b0 = b0_gate_fwd(g, A_log, dt_bias, beta, LB, 16)
        tag = f"B{B} T{T} H{H} g={gs}"
        print(f"[{tag}]")
        stats("g_cum  B0 vs FLA", gc_b0, gc_fla)
        stats("beta   B0 vs FLA", gb_b0, gb_fla)

    print("=" * 78)
    print("Part C  B0 反向前缀和 (APPLY_GATE=False, REVERSE=True) vs FLA chunk_local_cumsum")
    for (B, T, H, D, seed, gs) in CASES:
        torch.manual_seed(seed + 7)
        x = torch.randn(B, T, H, D).to(device) * 0.3
        rev_ref = chunk_local_cumsum(x, chunk_size=16, reverse=True)
        rev_b0, _ = b0_gate_fwd(x, torch.zeros(H, device=device), None, None, 0.0, 16,
                                scale=1.0, reverse=True, apply_gate=False)
        tag = f"B{B} T{T} H{H}"
        print(f"[{tag}]")
        stats("revcumsum  B0 vs FLA", rev_b0, rev_ref)

    print("=" * 78)
    print("Part D  B0 门控反向 vs FLA kda_gate_bwd vs torch.autograd")
    for (B, T, H, D, seed, gs) in CASES:
        g, beta, A_log, dt_bias = make_inputs(B, T, H, D, seed, gs)
        torch.manual_seed(seed + 11)
        dg_cum = torch.randn(B, T, H, D).to(device)          # 上游 dL/d g_cum
        # --- torch autograd golden ---
        gg = g.float().clone().requires_grad_(True)
        Al = A_log.clone().requires_grad_(True)
        db = dt_bias.clone().requires_grad_(True)
        gate_t = torch_gate(gg, Al, db, LB)
        gcum_t = torch_local_cumsum(gate_t, 16, LOG2E)
        (gcum_t * dg_cum).sum().backward()
        dg_ref, dA_ref, dbias_ref = gg.grad, Al.grad, db.grad
        # --- B0 pipeline: reverse cumsum (*log2e, 因为 g_cum=log2e*cumsum(gate)) -> gate bwd ---
        dyg, _ = b0_gate_fwd(dg_cum, torch.zeros(H, device=device), None, None, 0.0, 16,
                             scale=LOG2E, reverse=True, apply_gate=False)
        dg_b0, dA_b0, dbias_b0 = b0_gate_bwd(g, A_log, dt_bias, dyg, LB)
        # --- FLA gate bwd ---
        dg_fla, dA_fla, dbias_fla = kda_gate_bwd(g=g, A_log=A_log, dt_bias=dt_bias,
                                                 dyg=dyg, lower_bound=LB)
        tag = f"B{B} T{T} H{H} g={gs}"
        print(f"[{tag}]")
        stats("dg_logits  B0 vs autograd", dg_b0, dg_ref)
        stats("dg_logits  B0 vs FLA", dg_b0, dg_fla)
        stats("dg_logits  FLA vs autograd", dg_fla, dg_ref)
        stats("dg_logits  FLA vs autograd.bf16", dg_fla, dg_ref.bfloat16())
        stats("dA         B0 vs autograd", dA_b0, dA_ref)
        stats("dbias      B0 vs autograd", dbias_b0, dbias_ref)
        if dbias_fla is None:
            print("    (FLA kda_gate_bwd 未返回 dbias，跳过)")

    print("=" * 78)
    print("Part E  交叉验证：B0 的 exp2(g_cum[-1]) vs FlashKDA K1 workspace kGTotal")
    try:
        import flash_kda
        from flash_kda_C import get_workspace_size, fwd as _fwd_raw
        from flash_kda import fwd as _wrapped  # noqa: F401

        CHUNK, D = 16, 128
        WS_GT_OFF = (3 * CHUNK * D * 2)  # kKDecayed+kQDecayed+kKRestored
        for (B, T, H, D_, seed, gs) in [(1, 128, 4, 128, 0, 1.0),
                                        (1, 250, 2, 128, 2, 1.0),   # 非整除 T
                                        (1, 512, 2, 128, 1, 2.0)]:
            assert D_ == D
            g, beta, A_log, dt_bias = make_inputs(B, T, H, D, seed, gs)
            total_tiles = B * ((T + CHUNK - 1) // CHUNK)
            n_ht = H * total_tiles
            ws = torch.empty(get_workspace_size(B * T, H, B), dtype=torch.uint8, device=device)
            out = torch.empty(B, T, H, D, dtype=torch.bfloat16, device=device)
            _fwd_raw(g.contiguous(), g.contiguous(), torch.randn(B, T, H, D, dtype=torch.bfloat16, device=device).contiguous(),
                     g.contiguous(), beta.contiguous(), 1.0 / math.sqrt(D), out, ws,
                     A_log.clone(), dt_bias.clone(), LB)
            torch.cuda.synchronize()
            # kGTotal 是 E^tot = 2^{g_cum[-1]}，head-major: ws_idx = h*total_tiles + tile
            a = WS_GT_OFF * n_ht
            gt_flat = ws[a:a + D * 4 * n_ht].contiguous().view(torch.float32).view(H, total_tiles, D)
            # B0 recompute g_cum (log2), 取每 chunk 最后一有效行
            gc_b0, _ = b0_gate_fwd(g, A_log, dt_bias, None, LB, CHUNK)
            NT = (T + CHUNK - 1) // CHUNK
            # 每 tile 的最后有效 token 的 g_cum
            last = []
            for t in range(NT):
                tlast = min((t + 1) * CHUNK, T) - 1
                last.append(gc_b0[0, tlast])                 # [H,D]
            gc_last = torch.stack(last, 0)                   # [NT,H,D]
            etot_b0 = torch.exp2(gc_last.permute(1, 0, 2))   # [H,NT,D]
            # 只比有效 tile（total_tiles >= NT，多余 tile 为 pad）
            gt_eff = gt_flat[:, :NT, :]
            tag = f"B{B} T{T} H{H} g={gs}"
            print(f"[{tag}]  (K1 用 bf16 g + tanh.approx sigmoid，B0 用精确 sigmoid，"
                  f"差异应为 approx 底噪)")
            stats("E_tot  B0 vs K1-ws", etot_b0, gt_eff)
    except Exception as e:  # pragma: no cover
        print(f"  FlashKDA workspace 交叉验证跳过：{type(e).__name__}: {e}")

    print("=" * 78)
    print("Part F  B0 前向重算 kernel 计时（event 对包 REPS 次取均值）vs FLA kda_gate_chunk_cumsum")
    REPS, WARM = 100, 20
    for (B, T, H, D, seed, gs) in [(1, 512, 4, 128, 0, 1.0),
                                   (4, 2048, 8, 128, 1, 2.0),
                                   (1, 2048, 64, 128, 1, 2.0)]:
        g, beta, A_log, dt_bias = make_inputs(B, T, H, D, seed, gs)
        gc_fla = kda_gate_chunk_cumsum(g, A_log, chunk_size=16, scale=RCP_LN2,
                                       dt_bias=dt_bias, lower_bound=LB)
        for _ in range(WARM):
            b0_gate_fwd(g, A_log, dt_bias, beta, LB, 16)
            kda_gate_chunk_cumsum(g, A_log, chunk_size=16, scale=RCP_LN2, dt_bias=dt_bias, lower_bound=LB)

        def bench(fn):
            s, e = torch.cuda.Event(True), torch.cuda.Event(True)
            s.record()
            for _ in range(REPS):
                fn()
            e.record()
            torch.cuda.synchronize()
            return s.elapsed_time(e) / REPS * 1e3  # us

        t_b0 = bench(lambda: b0_gate_fwd(g, A_log, dt_bias, beta, LB, 16))
        t_fla = bench(lambda: kda_gate_chunk_cumsum(g, A_log, chunk_size=16, scale=RCP_LN2,
                                                    dt_bias=dt_bias, lower_bound=LB))
        tag = f"B{B} T{T} H{H} D{D}"
        print(f"[{tag}]  B0={t_b0:8.2f} us   FLA={t_fla:8.2f} us   ratio={t_b0/t_fla:.2f}x")

    print("\nDONE")


if __name__ == "__main__":
    main()
