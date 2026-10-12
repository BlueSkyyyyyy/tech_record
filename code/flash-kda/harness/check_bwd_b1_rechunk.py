"""P7-10: eliminate B1's CHUNK=16 *granularity* loss by re-chunking the cross-chunk state
recurrence to 64 — reconstructing the 64-chunk operands straight from FlashKDA's 16-chunk
K1 workspace (no forward recompute).

Motivation (docs/07 §11.5, "next step (2)"): P7-9 landed the cheap lever (B1 on native bf16
operands, 2.2-2.5x vs tf32) but B1 still eats `CHUNK=16` operands, so its serial state loop
runs NT=T/16 steps with 16-wide GEMMs, while FLA's `dhu` runs `chunk_size=64` (NT/4 steps,
64-wide).  §11.3 measured that granularity alone costs ~1.3-1.6x.

The key algebraic fact (this is what makes the re-chunk *workspace-reusable*): the chunked
state recurrence `S_{c+1} = diag(E_c) S_c + k_rest_c^T U_c` is *exact for any chunk size*,
and the 64-chunk operands are simple per-sub-chunk **rescalings** of the 16-chunk fields:

  for a 64-group built of 4 sub-chunks s=0..3 with sub-chunk gate totals G16_s (log2) and
  running offset off_s = sum_{s'<s} G16_s':
      q_dec64[t] = q_dec16[t] * 2^{+off_s}          (qg64)
      k_dec64[t] = k_dec16[t] * 2^{+off_s}          (kd64)
      ki64[t]    = ki16[t]    * 2^{-off_s},  ki16 = kr16 * 2^{-G16}   (unbounded, see below)
      k_rest64[t]= k_rest16[t]* 2^{sum_{s'>s} G16_{s'}}  (= ki64 * E64)  (kg64)
      E_tot64    = 2^{sum_s G16_s} = prod_s E16_s
  and the only *new* work is the 64x64 intra-block matrix `A64 = kd64 @ ki64^T` whose
  strict-lower `L` is needed to recompute `P64=(I+L)^{-1}` and hence `w64=P64 diag(beta) kd64`.

Numerics note: `ki16 = k_rest16 * 2^{-G16}` overflows fp32 (G16 can be ~ -58 for 16 tokens
at the harness's gate scale), so this prototype does the A64/P64/w64 reconstruction in
**fp64**; a production implementation computes A64 per sub-chunk block with a per-pair
rescale (bounded), or mirrors K1.  qg64/kg64 need no such care (their exponents are <= 0).

Run inside `kernel_lab` (GPU, FlashKDA built, FLA installed), idle GPU:
    CUDA_VISIBLE_DEVICES=1 python .../harness/check_bwd_b1_rechunk.py
    RC_MODE=correctness|timing
"""
import math
import os
import statistics
import sys

os.environ.setdefault("FLA_DISABLE_BACKEND_DISPATCH", "1")

import torch
import triton
import triton.language as tl

HARNESS = "~/proj/tech_record/code/flash-kda/harness"
SRC = "~/proj/tech_record/code/flash-kda/src"
sys.path.insert(0, HARNESS)
sys.path.insert(0, SRC)

from check_bwd_ws_reuse import (make_inputs, flash_forward, D, CHUNK, LOG2E,   # noqa: E402
                                cdiv, stats)
from check_bwd_b0_gate import b0_gate_fwd                                       # noqa: E402
from check_bwd_dhu_kernel import fla_forward                                    # noqa: E402
from check_bwd_b1_bf16_e2e import build_dhu_bf16, dhu_bf16_triton               # noqa: E402

BIG = 64


def b0_gate(inp, lower_bound):
    gc, gb = b0_gate_fwd(inp["g"], inp["A_log"], inp["dt_bias"], inp["beta"],
                         lower_bound, CHUNK, scale=LOG2E)
    return gc, gb


# --------------------------------------------------------------------- operand recovery
def recover_operands_64(fw, inp, lower_bound, scale, BT=BIG, dt=torch.float64, with_intra=True):
    """Build the BT-chunk operands B1 needs from the 16-chunk K1 workspace + B0 gate.

    ``dt`` selects the working precision of the reconstruction.  fp64 is the safe default
    (the naive `ki16 = kr*2^{-G16}` can reach ~2^58); fp32 additionally uses a **blockwise**
    A64 (per sub-chunk pair, with the pair's offset folded into the *smaller* factor) so no
    intermediate exceeds ~2^58 — this is the realistic production path.
    """
    B, T, H, _ = inp["q"].shape
    dev = inp["q"].device
    NT16 = fw["NT"]
    Sg = BT // CHUNK
    NG = cdiv(NT16, Sg)
    NTp = NG * Sg
    Tp = NG * BT
    g_cum16, beta16 = b0_gate(inp, lower_bound)          # [B,T,H,D], [B,T,H]

    def padf(x, last):
        z = torch.zeros(B, NTp, H, *last, device=dev, dtype=x.dtype)
        z[:, :NT16] = x
        return z

    kd = padf(fw["kd"], (CHUNK, D)).to(dt).view(B, NG, Sg, H, CHUNK, D)
    qd = padf(fw["qd"], (CHUNK, D)).to(dt).view(B, NG, Sg, H, CHUNK, D)
    kr = padf(fw["kr"], (CHUNK, D)).to(dt).view(B, NG, Sg, H, CHUNK, D)

    # padded 16-local g_cum: carry the last valid value (pad tokens have gate 0)
    gc16_full = torch.zeros(B, NT16 * CHUNK, H, D, device=dev, dtype=torch.float64)
    gc16_full[:, :T] = g_cum16.double()[:, :T]
    if NT16 * CHUNK > T:
        gc16_full[:, T:] = g_cum16.double()[:, T - 1:T]
    g16r = gc16_full.view(B, NT16, CHUNK, H, D)
    gate_tok = torch.empty_like(g16r)
    gate_tok[:, :, 0] = g16r[:, :, 0]
    gate_tok[:, :, 1:] = g16r[:, :, 1:] - g16r[:, :, :-1]
    g64 = torch.zeros(B, Tp, H, D, device=dev, dtype=torch.float64)
    g64[:, :NT16 * CHUNK] = gate_tok.reshape(B, NT16 * CHUNK, H, D)
    gc4 = g64.view(B, NG, Sg, CHUNK, H, D).cumsum(3)     # 16-local, on the fully padded axis
    G16 = gc4[:, :, :, -1].to(dt)                        # [B,NG,Sg,H,D] sub-chunk gate total
    off = torch.zeros_like(G16)
    if Sg > 1:
        off[:, :, 1:] = G16[:, :, :-1].cumsum(2)
    off_f = off.unsqueeze(4)                             # [B,NG,Sg,H,1,D]
    off_t = off.unsqueeze(3)                             # [B,NG,Sg,1,H,D]

    def to_tok(x):                                       # [B,NG,Sg,H,CH,D] -> [B,Tp,H,D]
        return x.permute(0, 1, 2, 4, 3, 5).reshape(B, Tp, H, D)

    qg64 = to_tok(qd * torch.exp2(off_f)) / scale
    ki16 = kr * torch.exp2(-G16.unsqueeze(4))            # = kn * 2^{-g_cum16}
    G64 = off[:, :, -1] + G16[:, :, -1]                  # [B,NG,H,D] block gate total (log2)
    E64 = torch.exp2(G64)
    # k_rest64 = kr16 * 2^{sum_{s'>s} G16} = kr16 * 2^{G64 - off_s - G16_s}  (exponent <= 0)
    kg64 = to_tok(kr * torch.exp2((G64[:, :, None] - off - G16).unsqueeze(4)))
    g_cum64 = (gc4 + off_t).reshape(B, Tp, H, D)

    if not with_intra:
        return dict(qg=qg64, kg=kg64, w=None, g_cum=g_cum64), dict(E64=E64, Tp=Tp,
                                                                  NT64=cdiv(T, BT), A64_Lmax=-1.0)

    # 64-block intra matrix A64 (blockwise, stable), inverse P64, and w64 = P64 diag(beta) kd64
    beta_g = torch.zeros(B, Tp, H, device=dev, dtype=dt)
    beta_g[:, :T] = beta16.to(dt)
    beta_g = beta_g.view(B, NG, BT, H).permute(0, 1, 3, 2)          # [B,NG,H,BT]
    A64 = torch.zeros(B, NG, H, BT, BT, dtype=dt, device=dev)
    for st in range(Sg):
        for su in range(st + 1):
            if st == su:
                sc = None
            else:
                sc = torch.exp2(G16[:, :, su:st].sum(2)).unsqueeze(3)     # [B,NG,H,1,D]
            kd_st = kd[:, :, st]                                          # [B,NG,H,CH,D]
            ki_su = ki16[:, :, su] if sc is None else ki16[:, :, su] * sc
            A64[:, :, :, st * CHUNK:(st + 1) * CHUNK, su * CHUNK:(su + 1) * CHUNK] = \
                torch.einsum("bnhtd,bnhud->bnhtu", kd_st, ki_su)
    kd_g = to_tok(kd * torch.exp2(off_f)).view(B, NG, BT, H, D).permute(0, 1, 3, 2, 4)
    L = torch.tril(A64, -1) * beta_g[..., None, :]
    eye = torch.eye(BT, device=dev, dtype=dt)
    P64 = torch.linalg.solve(eye + L, eye.expand_as(L).contiguous())
    w_g = torch.einsum("bnhtu,bnhud->bnhtd", P64, kd_g * beta_g[..., None])
    w64 = w_g.permute(0, 1, 3, 2, 4).reshape(B, Tp, H, D)

    fi = dict(qg=qg64, kg=kg64, w=w64, g_cum=g_cum64)
    diag = dict(A64_Lmax=torch.tril(A64, -1).abs().amax().item(),
                E64=E64, Tp=Tp, NT64=cdiv(T, BT))
    return fi, diag


def pad_tokens(x, Tp):
    """Pad [B,T,H,..] to [B,Tp,H,..] with zeros."""
    if x.shape[1] == Tp:
        return x
    z = torch.zeros(x.shape[0], Tp, *x.shape[2:], device=x.device, dtype=x.dtype)
    z[:, :x.shape[1]] = x
    return z


def b1_64(kernel, fi, do, dv, scale, dht=None):
    """Run the bf16-mma B1 at BT=64 on padded token axes (mask via zero operands)."""
    Tp = fi["qg"].shape[1]
    do_p = pad_tokens(do, Tp)
    dv_p = pad_tokens(dv, Tp)
    dh, dh0, dv2 = dhu_bf16_triton(kernel, fi, do_p, dv_p, scale, BIG, dht=dht)
    return dh, dh0, dv2


# ------------------------------------------------------------------------- FLA oracle
def fla_b1_64(v, qn, kn, beta_act, g_log2, h0, scale, do, dv, dht=None):
    """FLA `dhu` at chunk_size=64 on FLA's own 64-chunk forward operands."""
    from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_bwd_dhu
    F = fla_forward(qn, kn, v, g_log2, beta_act, h0, scale, BIG)
    dh, dh0, dv2 = chunk_gated_delta_rule_bwd_dhu(
        q=F["qg"], k=F["kg"], w=F["w"], gk=F["g_cum"], h0=h0, dht=dht,
        do=do, dv=dv, scale=scale, chunk_size=BIG)
    return dict(dh=dh, dh0=dh0, dv2=dv2, fi=F)


# ------------------------------------------------------- bf16 B1@16 baseline (P7-9 path)
def b1_16(kernel, fi16, do, dv, scale, dht=None):
    fi_b = dict(fi16)
    for kk in ("qg", "kg", "w"):
        fi_b[kk] = fi16[kk].to(torch.bfloat16)
    do_b = do.to(torch.bfloat16)
    dv_b = dv.to(torch.bfloat16)
    return dhu_bf16_triton(kernel, fi_b, do_b, dv_b, scale, CHUNK, dht=dht)


# --------------------------------------------------------------------------- timing
def cuda_kernel_us(fn, sub, reps=30):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(reps):
            fn()
        torch.cuda.synchronize()
    return sum(ev.self_device_time_total for ev in prof.key_averages()
               if sub in ev.key) / reps


def wall_us(fn, reps=30, rounds=3, warmup=8):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    med = []
    for _ in range(rounds):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(reps):
            fn()
        e.record()
        torch.cuda.synchronize()
        med.append(s.elapsed_time(e) / reps)
    return statistics.median(med) * 1e3


# --------------------------------------------------------------------------- runner
def prep(B, T, H, scale, lower_bound, seed=0, g_scale=1.0):
    inp = make_inputs(B, T, H, seed, "cuda", g_scale)
    fw = flash_forward(inp, scale, lower_bound, "cuda")
    qn = torch.nn.functional.normalize(inp["q"].float(), dim=-1)
    kn = torch.nn.functional.normalize(inp["k"].float(), dim=-1)
    beta_act = torch.sigmoid(inp["beta"].float())
    inner = inp["g"].float() + inp["dt_bias"].view(1, 1, H, D)
    g_fla = lower_bound * torch.sigmoid(torch.exp(inp["A_log"].view(1, 1, H, 1)) * inner)
    torch.manual_seed(seed + 77)
    do = torch.randn(B, T, H, D, device="cuda")
    dv = torch.randn(B, T, H, D, device="cuda") * 0.5
    return inp, fw, qn, kn, beta_act, g_fla, do, dv


def part_a(cases, scale, lower_bound):
    print("=" * 90)
    print("Part A: 64-chunk operands recovered from the 16-chunk workspace vs FLA forward@64")
    print("=" * 90)
    worst = {}
    for (name, B, T, H, seed, gs) in cases:
        inp, fw, qn, kn, beta_act, g_fla, do, dv = prep(B, T, H, scale, lower_bound, seed, gs)
        rec, diag = recover_operands_64(fw, inp, lower_bound, scale)
        qnb = qn.to(torch.bfloat16)
        knb = kn.to(torch.bfloat16)
        # FLA helper expects a *log2* gate; kda_gate output (lb*log2e*sigmoid) is log2.
        g2 = (g_fla * 1.4426950408889634).to(torch.bfloat16)
        F = fla_forward(qnb, knb, inp["v"].to(torch.bfloat16), g2,
                        beta_act.to(torch.bfloat16), inp["h0"], scale, BIG)
        print(f"\n# {name}  B={B} T={T} H={H}  |strict-lower A64|max={diag['A64_Lmax']:.3e}")
        for nm, fk in [("qg", "qg"), ("kg", "kg"), ("w", "w"), ("g_cum", "g_cum")]:
            ma, mr = stats(f"{nm:6s} rec64 vs FLA64", rec[nm][:, :T], F[fk].float()[:, :T])
            worst[nm] = max(worst.get(nm, 0.0), mr)
    print("\nworst-case max_rel across cases:")
    for nm in ["qg", "kg", "w", "g_cum"]:
        print(f"  {nm:6s} {worst[nm]:.3e}")


def part_b(cases, scale, lower_bound):
    print("\n" + "=" * 90)
    print("Part B: our B1@64 on recovered operands vs FLA `dhu`@64 (same do/dv/dht)")
    print("=" * 90)
    kernel = build_dhu_bf16()
    worst = {}
    for (name, B, T, H, seed, gs) in cases:
        inp, fw, qn, kn, beta_act, g_fla, do, dv = prep(B, T, H, scale, lower_bound, seed, gs)
        rec, diag = recover_operands_64(fw, inp, lower_bound, scale)
        rec32, _ = recover_operands_64(fw, inp, lower_bound, scale, dt=torch.float32)
        qnb = qn.to(torch.bfloat16)
        knb = kn.to(torch.bfloat16)
        g2 = (g_fla * 1.4426950408889634).to(torch.bfloat16)
        fb = fla_b1_64(inp["v"].to(torch.bfloat16), qnb, knb, beta_act.to(torch.bfloat16), g2,
                       inp["h0"], scale, do.to(torch.bfloat16), dv.to(torch.bfloat16))
        print(f"\n# {name}  B={B} T={T} H={H}  NT64={diag['NT64']}")
        for tag, rr in [("fp64", rec), ("fp32", rec32)]:
            dh, dh0, dv2 = b1_64(kernel, rr, do, dv, scale, dht=None)
            ma, mr = stats(f"dh  O64-{tag} vs FLA64", dh, fb["dh"].float()); worst["dh"] = max(worst.get("dh", 0.0), mr)
            ma, mr = stats(f"dv2 O64-{tag} vs FLA64", dv2[:, :T], fb["dv2"].float()[:, :T]); worst["dv2"] = max(worst.get("dv2", 0.0), mr)
            ma, mr = stats(f"dh0 O64-{tag} vs FLA64", dh0, fb["dh0"].float()); worst["dh0"] = max(worst.get("dh0", 0.0), mr)
    print("\nworst-case max_rel:")
    for nm in ["dh", "dv2", "dh0"]:
        print(f"  {nm:5s} {worst[nm]:.3e}")


def part_c(cases, scale, lower_bound):
    print("\n" + "=" * 90)
    print("Part C: B1 timing — ours@64(recovered) vs ours@16(workspace, P7-9) vs FLA dhu@64")
    print("=" * 90)
    from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_bwd_dhu
    kernel = build_dhu_bf16()
    print(f"  {'case':18s} | {'ours64 dev':>10s} {'ours16 dev':>10s} {'FLA64 dev':>9s} | "
          f"{'o64/o16':>8s} {'o64/FLA':>8s} | {'recover':>9s} {'rescale':>9s}")
    for (name, B, T, H, seed, gs) in cases:
        inp, fw, qn, kn, beta_act, g_fla, do, dv = prep(B, T, H, scale, lower_bound, seed, gs)
        rec, diag = recover_operands_64(fw, inp, lower_bound, scale)          # fp64 (correct)
        rec32, _ = recover_operands_64(fw, inp, lower_bound, scale, dt=torch.float32)  # prod path
        # 16-chunk workspace operands (bf16) exactly as the P7-9 chain feeds B1
        def toT(x):   # [B,NT,H,CH,D] -> [B,T,H,D]
            return x.permute(0, 1, 3, 2, 4).reshape(B, cdiv(T, CHUNK) * CHUNK, H, D)[:, :T]
        fi16 = dict(qg=toT(fw["qd"].float() / scale),
                    kg=toT(fw["kr"].float()),
                    w=None, g_cum=None)
        # w16 = P16 diag(beta) kd16 (workspace P16 = inv field), computed in fp32
        P16 = fw["inv"].float()                                   # [B,NT,H,16,16] P[o_t,o_A]
        kd16 = fw["kd"].float()                                   # [B,NT,H,16,D]
        beta_pad = torch.zeros(B, cdiv(T, CHUNK) * CHUNK, H, device=beta_act.device, dtype=beta_act.dtype)
        beta_pad[:, :T] = beta_act
        bc = beta_pad.view(B, cdiv(T, CHUNK), CHUNK, H).permute(0, 1, 3, 2)   # [B,NT,H,16]
        kdb = kd16 * bc[..., None]
        w16 = toT(torch.einsum("bnhtu,bnhuk->bnhtk", P16, kdb))
        fi16["w"] = w16
        fi16["g_cum"] = b0_gate(inp, lower_bound)[0]
        do_b = do.to(torch.bfloat16)
        dv_b = dv.to(torch.bfloat16)
        qnb = qn.to(torch.bfloat16)
        knb = kn.to(torch.bfloat16)
        g2 = (g_fla * 1.4426950408889634).to(torch.bfloat16)
        F = fla_forward(qnb, knb, inp["v"].to(torch.bfloat16), g2,
                        beta_act.to(torch.bfloat16), inp["h0"], scale, BIG)

        def run64():
            return b1_64(kernel, rec32, do, dv, scale)

        def run16():
            return b1_16(kernel, fi16, do, dv, scale)

        def runfla():
            return chunk_gated_delta_rule_bwd_dhu(
                q=F["qg"], k=F["kg"], w=F["w"], gk=F["g_cum"], h0=inp["h0"], dht=None,
                do=do_b, dv=dv_b, scale=scale, chunk_size=BIG)

        d64 = cuda_kernel_us(run64, "_dhu_kernel")
        d16 = cuda_kernel_us(run16, "_dhu_kernel")
        dfl = cuda_kernel_us(runfla, "bwd_kernel_dhu")
        drec = wall_us(lambda: recover_operands_64(fw, inp, lower_bound, scale, dt=torch.float32))
        dres = wall_us(lambda: recover_operands_64(fw, inp, lower_bound, scale, dt=torch.float32,
                                                   with_intra=False))
        print(f"  {name:18s} | {d64:10.1f} {d16:10.1f} {dfl:9.1f} | "
              f"{d64/d16:8.2f} {d64/dfl:8.2f} | {drec:9.1f} {dres:9.1f}")


def main():
    assert torch.cuda.is_available(), "need GPU"
    print(f"GPU: {torch.cuda.get_device_name()}  torch={torch.__version__}")
    scale = 1.0 / math.sqrt(D)
    lower_bound = -5.0
    cases = [
        ("T=256 H=2", 1, 256, 2, 0, 1.0),
        ("T=250 H=3 non-div", 1, 250, 3, 2, 1.0),
        ("B2 T=513 H=2 non-div", 2, 513, 2, 4, 1.0),
        ("T=192 H=2 g=0", 1, 192, 2, 5, 0.0),
    ]
    tcases = [
        ("T=512 H=4", 1, 512, 4, 0, 1.0),
        ("B2 T=1024 H=8", 2, 1024, 8, 1, 1.0),
        ("B8 T=2048 H=8", 8, 2048, 8, 2, 1.0),
        ("T=250 H=3 nondiv", 1, 250, 3, 3, 1.0),
    ]
    mode = os.environ.get("RC_MODE", "")
    if mode != "timing":
        part_a(cases, scale, lower_bound)
        part_b(cases, scale, lower_bound)
    if mode != "correctness":
        part_c(tcases, scale, lower_bound)
    print("\nDONE")


if __name__ == "__main__":
    main()
