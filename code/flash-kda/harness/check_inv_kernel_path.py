#!/usr/bin/env python
"""P7-112: close the historical ``待验证`` left in ``docs/05`` §7.4.

The question (stated there as "留到 P6"): the chunked **reference** backward uses
``torch.linalg.solve`` (cuSOLVER fp32) for the block inverse ``P = (I+L)^{-1}``,
while the real FlashKDA K1 kernel builds ``P`` with an **8x8 forward substitution
+ bf16 HMMA Schur merge** (``csrc/smxx/utils.cuh:213-369``).  Is the kernel path
"more stable" than cuSOLVER fp32, and does it matter for the gradients?

This script answers it on **real KDA block matrices** ``L`` (not the random ``L``
used by ``check_inv_precision.py``) and isolates the P-construction effect from
every other fp32 rounding by running the whole chunked backward in **fp64** and
plugging only ``_chunk_inv`` back in fp32.

Part A  -- P accuracy on real L: cuSOLVER GPU fp32 / CPU fp32 / kernel Schur path
           vs an fp64 golden inverse, over g-scale (=conditioning) sweeps.
Part B  -- gradient propagation: run the (fp64) chunked backward with ``_chunk_inv``
           patched to each fp32 P-construction path; compare the six gradients to the
           fp64-exact-P run.  Everything except P construction is fp64, so the delta
           is attributable to the inversion path alone.
Part C  -- token-autograd anchor: fp64 chunked (exact P) vs fp64 per-token autograd,
           to confirm the scaffolding itself is clean on the tested shapes.

Run:  code/flash-kda/scripts/lab.sh exec \
        "python3 ~/proj/tech_record/code/flash-kda/harness/check_inv_kernel_path.py"
"""
import math
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)                                   # check_inv_precision
sys.path.insert(0, os.path.join(HERE, "..", "src"))        # kda_ref / kda_bwd_ref

from check_inv_precision import schur_inv                  # noqa: E402
import kda_ref                                             # noqa: E402
import kda_bwd_ref                                         # noqa: E402


# --------------------------------------------------------------------------------------
# realistic block matrices L, built exactly as kda_chunked_fwd / FlashKDA K1 do
# --------------------------------------------------------------------------------------
def build_chunk_Ls(D, T, chunk, g_scale, beta_hi, seed, device="cpu", collinear=False):
    """Return a list of fp32 [chunk, chunk] strictly-lower L blocks from one (T,D) head.

    L = tril(kd @ ki^T, -1) * beta[:,None],  kd = k*2^gcum,  ki = k*2^-gcum.
    ``g_scale`` sets decay strength (small -> L O(1) -> ill-conditioned); ``collinear``
    makes every key in a chunk share a direction (intentionally nasty).
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    k = torch.randn(T, D, generator=g)
    k = k / k.pow(2).sum(-1, keepdim=True).add(1e-6).sqrt()
    if collinear:
        base = k[:1]                                        # one direction for all tokens
        k = base.expand(T, D).clone() + 0.02 * k
        k = k / k.pow(2).sum(-1, keepdim=True).add(1e-6).sqrt()
    gd = -torch.rand(T, D, generator=g) * g_scale
    beta = beta_hi * torch.sigmoid(torch.randn(T, generator=g))

    Ls = []
    for t0 in range(0, T, chunk):
        sl = slice(t0, t0 + chunk)
        kc, gc_ = k[sl], gd[sl]
        if kc.shape[0] < chunk:
            pad = chunk - kc.shape[0]
            kc = torch.cat([kc, kc.new_zeros(pad, D)], 0)
            gc_ = torch.cat([gc_, gc_.new_zeros(pad, D)], 0)
        gcum = gc_.cumsum(0)
        E = torch.exp2(gcum)
        kd = kc * E
        ki = kc * torch.exp2(-gcum)
        L = torch.tril(kd @ ki.transpose(0, 1), diagonal=-1)
        b = torch.cat([beta[sl], beta.new_zeros(chunk - min(chunk, T - t0))], 0)
        L = L * b[:, None]
        Ls.append(L.to(device))
    return Ls


def p_gold(L):
    C = L.shape[-1]
    eye = torch.eye(C, dtype=torch.float64, device=L.device)
    return torch.linalg.solve(eye + L.double(), eye)


def p_cusolver(L, device):
    """fp32 cuSOLVER on ``device`` (L is fp32)."""
    C = L.shape[-1]
    Ld = L.to(device)
    eye = torch.eye(C, dtype=torch.float32, device=device)
    return torch.linalg.solve(eye + Ld, eye)


def p_schur(L):
    """kernel path (utils.cuh:213-369): fp32 seed, 8x8 fwd-subst, bf16 HMMA merge."""
    return schur_inv(L.float().cpu(), quant_seed=False, quant_merge=True)


def stats(a, b):
    d = (a.double() - b.double()).abs()
    denom = b.double().abs().max().item() + 1e-30
    return d.max().item(), d.max().item() / denom


# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"torch {torch.__version__}  cuda={torch.cuda.is_available()}")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = False

    # =========================== Part A: P accuracy on real L ==========================
    print("\n================ Part A: P=(I+L)^-1 accuracy on REAL KDA block L ================")
    configs = [
        ("D=128 g=0.0 beta~1  (L O(1))", dict(D=128, g_scale=0.0, beta_hi=1.0, collinear=False)),
        ("D=128 g=0.5 beta~1", dict(D=128, g_scale=0.5, beta_hi=1.0, collinear=False)),
        ("D=128 g=2.0 beta~1", dict(D=128, g_scale=2.0, beta_hi=1.0, collinear=False)),
        ("D=128 g=8.0 beta~1  (L~0)", dict(D=128, g_scale=8.0, beta_hi=1.0, collinear=False)),
        ("D=16  g=0.0 beta~1", dict(D=16, g_scale=0.0, beta_hi=1.0, collinear=False)),
        ("D=16  g=0.3 collinear keys", dict(D=16, g_scale=0.3, beta_hi=1.0, collinear=True)),
    ]
    hdr = f"{'config':32s} {'cond':>10s} {'gpu_solve':>12s} {'cpu_solve':>12s} {'schur(kernel)':>14s}"
    print(hdr)
    print("-" * len(hdr))
    partA = {}
    for tag, kw in configs:
        Ls = build_chunk_Ls(T=512, chunk=16, seed=0, device=dev, **kw)
        acc = {k: 0.0 for k in ("gpu", "cpu", "schur")}
        conds = []
        for L in Ls:
            gold = p_gold(L)
            I = torch.eye(16, dtype=torch.float64, device=L.device) + L.double()
            conds.append(torch.linalg.cond(I).item())
            for name, P in (("gpu", p_cusolver(L, dev).cpu()),
                            ("cpu", p_cusolver(L, "cpu")),
                            ("schur", p_schur(L))):
                e, _ = stats(P, gold.cpu())
                acc[name] = max(acc[name], e)
        gmean = sum(conds) / len(conds)
        print(f"{tag:32s} {gmean:10.2e} {acc['gpu']:12.2e} {acc['cpu']:12.2e} {acc['schur']:14.2e}")
        partA[tag] = (gmean, acc["gpu"], acc["cpu"], acc["schur"])

    # docs/05 §7.4 blamed the GPU fp32 solve: does allow_tf32 explain a GPU>CPU gap?
    print("\n  [A2] GPU cuSOLVER fp32 vs allow_tf32 (D=128,g=0.0): ")
    Ls = build_chunk_Ls(D=128, T=512, chunk=16, g_scale=0.0, beta_hi=1.0, seed=0, device=dev)
    for tf32 in (False, True):
        torch.backends.cuda.matmul.allow_tf32 = tf32
        e_gpu = max(stats(p_cusolver(L, dev).cpu(), p_gold(L).cpu())[0] for L in Ls)
        e_cpu = max(stats(p_cusolver(L, "cpu"), p_gold(L).cpu())[0] for L in Ls)
        print(f"      allow_tf32={tf32!s:5s}  gpu_solve={e_gpu:.2e}  cpu_solve={e_cpu:.2e}")
    torch.backends.cuda.matmul.allow_tf32 = False

    # =========================== Part B: gradient propagation ===========================
    print("\n================ Part B: gradient isolation (whole backward in fp64) =============")
    print("(inputs fp64; only _chunk_inv is swapped to each fp32 P-construction path)")

    def run_chunked_bwd(q, k, v, g, beta, scale, dout, h0, pimpl):
        old = kda_ref._chunk_inv
        kda_ref._chunk_inv = pimpl
        try:
            return kda_bwd_ref.kda_chunked_backward(q, k, v, g, beta, scale, dout, h0=h0)
        finally:
            kda_ref._chunk_inv = old

    def inv_gold(L):
        C = L.shape[-1]
        eye = torch.eye(C, dtype=L.dtype, device=L.device)
        return torch.linalg.solve(eye + L, eye)

    def inv_solve32(L):
        return p_cusolver(L.float(), "cpu").to(L.dtype)

    def inv_schur32(L):
        return p_schur(L).to(L.dtype)

    T, D = 64, 32
    gen = torch.Generator().manual_seed(7)
    q = torch.randn(T, D, generator=gen).double()
    kk = torch.randn(T, D, generator=gen).double()
    q = q / q.norm(dim=-1, keepdim=True)                 # real KDA uses L2-normalised q/k
    kk = kk / kk.norm(dim=-1, keepdim=True)
    v = torch.randn(T, D, generator=gen).double()
    beta = torch.sigmoid(torch.randn(T, generator=gen)).double()
    scale = 1.0 / math.sqrt(D)
    dout = torch.randn(T, D, generator=gen).double()
    h0 = torch.randn(D, D, generator=gen).double()

    keys = ("dq", "dk", "dv", "dg", "dbeta", "dh0")
    for g_scale in (0.5, 2.0, 8.0):
        g = (-torch.rand(T, D, generator=gen) * g_scale).double()
        base = run_chunked_bwd(q, kk, v, g, beta, scale, dout, h0, inv_gold)
        gmax = base["dq"].abs().max().item()
        print(f"\n  g_scale={g_scale}  (|dq|max={gmax:.2e})")
        for name, impl in (("solve32(cuSOLVER)", inv_solve32), ("schur32(kernel)", inv_schur32)):
            got = run_chunked_bwd(q, kk, v, g, beta, scale, dout, h0, impl)
            row = " ".join(f"{k}={stats(got[k], base[k])[0]:.2e}" for k in keys)
            print(f"    {name:20s} {row}")

    # =========================== Part C: token-autograd anchor ==========================
    print("\n================ Part C: fp64 chunked (exact P) vs fp64 token autograd ===========")
    for g_scale in (0.5, 8.0):
        g = (-torch.rand(T, D, generator=gen) * g_scale).double()
        base = run_chunked_bwd(q, kk, v, g, beta, scale, dout, h0, inv_gold)
        qa, ka, va, ga, ba, ha = (x.clone().requires_grad_(True) for x in (q, kk, v, g, beta, h0))
        o, Sf = kda_ref.kda_token_recurrent(qa, ka, va, ga, ba, scale, ha)
        (o * dout).sum().backward()
        ref = dict(dq=qa.grad, dk=ka.grad, dv=va.grad, dg=ga.grad, dbeta=ba.grad, dh0=ha.grad)
        row = " ".join(f"{k}={stats(base[k], ref[k])[0]:.2e}" for k in keys)
        print(f"  g_scale={g_scale}: {row}")

    print("\nDONE")
