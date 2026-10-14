"""P5-4: validate the hand-derived *chunked* backward (``kda_chunked_backward``).

Three oracles, all on the same inputs:

1. ``kda_token_backward`` -- the per-token explicit adjoint already proven bit-exact
   against ``torch.autograd`` in P5-2.  In fp64 the chunked backward must agree to
   machine precision (different summation order, same math).
2. ``torch.autograd`` driven through ``kda_ref.kda_chunked_fwd`` itself -- this checks
   the derivation against the *actual blocked forward graph* (including the (I+L)^-1).
3. ``torch.autograd`` driven through ``kda_token_recurrent`` (does not reuse P/L, so an
   independent control).

Usage (inside kernel_lab, CPU or GPU):
    python ~/proj/tech_record/code/flash-kda/harness/check_bwd_chunked.py
"""

import math
import sys

import torch

SRC = "~/proj/tech_record/code/flash-kda/src"
sys.path.insert(0, SRC)
from kda_bwd_ref import kda_chunked_backward, kda_token_backward  # noqa: E402
from kda_ref import kda_chunked_fwd, kda_token_recurrent, l2_normalize  # noqa: E402


def stats(a, b):
    dt = torch.float64 if (a.dtype == torch.float64 or b.dtype == torch.float64) else torch.float32
    d = (a.to(dt) - b.to(dt)).abs()
    ref = b.to(dt).abs()
    return d.max().item(), (d.max() / (ref.max() + 1e-12)).item(), d.mean().item()


FIELDS = ["dq", "dk", "dv", "dg", "dbeta", "dh0"]


def make_inputs(T, D, g_scale, seed, dtype, device="cpu"):
    torch.manual_seed(seed)
    q = l2_normalize(torch.randn(T, D)).to(dtype).to(device)
    k = l2_normalize(torch.randn(T, D)).to(dtype).to(device)
    v = torch.randn(T, D).to(dtype).to(device)
    g = (-torch.rand(T, D) * g_scale).to(dtype).to(device)
    beta = torch.rand(T).to(dtype).to(device)
    h0 = torch.randn(D, D).to(dtype).to(device)
    scale = 1.0 / math.sqrt(D)
    dout = torch.randn(T, D).to(dtype).to(device)
    dfin = torch.randn(D, D).to(dtype).to(device)
    return q, k, v, g, beta, scale, h0, dout, dfin


def autograd_chunked(q, k, v, g, beta, scale, h0, dout, dfin, chunk):
    qa, ka, va, ga, ba, ha = [t.clone().requires_grad_(True) for t in (q, k, v, g, beta, h0)]
    o, Sf = kda_chunked_fwd(qa, ka, va, ga, ba, scale, ha, chunk=chunk)
    (o * dout).sum().add((Sf * dfin).sum()).backward()
    return dict(dq=qa.grad, dk=ka.grad, dv=va.grad, dg=ga.grad, dbeta=ba.grad, dh0=ha.grad)


def autograd_token(q, k, v, g, beta, scale, h0, dout, dfin):
    qa, ka, va, ga, ba, ha = [t.clone().requires_grad_(True) for t in (q, k, v, g, beta, h0)]
    o, Sf = kda_token_recurrent(qa, ka, va, ga, ba, scale, ha)
    (o * dout).sum().add((Sf * dfin).sum()).backward()
    return dict(dq=qa.grad, dk=ka.grad, dv=va.grad, dg=ga.grad, dbeta=ba.grad, dh0=ha.grad)


CASES = [
    # T,   D,  g_scale, seed, chunk
    (16,  16, 2.0, 0, 16),
    (17,  16, 2.0, 1, 16),
    (20,  12, 1.0, 2, 16),
    (33,  16, 2.0, 3, 16),
    (48,  16, 0.0, 4, 16),
    (64,  16, 4.0, 5, 16),
    (128, 16, 2.0, 6, 16),
    (100, 16, 2.0, 7, 15),   # non-16 chunk -> still consistent (single implementation)
    (200, 64, 2.0, 8, 16),
    (256, 128, 2.0, 10, 16),  # real-scale head dim
    (512, 128, 1.0, 11, 16),
]


def forward_precision_note(device):
    """Explain the fp32 gradient gap: the blocked reference's inverse solve.

    Compare ``kda_chunked_fwd`` vs ``kda_token_recurrent`` (same inputs) in fp32 and
    fp64.  NOTE (corrected by P7-113 / docs/07 §117): the fp32 GPU gap (~3e-4) is
    **not** the cuSOLVER ``(I+L)^-1`` -- it is the NGC container default
    ``torch.backends.cuda.matmul.allow_tf32 = True`` making every fp32 matmul in the
    blocked forward run in tf32.  cuSOLVER P is bit-identical CPU vs GPU; with
    ``allow_tf32=False`` the gap drops to ~1e-7 (see harness/check_fwd_gpu_cpu.py).
    """
    print("\n  forward chunked-vs-token precision (oracle conditioning, seed=0 T=16 D=16):")
    for dt in (torch.float32, torch.float64):
        torch.manual_seed(0)
        T, D = 16, 16
        q = l2_normalize(torch.randn(T, D)).to(dt).to(device)
        k = l2_normalize(torch.randn(T, D)).to(dt).to(device)
        v = torch.randn(T, D).to(dt).to(device)
        g = (-torch.rand(T, D) * 2.0).to(dt).to(device)
        beta = torch.rand(T).to(dt).to(device)
        h0 = torch.randn(D, D).to(dt).to(device)
        scale = 1.0 / math.sqrt(D)
        o1, S1 = kda_chunked_fwd(q, k, v, g, beta, scale, h0, chunk=16)
        o2, S2 = kda_token_recurrent(q, k, v, g, beta, scale, h0)
        ro = ((o1 - o2).abs().max() / o2.abs().max()).item()
        rs = ((S1 - S2).abs().max() / S2.abs().max()).item()
        tag = "fp64" if dt == torch.float64 else "fp32"
        print(f"    {tag} (device={device}): forward o max_rel={ro:.2e}  S max_rel={rs:.2e}")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 78)
    print(f"P5-4  chunked backward  vs  token backward  vs  autograd   (device={device})")
    print("      metric max_abs / max_rel;  case: chunked-bwd vs (A) token-bwd (B) autograd(chunked) (C) autograd(token)")
    print("=" * 78)

    agg_fp64 = {n: [0.0, 0.0] for n in FIELDS}
    agg_fp32 = {n: [0.0, 0.0] for n in FIELDS}
    agg_bf16 = {n: [0.0, 0.0] for n in FIELDS}

    for dtype, agg in [(torch.float64, agg_fp64), (torch.float32, agg_fp32), (torch.bfloat16, agg_bf16)]:
        print(f"\n----- dtype={str(dtype).replace('torch.','')} -----")
        for (T, D, gs, seed, ch) in CASES:
            q, k, v, g, beta, scale, h0, dout, dfin = make_inputs(T, D, gs, seed, dtype, device)
            mine = kda_chunked_backward(q, k, v, g, beta, scale, dout, h0, dfin, chunk=ch)
            ref_tok = kda_token_backward(q, k, v, g, beta, scale, dout, h0, dfin)
            au_chk = autograd_chunked(q, k, v, g, beta, scale, h0, dout, dfin, ch)
            au_tok = autograd_token(q, k, v, g, beta, scale, h0, dout, dfin)

            line = f"  T={T:<4d} D={D:<3d} g={gs:<4.1f} ch={ch:<3d} "
            for n in FIELDS:
                # primary comparison: vs independent per-token adjoint
                ma, mr, _ = stats(mine[n], ref_tok[n])
                agg[n][0] = max(agg[n][0], ma)
                agg[n][1] = max(agg[n][1], mr)
                # autograd (chunked graph) sanity
                ma_c, _, _ = stats(mine[n], au_chk[n])
                # autograd (token graph)
                ma_t, _, _ = stats(mine[n], au_tok[n])
                line += f"{n}:{ma:.1e}/{ma_c:.1e}/{ma_t:.1e} "
            print(line)

    print("\n  worst-case  chunked-bwd vs token-bwd  (max_abs / max_rel):")
    for tag, agg in [("fp64", agg_fp64), ("fp32", agg_fp32), ("bf16", agg_bf16)]:
        line = f"    {tag:5s}: "
        for n in FIELDS:
            line += f"{n}={agg[n][0]:.2e}/{agg[n][1]:.2e}  "
        print(line)

    forward_precision_note(device)

    print("\n(column legend per field: chunked-bwd vs token-bwd / vs autograd(chunked) / vs autograd(token))")
    print("\nDONE")


if __name__ == "__main__":
    main()
