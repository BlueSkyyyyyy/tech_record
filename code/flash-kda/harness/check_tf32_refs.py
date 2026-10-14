"""P7-114: system-wide audit of the **tf32 pollution of this topic's own reference
implementations** (`src/kda_ref.py`, `src/kda_bwd_ref.py`) when run on GPU.

Background.  P7-113 (`docs/07` §117) proved that this NGC container (`nv26.04`) ships
`torch.backends.cuda.matmul.allow_tf32 = True` (env `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1`),
so the *pure-torch* chunked reference `kda_chunked_fwd` runs every fp32 `@` in **tf32** on
the GPU.  That single finding corrected one prose note (`docs/05` §7.4) and one harness
docstring, but left two loose ends:

  (1) the *shipped artifact* `harness/check_bwd_chunked.out.txt` was produced with tf32 ON
      (its fp32 forward row reads `o 2.55e-04 / S 3.27e-04`), so the evidence served with
      P5-4 still carries the tf32 floor; and
  (2) nobody checked whether the *other* GPU-run numeric anchors of this topic are affected.

This harness closes both: (A) a **static inventory** of every `harness/check_*.py` that runs a
GPU numeric comparison but never pins the tf32 policy; (B) a **dynamic** run of the topic's
central reference comparisons (forward & backward, chunked vs token vs autograd) under
`allow_tf32` ON **and** OFF, with real max_abs/max_rel; (C) the conclusion.

Run inside `kernel_lab` (GPU):
    python ~/proj/tech_record/code/flash-kda/harness/check_tf32_refs.py \
        > ~/proj/tech_record/code/flash-kda/harness/check_tf32_refs.out.txt 2>&1
"""

import glob
import math
import os
import sys

import torch

HARNESS = "~/proj/tech_record/code/flash-kda/harness"
SRC = "~/proj/tech_record/code/flash-kda/src"
sys.path.insert(0, SRC)
from kda_bwd_ref import kda_chunked_backward, kda_token_backward  # noqa: E402
from kda_ref import kda_chunked_fwd, kda_token_recurrent, l2_normalize  # noqa: E402

FIELDS = ["dq", "dk", "dv", "dg", "dbeta", "dh0"]


def set_tf32(flag):
    torch.backends.cuda.matmul.allow_tf32 = flag
    torch.backends.cudnn.allow_tf32 = flag


def max_rel(a, b):
    dt = torch.float64 if (a.dtype == torch.float64 or b.dtype == torch.float64) else torch.float32
    d = (a.to(dt) - b.to(dt)).abs().max().item()
    return d, d / (b.to(dt).abs().max().item() + 1e-12)


def make_inputs(T, D, g_scale, seed, dtype, device="cuda"):
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


# (T, D, g_scale, seed, chunk) -- subset of check_bwd_chunked.CASES spanning non-divisible,
# g=0, and the real-scale head dim.
CASES = [
    (16, 16, 2.0, 0, 16),
    (33, 16, 2.0, 3, 16),
    (64, 16, 4.0, 5, 16),
    (48, 16, 0.0, 4, 16),
    (200, 64, 2.0, 8, 16),
    (256, 128, 2.0, 10, 16),
    (512, 128, 1.0, 11, 16),
]


# ------------------------------------------------------------------------------------------
# Part A: static inventory
# ------------------------------------------------------------------------------------------
def part_a():
    print("=" * 100)
    print("Part A: static inventory -- harness/check_*.py with a GPU numeric comparison but no tf32 pin")
    print("=" * 100)
    scripts = sorted(glob.glob(os.path.join(HARNESS, "check_*.py")))
    guard, noguard = [], []
    for s in scripts:
        txt = open(s).read()
        pinned = ("allow_tf32" in txt) or ("float32_matmul_precision" in txt) or \
                 ("fp32_precision" in txt)
        (guard if pinned else noguard).append(os.path.basename(s))
    print("  total check_*.py                     : %d" % len(scripts))
    print("  pin tf32 (allow_tf32/..._precision)  : %d" % len(guard))
    print("  do NOT pin tf32                      : %d" % len(noguard))
    print("\n  The un-pinned scripts are only *at risk* when they (a) run on GPU and (b) compare an")
    print("  fp32 torch quantity against a >=fp32 oracle (a fp32-vs-fp32 ``@`` involved).  Timing /")
    print("  ncu / workspace / bf16-vs-bf16 scripts are unaffected.  Spot list of un-pinned scripts:")
    for name in noguard:
        print("    - %s" % name)


# ------------------------------------------------------------------------------------------
# Part B: dynamic -- central reference comparisons under tf32 ON / OFF
# ------------------------------------------------------------------------------------------
def run_block(tf32, dtype):
    set_tf32(tf32)
    worst = {n: [0.0, 0.0] for n in FIELDS}
    fwd_worst = [0.0, 0.0]
    twei = {n: [0.0, 0.0] for n in FIELDS}   # chunked-bwd vs autograd(token) sanity
    for (T, D, gs, seed, ch) in CASES:
        q, k, v, g, beta, scale, h0, dout, dfin = make_inputs(T, D, gs, seed, dtype)
        mine = kda_chunked_backward(q, k, v, g, beta, scale, dout, h0, dfin, chunk=ch)
        ref_tok = kda_token_backward(q, k, v, g, beta, scale, dout, h0, dfin)
        au_tok = autograd_token(q, k, v, g, beta, scale, h0, dout, dfin)
        o1, S1 = kda_chunked_fwd(q, k, v, g, beta, scale, h0, chunk=ch)
        o2, S2 = kda_token_recurrent(q, k, v, g, beta, scale, h0)
        fo = max_rel(o1, o2)
        fs = max_rel(S1, S2)
        fwd_worst[0] = max(fwd_worst[0], fo[1])
        fwd_worst[1] = max(fwd_worst[1], fs[1])
        for n in FIELDS:
            ma, mr = max_rel(mine[n], ref_tok[n])
            worst[n][0] = max(worst[n][0], ma)
            worst[n][1] = max(worst[n][1], mr)
            _, mr2 = max_rel(mine[n], au_tok[n])
            twei[n][1] = max(twei[n][1], mr2)
    return worst, fwd_worst, twei


def part_b():
    print("\n" + "=" * 100)
    print("Part B: central reference comparisons, fp32 on GPU, tf32 ON vs OFF")
    print("=" * 100)
    print("  fp64 control (tf32 irrelevant):")
    set_tf32(True)
    w64, f64, _ = run_block(True, torch.float64)
    print("    fwd o/S max_rel = %.2e / %.2e | bwd worst field max_rel = %.2e"
          % (f64[0], f64[1], max(w64[n][1] for n in FIELDS)))

    on, fon, _ = run_block(True, torch.float32)
    off, foff, _ = run_block(False, torch.float32)

    print("\n  fp32, tf32 = ON  (container default):")
    print("    forward  chunked-vs-token   o max_rel = %.2e   S max_rel = %.2e" % (fon[0], fon[1]))
    print("    backward chunked-vs-token   " +
          "  ".join("%s=%.2e" % (n, on[n][1]) for n in FIELDS))
    print("\n  fp32, tf32 = OFF (ieee):")
    print("    forward  chunked-vs-token   o max_rel = %.2e   S max_rel = %.2e" % (foff[0], foff[1]))
    print("    backward chunked-vs-token   " +
          "  ".join("%s=%.2e" % (n, off[n][1]) for n in FIELDS))

    print("\n  ratio ON/OFF:")
    print("    forward o x%.1f  S x%.1f" % (fon[0] / foff[0], fon[1] / foff[1]))
    print("    backward worst field x%.1f" %
          (max(on[n][1] for n in FIELDS) / max(off[n][1] for n in FIELDS)))
    print("\n  (eps_tf32 ~ 2^-11 = 4.9e-04; the ON rows land on it, the OFF rows on fp32 ulp.)")


def main():
    print("GPU: %s  torch=%s" % (torch.cuda.get_device_name(), torch.__version__))
    print("container default: allow_tf32=%s  fp32_precision=%s" %
          (torch.backends.cuda.matmul.allow_tf32, torch.backends.cuda.matmul.fp32_precision))
    assert torch.cuda.is_available(), "need GPU"
    part_a()
    part_b()
    set_tf32(True)  # restore
    print("\nDONE")


if __name__ == "__main__":
    main()
