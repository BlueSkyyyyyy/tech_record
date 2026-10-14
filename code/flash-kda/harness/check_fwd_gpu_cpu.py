"""Close the residual ``docs/05`` §7.4 / §116.6 ``待验证``: why does the fp32
blocked forward (``kda_ref.kda_chunked_fwd``) differ from the per-token golden
(``kda_ref.kda_token_recurrent``) by ~3e-4 on **GPU** while only ~2e-7 on **CPU**?

The old write-up (``check_bwd_chunked.py`` docstring + ``docs/05`` §7.4) attributed
it to "GPU cuSOLVER-fp32 ``(I+L)^{-1}`` being much less accurate than CPU LAPACK".
The real cause is that on this container the **NVIDIA NGC build defaults
``torch.backends.cuda.matmul.allow_tf32 = True``** (env ``TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1``),
so every fp32 matmul inside the blocked forward runs in tf32 on GPU.

Parts:
  A  forward chunked-vs-token precision: {CPU, GPU} x {tf32, ieee} x several (T,D).
  B  isolate the solve: (B1) cuSOLVER P on a real KDA block, CPU vs GPU;
     (B2) GPU fp32 with tf32 ON but ``_chunk_inv`` forced to an fp64 solve -> if the
     gap persists the solve is exonerated; (B3) tf32 OFF restores ~1e-7.
  C  knock-on: the fp32 *chunked backward* vs token backward gap (P5-4 table) collapses
     when tf32 is off.

Usage (inside kernel_lab; CPU-only works, GPU parts skipped):
    python ~/proj/tech_record/code/flash-kda/harness/check_fwd_gpu_cpu.py
"""

import os
import math
import sys

import torch

SRC = os.path.expanduser("~/proj/tech_record/code/flash-kda/src")
sys.path.insert(0, SRC)
import kda_ref  # noqa: E402
from kda_bwd_ref import kda_chunked_backward, kda_token_backward  # noqa: E402
from kda_ref import kda_chunked_fwd, kda_token_recurrent, l2_normalize  # noqa: E402


def set_tf32(flag):
    torch.backends.cuda.matmul.allow_tf32 = flag
    try:
        torch.backends.cuda.matmul.fp32_precision = "tf32" if flag else "ieee"
    except Exception:
        pass


def mk(T, D, dt, dev, seed=0):
    torch.manual_seed(seed)
    q = l2_normalize(torch.randn(T, D)).to(dt).to(dev)
    k = l2_normalize(torch.randn(T, D)).to(dt).to(dev)
    v = torch.randn(T, D).to(dt).to(dev)
    g = (-torch.rand(T, D) * 2.0).to(dt).to(dev)
    beta = torch.rand(T).to(dt).to(dev)
    h0 = torch.randn(D, D).to(dt).to(dev)
    return q, k, v, g, beta, h0, 1.0 / math.sqrt(D)


def rel(a, b):
    return ((a - b).abs().max() / b.abs().max()).item()


def part_a(has_cuda):
    print("=" * 78)
    print("Part A  forward  kda_chunked_fwd  vs  kda_token_recurrent   (max_rel of o)")
    print("=" * 78)
    print(f"  torch.backends.cuda.matmul.allow_tf32 (container default) = {torch.backends.cuda.matmul.allow_tf32}")
    print(f"  {'T':>4} {'D':>4} | {'fp64 cpu':>10} {'fp32 cpu':>10} | "
          f"{'fp32 gpu ieee':>13} {'fp32 gpu tf32':>13} | {'fp64 gpu':>10}")
    for (T, D) in [(16, 16), (64, 64), (256, 128), (512, 128)]:
        def fwd(dt, dev, tf32=None):
            if dev == "cuda" and tf32 is not None:
                set_tf32(tf32)
            qq, kk, vv, gg, bb, hh, ss = mk(T, D, dt, dev)
            o1 = kda_chunked_fwd(qq, kk, vv, gg, bb, ss, hh, chunk=16)[0]
            o2 = kda_token_recurrent(qq, kk, vv, gg, bb, ss, hh)[0]
            return rel(o1, o2)
        r64c = fwd(torch.float64, "cpu")
        r32c = fwd(torch.float32, "cpu")
        if has_cuda:
            r32g_ieee = fwd(torch.float32, "cuda", tf32=False)
            r32g_tf32 = fwd(torch.float32, "cuda", tf32=True)
            r64g = fwd(torch.float64, "cuda", tf32=False)
        else:
            r32g_ieee = r32g_tf32 = r64g = float("nan")
        print(f"  {T:>4} {D:>4} | {r64c:>10.2e} {r32c:>10.2e} | "
              f"{r32g_ieee:>13.2e} {r32g_tf32:>13.2e} | {r64g:>10.2e}")
    set_tf32(True)  # restore container default


def part_b(dev):
    print("=" * 78)
    print("Part B  isolate the solve  (T=64 D=64, real KDA block, fp32)")
    print("=" * 78)
    q, k, v, g, beta, h0, sc = mk(64, 64, torch.float32, dev)
    gc = g.cumsum(0)
    E = torch.exp2(gc)
    kd = k * E
    ki = k * torch.exp2(-gc)
    L = torch.tril(kd @ ki.transpose(0, 1), diagonal=-1) * beta[:, None]
    eye = torch.eye(64, device=dev)
    Pg = torch.linalg.solve(eye + L, eye)
    Pc = torch.linalg.solve((eye + L).cpu(), eye.cpu())
    print(f"  B1 cuSOLVER P, cpu vs gpu max_rel = {rel(Pc.cpu(), Pg.cpu()):.2e}   "
          f"cond(I+L) = {torch.linalg.cond(eye + L).item():.4f}")

    orig = kda_ref._chunk_inv

    def inv_fp64(Lx):
        n = Lx.shape[0]
        e = torch.eye(n, dtype=torch.float64, device=Lx.device)
        return torch.linalg.solve(e + Lx.double(), e).to(Lx.dtype)

    set_tf32(True)
    o1 = kda_chunked_fwd(q, k, v, g, beta, sc, h0, chunk=16)[0]
    o2 = kda_token_recurrent(q, k, v, g, beta, sc, h0)[0]
    print(f"  B2 tf32=ON  + fp32 solve  : o max_rel = {rel(o1, o2):.2e}")
    kda_ref._chunk_inv = inv_fp64
    o1b = kda_chunked_fwd(q, k, v, g, beta, sc, h0, chunk=16)[0]
    print(f"  B2 tf32=ON  + fp64 solve  : o max_rel = {rel(o1b, o2):.2e}   "
          f"(unchanged -> solve exonerated)")
    kda_ref._chunk_inv = orig
    set_tf32(False)
    o1c = kda_chunked_fwd(q, k, v, g, beta, sc, h0, chunk=16)[0]
    print(f"  B3 tf32=OFF + fp32 solve  : o max_rel = {rel(o1c, o2):.2e}")
    set_tf32(True)


BWD_CASES = [(16, 16, 2.0, 0), (64, 16, 4.0, 5), (256, 128, 2.0, 10)]


def grad_gap(dev, dt, tf32):
    set_tf32(tf32)
    worst = 0.0
    for (T, D, gs, seed) in BWD_CASES:
        torch.manual_seed(seed)
        q = l2_normalize(torch.randn(T, D)).to(dt).to(dev)
        k = l2_normalize(torch.randn(T, D)).to(dt).to(dev)
        v = torch.randn(T, D).to(dt).to(dev)
        g = (-torch.rand(T, D) * gs).to(dt).to(dev)
        beta = torch.rand(T).to(dt).to(dev)
        h0 = torch.randn(D, D).to(dt).to(dev)
        sc = 1.0 / math.sqrt(D)
        dout = torch.randn(T, D).to(dt).to(dev)
        dfin = torch.randn(D, D).to(dt).to(dev)
        mine = kda_chunked_backward(q, k, v, g, beta, sc, dout, h0, dfin, chunk=16)
        ref = kda_token_backward(q, k, v, g, beta, sc, dout, h0, dfin)
        for n in ("dq", "dk", "dv", "dg", "dbeta", "dh0"):
            worst = max(worst, rel(mine[n], ref[n]))
    return worst


def part_c(has_cuda):
    print("=" * 78)
    print("Part C  knock-on: chunked-bwd vs token-bwd  (worst max_rel over 6 grads, 3 cases)")
    print("=" * 78)
    print(f"  cpu fp32 ieee : {grad_gap('cpu', torch.float32, False):.2e}")
    if has_cuda:
        print(f"  gpu fp32 tf32 : {grad_gap('cuda', torch.float32, True):.2e}")
        print(f"  gpu fp32 ieee : {grad_gap('cuda', torch.float32, False):.2e}")
        print(f"  gpu fp64      : {grad_gap('cuda', torch.float64, False):.2e}")
    set_tf32(True)


def main():
    has_cuda = torch.cuda.is_available()
    print(f"device: cuda={has_cuda}  torch={torch.__version__}")
    part_a(has_cuda)
    if has_cuda:
        part_b("cuda")
    part_c(has_cuda)
    print("\nDONE")


if __name__ == "__main__":
    main()
