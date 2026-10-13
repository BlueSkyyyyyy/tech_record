"""P7-44: first slice of the K2-shaped CUDA port of the fused B2/B3 -- the B2
(`wy_dqkg`) **V-reduction GEMM core** in plain, high-occupancy SIMT CUDA.

Context (docs/07 §46 / P7-43).  §46.6 named the next real engineering step as
porting the fused B2/B3 to a K2-shaped CUDA kernel whose *first* action is

    "把 `dq/dk/dw [BT,K]` 累加器按 warp 分片常驻寄存器（先），TMA 输入流水（次）".

§46 also measured *why* the Triton fused kernel cannot do that itself: at the
deployed `B8 T2048 H8` it is pinned to 1 block/SM by **both** 255 registers and
114 KB smem, so `Eligible Warps/Scheduler` is 0.18 and it runs 1.94x above a
same-traffic no-`tl.dot` kernel.  This round takes the first, cheapest slice of the
CUDA port -- the three V-reduction GEMMs `dq = do@b_h`, `dk = v_new@b_dh`,
`dw = dv@b_h` plus the column reduction `dgk = sum_v h*dh` -- and asks the
*decisive, isolated* question:

    Does a plain SIMT CUDA kernel that keeps `dq/dk/dw [BT,K]` register-resident and
    runs at high occupancy beat an *identical-op* Triton kernel in the fused
    kernel's program shape?

  Part A  correctness: CUDA core vs torch (fp32) and vs the identical-op Triton core.
  Part B  head-to-head kernel-only timing (CUPTI dev + wall): CUDA vs Triton core.
  Part C  context: the CUDA core vs the *full* Triton B2 and the *full* fused B2/B3
          on the same inputs (the core is a strict subset -> an upper bound on what
          the port must still add), and effective FLOP/s.
  Part D  static cost: CUDA smem/regs vs the Triton core/fused static metadata.

Scope boundary (honest): this file implements *only* the GEMM core; the B2 finalize
(gate chain, `dkgb = A@dw`, `dA = dv@v^T`), `dv2`/`db`, and the intra-diagonal fold
are **not** ported here -- see docs/07 §47.6.  What it establishes is whether the
"register-resident accumulators + occupancy" premise holds at the dominant GEMMs.

Run inside `kernel_lab` (idle card), absolute paths:
    CUDA_VISIBLE_DEVICES=2 python .../harness/check_bwd_b2_core.py \
        > .../harness/check_bwd_b2_core.out.txt 2>&1
    B2C_MODE=a | b | c | d
"""
import ctypes
import math
import os
import subprocess
import sys

os.environ.setdefault("FLA_DISABLE_BACKEND_DISPATCH", "1")

import torch

# The CUDA core accumulates in true fp32; make the torch reference exact too, so the
# CUDA-vs-torch delta is only summation order (~1e-6), not torch's cublas tf32 default
# (~1e-3, which is what Triton's `tl.dot` also is).
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

HARNESS = "~/proj/tech_record/code/flash-kda/harness"
SRC = "~/proj/tech_record/code/flash-kda/src"
sys.path.insert(0, HARNESS)
sys.path.insert(0, SRC)

from check_bwd_dqkg_kernel import (make_inputs as dqkg_make_inputs,   # noqa: E402
                                   build_kernel as build_dqkg, wy_dqkg_triton)
from check_bwd_b2b3_fused import build_fused_b2b3, fused_b2b3_triton   # noqa: E402
from check_bwd_ws_reuse import stats, cdiv                             # noqa: E402
from check_bwd_b1_bf16_e2e import time_wall, profile_dev               # noqa: E402

D = 128
SO = "/tmp/b2_core.so"
NTHR = 256


def build():
    src = os.path.join(HARNESS, "b2_core_probe.cu")
    extra = os.environ.get("B2C_CFLAGS", "").split()
    cmd = ["nvcc", "-std=c++17", "-O3", "-arch=sm_90a", "-Xcompiler", "-fPIC",
           "-shared"] + extra + [src, "-o", SO]
    print("build:", " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout)
        print(r.stderr)
        raise SystemExit("nvcc failed")


def load():
    lib = ctypes.CDLL(SO)
    lib.b2_core_launch.restype = ctypes.c_int
    lib.b2_core_launch.argtypes = [ctypes.c_void_p] * 10 + [ctypes.c_int] * 4 + \
        [ctypes.c_void_p]
    lib.b2_core_smem_bytes.restype = ctypes.c_int
    return lib


def ptr(t):
    return ctypes.c_void_p(t.data_ptr())


def run_cuda(lib, inp, BT):
    B, T, H, K = inp["q"].shape
    V = inp["v"].shape[-1]
    assert K == 128 and V == 128, "kernel is compiled for K=V=128 (deployment)"
    NT = cdiv(T, BT)
    dev = inp["q"].device
    for name in ("v", "v_new", "h", "dh", "do", "dv"):
        assert inp[name].dtype == torch.float32
    dq = torch.empty(B, T, H, K, device=dev, dtype=torch.float32)
    dk = torch.empty(B, T, H, K, device=dev, dtype=torch.float32)
    dw = torch.empty(B, T, H, K, device=dev, dtype=torch.float32)
    dgk = torch.empty(B, NT, H, K, device=dev, dtype=torch.float32)
    stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
    rc = lib.b2_core_launch(
        ptr(inp["v"]), ptr(inp["v_new"]), ptr(inp["h"]), ptr(inp["dh"]),
        ptr(inp["do"]), ptr(inp["dv"]), ptr(dq), ptr(dk), ptr(dw), ptr(dgk),
        B, T, H, NT, stream)
    if rc != 0:
        raise RuntimeError(f"b2_core_launch rc={rc}")
    return dict(dq=dq, dk=dk, dw=dw, dgk=dgk)


# --------------------------------------------------------------- torch reference
def _pad_rows(x, B, T, NT, BT, H, W):
    """Pad the token axis [B,T,H,W] to NT*BT (zero pad) so chunk views are legal."""
    dev = x.device
    out = torch.zeros(B, NT * BT, H, W, device=dev, dtype=torch.float32)
    out[:, :T] = x.float()
    return out.view(B, NT, BT, H, W)


def ref_core(inp, BT):
    q = inp["q"].float()
    B, T, H, K = q.shape
    V = inp["v"].shape[-1]
    NT = cdiv(T, BT)
    do4 = _pad_rows(inp["do"], B, T, NT, BT, H, V)
    vn4 = _pad_rows(inp["v_new"], B, T, NT, BT, H, V)
    dv4 = _pad_rows(inp["dv"], B, T, NT, BT, H, V)
    # h/dh are already chunk-granular [B,NT,H,K,V]
    h4 = inp["h"].float().view(B, NT, H, K, V)
    dh4 = inp["dh"].float().view(B, NT, H, K, V)
    # dq[b,c,t,h,k] = sum_v do[b,c,t,h,v] * h[b,c,k,h,v]
    dq = torch.einsum("bcthv,bchkv->bcthk", do4, h4).reshape(B, NT * BT, H, K)[:, :T]
    dk = torch.einsum("bcthv,bchkv->bcthk", vn4, dh4).reshape(B, NT * BT, H, K)[:, :T]
    dw = torch.einsum("bcthv,bchkv->bcthk", dv4, h4).reshape(B, NT * BT, H, K)[:, :T]
    dgk = (h4 * dh4).sum(dim=-1)                                   # [B,NT,H,K]
    return dict(dq=dq.contiguous(), dk=dk.contiguous(), dw=dw.contiguous(), dgk=dgk)


# --------------------------------------------------------------- identical-op triton
def build_core_triton():
    import triton
    import triton.language as tl

    @triton.jit
    def _b2core_kernel(vn_ptr, h_ptr, dh_ptr, do_ptr, dv_ptr,
                       dq_ptr, dk_ptr, dw_ptr, dgk_ptr,
                       T, NT, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
                       BT: tl.constexpr):
        i_t = tl.program_id(0)
        i_bh = tl.program_id(1)
        i_b = i_bh // H
        i_h = i_bh % H
        o_t = i_t * BT + tl.arange(0, BT)
        m_t = o_t < T
        o_k = tl.arange(0, K)
        o_v = tl.arange(0, V)
        m_tv = m_t[:, None] & (o_v < V)[None, :]
        pv = ((i_b * T + o_t)[:, None]) * (H * V) + i_h * V + o_v[None, :]
        b_do = tl.load(do_ptr + pv, mask=m_tv, other=0.0)
        b_vn = tl.load(vn_ptr + pv, mask=m_tv, other=0.0)
        b_dv = tl.load(dv_ptr + pv, mask=m_tv, other=0.0)
        ph = ((i_b * NT + i_t) * H + i_h) * K * V + o_k[None, :] * V + o_v[:, None]
        b_h = tl.load(h_ptr + ph, mask=(o_v[:, None] < V) & (o_k[None, :] < K), other=0.0)
        b_dh = tl.load(dh_ptr + ph, mask=(o_v[:, None] < V) & (o_k[None, :] < K), other=0.0)
        b_dq = tl.dot(b_do, b_h)
        b_dk = tl.dot(b_vn, b_dh)
        b_dw = tl.dot(b_dv, b_h)
        b_dgk = tl.sum(b_h * b_dh, axis=0)
        pq = ((i_b * T + o_t)[:, None]) * (H * K) + i_h * K + o_k[None, :]
        mtk = m_t[:, None] & (o_k < K)[None, :]
        tl.store(dq_ptr + pq, b_dq, mask=mtk)
        tl.store(dk_ptr + pq, b_dk, mask=mtk)
        tl.store(dw_ptr + pq, b_dw, mask=mtk)
        tl.store(dgk_ptr + ((i_b * NT + i_t) * H + i_h) * K + o_k, b_dgk)

    return _b2core_kernel


def run_triton_core(k, inp, BT, num_warps=4, num_stages=1):
    B, T, H, K = inp["q"].shape
    V = inp["v"].shape[-1]
    NT = cdiv(T, BT)
    dev = inp["q"].device
    dq = torch.empty(B, T, H, K, device=dev, dtype=torch.float32)
    dk = torch.empty(B, T, H, K, device=dev, dtype=torch.float32)
    dw = torch.empty(B, T, H, K, device=dev, dtype=torch.float32)
    dgk = torch.empty(B, NT, H, K, device=dev, dtype=torch.float32)
    k[(NT, B * H)](inp["v_new"], inp["h"], inp["dh"], inp["do"], inp["dv"],
                   dq, dk, dw, dgk, T, NT, H=H, K=K, V=V, BT=BT,
                   num_warps=num_warps, num_stages=num_stages)
    return dict(dq=dq, dk=dk, dw=dw, dgk=dgk)


def kern_info(kern):
    try:
        cache = list(kern.device_caches.values())[0][0]
        c = next(iter(cache.values()))
        return int(c.n_regs), int(c.n_spills), int(c.metadata.shared)
    except Exception as e:  # noqa: BLE001
        return None, None, str(e)


CASES = [
    ("B8 T2048 H8", 8, 2048, 8, 16),
    ("B2 T1024 H8", 2, 1024, 8, 16),
    ("T512 H4", 1, 512, 4, 16),
    ("T250 H3 (non-div)", 1, 250, 3, 16),
]
BT = 16


def make(B, T, H, seed=0):
    inp = dqkg_make_inputs(B, T, H, 128, 128, BT, "cuda", dtype=torch.float32, seed=seed)
    inp = {k: (v if v.is_contiguous() else v.contiguous()) for k, v in inp.items()}
    return inp


def part_a(lib):
    print("=" * 92)
    print("Part A: correctness -- CUDA core vs torch(fp32) and vs identical-op Triton core")
    print("=" * 92)
    worst = {}
    for (name, B, T, H, btc) in CASES:
        inp = make(B, T, H, seed=1)
        ref = ref_core(inp, BT)
        cu = run_cuda(lib, inp, BT)
        tk = build_core_triton()
        tri = run_triton_core(tk, inp, BT)
        print(f"\n# {name}  grid=({cdiv(T, BT)}, {B*H})")
        for fld in ("dq", "dk", "dw", "dgk"):
            _, mr_c = stats(f"  cuda {fld} vs torch", cu[fld], ref[fld])
            _, mr_t = stats(f"  trit {fld} vs torch", tri[fld], ref[fld])
            stats(f"  cuda {fld} vs trit", cu[fld], tri[fld])
            worst[f"cuda_{fld}"] = max(worst.get(f"cuda_{fld}", 0.0), mr_c)
            worst[f"trit_{fld}"] = max(worst.get(f"trit_{fld}", 0.0), mr_t)
    print("\nworst max_rel vs torch:")
    for kk in sorted(worst):
        print(f"  {kk:12s} {worst[kk]:.3e}")


def part_b(lib):
    print("\n" + "=" * 92)
    print("Part B: head-to-head identical-op timing -- plain SIMT CUDA vs Triton core")
    print("=" * 92)
    print(f"  {'case':18s} | {'CUDA dev':>9s} {'CUDA wall':>9s} | "
          f"{'trit nw4 dev':>12s} {'trit nw8 dev':>12s} | {'CUDA/trit':>9s}")
    for (name, B, T, H, btc) in CASES:
        inp = make(B, T, H, seed=1)
        run_cuda(lib, inp, BT); torch.cuda.synchronize()
        dcu, _ = profile_dev(lambda: run_cuda(lib, inp, BT), reps=30)
        wcu = time_wall(lambda: run_cuda(lib, inp, BT), reps=50)
        tk = build_core_triton()
        run_triton_core(tk, inp, BT, num_warps=4); torch.cuda.synchronize()
        d4, _ = profile_dev(lambda: run_triton_core(tk, inp, BT, num_warps=4), reps=30)
        tk8 = build_core_triton()
        run_triton_core(tk8, inp, BT, num_warps=8); torch.cuda.synchronize()
        d8, _ = profile_dev(lambda: run_triton_core(tk8, inp, BT, num_warps=8), reps=30)
        best = min(d4, d8)
        print(f"  {name:18s} | {dcu:9.1f} {wcu:9.1f} | {d4:12.1f} {d8:12.1f} | "
              f"{dcu/best:8.2f}x")


def part_c(lib):
    print("\n" + "=" * 92)
    print("Part C: context -- CUDA core vs full Triton B2 and full fused B2/B3 (same inputs)")
    print("=" * 92)
    dqkg_k = build_dqkg()
    fused_k = build_fused_b2b3()
    tk = build_core_triton()
    print(f"  {'case':18s} | {'CUDAcore':>9s} {'TritCore':>9s} {'fullB2':>9s} "
          f"{'fusedB2B3':>10s} | {'GF/s(core)':>10s} {'core/fullB2':>11s}")
    for (name, B, T, H, btc) in CASES:
        inp = make(B, T, H, seed=1)
        scale = 1.0 / math.sqrt(128)
        torch.manual_seed(7)
        dAqk = torch.randn(B, T, H, BT, device="cuda") * 0.05

        run_cuda(lib, inp, BT); torch.cuda.synchronize()
        dcu, _ = profile_dev(lambda: run_cuda(lib, inp, BT), reps=20)
        run_triton_core(tk, inp, BT, num_warps=8); torch.cuda.synchronize()
        dtr, _ = profile_dev(lambda: run_triton_core(tk, inp, BT, num_warps=8), reps=20)
        wy_dqkg_triton(dqkg_k, inp, scale, BT, num_warps=4); torch.cuda.synchronize()
        d2, _ = profile_dev(lambda: wy_dqkg_triton(dqkg_k, inp, scale, BT, num_warps=4), reps=20)
        fused_b2b3_triton(fused_k, inp, dAqk, scale, BT); torch.cuda.synchronize()
        df, _ = profile_dev(lambda: fused_b2b3_triton(fused_k, inp, dAqk, scale, BT), reps=20)
        # FLOP of the core = 3 GEMMs of [BT,V]x[V,K] over all chunks = 6*B*T*H*K*V
        flop = 6.0 * B * T * H * 128 * 128
        gfs = flop / (dcu * 1e-6) / 1e9
        print(f"  {name:18s} | {dcu:9.1f} {dtr:9.1f} {d2:9.1f} {df:10.1f} | "
              f"{gfs:10.1f} {dcu/d2:10.2f}x")


def part_d(lib):
    print("\n" + "=" * 92)
    print("Part D: static cost")
    print("=" * 92)
    print(f"  CUDA core smem = {lib.b2_core_smem_bytes()} B, block = {NTHR} threads (8 warps)")
    # 228 KB smem/SM, 65536 regs/SM -> compute headroom under an assumed reg count.
    smem = lib.b2_core_smem_bytes()
    print(f"  smem-limited blocks/SM (228KB/{smem / 1024:.1f}KB) ~ "
          f"{233472 // (smem + 1024)}")
    tk = build_core_triton()
    inp = make(8, 2048, 8, seed=1)
    run_triton_core(tk, inp, BT, num_warps=4)
    r4 = kern_info(tk)
    tk8 = build_core_triton()
    run_triton_core(tk8, inp, BT, num_warps=8)
    r8 = kern_info(tk8)
    fk = build_fused_b2b3()
    torch.manual_seed(7)
    dAqk = torch.randn(8, 2048, 8, BT, device="cuda") * 0.05
    fused_b2b3_triton(fk, inp, dAqk, 1.0 / math.sqrt(128), BT)
    rf = kern_info(fk)
    print(f"  triton core nw=4 : regs={r4[0]} spills={r4[1]} smem={r4[2]/1024:.1f}KB" if r4[0] is not None else str(r4))
    print(f"  triton core nw=8 : regs={r8[0]} spills={r8[1]} smem={r8[2]/1024:.1f}KB" if r8[0] is not None else str(r8))
    print(f"  triton fused nw=8: regs={rf[0]} spills={rf[1]} smem={rf[2]/1024:.1f}KB" if rf[0] is not None else str(rf))
    try:
        import triton
        print(f"  nvcc/triton: triton={triton.__version__}")
    except Exception:  # noqa: BLE001
        pass


def main():
    assert torch.cuda.is_available(), "need GPU"
    print(f"GPU: {torch.cuda.get_device_name()}  torch={torch.__version__}")
    build()
    lib = load()
    mode = os.environ.get("B2C_MODE", "")
    if mode in ("", "a"):
        part_a(lib)
    if mode in ("", "b"):
        part_b(lib)
    if mode in ("", "c"):
        part_c(lib)
    if mode in ("", "d"):
        part_d(lib)
    print("\nDONE")


if __name__ == "__main__":
    main()
