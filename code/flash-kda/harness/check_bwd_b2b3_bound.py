"""P7-43: before committing to the "port the fused B2/B3 to a K2-shaped CUDA kernel"
item named by §45.3, measure *what the fused Triton kernel is actually bound on* and
*how much of its time a latency-hiding restructure could plausibly recover*.

§30 (P7-30) already recorded one `ncu` snapshot of the fused kernel (`probe_b2b3_fused_ncu.out.txt`):
at `B8 T2048 H8`, `BT=16`, `nw=8`, `BV=128` (deploy config) it reads 1.58 GB / writes 0.38 GB,
runs at **DRAM 55.8% / L2 62.2% SOL, tensor 6.2%, achieved occupancy 12.5%** (1 block/SM, pinned
by *both* 255 registers and 165.9 KB smem), with `long_scoreboard` 5.80 the top stall, and ncu
labels it "latency issues".  That is suggestive but not decisive: the same "low SOL => latency"
heuristic was **wrong twice before** in this project (K2 §8.4, B1 §16), where raising occupancy
was impossible and the real lever was a K2-shaped restructure (register-resident state + TMA
pipeline).  So this round answers three cheap questions *without building the CUDA kernel*:

  Part A  baseline: isolated deploy-config dev + static per-thread cost (`regs`/`spills`/`smem`)
          of the Triton fused kernel, at the big deploy shape and a small one.
  Part B  register-pressure knob: launch with an explicit `maxnreg` ceiling.  This isolates
          whether the 255-register cost is *causal* (spills -> slow) or merely an occupancy cap.
  Part C  memory floor: a **traffic-equivalent kernel** -- identical pointers/masks/bytes but
          *no* `tl.dot` (pure load + trivial combine + store), built to run at high occupancy.
          Its dev is the best case a perfectly latency-hidden kernel of the same DRAM traffic
          could reach, i.e. the ceiling a K2-shape port would chase.
  Part D  conclusion / design input for the target-card CUDA port (and the DRAM floor from the
          §30 byte counts).

Run inside `kernel_lab` (idle card), absolute paths:
    CUDA_VISIBLE_DEVICES=2 python .../harness/check_bwd_b2b3_bound.py \
        > .../harness/check_bwd_b2b3_bound.out.txt 2>&1
    B23_MODE=a | b | c
"""
import math
import os
import sys

os.environ.setdefault("FLA_DISABLE_BACKEND_DISPATCH", "1")

import torch
import triton
import triton.language as tl

HARNESS = "~/proj/tech_record/code/flash-kda/harness"
SRC = "~/proj/tech_record/code/flash-kda/src"
sys.path.insert(0, HARNESS)
sys.path.insert(0, SRC)

from check_bwd_dqkg_kernel import make_inputs as dqkg_make_inputs          # noqa: E402
from check_bwd_b2b3_fused import build_fused_b2b3, fused_b2b3_triton        # noqa: E402
from check_bwd_b1_bf16_e2e import time_wall, profile_dev                    # noqa: E402

D = 128


# --------------------------------------------------------------------------- static metadata
def kern_info(kern):
    """(regs, spills, shared_bytes) of the last (only) compiled variant."""
    try:
        cache = list(kern.device_caches.values())[0][0]
        c = next(iter(cache.values()))
        return int(c.n_regs), int(c.n_spills), int(c.metadata.shared)
    except Exception as e:  # noqa: BLE001
        return None, None, str(e)


def occupancy(regs, shared, threads):
    """Analytic blocks/SM and warps/SM on H100 (65536 regs, 228 KB smem, 64 warps/SM)."""
    if regs is None or isinstance(shared, str):
        return None
    by_reg = 65536 // (regs * threads)
    by_smem = (228 * 1024) // (shared + 1024)
    blocks = max(1, min(by_reg, by_smem, 32))
    return blocks, blocks * (threads // 32), 100.0 * blocks * (threads // 32) / 64.0


# --------------------------------------------------------------------------- traffic kernel
def build_traffic():
    """Same pointers/masks/bytes as `_fused_b2b3_kernel`, but no `tl.dot`: every operand is
    loaded once and folded into a store, so the DRAM traffic is byte-identical while the
    arithmetic is trivial -> high occupancy, pure memory ceiling."""

    @triton.jit
    def _traffic_kernel(
        q_ptr, k_ptr, v_ptr, vn_ptr, g_ptr, beta_ptr, A_ptr, h_ptr,
        do_ptr, dh_ptr, dv_ptr, dAqk_ptr,
        dq_ptr, dk_ptr, dv2_ptr, db_ptr, dg_ptr,
        T, NT, scale,
        H: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BT: tl.constexpr,
        BV: tl.constexpr,
    ):
        i_t = tl.program_id(0)
        i_bh = tl.program_id(1)
        i_b = i_bh // H
        i_h = i_bh % H
        o_t = i_t * BT + tl.arange(0, BT)
        m_t = o_t < T
        o_k = tl.arange(0, K)
        m_k = o_k < K
        m_tk = m_t[:, None] & m_k[None, :]

        b_q = tl.load(q_ptr + (i_b * T + o_t)[:, None] * (H * K) + i_h * K + o_k[None, :],
                      mask=m_tk, other=0.0)
        b_k = tl.load(k_ptr + (i_b * T + o_t)[:, None] * (H * K) + i_h * K + o_k[None, :],
                      mask=m_tk, other=0.0)
        b_g = tl.load(g_ptr + (i_b * T + o_t)[:, None] * (H * K) + i_h * K + o_k[None, :],
                      mask=m_tk, other=0.0)
        b_vn = tl.load(vn_ptr + (i_b * T + o_t)[:, None] * (H * V) + i_h * V + o_k[None, :],
                       mask=m_tk, other=0.0)
        # beta [BT]
        b_beta = tl.load(beta_ptr + (i_b * T + o_t) * H + i_h, mask=m_t, other=0.0)
        # A / dAqk [BT,BT]
        p_A = A_ptr + ((i_b * T + i_t * BT) * H + i_h) * BT \
            + tl.arange(0, BT)[:, None] + tl.arange(0, BT)[None, :] * (H * BT)
        b_A = tl.load(p_A, mask=m_t[None, :], other=0.0)
        o_i = tl.arange(0, BT)
        m_jd = (i_t * BT + o_i) < T
        m_diag = m_t[:, None] & m_jd[None, :]
        p_dAq = dAqk_ptr + ((i_b * T + o_t)[:, None] * H + i_h) * BT + o_i[None, :]
        b_dAq = tl.load(p_dAq, mask=m_diag, other=0.0)
        s_aux = tl.sum(b_A) + tl.sum(b_dAq)

        o_v = tl.arange(0, BV)
        m_v = o_v < V
        m_tv = m_t[:, None] & m_v[None, :]
        m_h = m_v[:, None] & m_k[None, :]
        b_v = tl.load(v_ptr + (i_b * T + o_t)[:, None] * (H * V) + i_h * V + o_v[None, :],
                      mask=m_tv, other=0.0)
        b_do = tl.load(do_ptr + (i_b * T + o_t)[:, None] * (H * V) + i_h * V + o_v[None, :],
                       mask=m_tv, other=0.0)
        b_dv = tl.load(dv_ptr + (i_b * T + o_t)[:, None] * (H * V) + i_h * V + o_v[None, :],
                       mask=m_tv, other=0.0)
        b_h = tl.load(h_ptr + ((i_b * NT + i_t) * H + i_h) * K * V
                      + o_k[None, :] * V + o_v[:, None], mask=m_h, other=0.0)
        b_dh = tl.load(dh_ptr + ((i_b * NT + i_t) * H + i_h) * K * V
                       + o_k[None, :] * V + o_v[:, None], mask=m_h, other=0.0)
        s_h = tl.sum(b_h * b_dh)

        dq = b_q + b_k + b_vn + b_g
        dk = b_k + b_q
        dg = b_g + b_q
        dv2 = b_v + b_do + b_dv
        db = b_beta + s_aux + s_h

        tl.store(dq_ptr + (i_b * T + o_t)[:, None] * (H * K) + i_h * K + o_k[None, :],
                 dq, mask=m_tk)
        tl.store(dk_ptr + (i_b * T + o_t)[:, None] * (H * K) + i_h * K + o_k[None, :],
                 dk, mask=m_tk)
        tl.store(dg_ptr + (i_b * T + o_t)[:, None] * (H * K) + i_h * K + o_k[None, :],
                 dg, mask=m_tk)
        tl.store(dv2_ptr + (i_b * T + o_t)[:, None] * (H * V) + i_h * V + o_v[None, :],
                 dv2, mask=m_tv)
        tl.store(db_ptr + (i_b * T + o_t) * H + i_h, db, mask=m_t)

    return _traffic_kernel


def run_traffic(k, inp, dAqk, scale, BT, BV=128, num_warps=4, num_stages=1):
    B, T, H, K = inp["q"].shape
    V = inp["v"].shape[-1]
    NT = (T + BT - 1) // BT
    dev = inp["q"].device
    dq = torch.empty(B, T, H, K, device=dev, dtype=torch.float32)
    dk = torch.empty(B, T, H, K, device=dev, dtype=torch.float32)
    dg = torch.empty(B, T, H, K, device=dev, dtype=torch.float32)
    dv2 = torch.empty(B, T, H, V, device=dev, dtype=torch.float32)
    db = torch.empty(B, T, H, device=dev, dtype=torch.float32)
    k[(NT, B * H)](
        inp["q"], inp["k"], inp["v"], inp["v_new"], inp["g"], inp["beta"], inp["A"], inp["h"],
        inp["do"], inp["dh"], inp["dv"], dAqk,
        dq, dk, dv2, db, dg,
        T, NT, scale, H=H, K=K, V=V, BT=BT, BV=min(BV, V),
        num_warps=num_warps, num_stages=num_stages)


def launch_fused(kernel, inp, dAqk, scale, BT, BV=128, num_warps=8, num_stages=1,
                 maxnreg=None):
    """`fused_b2b3_triton` with an extra `maxnreg` pass-through (Triton >=3.1)."""
    B, T, H, K = inp["q"].shape
    V = inp["v"].shape[-1]
    NT = (T + BT - 1) // BT
    dev = inp["q"].device
    dq = torch.empty(B, T, H, K, device=dev, dtype=torch.float32)
    dk = torch.empty(B, T, H, K, device=dev, dtype=torch.float32)
    dg = torch.empty(B, T, H, K, device=dev, dtype=torch.float32)
    dv2 = torch.empty(B, T, H, V, device=dev, dtype=torch.float32)
    db = torch.empty(B, T, H, device=dev, dtype=torch.float32)
    kw = dict(num_warps=num_warps, num_stages=num_stages)
    if maxnreg is not None:
        kw["maxnreg"] = maxnreg
    kernel[(NT, B * H)](
        inp["q"], inp["k"], inp["v"], inp["v_new"], inp["g"], inp["beta"], inp["A"], inp["h"],
        inp["do"], inp["dh"], inp["dv"], dAqk,
        dq, dk, dv2, db, dg,
        T, NT, scale, H=H, K=K, V=V, BT=BT, BV=min(BV, V), **kw)
    return dict(dq=dq, dk=dk, dv2=dv2, db=db, dg=dg)


# --------------------------------------------------------------------------- inputs
def make_ops(B, T, H, K, V, BT, seed=0):
    inp = dqkg_make_inputs(B, T, H, K, V, BT, "cuda", dtype=torch.float32, seed=seed)
    inp = {k2: (v if v.is_contiguous() else v.contiguous()) for k2, v in inp.items()}
    # deployment dtype mix: dh is bf16 out of CUDA B1 (P7-42); everything else fp32.
    inp["dh"] = inp["dh"].to(torch.bfloat16)
    torch.manual_seed(seed + 100)
    dAqk = torch.randn(B, T, H, BT, device="cuda") * 0.05
    return inp, dAqk


# --------------------------------------------------------------------------- parts
CASES = [
    ("B8 T2048 H8", 8, 2048, 8, 128, 128, 16),
    ("T512 H4", 1, 512, 4, 128, 128, 16),
]
DEPLOY = dict(BV=128, num_warps=8)


def part_a():
    print("=" * 90)
    print("Part A: fused B2/B3 isolated baseline (deploy config BV=128 nw=8) + static cost")
    print("=" * 90)
    fused_k = build_fused_b2b3()
    for (name, B, T, H, K, V, BT) in CASES:
        inp, dAqk = make_ops(B, T, H, K, V, BT, seed=0)
        scale = 1.0 / math.sqrt(K)

        def ff():
            fused_b2b3_triton(fused_k, inp, dAqk, scale, BT, **DEPLOY)

        ff()
        torch.cuda.synchronize()
        dev, _ = profile_dev(ff, reps=20)
        w = time_wall(ff, reps=50)
        regs, spills, shared = kern_info(fused_k)
        occ = occupancy(regs, shared, 256)
        print(f"\n# {name}  grid=({(T + BT - 1)//BT}, {B*H})")
        print(f"  fused dev={dev:8.1f}us  wall={w:8.1f}us  regs={regs}  spills={spills}  "
              f"smem={shared/1024:.1f}KB")
        if occ:
            print(f"  analytic occupancy: {occ[0]} block/SM  {occ[1]} warps/SM  {occ[2]:.2f}%")


def part_b():
    print("\n" + "=" * 90)
    print("Part B: register-pressure knob -- explicit `maxnreg` ceiling on the fused kernel")
    print("  (does the 255-register / 1-block/SM cost buy speed, or is it just an occupancy cap?)")
    print("=" * 90)
    name, B, T, H, K, V, BT = CASES[0]
    inp, dAqk = make_ops(B, T, H, K, V, BT, seed=0)
    scale = 1.0 / math.sqrt(K)

    base_dev = None
    print(f"  {'maxnreg':>8s} {'dev us':>9s} {'regs':>5s} {'spills':>7s} {'smem KB':>8s}  occ(blk,warps,%)")
    for mr in [None, 224, 192, 160, 128, 96, 80]:
        fused_k = build_fused_b2b3()   # fresh: one compiled variant in the cache

        def ff(k=fused_k, mr=mr):
            launch_fused(k, inp, dAqk, scale, BT, BV=128, num_warps=8, maxnreg=mr)

        try:
            ff()
            torch.cuda.synchronize()
        except Exception as e:  # noqa: BLE001
            print(f"  {str(mr):>8s}  FAILED: {type(e).__name__}: {str(e)[:70]}")
            continue
        d, _ = profile_dev(ff, reps=20)
        regs, spills, shared = kern_info(fused_k)
        occ = occupancy(regs, shared, 256)
        tag = "base" if mr is None else str(mr)
        if mr is None:
            base_dev = d
        ratio = f"{d/base_dev:.2f}x" if base_dev else ""
        print(f"  {tag:>8s} {d:9.1f} {regs:>5} {spills:>7} {shared/1024:>8.1f}  "
              f"({occ[0]},{occ[1]},{occ[2]:.1f}) {ratio}")


def part_c():
    print("\n" + "=" * 90)
    print("Part C: memory floor -- traffic-equivalent kernel (same bytes, no tl.dot), high occ")
    print("=" * 90)
    fused_k = build_fused_b2b3()
    for (name, B, T, H, K, V, BT) in CASES:
        inp, dAqk = make_ops(B, T, H, K, V, BT, seed=0)
        scale = 1.0 / math.sqrt(K)

        def ff():
            fused_b2b3_triton(fused_k, inp, dAqk, scale, BT, **DEPLOY)

        ff()
        torch.cuda.synchronize()
        df, _ = profile_dev(ff, reps=20)
        print(f"\n# {name}")
        print(f"  fused                       dev={df:8.1f}us")
        nw_best = None
        for nw in (4, 8):
            tk = build_traffic()

            def fn(tk=tk, nw=nw):
                run_traffic(tk, inp, dAqk, scale, BT, BV=128, num_warps=nw)

            fn()
            torch.cuda.synchronize()
            d, _ = profile_dev(fn, reps=20)
            w = time_wall(fn, reps=50)
            regs, spills, shared = kern_info(tk)
            occ = occupancy(regs, shared, nw * 32)
            nw_best = d if nw_best is None else min(nw_best, d)
            print(f"  traffic nw={nw} (no dot)      dev={d:8.1f}us  wall={w:7.1f}  "
                  f"regs={regs} spills={spills} smem={shared/1024:.1f}KB  "
                  f"occ=({occ[0]},{occ[1]},{occ[2]:.1f})  fused/traffic={df/d:.2f}x")
        print(f"  --> best traffic/quasi-floor {nw_best:.1f}us; fused is {df/nw_best:.2f}x above it")


def part_d():
    print("\n" + "=" * 90)
    print("Part D: DRAM floor from the §30 byte counts (fused B8T2048H8)")
    print("=" * 90)
    reads, writes, peak = 1.58e9, 0.383e9, 3.35e12      # bytes; H100 HBM3 peak
    floor = (reads + writes) / peak
    print(f"  reads={reads/1e9:.2f}GB writes={writes/1e9:.2f}GB total={(reads+writes)/1e9:.2f}GB")
    print(f"  DRAM-peak floor = (reads+writes)/peak = {floor*1e6:.0f}us   "
          f"(§30 fused dev ~1050us => {1050/ (floor*1e6):.2f}x above floor)")
    print("  breakdown of reads: h(fp32,0.5G)+dh(bf16,0.25G) ~48%, "
          "q/k/g/v_new/do/dv(fp32) ~ answer Part A/C.")


def main():
    assert torch.cuda.is_available(), "need GPU"
    print(f"GPU: {torch.cuda.get_device_name()}  torch={torch.__version__}  "
          f"triton={triton.__version__}")
    mode = os.environ.get("B23_MODE", "")
    if mode in ("", "a"):
        part_a()
    if mode in ("", "b"):
        part_b()
    if mode in ("", "c"):
        part_c()
    if mode in ("", "d"):
        part_d()
    print("\nDONE")


if __name__ == "__main__":
    main()
