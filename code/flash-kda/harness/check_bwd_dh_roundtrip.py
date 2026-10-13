"""P7-31: how much is the B1->B2 `dh` DRAM round-trip worth?

§30.5 concluded that the only remaining *true structural* item is the "K2-shaped
B1->B2 fusion" (register-resident state + TMA + same-CTA handoff), and left it as
**待验证** because a monolithic `BV=V` Triton fusion is register-bound (P7-27).  But
that leaves an obvious open question: *what is the upper bound on its payoff?*  A K2
shape keeps `dS` in registers, so the only traffic it can remove (vs the two-kernel
chain) is the per-chunk `dh` matrix:

    B1 writes dh[c]  ->  DRAM  (bf16 [K,V] per chunk)
    B2 reads  dh[c]  <-  DRAM

Instead of building the (large) fused kernel, this script *measures* the two halves in
isolation with controlled ablations:

  Part A  K2-shaped TMA B1 (`b1_tma_probe.cu`, best `-DB1T_ETMA` variant) built twice:
          baseline vs `-DB1T_NODH` (the global `dh_g` store removed; the per-warp bf16
          `sdh` mirror, needed by op1, is *kept*).  Verify `dv2/dh0` are bit-identical,
          then CUPTI-time both -> the `dh`-store cost.
  Part B  Triton B2 (`wy_dqkg`, `build_kernel(nodh=True)`) built twice: baseline vs a
          no-`dh`-load variant (reuses the already-loaded `b_h`; wrong value, timing
          only).  CUPTI-time -> the `dh`-load cost.
  Part C  synthesize the round-trip upper bound and compare it against the measured
          B1 / B2+B3 chain split (§26), to decide whether the K2-shaped fusion is
          worth its structural cost, or whether (as §26/§29 found for `h/dh` bf16)
          it is only a sub-10% lever.

Run inside `kernel_lab` (idle card):
    CUDA_VISIBLE_DEVICES=1 python .../harness/check_bwd_dh_roundtrip.py \
        > .../harness/check_bwd_dh_roundtrip.out.txt 2>&1
"""
import ctypes
import os
import subprocess
import sys

os.environ.setdefault("FLA_DISABLE_BACKEND_DISPATCH", "1")

import torch

HARNESS = "~/proj/tech_record/code/flash-kda/harness"
SRC = "~/proj/tech_record/code/flash-kda/src"
sys.path.insert(0, HARNESS)
sys.path.insert(0, SRC)

from check_bwd_dhu_dtype import (                                          # noqa: E402
    cuda_kernel_us, stats, make_inputs, as_dtype, fla_dhu, cdiv,
)

D = 128
B1_BASE_SO = "/tmp/b1_tma_e.so"      # ETMA baseline  (§24 best)
B1_NODH_SO = "/tmp/b1_tma_en.so"     # ETMA + B1T_NODH


def build(src_name, so, extra=()):
    src = os.path.join(HARNESS, src_name)
    cmd = ["nvcc", "-std=c++17", "-O3", "-arch=sm_90a", "-Xcompiler", "-fPIC",
           "-shared"] + list(extra) + [src, "-o", so]
    print("build:", " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout)
        print(r.stderr)
        raise SystemExit(f"nvcc failed: {src_name} {' '.join(extra)}")


def load(so):
    lib = ctypes.CDLL(so)
    f = lib.b1t_launch
    f.restype = ctypes.c_int
    f.argtypes = [ctypes.c_int, ctypes.c_int] + [ctypes.c_void_p] * 9 + \
        [ctypes.c_int] * 5 + [ctypes.c_float, ctypes.c_void_p]
    return lib


def ptr(t):
    return ctypes.c_void_p(t.data_ptr())


def run_b1(lib, fi, do, dv, scale, BT, NSTAGE):
    qg, kg, w, g_cum = fi["qg"], fi["kg"], fi["w"], fi["g_cum"]
    B, T, HV, Kk = qg.shape
    V = do.shape[-1]
    NT = cdiv(T, BT)
    last_idx = torch.clamp((torch.arange(NT, device=qg.device) + 1) * BT - 1, max=T - 1)
    etot = torch.exp2(g_cum.float()[:, last_idx]).contiguous()
    dh = torch.empty(B, NT, HV, Kk, V, dtype=torch.bfloat16, device=qg.device)
    dh0 = torch.empty(B, HV, Kk, V, dtype=torch.float32, device=qg.device)
    dv2 = torch.empty_like(do)
    stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
    rc = lib.b1t_launch(BT, NSTAGE, ptr(qg.contiguous()), ptr(kg.contiguous()),
                        ptr(w.contiguous()), ptr(do.contiguous()), ptr(dv.contiguous()),
                        ptr(etot), ptr(dh), ptr(dh0), ptr(dv2),
                        B, T, HV, V, NT, scale, stream)
    if rc != 0:
        raise RuntimeError(f"b1t_launch rc={rc} (BT={BT}, NSTAGE={NSTAGE})")
    return dh, dh0, dv2


def cuda_kernel_us_named(fn, sub, reps=50):
    return cuda_kernel_us(fn, sub, reps)


def main():
    assert torch.cuda.is_available(), "need GPU"
    print(f"GPU: {torch.cuda.get_device_name()}  torch={torch.__version__}")
    build("b1_tma_probe.cu", B1_BASE_SO, extra=["-lcuda", "-DB1T_ETMA"])
    build("b1_tma_probe.cu", B1_NODH_SO, extra=["-lcuda", "-DB1T_ETMA", "-DB1T_NODH"])
    lib_base = load(B1_BASE_SO)
    lib_nodh = load(B1_NODH_SO)
    scale = D ** -0.5

    shapes = [(1, 512, 4), (2, 1024, 8), (8, 2048, 8), (1, 250, 3)]

    # ------------------------------------------------------------------ Part A
    print("\n" + "=" * 92)
    print("Part A: K2-shaped TMA B1 (ETMA) — `dh` store removed (B1T_NODH)")
    print("   correctness: dv2/dh0 must be bit-identical; timing: CUPTI us/call")
    print("=" * 92)
    print(f"  {'shape':16s} {'BT':>3s} {'NS':>3s} | {'base us':>9s} {'nodh us':>9s} "
          f"{'nodh/base':>9s} | dv2 max_abs  dh0 max_abs")
    b1_res = {}
    for (B, T, H) in shapes:
        q, k, v, g_log2, beta, h0 = make_inputs(B, T, H)
        q, k, v, g_log2, beta, h0 = (as_dtype(x, torch.bfloat16)
                                     for x in (q, k, v, g_log2, beta, h0))
        torch.manual_seed(99)
        do = as_dtype(torch.randn(B, T, H, D), torch.bfloat16)
        for BT in (32, 64):
            F = fla_dhu(q, k, v, g_log2, beta, h0, scale, BT, do)
            for NS in (1, 2):
                dh_b, dh0_b, dv2_b = run_b1(lib_base, F["fi"], do, F["dv"], scale, BT, NS)
                dh_n, dh0_n, dv2_n = run_b1(lib_nodh, F["fi"], do, F["dv"], scale, BT, NS)
                us_b = cuda_kernel_us(
                    lambda: run_b1(lib_base, F["fi"], do, F["dv"], scale, BT, NS),
                    "b1_tma_kernel")
                us_n = cuda_kernel_us(
                    lambda: run_b1(lib_nodh, F["fi"], do, F["dv"], scale, BT, NS),
                    "b1_tma_kernel")
                d_dv2 = (dv2_n.float() - dv2_b.float()).abs().max().item()
                d_dh0 = (dh0_n.float() - dh0_b.float()).abs().max().item()
                print(f"  B{B} T{T} H{H:<6} {BT:3d} {NS:3d} | {us_b:9.1f} {us_n:9.1f} "
                      f"{us_n/us_b:9.3f} | {d_dv2:11.2e}  {d_dh0:10.2e}")
                b1_res[(B, T, H, BT, NS)] = (us_b, us_n)
                del dh_b, dh0_b, dv2_b, dh_n, dh0_n, dv2_n

    # ------------------------------------------------------------------ Part B
    print("\n" + "=" * 92)
    print("Part B: Triton B2 `wy_dqkg` — `dh` load removed (build_kernel(nodh=True))")
    print("   (nodh is numerically wrong; timing-only).  CUPTI us/call")
    print("=" * 92)
    import check_bwd_dqkg_kernel as B2
    k_base = B2.build_kernel(nodh=False)
    k_nodh = B2.build_kernel(nodh=True)
    print(f"  {'shape':16s} {'K=V':>5s} {'BT':>3s} | {'base us':>9s} {'nodh us':>9s} "
          f"{'nodh/base':>9s} | delta check (dq max_abs)")
    b2_res = {}
    for (B, T, H, K, V, BT, seed) in [(2, 512, 4, 128, 128, 64, 1),
                                      (2, 2048, 8, 128, 128, 64, 1),
                                      (1, 512, 4, 128, 128, 64, 1),
                                      (1, 250, 3, 64, 128, 64, 2)]:
        inp = B2.make_inputs(B, T, H, K, V, BT, "cuda", dtype=torch.bfloat16, seed=seed)
        inp = {kk: (vv if vv.is_contiguous() else vv.contiguous()) for kk, vv in inp.items()}
        sc = scale if K == 128 else (1.0 / (K ** 0.5))
        out_b = B2.wy_dqkg_triton(k_base, inp, sc, BT)
        out_n = B2.wy_dqkg_triton(k_nodh, inp, sc, BT)
        d_dq = (out_n["dq"].float() - out_b["dq"].float()).abs().max().item()
        us_b = cuda_kernel_us(lambda: B2.wy_dqkg_triton(k_base, inp, sc, BT), "_wy_dqkg_kernel")
        us_n = cuda_kernel_us(lambda: B2.wy_dqkg_triton(k_nodh, inp, sc, BT), "_wy_dqkg_kernel")
        print(f"  B{B} T{T} H{H:<6} {K:3d}/{V:3d} {BT:3d} | {us_b:9.1f} {us_n:9.1f} "
              f"{us_n/us_b:9.3f} | {d_dq:11.2e}")
        b2_res[(B, T, H, K, V, BT)] = (us_b, us_n)
        del inp, out_b, out_n

    # ------------------------------------------------------------------ Part C
    print("\n" + "=" * 92)
    print("Part C: upper bound on a K2-shaped B1->B2 fusion (only the `dh` round-trip)")
    print("   key = (dh store cost in B1) + (dh load cost in B2); chain refs from docs/07 §26")
    print("=" * 92)
    for (B, T, H) in [(2, 1024, 8), (8, 2048, 8)]:
        # B1 at the deployed BT (16->we use 64 as the target-card-shaped prototype);
        # report both NS=1/2.
        for NS in (1, 2):
            kb = (B, T, H, 64, NS)
            if kb not in b1_res:
                continue
            us_b, us_n = b1_res[kb]
            store = us_b - us_n
            print(f"  B{B} T{T} H{H} B1(BT=64 NS={NS}): base {us_b:7.1f}us  "
                  f"nodh {us_n:7.1f}us  => dh-store {store:6.1f}us ({store/us_b*100:4.1f}%)")
        k2 = (B, T, H, 128, 128, 64)
        if k2 in b2_res:
            ub, un = b2_res[k2]
            dh_load = ub - un
            print(f"  B{B} T{T} H{H} B2(wy_dqkg):     base {ub:7.1f}us  "
                  f"nodh {un:7.1f}us  => dh-load  {dh_load:6.1f}us ({dh_load/ub*100:4.1f}%)")

    print("\nDONE")


if __name__ == "__main__":
    main()
