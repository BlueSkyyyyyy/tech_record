"""P7-40: close P7-39's host-tax open item -- is the CUDA-B1 small-shape regression really
the per-launch `cuTensorMapEncodeTiled` re-encoding?

P7-39 (`docs/07` §42.4/§42.6) landed the K2-shaped CUDA B1 into the assembled reverse chain
and saw a `T=250 H=3` chain-wall ratio of **1.65x** vs the Triton-B1 chain, attributing it to
"`launch_impl` re-encodes 5-8 tensor maps every launch + a `cudaFuncSetAttribute` check; a
small shape is host-bound".  That attribution was a *hypothesis* (marked 待验证).  This round
tests it directly:

  * add an **encode-once** tensor-map cache to `b1_tma_probe.cu` (`-DB1T_MAPCACHE`): while
    every operand pointer and shape is unchanged the `MapSet` is reused; a miss counter
    (`b1t_mapcache_miss`) proves the cache actually hits;
  * a probe entry `b1t_encode_only(...)` times the *pure host-side* encoding (no launch), so
    the encode cost can be separated from the launch/driver cost;
  * re-measure the chain wall on an idle card, several rounds, to see whether the 1.65x
    reproduces at all.

Parts
-----
  A  host-cost decomposition: pure encode (`b1t_encode_only`) vs full launch
     (`b1t_launch`), each with/without `B1T_MAPCACHE`, on a tiny host-bound shape and on
     `T=250 H=3`.  `process_time` (P7-6 methodology: single-threaded CPU time).
  B  correctness of the cache: with nonzero `dht`, cache vs no-cache `dh`/`dv2`/`dh0`
     must be bitwise identical (the cache is a pure host-side reuse).
  C  chain-wall reproducibility: Triton-B1 vs CUDA-B1 (no-cache / cache) chain, R rounds,
     to check the 1.65x and the run-to-run spread of the Triton baseline.

Run inside `kernel_lab` (idle card):
    CUDA_VISIBLE_DEVICES=1 python .../harness/check_bwd_b1_launch_tax.py \
        > .../harness/check_bwd_b1_launch_tax.out.txt 2>&1
    B1LT_MODE=abc   # subset of parts
"""
import ctypes
import os
import statistics
import subprocess
import sys
import time

os.environ.setdefault("FLA_DISABLE_BACKEND_DISPATCH", "1")

import torch

HARNESS = os.path.expanduser("~/proj/tech_record/code/flash-kda/harness")
sys.path.insert(0, HARNESS)

D = 128
DEPLOY_FLAGS = ["-lcuda", "-DB1T_ETMA", "-DB1T_TMADH", "-DB1T_TMADV",
                "-DB1T_EPROXY", "-DB1T_SPROXY"]
SO_NOCACHE = "/tmp/b1_tma_tax_nc.so"
SO_CACHE = "/tmp/b1_tma_tax_c.so"


def build(extra, so):
    src = os.path.join(HARNESS, "b1_tma_probe.cu")
    cmd = ["nvcc", "-std=c++17", "-O3", "-arch=sm_90a", "-Xcompiler", "-fPIC",
           "-shared"] + DEPLOY_FLAGS + list(extra) + [src, "-o", so]
    print("build:", " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout)
        print(r.stderr)
        raise SystemExit("nvcc failed")
    return so


def load(so):
    lib = ctypes.CDLL(so)
    lib.b1t_launch.restype = ctypes.c_int
    lib.b1t_launch.argtypes = [ctypes.c_int, ctypes.c_int] + [ctypes.c_void_p] * 9 + \
        [ctypes.c_int] * 5 + [ctypes.c_float, ctypes.c_void_p]
    lib.b1t_launch_dht.restype = ctypes.c_int
    lib.b1t_launch_dht.argtypes = [ctypes.c_int, ctypes.c_int] + [ctypes.c_void_p] * 10 + \
        [ctypes.c_int] * 5 + [ctypes.c_float, ctypes.c_void_p]
    lib.b1t_encode_only.restype = ctypes.c_int
    lib.b1t_encode_only.argtypes = [ctypes.c_int] + [ctypes.c_void_p] * 8 + [ctypes.c_int] * 4
    return lib


def ptr(t):
    return ctypes.c_void_p(t.data_ptr())


def mk_shape(B, T, H, BT, dev="cuda"):
    NT = (T + BT - 1) // BT
    K = V = D
    g = 0.15
    q = (torch.randn(B, T, H, K, device=dev) * g).to(torch.bfloat16)
    k = (torch.randn(B, T, H, K, device=dev) * g).to(torch.bfloat16)
    w = (torch.randn(B, T, H, K, device=dev) * g).to(torch.bfloat16)
    do = (torch.randn(B, T, H, V, device=dev) * g).to(torch.bfloat16)
    dv = (torch.randn(B, T, H, V, device=dev) * g).to(torch.bfloat16)
    etot = (0.5 + 0.5 * torch.rand(B, NT, H, K, device=dev))   # decay in [0.5,1)
    dh = torch.empty(B, NT, H, K, V, device=dev, dtype=torch.bfloat16)
    dh0 = torch.empty(B, H, K, V, device=dev, dtype=torch.float32)
    dv2 = torch.empty(B, T, H, V, device=dev, dtype=torch.bfloat16)
    return dict(q=q, k=k, w=w, do=do, dv=dv, etot=etot, dh=dh, dh0=dh0, dv2=dv2,
                NT=NT, B=B, T=T, H=H, V=V)


def _bench(fn, n=5000, warm=100, sync=True):
    for _ in range(warm):
        fn()
    if sync:
        torch.cuda.synchronize()
    t0 = time.process_time()
    for _ in range(n):
        fn()
    t1 = time.process_time()
    if sync:
        torch.cuda.synchronize()
    return (t1 - t0) / n * 1e6


# --------------------------------------------------------------------------- Part A
def part_a(libs):
    print("=" * 96)
    print("Part A: host cost decomposition -- pure tensor-map encode vs full launch")
    print("  process_time / call; 'cache' = -DB1T_MAPCACHE (encode-once reuse)")
    print("=" * 96)
    print(f"  {'shape':22s} | {'launch nc':>10s} {'launch c':>9s} {'enc nc':>8s} {'enc c?':>8s}")
    for name, (B, T, H) in [("tiny B1 T16 H1", (1, 16, 1)), ("T250 H3", (1, 250, 3)),
                            ("B2 T1024 H8", (2, 1024, 8))]:
        BT = 16
        s = mk_shape(B, T, H, BT)
        stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
        scale = 1.0 / (D ** 0.5)

        def launch(L):
            return lambda: L.b1t_launch(BT, 2, ptr(s["q"]), ptr(s["k"]), ptr(s["w"]),
                                        ptr(s["do"]), ptr(s["dv"]), ptr(s["etot"]),
                                        ptr(s["dh"]), ptr(s["dh0"]), ptr(s["dv2"]),
                                        B, T, H, s["V"], s["NT"], scale, stream)

        def enc(L):
            return lambda: L.b1t_encode_only(BT, ptr(s["q"]), ptr(s["k"]), ptr(s["w"]),
                                             ptr(s["do"]), ptr(s["dv"]), ptr(s["etot"]),
                                             ptr(s["dh"]), ptr(s["dv2"]), B, T, H, s["NT"])

        nc, c = libs["nocache"], libs["cache"]
        lnc = _bench(launch(nc))
        lc = _bench(launch(c))
        enc_ = _bench(enc(nc), n=200000, warm=1000, sync=False)
        if hasattr(c, "b1t_mapcache_miss"):
            c.b1t_mapcache_miss.restype = ctypes.c_long
        print(f"  {name:22s} | {lnc:10.2f} {lc:9.2f} {enc_:8.2f} {'(pure)':>8s}")
    # cache-hit proof
    c = libs["cache"]
    if hasattr(c, "b1t_mapcache_miss"):
        c.b1t_mapcache_miss.restype = ctypes.c_long
        s = mk_shape(1, 250, 3, 16)
        stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
        fn = lambda: c.b1t_launch(16, 2, ptr(s["q"]), ptr(s["k"]), ptr(s["w"]), ptr(s["do"]),
                                  ptr(s["dv"]), ptr(s["etot"]), ptr(s["dh"]), ptr(s["dh0"]),
                                  ptr(s["dv2"]), 1, 250, 3, s["V"], s["NT"],
                                  1.0 / (D ** 0.5), stream)
        before = c.b1t_mapcache_miss()
        for _ in range(200):
            fn()
        torch.cuda.synchronize()
        after = c.b1t_mapcache_miss()
        print(f"  cache-hit proof: miss counter {before} -> {after} over 200 launches "
              f"(so {200 - (after - before)} hits)")


# --------------------------------------------------------------------------- Part B
def part_b(libs):
    print("\n" + "=" * 96)
    print("Part B: cache correctness -- cache vs no-cache (nonzero dht), max_abs must be 0")
    print("=" * 96)
    scale = 1.0 / (D ** 0.5)

    def run(L, s, dht, BT=16):
        stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
        rc = L.b1t_launch_dht(BT, 4, ptr(s["q"]), ptr(s["k"]), ptr(s["w"]), ptr(s["do"]),
                              ptr(s["dv"]), ptr(s["etot"]), ptr(dht), ptr(s["dh"]),
                              ptr(s["dh0"]), ptr(s["dv2"]), s["B"], s["T"], s["H"],
                              s["V"], s["NT"], scale, stream)
        assert rc == 0, rc
        return s["dh"].float(), s["dh0"].clone(), s["dv2"].float()

    for name, (B, T, H) in [("T128 H2", (1, 128, 2)), ("T250 H3 (non-div)", (1, 250, 3)),
                            ("B2 T513 H2 (non-div)", (2, 513, 2))]:
        s1 = mk_shape(B, T, H, 16)
        s2 = mk_shape(B, T, H, 16)
        # same operand *values* on both shapes (different pointers => cache cannot bleed)
        for key in ("q", "k", "w", "do", "dv", "etot"):
            s2[key].copy_(s1[key])
        dht = torch.randn(B, H, D, D, device="cuda") * 0.01
        a = run(libs["nocache"], s1, dht)
        b = run(libs["cache"], s2, dht)
        cols = []
        for n, x, y in zip(("dh", "dh0", "dv2"), a, b):
            nan_ok = bool((torch.isnan(x) == torch.isnan(y)).all())
            d = (x - y).abs()
            d = torch.nan_to_num(d)
            cols.append(f"{n} max_abs={float(d.max()):.2e}{'' if nan_ok else '/NANPAT'}")
        print(f"  {name:22s} | " + "  ".join(cols))


# --------------------------------------------------------------------------- Part C
def part_c(libs, rounds=3):
    import check_bwd_b1_cuda_e2e as X
    from check_bwd_b1_bf16_e2e import prep, time_wall
    from check_bwd_b1_pipe_e2e import make_chain_fn_pipe, CHUNK  # noqa: F401
    print("\n" + "=" * 100)
    print("Part C: chain-wall reproducibility (idle card). Triton-B1 vs CUDA-B1(nc) vs CUDA-B1(c)")
    print("  wall = event-pair / reps, R rounds; report median and min..max")
    print("=" * 100)
    scale = 1.0 / (D ** 0.5)
    lower_bound = -5.0
    cases = [("T=512 H=4", 1, 512, 4, 0, 1.0),
             ("B8 T=2048 H=8", 8, 2048, 8, 2, 1.0),
             ("T=250 H=3 nondiv", 1, 250, 3, 3, 1.0)]
    print(f"  {'case':18s} |       {'Triton-B1':>22s} |       {'CUDA nc':>22s} |"
          f"       {'CUDA c':>22s} | {'c/nc':>6s} {'c/tri':>6s}")
    for (name, B, T, H, seed, gs) in cases:
        inp, fi, qn, kn, beta_act, g_fla, do, dfin = prep(B, T, H, scale, lower_bound, seed, gs)
        chain_t = make_chain_fn_pipe(fi, inp, qn, kn, do, dfin, scale, NS=3)
        chain_nc = X.make_chain_cuda(libs["nocache"], fi, inp, qn, kn, do, dfin, scale, NSTAGE=4)
        chain_c = X.make_chain_cuda(libs["cache"], fi, inp, qn, kn, do, dfin, scale, NSTAGE=4)
        for c in (chain_t, chain_nc, chain_c):
            c()
        torch.cuda.synchronize()

        def mult(fn):
            vals = [time_wall(fn, reps=30, rounds=1) for _ in range(rounds)]
            return vals

        vt, vn, vc = mult(chain_t), mult(chain_nc), mult(chain_c)
        med = statistics.median
        print(f"  {name:18s} | {med(vt):9.1f} ({min(vt):.0f}..{max(vt):.0f}) |"
              f" {med(vn):9.1f} ({min(vn):.0f}..{max(vn):.0f}) |"
              f" {med(vc):9.1f} ({min(vc):.0f}..{max(vc):.0f}) |"
              f" {med(vc)/med(vn):6.3f} {med(vc)/med(vt):6.3f}")


def main():
    assert torch.cuda.is_available(), "need GPU"
    print(f"GPU: {torch.cuda.get_device_name()}  torch={torch.__version__}")
    so_nc = build([], SO_NOCACHE)
    so_c = build(["-DB1T_MAPCACHE"], SO_CACHE)
    libs = {"nocache": load(so_nc), "cache": load(so_c)}
    mode = os.environ.get("B1LT_MODE", "")
    if "a" in mode or mode == "":
        part_a(libs)
    if "b" in mode or mode == "":
        part_b(libs)
    if "c" in mode or mode == "":
        part_c(libs)
    print("\nDONE")


if __name__ == "__main__":
    main()
