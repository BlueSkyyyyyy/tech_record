"""P7-70: close the last named boundary of §62 -- vectorize the `[BT,BT]` **`dA`/`dApre` stores**
in the K2-shaped CUDA B2 (`b2_mma_probe.cu`).

P7-59 (§62) applied the `st2()` pairwise-store helper to the *large* non-GEMM B2 outputs
(`dv2/dvb`, `dkgb`, `dqq/dkg/dg`) and closed every "本段 store 未接 `-DB2M_VECST`" boundary,
but explicitly **left the `[BT,BT]` `dA`/`dApre` stores scalar** ("small, one-thread-per-entry
layout").  That is the one remaining "store not vectorized" note in the topic's boundary list
(§57.6/§58.6/§62.6).  This round closes it: the `dApre` store (`-DB2M_DA`, P7-54) and the final
`dA` sandwich store (`-DB2M_SAND`, P7-55) now write adjacent column pairs through `st2` under
`-DB2M_VECST`.

Key distinction from P7-59: those stores were **fragment-strided** (4 lanes writing bytes
`{0,8,16,24}` of a 32-B sector -> `l1tex st` sectors = 1.98x the data sectors).  The `dA`/`dApre`
stores are `row = idx/BT`, `col = idx%BT` with `idx = threadIdx.x`, i.e. **consecutive threads
write consecutive columns** -- already fully coalesced (warp spans two rows of `BT` floats = two
64-B segments, exactly 128 B / 4 sectors for 128 B requested).  So vectorizing them can only
halve the *instruction count*, not the sectors; the honest expectation is a **wash**.  This script
measures that and pins it, so the boundary is closed with data rather than left as "未做".

  Part A  correctness (standalone `da` / `sand`, 4 cases incl. non-div T=250):
          base vs vecst -> **bit-identical** (`max_abs=0`), and `sand` vs torch golden unchanged.
  Part B  timing (standalone `da` / `sand`): CUPTI `dev` base vs vecst.
  Part C  production (`-DB2M_PROD -DB2M_HDW`): base vs vecst **bit-identical** on all ten
          outputs + CUPTI `dev` base vs vecst.
  Part D  SASS: count `STG.E` (32-bit) vs `STG.E.64` in the `sand` build, base vs vecst.

Run inside `kernel_lab` (idle card), absolute paths:
    CUDA_VISIBLE_DEVICES=1 python .../harness/check_bwd_b2_vecst_da.py \
        > .../harness/check_bwd_b2_vecst_da.out.txt 2>&1
    B2VD_MODE=a | b | c | d
"""
import ctypes
import os
import re
import subprocess
import sys

os.environ.setdefault("FLA_DISABLE_BACKEND_DISPATCH", "1")

import torch

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

HARNESS = "~/proj/tech_record/code/flash-kda/harness"
SRC = "~/proj/tech_record/code/flash-kda/src"
sys.path.insert(0, HARNESS)
sys.path.insert(0, SRC)

from check_bwd_dqkg_kernel import make_inputs as dqkg_make_inputs   # noqa: E402
from check_bwd_ws_reuse import cdiv, stats                          # noqa: E402
from check_bwd_b1_bf16_e2e import profile_dev                       # noqa: E402

CASES = [
    ("B8 T2048 H8", 8, 2048, 8, 16),
    ("B2 T1024 H8", 2, 1024, 8, 16),
    ("T512 H4", 1, 512, 4, 16),
    ("T250 H3 (non-div)", 1, 250, 3, 16),
]
BT = 16
SCALE = 1.0 / (128 ** 0.5)
CU = os.path.join(HARNESS, "b2_mma_probe.cu")

SEGS = {
    "da":   ["-DB2M_DA=1"],
    "sand": ["-DB2M_DA=1", "-DB2M_SAND=1"],
    "prod": ["-DB2M_PROD=1", "-DB2M_HDW=1"],
}


def ptr(t):
    return ctypes.c_void_p(t.data_ptr())


def stream():
    return ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)


def build_v(tag, extra, vecst):
    so = f"/tmp/b2_vda_{tag}_{'v' if vecst else 'b'}.so"
    flags = list(extra) + (["-DB2M_VECST=1"] if vecst else [])
    cmd = ["nvcc", "-std=c++17", "-O3", "-arch=sm_90a", "-Xcompiler", "-fPIC",
           "-shared", "-Xptxas", "-v"] + flags + ["-lcuda", CU, "-o", so]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr)
        raise SystemExit(f"nvcc failed for {tag} vecst={vecst}")
    return so, r.stderr


def build(tag, extra, vecst):
    return build_v(tag, extra, vecst)[0]


def make(B, T, H, seed=1):
    inp = dqkg_make_inputs(B, T, H, 128, 128, BT, "cuda", dtype=torch.float32, seed=seed)
    return {k: (v if v.is_contiguous() else v.contiguous()) for k, v in inp.items()}


def run_core(lib, inp):
    lib.b2mma_launch.restype = ctypes.c_int
    lib.b2mma_launch.argtypes = [ctypes.c_void_p] * 10 + [ctypes.c_int] * 4 + [ctypes.c_void_p]
    B, T, H, K = inp["q"].shape
    NT = cdiv(T, BT)
    dev = inp["q"].device
    dq = torch.empty(B, T, H, K, device=dev, dtype=torch.float32)
    dk = torch.empty(B, T, H, K, device=dev, dtype=torch.float32)
    dw = torch.empty(B, T, H, K, device=dev, dtype=torch.float32)
    dgk = torch.empty(B, NT, H, K, device=dev, dtype=torch.float32)
    rc = lib.b2mma_launch(ptr(inp["v"]), ptr(inp["v_new"]), ptr(inp["h"]), ptr(inp["dh"]),
                          ptr(inp["do"]), ptr(inp["dv"]), ptr(dq), ptr(dk), ptr(dw), ptr(dgk),
                          B, T, H, NT, stream())
    if rc != 0:
        raise RuntimeError(f"b2mma_launch rc={rc}")
    return dw


def run_da(lib, inp):
    lib.b2da_launch.restype = ctypes.c_int
    lib.b2da_launch.argtypes = [ctypes.c_void_p] * 13 + [ctypes.c_int] * 4 + [ctypes.c_void_p]
    B, T, H, K = inp["q"].shape
    NT = cdiv(T, BT)
    dev = inp["q"].device
    dq = torch.empty(B, T, H, K, device=dev, dtype=torch.float32)
    dk = torch.empty(B, T, H, K, device=dev, dtype=torch.float32)
    dw = torch.empty(B, T, H, K, device=dev, dtype=torch.float32)
    dgk = torch.empty(B, NT, H, K, device=dev, dtype=torch.float32)
    dApre = torch.empty(B, T, H, BT, device=dev, dtype=torch.float32)
    rc = lib.b2da_launch(ptr(inp["v"]), ptr(inp["v_new"]), ptr(inp["h"]), ptr(inp["dh"]),
                         ptr(inp["do"]), ptr(inp["dv"]), ptr(inp["k"]), ptr(inp["g"]),
                         ptr(dq), ptr(dk), ptr(dw), ptr(dgk), ptr(dApre),
                         B, T, H, NT, stream())
    if rc != 0:
        raise RuntimeError(f"b2da_launch rc={rc}")
    return dict(dApre=dApre)


def run_sand(lib, inp):
    lib.b2sand_launch.restype = ctypes.c_int
    lib.b2sand_launch.argtypes = [ctypes.c_void_p] * 16 + [ctypes.c_int] * 4 + [ctypes.c_void_p]
    B, T, H, K = inp["q"].shape
    NT = cdiv(T, BT)
    dev = inp["q"].device
    dq = torch.empty(B, T, H, K, device=dev, dtype=torch.float32)
    dk = torch.empty(B, T, H, K, device=dev, dtype=torch.float32)
    dw = torch.empty(B, T, H, K, device=dev, dtype=torch.float32)
    dgk = torch.empty(B, NT, H, K, device=dev, dtype=torch.float32)
    dApre = torch.empty(B, T, H, BT, device=dev, dtype=torch.float32)
    dA = torch.empty(B, T, H, BT, device=dev, dtype=torch.float32)
    rc = lib.b2sand_launch(ptr(inp["v"]), ptr(inp["v_new"]), ptr(inp["h"]), ptr(inp["dh"]),
                           ptr(inp["do"]), ptr(inp["dv"]), ptr(inp["k"]), ptr(inp["g"]),
                           ptr(inp["A"]), ptr(inp["beta"]),
                           ptr(dq), ptr(dk), ptr(dw), ptr(dgk), ptr(dApre), ptr(dA),
                           B, T, H, NT, stream())
    if rc != 0:
        raise RuntimeError(f"b2sand_launch rc={rc}")
    return dict(dApre=dApre, dA=dA)


def run_prod(lib, inp, dw):
    lib.b2prod_launch.restype = ctypes.c_int
    lib.b2prod_launch.argtypes = ([ctypes.c_void_p] * 24 + [ctypes.c_float] +
                                  [ctypes.c_void_p] * 4 + [ctypes.c_int] * 4 +
                                  [ctypes.c_void_p])
    B, T, H, K = inp["q"].shape
    V = inp["v"].shape[-1]
    NT = cdiv(T, BT)
    dev = inp["q"].device
    dq = torch.empty(B, T, H, K, device=dev, dtype=torch.float32)
    dk = torch.empty(B, T, H, K, device=dev, dtype=torch.float32)
    dw_out = torch.empty(B, T, H, K, device=dev, dtype=torch.float32)
    dgk = torch.empty(B, NT, H, K, device=dev, dtype=torch.float32)
    dv2 = torch.empty(B, T, H, V, device=dev, dtype=torch.float32)
    dvb = torch.empty(B, T, H, V, device=dev, dtype=torch.float32)
    dkgb = torch.empty(B, T, H, K, device=dev, dtype=torch.float32)
    dqg = torch.empty(B, T, H, K, device=dev, dtype=torch.float32)
    dkg = torch.empty(B, T, H, K, device=dev, dtype=torch.float32)
    dg = torch.empty(B, T, H, K, device=dev, dtype=torch.float32)
    db_out = torch.zeros(B, T, H, device=dev, dtype=torch.float32)
    dApre = torch.empty(B, T, H, BT, device=dev, dtype=torch.float32)
    dA = torch.empty(B, T, H, BT, device=dev, dtype=torch.float32)
    rc = lib.b2prod_launch(ptr(inp["v"]), ptr(inp["v_new"]), ptr(inp["h"]), ptr(inp["dh"]),
                           ptr(inp["do"]), ptr(inp["dv"]), ptr(inp["k"]), ptr(inp["g"]),
                           ptr(dw),
                           ptr(dq), ptr(dk), ptr(dw_out), ptr(dgk),
                           ptr(inp["A"]), ptr(inp["beta"]), ptr(dv2), ptr(dvb),
                           ptr(dkgb), ptr(inp["q"]), ptr(inp["k"]), ptr(inp["g"]),
                           ptr(dqg), ptr(dkg), ptr(dg), SCALE,
                           ptr(db_out), ptr(dApre), ptr(inp["A"]), ptr(dA),
                           B, T, H, NT, stream())
    if rc != 0:
        raise RuntimeError(f"b2prod_launch rc={rc}")
    return dict(dw=dw_out, dgk=dgk, dkgb=dkgb, dv2=dv2, dvb=dvb, db=db_out,
                dApre=dApre, dA=dA, dqg=dqg, dkg=dkg, dg=dg)


def bits(name, a, b):
    d = (a.float() - b.float()).abs().max().item()
    print(f"    {name:40s} max_abs={d:.3e}  [{'OK' if d == 0.0 else 'DIFF'}]")
    return d


# ------------------------------------------------------------------------- parts
def part_a():
    print("=" * 92)
    print("Part A: standalone `da` / `sand` base vs vecst -- expect bit-identical (max_abs=0)")
    print("=" * 92)
    libs = {t: (ctypes.CDLL(build(t, e, False)), ctypes.CDLL(build(t, e, True)))
            for t, e in SEGS.items() if t != "prod"}
    worst = 0.0
    for (name, B, T, H, seed) in CASES:
        print(f"\n# {name}")
        inp = make(B, T, H, seed)
        for tag, run in (("da", run_da), ("sand", run_sand)):
            ob = run(libs[tag][0], inp)
            ov = run(libs[tag][1], inp)
            for f in ob:
                worst = max(worst, bits(f"{tag}.{f} base vs vecst", ob[f], ov[f]))
    print(f"\n  worst: max_abs={worst:.3e}  [{'BIT-IDENTICAL' if worst == 0.0 else 'MISMATCH'}]")


def part_b():
    print("\n" + "=" * 92)
    print("Part B: CUPTI dev (us) standalone `da` / `sand` base vs vecst")
    print("=" * 92)
    libs = {t: (ctypes.CDLL(build(t, e, False)), ctypes.CDLL(build(t, e, True)))
            for t, e in SEGS.items() if t != "prod"}
    for tag, run in (("da", run_da), ("sand", run_sand)):
        print(f"\n# segment: {tag}")
        print(f"  {'case':18s} | {'base':>9s} {'vecst':>9s} | {'vecst/base':>10s}")
        for (name, B, T, H, seed) in CASES:
            inp = make(B, T, H, seed)
            run(libs[tag][0], inp)
            run(libs[tag][1], inp)
            torch.cuda.synchronize()
            db, _ = profile_dev(lambda: run(libs[tag][0], inp), reps=40)
            dv, _ = profile_dev(lambda: run(libs[tag][1], inp), reps=40)
            print(f"  {name:18s} | {db:9.1f} {dv:9.1f} | {dv/db:9.3f}x")


def part_c():
    print("\n" + "=" * 92)
    print("Part C: production `-DB2M_PROD -DB2M_HDW` base vs vecst (10 outputs + dev)")
    print("=" * 92)
    lb = ctypes.CDLL(build("prod", SEGS["prod"], False))
    lv = ctypes.CDLL(build("prod", SEGS["prod"], True))
    cl = ctypes.CDLL(build("core", [], False))
    worst = 0.0
    print(f"  {'case':18s} | {'base':>9s} {'vecst':>9s} | {'vecst/base':>10s}")
    for (name, B, T, H, seed) in CASES:
        inp = make(B, T, H, seed)
        dw = run_core(cl, inp)
        ob = run_prod(lb, inp, dw)
        ov = run_prod(lv, inp, dw)
        for f in ("dv2", "dvb", "dkgb", "db", "dApre", "dA", "dqg", "dkg", "dg", "dgk"):
            worst = max(worst, bits(f"prod.{f} base vs vecst", ob[f], ov[f]))
        run_prod(lb, inp, dw)
        run_prod(lv, inp, dw)
        torch.cuda.synchronize()
        db, _ = profile_dev(lambda: run_prod(lb, inp, dw), reps=40)
        dv, _ = profile_dev(lambda: run_prod(lv, inp, dw), reps=40)
        print(f"  {name:18s} | {db:9.1f} {dv:9.1f} | {dv/db:9.3f}x")
    print(f"\n  worst: max_abs={worst:.3e}  [{'BIT-IDENTICAL' if worst == 0.0 else 'MISMATCH'}]")


def part_d():
    print("\n" + "=" * 92)
    print("Part D: SASS store-instruction counts (sand build, base vs vecst)")
    print("=" * 92)
    for vecst in (False, True):
        so = build("sand", SEGS["sand"], vecst)
        r = subprocess.run(["cuobjdump", "-sass", so], capture_output=True, text=True)
        txt = r.stdout
        stg = len(re.findall(r"\bSTG\.E\b", txt))
        stg64 = len(re.findall(r"\bSTG\.E\.64\b", txt))
        print(f"  sand {'vecst' if vecst else ' base'} : STG.E(32b)={stg:4d}  STG.E.64={stg64:4d}")


def main():
    mode = os.environ.get("B2VD_MODE", "")
    assert torch.cuda.is_available(), "need GPU"
    print(f"GPU: {torch.cuda.get_device_name()}  torch={torch.__version__}")
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
