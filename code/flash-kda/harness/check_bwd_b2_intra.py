"""P7-63: fold the B3 `intra` diagonal into the **production launch** `-DB2M_PROD`.

ROADMAP / docs/07 §65.6 (P7-62) left exactly one engineering step: take the K2-shape CUDA
`intra` (standalone `b3_intra_probe.cu`) and merge it into the single production B2 launch --
using the *in-kernel* sandwiched `dA` (SAND's `daB`), the `dAqk` input and the already-available
`q/k/g` to produce the diagonal contribution and add it into the gate's final `dq/dk/dg` and the
`db` reduction.  `-DB2M_INTRA` (requires `-DB2M_PROD`) does exactly that; `b2prod_intra_launch`
takes one extra `dAqk` input.  No `dA` round-trip and no separate B3 kernel.

Parts:
  A  bit-check: `prod_intra.dqg/dkg/dg/db` vs `prod.<...> + b3intra.<...>` (the two things it
     replaces) -- should be bit-identical (dg up to fp32 add-order ulp: the fold adds `idg`
     before the gate's `m_last` term, the reference after).
  A2 sanity: `prod_intra` vs `prod` alone differs (the intra contribution is really added).
  B  resources: registers / spills / smem of `prod` vs `prod_intra`.
  C  timing: one `prod_intra` launch vs `prod` + standalone `b3intra` (CUPTI `dev` + event `wall`).

Run inside `kernel_lab` (idle card), absolute paths:
    CUDA_VISIBLE_DEVICES=1 python .../harness/check_bwd_b2_intra.py \
        > .../harness/check_bwd_b2_intra.out.txt 2>&1
    B2I_MODE=a | a2 | b | c
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

HARNESS = os.path.expanduser("~/proj/tech_record/code/flash-kda/harness")
sys.path.insert(0, HARNESS)

from check_bwd_dqkg_kernel import make_inputs as dqkg_make_inputs      # noqa: E402
from check_bwd_ws_reuse import cdiv                                    # noqa: E402
from check_bwd_b1_bf16_e2e import profile_dev, time_wall              # noqa: E402

BT = 16
SCALE = 1.0 / (128 ** 0.5)
CU = os.path.join(HARNESS, "b2_mma_probe.cu")
SO_PROD = "/tmp/b2_intra_prod.so"
SO_INTRA = "/tmp/b2_intra_fold.so"
SO_B3 = "/tmp/b2_intra_b3.so"

CASES = [
    ("B8 T2048 H8", 8, 2048, 8, 16),
    ("B2 T1024 H8", 2, 1024, 8, 16),
    ("T512 H4", 1, 512, 4, 16),
    ("T256 H2", 1, 256, 2, 16),
    ("T250 H3 (non-div)", 1, 250, 3, 16),
    ("T64 H1 (single chunk)", 1, 64, 1, 16),
]


def ptr(t):
    return ctypes.c_void_p(t.data_ptr())


def stream():
    return ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)


def build(tag, src, extra, so):
    cmd = ["nvcc", "-std=c++17", "-O3", "-arch=sm_90a", "-Xcompiler", "-fPIC",
           "-shared", "-Xptxas", "-v"] + extra + ["-lcuda", src, "-o", so]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr)
        raise SystemExit(f"nvcc failed for {tag}")
    return r.stderr


def ptxas_info(txt):
    reg = re.search(r"Used (\d+) registers", txt)
    spill = re.search(r"(\d+) bytes spill stores", txt)
    return (int(reg.group(1)) if reg else -1, int(spill.group(1)) if spill else 0)


def load(so):
    return ctypes.CDLL(so)


def make(B, T, H, seed):
    inp = dqkg_make_inputs(B, T, H, 128, 128, BT, "cuda", dtype=torch.float32, seed=seed)
    inp = {k: (v if v.is_contiguous() else v.contiguous()) for k, v in inp.items()}
    # `dAqk = tril(randn, incl diag)` in the [B,T,H,BT] layout the intra kernel consumes.
    torch.manual_seed(seed + 100)
    NT = cdiv(T, BT)
    o_t = torch.arange(BT, device="cuda")
    low_incl = (o_t[:, None] >= o_t[None, :]).float()
    dAqk = torch.randn(B, NT, H, BT, BT, device="cuda") * low_incl
    dAqk = dAqk.permute(0, 1, 3, 2, 4).reshape(B, NT * BT, H, BT)[:, :T].contiguous()
    inp["dAqk"] = dAqk
    return inp


# ------------------------------------------------------------------ launchers
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


def run_prod_intra(lib, inp, dw):
    lib.b2prod_intra_launch.restype = ctypes.c_int
    lib.b2prod_intra_launch.argtypes = ([ctypes.c_void_p] * 24 + [ctypes.c_float] +
                                        [ctypes.c_void_p] * 5 + [ctypes.c_int] * 4 +
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
    rc = lib.b2prod_intra_launch(ptr(inp["v"]), ptr(inp["v_new"]), ptr(inp["h"]), ptr(inp["dh"]),
                                 ptr(inp["do"]), ptr(inp["dv"]), ptr(inp["k"]), ptr(inp["g"]),
                                 ptr(dw),
                                 ptr(dq), ptr(dk), ptr(dw_out), ptr(dgk),
                                 ptr(inp["A"]), ptr(inp["beta"]), ptr(dv2), ptr(dvb),
                                 ptr(dkgb), ptr(inp["q"]), ptr(inp["k"]), ptr(inp["g"]),
                                 ptr(dqg), ptr(dkg), ptr(dg), SCALE,
                                 ptr(db_out), ptr(dApre), ptr(inp["A"]), ptr(dA),
                                 ptr(inp["dAqk"]),
                                 B, T, H, NT, stream())
    if rc != 0:
        raise RuntimeError(f"b2prod_intra_launch rc={rc}")
    return dict(dw=dw_out, dgk=dgk, dkgb=dkgb, dv2=dv2, dvb=dvb, db=db_out,
                dApre=dApre, dA=dA, dqg=dqg, dkg=dkg, dg=dg)


def load_b3(lib):
    lib.b3intra_launch.restype = ctypes.c_int
    lib.b3intra_launch.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_void_p] * 4 + \
        [ctypes.c_int] * 4 + [ctypes.c_void_p]
    return lib


def run_b3(lib, inp, dAkk):
    B, T, H, K = inp["q"].shape
    NT = cdiv(T, BT)
    dev = inp["q"].device
    dq = torch.empty(B, T, H, K, device=dev, dtype=torch.float32)
    dk = torch.empty(B, T, H, K, device=dev, dtype=torch.float32)
    db = torch.empty(B, T, H, device=dev, dtype=torch.float32)
    dg = torch.empty(B, T, H, K, device=dev, dtype=torch.float32)
    rc = lib.b3intra_launch(ptr(inp["q"]), ptr(inp["k"]), ptr(inp["g"]), ptr(inp["beta"]),
                            ptr(inp["dAqk"]), ptr(dAkk), ptr(dq), ptr(dk), ptr(db), ptr(dg),
                            B, T, H, NT, stream())
    if rc != 0:
        raise RuntimeError(f"b3intra_launch rc={rc}")
    return dict(dq=dq, dk=dk, db=db, dg=dg)


def stats(name, a, b):
    d = (a.float() - b.float()).abs().max().item()
    denom = b.float().abs().max().item()
    print(f"  {name:28s} max_abs={d:.3e}  rel={d/(denom+1e-12):.3e}  ref_max={denom:.3e}")
    return d


# ------------------------------------------------------------------------- parts
def build_all():
    txt_prod = build("prod", CU, ["-DB2M_PROD=1"], SO_PROD)
    txt_intra = build("intra", CU, ["-DB2M_PROD=1", "-DB2M_INTRA=1"], SO_INTRA)
    build("b3", os.path.join(HARNESS, "b3_intra_probe.cu"), [], SO_B3)
    return txt_prod, txt_intra


def part_a(cases, verbose=True):
    txt_prod, txt_intra = build_all()
    libp = load(SO_PROD)
    libi = load(SO_INTRA)
    libb = load_b3(load(SO_B3))
    worst = {}
    for (name, B, T, H, seed) in cases:
        inp = make(B, T, H, seed)
        # raw `dw` from the core (b2prod expects the core's dw as input for the dkgb path)
        p = run_prod(libp, inp, _core_dw(inp))
        pi = run_prod_intra(libi, inp, _core_dw(inp))
        b3 = run_b3(libb, inp, p["dA"])
        print(f"\n# {name}  NT={cdiv(T,BT)}")
        # sanity: the intra contribution must be non-zero and added
        _sa = (pi["dqg"].float() - p["dqg"].float()).abs().max().item()
        _ra = (p["dqg"].float() + b3["dq"]).abs().max().item()
        print(f"  (prod_intra - prod).dqg max_abs={_sa:.3e}  "
              f"|prod+b3|.dqg max_abs={_ra:.3e}")
        for fld, ref in (("dqg", lambda: p["dqg"] + b3["dq"]),
                         ("dkg", lambda: p["dkg"] + b3["dk"]),
                         ("dg",  lambda: p["dg"] + b3["dg"]),
                         ("db",  lambda: p["db"] + b3["db"])):
            d = stats(f"prod_intra.{fld} vs sum", pi[fld], ref())
            worst[fld] = max(worst.get(fld, 0.0), d)
        # unchanged outputs must stay bit-identical
        for fld in ("dv2", "dvb", "dkgb", "dA", "dApre", "dw", "dgk"):
            d = (pi[fld].float() - p[fld].float()).abs().max().item()
            worst["un_" + fld] = max(worst.get("un_" + fld, 0.0), d)
    print("\nworst-case (fold vs prod+standalone-intra; 0 == bit-identical):")
    for fld in ("dqg", "dkg", "dg", "db"):
        print(f"  {fld:4s} {worst.get(fld,0):.3e}")
    print("unchanged outputs (must be 0):")
    for fld in ("dv2", "dvb", "dkgb", "dA", "dApre", "dw", "dgk"):
        print(f"  {fld:6s} {worst.get('un_'+fld,0):.3e}")


_CORE = {}


def _core_dw(inp):
    """The `dw` the production kernel expects as `dwin` (from the bare GEMM core)."""
    key = (inp["q"].shape, inp["q"].data_ptr())
    if key not in _CORE:
        lib = _CORE.setdefault("__lib__", load(SO_CORE))
        B, T, H, K = inp["q"].shape
        NT = cdiv(T, BT)
        dev = inp["q"].device
        dq = torch.empty(B, T, H, K, device=dev, dtype=torch.float32)
        dk = torch.empty(B, T, H, K, device=dev, dtype=torch.float32)
        dw = torch.empty(B, T, H, K, device=dev, dtype=torch.float32)
        dgk = torch.empty(B, NT, H, K, device=dev, dtype=torch.float32)
        lib.b2mma_launch.restype = ctypes.c_int
        lib.b2mma_launch.argtypes = [ctypes.c_void_p] * 10 + [ctypes.c_int] * 4 + [ctypes.c_void_p]
        lib.b2mma_launch(ptr(inp["v"]), ptr(inp["v_new"]), ptr(inp["h"]), ptr(inp["dh"]),
                         ptr(inp["do"]), ptr(inp["dv"]), ptr(dq), ptr(dk), ptr(dw), ptr(dgk),
                         B, T, H, NT, stream())
        _CORE[key] = dw
    return _CORE[key]


SO_CORE = "/tmp/b2_intra_core.so"


def part_b(txt_prod, txt_intra):
    print("=" * 78)
    print("Part B: resources (registers / spill / smem)")
    print("=" * 78)
    print(f"  {'build':16s} {'regs':>6s} {'spill':>7s} {'smem':>9s}")
    for tag, txt, so in (("prod", txt_prod, SO_PROD), ("prod_intra", txt_intra, SO_INTRA)):
        r, sp = ptxas_info(txt)
        lib = load(so)
        lib.b2mma_smem_bytes.restype = ctypes.c_int
        smem = lib.b2mma_smem_bytes()
        print(f"  {tag:16s} {r:6d} {sp:7d} {smem:9d}")


def part_c(cases):
    libp = load(SO_PROD)
    libi = load(SO_INTRA)
    libb = load_b3(load(SO_B3))
    print("=" * 78)
    print("Part C: timing -- one prod_intra launch vs prod + standalone b3intra")
    print("=" * 78)
    print(f"  {'case':22s} | {'prod':>8s} {'prod_intra':>10s} | "
          f"{'prod+b3':>9s} | {'incl/prod':>9s} {'fold/(p+b3)':>11s}")
    for (name, B, T, H, seed) in cases:
        inp = make(B, T, H, seed)
        dw = _core_dw(inp)

        def f_prod():
            run_prod(libp, inp, dw)

        def f_fold():
            run_prod_intra(libi, inp, dw)

        # need `dA` for b3; get it once
        p0 = run_prod(libp, inp, dw)

        def f_old():
            run_prod(libp, inp, dw)
            run_b3(libb, inp, p0["dA"])

        f_prod(); f_fold(); f_old(); torch.cuda.synchronize()
        dp, _ = profile_dev(f_prod, reps=30)
        di, _ = profile_dev(f_fold, reps=30)
        do, _ = profile_dev(f_old, reps=30)
        print(f"  {name:22s} | {dp:8.1f} {di:10.1f} | "
              f"{do:9.1f} | {di/dp:9.3f} {di/do:11.3f}")


def main():
    assert torch.cuda.is_available(), "need GPU"
    print(f"GPU: {torch.cuda.get_device_name()}  torch={torch.__version__}")
    # core build (for the `dw` input), used by every part
    build("core", CU, [], SO_CORE)
    mode = os.environ.get("B2I_MODE", "")
    tp = ti = None
    if mode in ("", "a"):
        part_a(CASES)
    if mode in ("", "b"):
        tp, ti = build_all()
        part_b(tp, ti)
    if mode in ("", "c"):
        part_c(CASES)
    print("\nDONE")


if __name__ == "__main__":
    main()
