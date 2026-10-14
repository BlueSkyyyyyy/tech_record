"""P7-82: close the "six non-GEMM segments still require VD==KD" boundary of the K2-shaped
CUDA B2 production kernel (`b2_mma_probe.cu`).

P7-81 (docs/07 section 84) parameterized `KD`/`VD` and verified the **base GEMM core**
(`dq/dk/dw/dgk`) is generic in K != V, but left the six non-GEMM segments guarded by an
`#error` (`VD==KD`) because they partitioned the V axis by the warp output width `BW=KD/NW`.

This round generalizes the V-axis segments: `OUT` (`dv2/dvb`), `DA`'s `dv@v^T`, and `DB`'s
raw-v mirror now partition V by their own warp width `BVW = VD/NW` (the K-axis segments --
`DKGB`'s `dkgb`, the gate chain, the `dgk` reduction, `DA`'s `dw@kg^T`, folded `intra` --
keep `BW`).  The single remaining coupling is the **global-`dw` DKGB staging** (`dwm`): it is
K-tiled but rides the V-tile loop, so `-DB2M_DKGB` without `-DB2M_HDW` still requires
`VD==KD` (now a narrowed `#error`).  The same-CTA handoff (`-DB2M_HDW`, the production form)
is K-axis (`swall`) and therefore K!=V-clean.

Parts
  Part A  regression: at K==V (VD=128) the modified kernel is **bit-identical** to the
          pre-change source (compiled from git HEAD) on all ten production outputs.
  Part B  correctness at K!=V: `prod -DB2M_HDW` (VD in {64,128,256}) vs torch fp32
          references (`dv2/dvb` = A@dv; `db`; `dApre` = dv@v^T - dw@kg^T), with `dw`
          recomputed in torch from the operands (`dw[t,k] = sum_v dv[t,v]*h[k,v]`).
  Part C  stage-layout identity: prod vs prod+TMA bit-identity at each VD.
  Part D  static cost (smem vs VD).

Run inside `kernel_lab` (idle card), absolute paths:
    CUDA_VISIBLE_DEVICES=3 python .../harness/check_bwd_b2_kv_prod.py \
        > .../harness/check_bwd_b2_kv_prod.out.txt 2>&1
    B2KVP_MODE=a | b | c | d
"""
import ctypes
import os
import subprocess
import sys

os.environ.setdefault("FLA_DISABLE_BACKEND_DISPATCH", "1")

import torch

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

HARNESS = os.path.expanduser("~/proj/tech_record/code/flash-kda/harness")
SRC = os.path.expanduser("~/proj/tech_record/code/flash-kda/src")
sys.path.insert(0, HARNESS)
sys.path.insert(0, SRC)

from check_bwd_dqkg_kernel import make_inputs as dqkg_make_inputs   # noqa: E402
from check_bwd_ws_reuse import stats, cdiv                          # noqa: E402
from check_bwd_b1_bf16_e2e import profile_dev, time_wall            # noqa: E402
from check_bwd_b2_out import ref_out                                # noqa: E402
from check_bwd_b2_db import ref_operands                            # noqa: E402
from check_bwd_b2_da import ref_dapre                               # noqa: E402

K = 128
BT = 16
SCALE = 1.0 / (128 ** 0.5)
CU = os.path.join(HARNESS, "b2_mma_probe.cu")

CASES = [
    ("B2 T1024 H8", 2, 1024, 8),
    ("T512 H4", 1, 512, 4),
    ("T250 H3 (non-div)", 1, 250, 3),
]
VS = [64, 128, 256]


def ptr(t):
    return ctypes.c_void_p(t.data_ptr())


def stream():
    return ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)


def build(src, so, V, extra):
    cmd = ["nvcc", "-std=c++17", "-O3", "-arch=sm_90a", "-Xcompiler", "-fPIC",
           "-shared", "-DB2M_VD=%d" % V] + extra + ["-lcuda", src, "-o", so]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout)
        print(r.stderr)
        raise SystemExit("nvcc failed: %s (V=%d %s)" % (src, V, extra))
    return so


def orig_source():
    """Extract the pre-P7-82 source from git HEAD into /tmp for the K==V regression."""
    out = "/tmp/b2_mma_probe_orig.cu"
    r = subprocess.run(["git", "-C", os.path.dirname(HARNESS), "show",
                        "HEAD:code/flash-kda/harness/b2_mma_probe.cu"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        # fall back: the harness dir's git root may differ; try the repo root
        r = subprocess.run(["git", "-C", os.path.expanduser("~/proj/tech_record"), "show",
                            "HEAD:code/flash-kda/harness/b2_mma_probe.cu"],
                           capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit("cannot read HEAD source: " + r.stderr)
    open(out, "w").write(r.stdout)
    return out


def make(B, T, H, V, seed=1):
    inp = dqkg_make_inputs(B, T, H, K, V, BT, "cuda", dtype=torch.float32, seed=seed)
    return {k: (v if v.is_contiguous() else v.contiguous()) for k, v in inp.items()}


def run_prod(lib, inp):
    lib.b2prod_launch.restype = ctypes.c_int
    lib.b2prod_launch.argtypes = ([ctypes.c_void_p] * 24 + [ctypes.c_float] +
                                  [ctypes.c_void_p] * 4 + [ctypes.c_int] * 4 +
                                  [ctypes.c_void_p])
    B, T, H, K_ = inp["q"].shape
    V = inp["v"].shape[-1]
    NT = cdiv(T, BT)
    dev = inp["q"].device
    dq = torch.empty(B, T, H, K_, device=dev, dtype=torch.float32)
    dk = torch.empty(B, T, H, K_, device=dev, dtype=torch.float32)
    dw_in = torch.zeros(B, T, H, K_, device=dev, dtype=torch.float32)   # scratch under HDW
    dw_out = torch.empty(B, T, H, K_, device=dev, dtype=torch.float32)  # unwritten under HDW
    dgk = torch.empty(B, NT, H, K_, device=dev, dtype=torch.float32)
    dv2 = torch.empty(B, T, H, V, device=dev, dtype=torch.float32)
    dvb = torch.empty(B, T, H, V, device=dev, dtype=torch.float32)
    dkgb = torch.empty(B, T, H, K_, device=dev, dtype=torch.float32)
    dqg = torch.empty(B, T, H, K_, device=dev, dtype=torch.float32)
    dkg = torch.empty(B, T, H, K_, device=dev, dtype=torch.float32)
    dg = torch.empty(B, T, H, K_, device=dev, dtype=torch.float32)
    db_out = torch.zeros(B, T, H, device=dev, dtype=torch.float32)
    dApre = torch.empty(B, T, H, BT, device=dev, dtype=torch.float32)
    dA = torch.empty(B, T, H, BT, device=dev, dtype=torch.float32)
    rc = lib.b2prod_launch(ptr(inp["v"]), ptr(inp["v_new"]), ptr(inp["h"]), ptr(inp["dh"]),
                           ptr(inp["do"]), ptr(inp["dv"]), ptr(inp["k"]), ptr(inp["g"]),
                           ptr(dw_in),
                           ptr(dq), ptr(dk), ptr(dw_out), ptr(dgk),
                           ptr(inp["A"]), ptr(inp["beta"]), ptr(dv2), ptr(dvb),
                           ptr(dkgb), ptr(inp["q"]), ptr(inp["k"]), ptr(inp["g"]),
                           ptr(dqg), ptr(dkg), ptr(dg), SCALE,
                           ptr(db_out), ptr(dApre), ptr(inp["A"]), ptr(dA),
                           B, T, H, NT, stream())
    if rc != 0:
        raise RuntimeError("b2prod_launch rc=%d" % rc)
    return dict(dgk=dgk, dkgb=dkgb, dv2=dv2, dvb=dvb, db=db_out, dApre=dApre, dA=dA,
                dqg=dqg, dkg=dkg, dg=dg)


def torch_dw(inp, BT):
    """The kernel's `dw[t,k] = sum_v dv[t,v]*h[k,v]` (per chunk), recomputed in torch fp32."""
    B, T, H, K_ = inp["q"].shape
    V = inp["v"].shape[-1]
    NT = cdiv(T, BT)
    dev = inp["q"].device

    def pad4(x):   # [B,T,H,W] -> [B,NT,H,BT,W]
        o = torch.zeros((B, NT * BT, H) + x.shape[3:], device=dev, dtype=torch.float32)
        o[:, :T] = x.float()
        return o.view((B, NT, BT, H) + x.shape[3:]).permute(0, 1, 3, 2, 4).contiguous()

    dv4 = pad4(inp["dv"])            # [B,NT,H,BT,V]
    h4 = inp["h"].float()            # [B,NT,H,K,V]
    dw = torch.einsum("bchtv,bchkv->bchtk", dv4, h4)   # [B,NT,H,BT,K]
    return dw.permute(0, 1, 3, 2, 4).reshape(B, NT * BT, H, K_)[:, :T].contiguous()


def bits(name, a, b):
    d = (a.float() - b.float()).abs().max().item()
    flag = "OK " if d == 0.0 else "DIFF"
    print("    %-28s max_abs=%.3e  [%s]" % (name, d, flag))
    return d


def part_a():
    print("=" * 92)
    print("Part A: K==V (VD=128) regression -- modified source vs pre-change (git HEAD)")
    print("=" * 92)
    orig = build(orig_source(), "/tmp/b2kvp_orig128.so", 128, ["-DB2M_PROD", "-DB2M_HDW"])
    new = build(CU, "/tmp/b2kvp_new128.so", 128, ["-DB2M_PROD", "-DB2M_HDW"])
    lo, ln = ctypes.CDLL(orig), ctypes.CDLL(new)
    worst = 0.0
    for (name, B, T, H) in CASES:
        inp = make(B, T, H, 128, seed=1)
        o = run_prod(lo, inp)
        n = run_prod(ln, inp)
        print("\n# %s" % name)
        for fld in ("dv2", "dvb", "dkgb", "db", "dApre", "dA", "dqg", "dkg", "dg", "dgk"):
            worst = max(worst, bits("orig vs new %s" % fld, o[fld], n[fld]))
    print("\nPart A worst max_abs over all fields/cases: %.3e" % worst)
    return worst


def part_b():
    print("\n" + "=" * 92)
    print("Part B: K!=V six-segment correctness -- prod(HDW) vs torch fp32 (V-axis outputs)")
    print("=" * 92)
    libs = {V: ctypes.CDLL(build(CU, "/tmp/b2kvp_v%d.so" % V, V,
                                 ["-DB2M_PROD", "-DB2M_HDW"])) for V in VS}
    worst = {}
    for V in VS:
        lib = libs[V]
        print("\n#### V = %d  (K = %d fixed) ####" % (V, K))
        for (name, B, T, H) in CASES:
            inp = make(B, T, H, V, seed=1)
            out = run_prod(lib, inp)
            dw = torch_dw(inp, BT)
            ro = ref_out(inp, BT)
            rd = ref_operands(inp, dw, BT)
            rp = ref_dapre(inp, dw, BT)
            print("\n# %s  grid=(%d, %d)" % (name, cdiv(T, BT), B * H))
            for fld, ref, tag in (("dv2", ro["dv2"], "out"), ("dvb", ro["dvb"], "out"),
                                  ("db", rd["db"], "db"), ("dApre", rp["dApre"], "da")):
                _, mr = stats("  [%d] %-6s vs torch" % (V, fld), out[fld], ref)
                worst["%d_%s" % (V, fld)] = max(worst.get("%d_%s" % (V, fld), 0.0), mr)
                worst["%d_%s_tag" % (V, fld)] = tag
            # dvb also gets the operand-based reference (checks the A@dv algebra independently)
            _, mr = stats("  [%d] %-6s vs torch(ops)" % (V, "dvb"), out["dvb"], rd["dvb"])
            worst["%d_dvb_ops" % V] = max(worst.get("%d_dvb_ops" % V, 0.0), mr)
    print("\nworst max_rel over all cases:")
    print("  %4s  %-8s %11s" % ("V", "field", "vs torch"))
    for V in VS:
        for fld in ("dv2", "dvb", "db", "dApre"):
            print("  %4d  %-8s %11.3e" % (V, fld, worst["%d_%s" % (V, fld)]))
        print("  %4d  %-8s %11.3e" % (V, "dvb/ops", worst["%d_dvb_ops" % V]))
    return worst


def part_c():
    print("\n" + "=" * 92)
    print("Part C: stage-layout identity -- prod vs prod+TMA (bit-identical) at each VD")
    print("=" * 92)
    worst = 0.0
    for V in VS:
        lib = ctypes.CDLL(build(CU, "/tmp/b2kvp_v%d.so" % V, V, ["-DB2M_PROD", "-DB2M_HDW"]))
        libt = ctypes.CDLL(build(CU, "/tmp/b2kvp_v%d_tma.so" % V, V,
                                 ["-DB2M_PROD", "-DB2M_HDW", "-DB2M_TMA"]))
        print("\n#### V = %d ####" % V)
        for (name, B, T, H) in CASES:
            inp = make(B, T, H, V, seed=1)
            a = run_prod(lib, inp)
            b = run_prod(libt, inp)
            d = 0.0
            for fld in ("dv2", "dvb", "dkgb", "db", "dApre", "dA", "dqg", "dkg", "dg", "dgk"):
                d = max(d, bits("%s %s" % (name, fld), a[fld], b[fld]))
            worst = max(worst, d)
    print("\nPart C worst max_abs: %.3e" % worst)
    return worst


def part_d():
    print("\n" + "=" * 92)
    print("Part D: static cost (smem scales with KD rows x BVK V-tile; V only sets nvt)")
    print("=" * 92)
    for V in VS:
        lib = ctypes.CDLL(build(CU, "/tmp/b2kvp_v%d.so" % V, V, ["-DB2M_PROD", "-DB2M_HDW"]))
        lib.b2mma_smem_bytes.restype = ctypes.c_int
        smem = lib.b2mma_smem_bytes()
        print("  V=%4d: smem = %d B (%.1f KB), nvt=%d (BVK=32), BVW=%d, NBV=%d"
              % (V, smem, smem / 1024.0, V // 32, V // 8, V // 8 // 8))


def main():
    assert torch.cuda.is_available(), "need GPU"
    print("GPU: %s  torch=%s" % (torch.cuda.get_device_name(), torch.__version__))
    mode = os.environ.get("B2KVP_MODE", "")
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
