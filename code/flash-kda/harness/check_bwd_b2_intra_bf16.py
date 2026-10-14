"""P7-109: the bf16-staging counterfactual for the folded B3 `intra` (docs/07 §112).

docs/07 §97 (P7-94) ruled the **pad/swizzle** lever out as a ~1.02x item and closed with
"K=128 折叠翻正要靠把它移出关键路径/减少这段活；K=256 的 **bf16 staging 是容量（非速度）杠杆**"
-- a lever that was named three times (§95.3 / §96.3 / §97) but **never run**.  `-DB2M_INTRA_BF16`
implements exactly that counterfactual: store the intra's raw `q/k/g` staging tiles as bf16 and
widen on read.  bf16 is a subset of tf32, so the widened values feed the *unchanged* tf32 mma;
the only effect is quantizing the *stored* `q/k/g`.

Two variants:
  `-DB2M_INTRA_BF16=1`  bf16 `q/k/g` (the full 1/2 smem saving)
  `-DB2M_INTRA_BF16=2`  bf16 `q/k`, `g` kept fp32 (g drives `exp2f(+-g)` in the gate)

Parts:
  A  correctness: `prod_intra_bf16{1,2}` vs the fp32-staging `prod_intra` -- the deltas are the
     pure cost of quantizing the stored `q/k/g`; the seven untouched outputs must stay bit-identical.
  B  capacity: `sizeof(Shared)` ledger (`prod` / `prod_intra` / bf16 variants) and the K=256 launch
     rc -- does halving the `3*BT*KD*4` staging actually make K=256 foldable on its own?
  C  timing: isolated `prod_intra` vs the two bf16 variants (CUPTI `dev`), K=V=128.

Run inside `kernel_lab` (idle card), absolute paths:
    CUDA_VISIBLE_DEVICES=1 python .../harness/check_bwd_b2_intra_bf16.py \
        > .../harness/check_bwd_b2_intra_bf16.out.txt 2>&1
    B2IB_MODE=a | b | c
"""
import ctypes
import os
import sys

os.environ.setdefault("FLA_DISABLE_BACKEND_DISPATCH", "1")

import torch

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

HARNESS = "~/proj/tech_record/code/flash-kda/harness"
sys.path.insert(0, HARNESS)

import check_bwd_b2_intra as I                                       # noqa: E402
from check_bwd_dqkg_kernel import make_inputs as dqkg_make_inputs    # noqa: E402
from check_bwd_ws_reuse import cdiv                                  # noqa: E402
from check_bwd_b1_bf16_e2e import profile_dev                        # noqa: E402

BT = 16
CU = os.path.join(HARNESS, "b2_mma_probe.cu")
SO_PROD = "/tmp/b2ib_prod.so"
SO_FP32 = "/tmp/b2ib_intra_fp32.so"
SO_B16 = "/tmp/b2ib_intra_b16.so"
SO_B16QK = "/tmp/b2ib_intra_b16qk.so"
SO_CORE = "/tmp/b2ib_core.so"

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


def build(tag, extra, so, K=128, V=128):
    """Same nvcc wrapper as `check_bwd_b2_kvcross.py` (with the K/V macros)."""
    cmd = ["nvcc", "-std=c++17", "-O3", "-arch=sm_90a", "-Xcompiler", "-fPIC",
           "-shared", "-Xptxas", "-v", "-DB2M_KD=%d" % K, "-DB2M_VD=%d" % V] + extra + \
          ["-lcuda", CU, "-o", so]
    import subprocess
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr)
        raise SystemExit(f"nvcc failed for {tag}")
    return r.stderr


def smem_of(so):
    lib = ctypes.CDLL(so)
    lib.b2mma_smem_bytes.restype = ctypes.c_int
    return lib.b2mma_smem_bytes()


def make(B, T, H, seed, K=128, V=128):
    inp = dqkg_make_inputs(B, T, H, K, V, BT, "cuda", dtype=torch.float32, seed=seed)
    inp = {k: (v if v.is_contiguous() else v.contiguous()) for k, v in inp.items()}
    torch.manual_seed(seed + 100)
    NT = cdiv(T, BT)
    o_t = torch.arange(BT, device="cuda")
    low_incl = (o_t[:, None] >= o_t[None, :]).float()
    dAqk = torch.randn(B, NT, H, BT, BT, device="cuda") * low_incl
    dAqk = dAqk.permute(0, 1, 3, 2, 4).reshape(B, NT * BT, H, BT)[:, :T].contiguous()
    inp["dAqk"] = dAqk
    return inp


def core_dw(inp):
    """The `dw` the production kernel expects as `dwin` (bare GEMM core), built per-K."""
    B, T, H, K = inp["q"].shape
    key = (K, inp["q"].data_ptr())
    if core_dw.cache.get("k") != K:
        build("core", [], SO_CORE, K=K, V=inp["v"].shape[-1])
        core_dw.cache = {"k": K, "lib": ctypes.CDLL(SO_CORE)}
    lib = core_dw.cache["lib"]
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
                     B, T, H, NT, I.stream())
    return dw


core_dw.cache = {}


# ------------------------------------------------------------------------- parts
def build_all():
    txt = {}
    txt["prod"] = build("prod", ["-DB2M_PROD=1"], SO_PROD)
    txt["fp32"] = build("fp32", ["-DB2M_PROD=1", "-DB2M_INTRA=1"], SO_FP32)
    txt["b16"] = build("b16", ["-DB2M_PROD=1", "-DB2M_INTRA=1", "-DB2M_INTRA_BF16=1"], SO_B16)
    txt["b16qk"] = build("b16qk", ["-DB2M_PROD=1", "-DB2M_INTRA=1", "-DB2M_INTRA_BF16=2"], SO_B16QK)
    build("core", [], SO_CORE)
    return txt


def part_a():
    build_all()
    libf = ctypes.CDLL(SO_FP32)
    lib1 = ctypes.CDLL(SO_B16)
    lib2 = ctypes.CDLL(SO_B16QK)
    worst = {"1": {}, "2": {}}
    for (name, B, T, H, seed) in CASES:
        inp = make(B, T, H, seed)
        dw = core_dw(inp)
        ref = I.run_prod_intra(libf, inp, dw)
        print(f"\n# {name}  NT={cdiv(T,BT)}")
        for tag, lib in (("1", lib1), ("2", lib2)):
            got = I.run_prod_intra(lib, inp, dw)
            for fld in ("dqg", "dkg", "dg", "db"):
                d = I.stats(f"bf16({tag}).{fld} vs fp32", got[fld], ref[fld])
                worst[tag][fld] = max(worst[tag].get(fld, 0.0), d)
            untouched = 0.0
            for fld in ("dv2", "dvb", "dkgb", "dA", "dApre", "dgk"):
                untouched = max(untouched, (got[fld].float() - ref[fld].float()).abs().max().item())
            print(f"  bf16({tag}) untouched outputs max_abs={untouched:.3e}")
            worst[tag]["untouched"] = max(worst[tag].get("untouched", 0.0), untouched)
    for tag in ("1", "2"):
        print(f"\nworst-case bf16({tag}) vs fp32-staging prod_intra (K=V=128):")
        for fld in ("dqg", "dkg", "dg", "db", "untouched"):
            print(f"  {fld:10s} {worst[tag].get(fld,0):.3e}")


def part_b():
    txt = build_all()
    print("Part B: sizeof(Shared) ledger (bytes).  K=V=128 default, then K=V=256.")
    print(f"  {'build':22s} {'regs':>5s} {'spill':>6s} {'smem@128':>9s}")
    rows = [("prod", "prod", SO_PROD), ("prod_intra(fp32)", "fp32", SO_FP32),
            ("prod_intra(bf16=1)", "b16", SO_B16), ("prod_intra(bf16=2)", "b16qk", SO_B16QK)]
    for label, key, so in rows:
        r, sp = I.ptxas_info(txt[key])
        print(f"  {label:22s} {r:5d} {sp:6d} {smem_of(so):9d}")
    intra_fp32 = smem_of(SO_FP32) - smem_of(SO_PROD)
    intra_b16 = smem_of(SO_B16) - smem_of(SO_PROD)
    print(f"\n  intra staging smem increment: fp32={intra_fp32} B  bf16(1)={intra_b16} B "
          f"(saved {intra_fp32-intra_b16} B, {100*(intra_fp32-intra_b16)/intra_fp32:.1f}%)")

    print(f"\n  K=V=256 ledger (the production deploy config adds `-DB2M_HDW`, matching §93/P7-102);"
          f" cap = {232448} B (227 KiB dyn smem):")
    k = 256
    bases = {}
    for tag, extra in (("prod", ["-DB2M_PROD=1", "-DB2M_HDW=1"]),
                       ("intra_fp32", ["-DB2M_PROD=1", "-DB2M_HDW=1", "-DB2M_INTRA=1"]),
                       ("intra_b16", ["-DB2M_PROD=1", "-DB2M_HDW=1", "-DB2M_INTRA=1",
                                      "-DB2M_INTRA_BF16=1"])):
        so = f"/tmp/b2ib_k256_{tag}.so"
        try:
            build(f"k256_{tag}", extra, so, K=k, V=k)
        except SystemExit:
            print(f"    {tag:12s} K=256: nvcc FAILED")
            continue
        bases[tag] = so
        sm = smem_of(so)
        print(f"    {tag:12s} K=256: sizeof(Shared)={sm:7d} B  "
              f"(< {232448}: {sm < 232448})")
    if "prod" in bases and "intra_fp32" in bases and "intra_b16" in bases:
        print(f"    -> intra increment @K=256: fp32="
              f"{smem_of(bases['intra_fp32'])-smem_of(bases['prod'])} B, "
              f"bf16={smem_of(bases['intra_b16'])-smem_of(bases['prod'])} B")
    # launch rc at K=256 (non-HDW launcher signature; only for the non-HDW builds above is not
    # available, so we probe the non-HDW K=256 builds separately)
    print("\n  K=V=256 non-HDW launch probe (deployment uses HDW, but rc is the same smem gate):")
    inp = make(1, 256, 2, 7, K=k, V=k)
    NT = cdiv(256, BT)
    dev = inp["q"].device
    zeros = torch.zeros(1, 256, 2, k, device=dev, dtype=torch.float32)
    for tag, extra in (("prod", ["-DB2M_PROD=1"]),
                       ("intra_fp32", ["-DB2M_PROD=1", "-DB2M_INTRA=1"]),
                       ("intra_b16", ["-DB2M_PROD=1", "-DB2M_INTRA=1", "-DB2M_INTRA_BF16=1"])):
        so = f"/tmp/b2ib_k256nh_{tag}.so"
        try:
            build(f"k256nh_{tag}", extra, so, K=k, V=k)
        except SystemExit:
            print(f"    {tag:12s} K=256: nvcc FAILED")
            continue
        print(f"    {tag:12s} K=256: sizeof(Shared)={smem_of(so):7d} B")
        if tag == "prod":
            continue
        lib = ctypes.CDLL(so)
        try:
            I.run_prod_intra(lib, inp, zeros)
            print(f"    {tag:12s} K=256 launch: rc=0 (OK)")
        except RuntimeError as e:
            print(f"    {tag:12s} K=256 launch: {e}  (NT={NT})")


def part_c():
    build_all()
    libf = ctypes.CDLL(SO_FP32)
    lib1 = ctypes.CDLL(SO_B16)
    lib2 = ctypes.CDLL(SO_B16QK)
    print("Part C: isolated `prod_intra` dev (us), fp32 staging vs bf16 variants, K=V=128")
    print(f"  {'case':22s} | {'fp32':>9s} {'bf16=1':>9s} {'bf16=2':>9s} | "
          f"{'1/fp32':>7s} {'2/fp32':>7s}")
    for (name, B, T, H, seed) in CASES:
        inp = make(B, T, H, seed)
        dw = core_dw(inp)

        def f1():
            I.run_prod_intra(libf, inp, dw)

        def f2():
            I.run_prod_intra(lib1, inp, dw)

        def f3():
            I.run_prod_intra(lib2, inp, dw)

        f1(); f2(); f3(); torch.cuda.synchronize()
        d0, _ = profile_dev(f1, reps=30)
        d1, _ = profile_dev(f2, reps=30)
        d2, _ = profile_dev(f3, reps=30)
        print(f"  {name:22s} | {d0:9.1f} {d1:9.1f} {d2:9.1f} | {d1/d0:7.3f} {d2/d0:7.3f}")


def main():
    assert torch.cuda.is_available(), "need GPU"
    print(f"GPU: {torch.cuda.get_device_name()}  torch={torch.__version__}")
    mode = os.environ.get("B2IB_MODE", "")
    if mode in ("", "a"):
        print("=" * 78)
        print("Part A: correctness -- bf16 q/k/g staging vs fp32 staging")
        print("=" * 78)
        part_a()
    if mode in ("", "b"):
        print("=" * 78)
        print("Part B: capacity -- smem ledger and the K=256 launch")
        print("=" * 78)
        part_b()
    if mode in ("", "c"):
        print("=" * 78)
        print("Part C: timing")
        print("=" * 78)
        part_c()
    print("\nDONE")


if __name__ == "__main__":
    main()
