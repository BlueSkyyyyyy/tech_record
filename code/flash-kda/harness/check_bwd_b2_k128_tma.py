"""P7-88 cross-check (appendix to `check_bwd_b2_k128.py`): the K!=128 base GEMM core under
the **TMA staging path** (`-DB2M_TMA`) as well as the default `cp.async` path.

The TMA tensor maps already carry `{VD,KD}` separately (`make_map_dov`/`make_map_hdh`), so this
checks that the `KD`-generic core + generic `dgk` reduction stay correct when the operands are
streamed with `cp.async.bulk.tensor` (SWIZZLE_128B) instead of `cp.async`.  Same references as
the main harness (torch fp32 golden).  Reuses the main harness's build/load/run helpers with a
TMA-tagged shared object.

Run inside `kernel_lab` (idle card), absolute paths:
    CUDA_VISIBLE_DEVICES=3 python .../harness/check_bwd_b2_k128_tma.py \
        > .../harness/check_bwd_b2_k128_tma.out.txt 2>&1
"""
import os
import sys

os.environ["B2K_CFLAGS"] = "-DB2M_TMA"   # must be set before importing the harness

HARNESS = os.path.expanduser("~/proj/tech_record/code/flash-kda/harness")
sys.path.insert(0, HARNESS)

import torch                                                             # noqa: E402
import check_bwd_b2_k128 as m                                           # noqa: E402

# TMA-tagged shared objects so we never clobber the cp.async evidence.
m.so_path = lambda K, V: os.path.join("/tmp", "b2k128_tma_%d_%d.so" % (K, V))


def main():
    assert torch.cuda.is_available(), "need GPU"
    print(f"GPU: {torch.cuda.get_device_name()}  torch={torch.__version__}")
    print("cross: base GEMM core under -DB2M_TMA (TMA staging), K in {64,128,256}")
    libs = {}
    for (K, V) in m.KV:
        m.build(K, V)
        libs[(K, V)] = m.load(K, V)
    worst = {}
    for (K, V) in m.KV:
        lib = libs[(K, V)]
        print(f"\n#### K = {K}, V = {V} (TMA) ####")
        for (name, B, T, H) in m.CASES:
            inp = m.make(B, T, H, K, V, seed=1)
            ref = m.ref_core(inp, m.BT)
            cu = m.run_cuda(lib, inp, m.BT, K, V)
            print(f"\n# {name}")
            for fld in ("dq", "dk", "dw", "dgk"):
                _, mr = m.stats(f"  tma[K{K}V{V}] {fld} vs torch", cu[fld], ref[fld])
                worst[f"{K}_{V}_{fld}"] = max(worst.get(f"{K}_{V}_{fld}", 0.0), mr)
    print("\nworst max_rel (TMA vs torch fp32):")
    for (K, V) in m.KV:
        print(f"  K={K:4d} V={V:4d}: " + "  ".join(
            f"{fld} {worst[f'{K}_{V}_{fld}']:.3e}" for fld in ("dq", "dk", "dw", "dgk")))

    # Part B: TMA vs cp.async -- the two staging paths must agree (same mma accumulation order,
    # same staged values) or the K-generic reduction would show a stage-dependent drift.
    print("\n" + "=" * 92)
    print("Part B: TMA vs cp.async staging -- bit-identity of the K!=128 base GEMM core")
    print("=" * 92)
    tma_so = m.so_path
    m.so_path = lambda K, V: os.path.join("/tmp", "b2k128_%d_%d.so" % (K, V))  # main harness's cp.async
    cs_libs = {}
    for (K, V) in m.KV:
        m.build(K, V)
        cs_libs[(K, V)] = m.load(K, V)
    worst_b = {}
    for (K, V) in m.KV:
        for (name, B, T, H) in m.CASES:
            inp = m.make(B, T, H, K, V, seed=1)
            a = m.run_cuda(libs[(K, V)], inp, m.BT, K, V)
            b = m.run_cuda(cs_libs[(K, V)], inp, m.BT, K, V)
            for fld in ("dq", "dk", "dw", "dgk"):
                ma = (a[fld] - b[fld]).abs().max().item()
                worst_b[f"{fld}"] = max(worst_b.get(fld, 0.0), ma)
    print("  max |TMA - cp.async| over all (K,V,case):")
    for fld in ("dq", "dk", "dw", "dgk"):
        print(f"    {fld:4s}  {worst_b[fld]:.3e}")
    print("\nDONE")


if __name__ == "__main__":
    main()
