"""megatron-port M8-2: anchor the **context-parallel (CP) backward** contract -- the
piece M8-1 explicitly left open (its Part C anchored only the CP *forward*).

Background (docs/06 §5, docs/07 §1).  With ``cp_context`` set, ``chunk_kda``:
  * forbids ``initial_state`` / ``output_final_state`` (``chunk.py:362-364``);
  * overrides ``cu_seqlens`` with the rank-local (clamped) one (``chunk.py:365-369``);
  * forwards through ``chunk_gated_delta_rule_fwd_h_pre_process`` which all-gathers the
    per-rank partial states and merges the ``pre_num_ranks`` predecessors'
    contributions (``ops/cp/chunk_delta_h.py:747-828``).

The **backward** has an entirely separate assembly (``ops/kda/chunk_bwd.py:522-539``):
after ``dAqk/dv`` it calls ``chunk_gated_delta_rule_bwd_dhu_pre_process``
(``ops/cp/chunk_delta_h.py:831-920``) which (a) recomputes each rank's own state-gradient
partial ``dhm`` w.r.t. ``(q,k,w,do,dv,g)`` via ``pre_process_bwd_kernel_merged``, (b)
all-gathers ``dhm`` across ranks and merges the ``post_num_ranks`` successors into the
incoming ``dht`` via ``merge_fwd_bwd_kernel(..., FORWARD=False)``.  That ``dht`` then
feeds the ordinary ``chunk_gated_delta_rule_bwd_dhu``.  This file checks the *whole*
assembly end-to-end: gathered per-rank gradients ``dq/dk/dv/dg/dbeta`` must equal the
non-CP full-sequence gradients.

Methodology identical to M8-1: same bf16 operands on both sides; the comparison is
CP (FLA kernels, real NCCL ranks) vs non-CP (FLA kernels, full sequence).  Because CP
and non-CP share the same bf16 kernel family, an *aligned* partition is expected to be
bit-identical (as the forward was in M8-1 §5); the point of the test is exactly that.

Run (inside kernel_lab, idle GPUs):
    CUDA_VISIBLE_DEVICES=3,4,5,6,7 FLA_DISABLE_BACKEND_DISPATCH=1 python3 \
        ~/proj/tech_record/code/flash-kda/megatron-port/harness/check_port_fla_m8b.py
"""

import math
import os
import sys

import torch
import torch.nn.functional as F

os.environ.setdefault("FLA_DISABLE_BACKEND_DISPATCH", "1")

LN2 = math.log(2.0)


def stats(gold, pred):
    g = gold.float()
    p = pred.float()
    d = (g - p).abs()
    return d.max().item(), (d.max() / (g.abs().max() + 1e-12)).item()


# --------------------------------------------------------------------------------------
# Part 0: reference-internal identity (fp64 CPU) -- the chunked backward is
# invariant to the zero-padded chunking (the algebra CP relies on: tokens are only
# regrouped, the per-token adjoint is unchanged).
# --------------------------------------------------------------------------------------
def part0():
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
    from kda_ref import kda_chunked_fwd, kda_token_recurrent  # noqa: E402
    print("\n" + "=" * 96)
    print("[Part 0] reference-internal identity (fp64 CPU): chunked backward == token backward")
    print("=" * 96)
    for C in (32, 64):
        for T in (33, 130, 250):
            B, H, K, V = 1, 2, 16, 20
            gen = torch.Generator().manual_seed(700 + T)
            q = torch.randn(B, T, H, K, generator=gen, dtype=torch.float64)
            k = torch.randn(B, T, H, K, generator=gen, dtype=torch.float64)
            v = torch.randn(B, T, H, V, generator=gen, dtype=torch.float64)
            g = -torch.rand(B, T, H, K, generator=gen, dtype=torch.float64)
            beta = torch.rand(B, T, H, generator=gen, dtype=torch.float64)
            dout = torch.randn(B, T, H, V, generator=gen, dtype=torch.float64)

            names = ["q", "k", "v", "g", "beta"]
            def run(fn):
                xs = [x.clone().requires_grad_(True) for x in (q, k, v, g, beta)]
                o, _ = fn(*xs)
                (o * dout).sum().backward()
                return [x.grad for x in xs]
            gt = run(lambda qq, kk, vv, gg, bb: kda_token_recurrent(qq, kk, vv, gg, bb))
            gc = run(lambda qq, kk, vv, gg, bb: kda_chunked_fwd(qq, kk, vv, gg, bb, chunk_size=C))
            gs = ""
            for nm, a, b in zip(names, gt, gc):
                _, rr = stats(a, b)
                gs += f" {nm} {rr:.1e}"
            print(f"      C={C:>2} T={T:>3} pad={(-T) % C:>3}  |{gs}")


def cu_of(lens):
    c = [0]
    for L in lens:
        c.append(c[-1] + L)
    return c


def make_cp_inputs(lens, C, H=2, K=128, V=128, hv=None, seed=0):
    """Deterministic CPU-built bf16 operands, so every rank and the reference see the
    *same* bits (Philox seeded on CPU, no per-device RNG divergence)."""
    HV = hv or H
    T = sum(lens)
    g_cpu = torch.Generator().manual_seed(seed)
    q = torch.randn(1, T, H, K, generator=g_cpu, dtype=torch.float32)
    k = torch.randn(1, T, H, K, generator=g_cpu, dtype=torch.float32)
    v = torch.randn(1, T, HV, V, generator=g_cpu, dtype=torch.float32)
    qn = (q / q.pow(2).sum(-1, keepdim=True).add(1e-6).sqrt()).to(torch.bfloat16)
    kn = (k / k.pow(2).sum(-1, keepdim=True).add(1e-6).sqrt()).to(torch.bfloat16)
    vb = v.to(torch.bfloat16)
    A = torch.log(torch.rand(HV, generator=g_cpu) * 15.0 + 1.0)
    dt = 0.1 * torch.randn(HV * K, generator=g_cpu)
    gl = torch.randn(1, T, HV, K, generator=g_cpu)
    beta = torch.rand(1, T, HV, generator=g_cpu).to(torch.bfloat16)
    dout = (0.1 * torch.randn(1, T, HV, V, generator=g_cpu)).to(torch.bfloat16)
    return {
        "qn": qn, "kn": kn, "vb": vb, "beta": beta, "dout": dout,
        "gl": gl, "A": A, "dt": dt,
    }


def g_activated(inp):
    """explicit pre-computed decay g (natural log), bf16."""
    gl, A, dt = inp["gl"], inp["A"], inp["dt"]
    z = gl + dt.view(A.shape[0], -1)
    return (-A.view(-1, 1).exp() * F.softplus(z)).to(torch.bfloat16)


# --------------------------------------------------------------------------------------
# non-CP reference: full sequence, FLA kernels, backward
# --------------------------------------------------------------------------------------
def run_noncp(inp, lens, C, device, fused=False, svf=False):
    from fla.ops.kda import chunk_kda
    cu = torch.tensor(cu_of(lens), dtype=torch.int32, device=device)
    qn = inp["qn"].to(device)
    kn = inp["kn"].to(device)
    vb = inp["vb"].to(device)
    beta = inp["beta"].to(device)
    dout = inp["dout"].to(device)
    scale = qn.shape[-1] ** -0.5

    qa = qn.clone().requires_grad_(True)
    ka = kn.clone().requires_grad_(True)
    va = vb.clone().requires_grad_(True)
    ba = beta.clone().requires_grad_(True)
    kw = {}
    if fused:
        ga = inp["gl"].to(device).clone().requires_grad_(True)
        bl = beta.clone().requires_grad_(True)
        ba = bl
        A = inp["A"].to(device).float().clone().requires_grad_(True)
        dt = inp["dt"].to(device).float().clone().requires_grad_(True)
        kw = dict(use_gate_in_kernel=True, use_beta_sigmoid_in_kernel=True, A_log=A, dt_bias=dt)
    else:
        ga = g_activated(inp).to(device).clone().requires_grad_(True)
    o, _ = chunk_kda(qa, ka, va, ga, ba, scale=scale, output_final_state=False,
                     cu_seqlens=cu, chunk_size=C, state_v_first=svf, **kw)
    (o.float() * dout.float()).sum().backward()
    return {
        "o": o.detach().float().cpu(),
        "dq": qa.grad.float().cpu(), "dk": ka.grad.float().cpu(), "dv": va.grad.float().cpu(),
        "dg": ga.grad.float().cpu(), "dbeta": ba.grad.float().cpu(),
    }


# --------------------------------------------------------------------------------------
# CP worker: real NCCL rank, local token slice, backward, save gradients
# --------------------------------------------------------------------------------------
def _cp_slice(lens, world, rank):
    total = sum(lens)
    part = total // world
    return rank * part, (rank + 1) * part, part


def _cp_worker(rank, world, lens, C, inp, fused, svf, out_dir):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29678"
    os.environ["NCCL_NVLS_ENABLE"] = "0"
    import torch.distributed as dist
    dist.init_process_group("nccl", rank=rank, world_size=world)
    torch.cuda.set_device(rank % torch.cuda.device_count())
    dev = torch.cuda.current_device()
    from fla.ops.cp import build_cp_context
    from fla.ops.kda import chunk_kda

    cu_cpu = cu_of(lens)
    cu = torch.tensor(cu_cpu, dtype=torch.int32, device=dev)
    ctx = build_cp_context(cu, group=dist.group.WORLD)
    s, e, part = _cp_slice(lens, world, rank)

    qn = inp["qn"][:, s:e].to(dev)
    kn = inp["kn"][:, s:e].to(dev)
    vb = inp["vb"][:, s:e].to(dev)
    beta = inp["beta"][:, s:e].to(dev)
    dout = inp["dout"][:, s:e].to(dev)
    scale = qn.shape[-1] ** -0.5

    qa = qn.clone().contiguous().requires_grad_(True)
    ka = kn.clone().contiguous().requires_grad_(True)
    va = vb.clone().contiguous().requires_grad_(True)
    ba = beta.clone().requires_grad_(True)
    kw = {}
    if fused:
        ga = inp["gl"][:, s:e].to(dev).clone().requires_grad_(True)
        ba = beta.clone().requires_grad_(True)
        A = inp["A"].to(dev).float().clone().requires_grad_(True)
        dt = inp["dt"].to(dev).float().clone().requires_grad_(True)
        kw = dict(use_gate_in_kernel=True, use_beta_sigmoid_in_kernel=True, A_log=A, dt_bias=dt)
    else:
        ga = g_activated(inp)[:, s:e].to(dev).clone().requires_grad_(True)

    o, _ = chunk_kda(qa, ka, va, ga, ba, scale=scale, cp_context=ctx,
                     chunk_size=C, state_v_first=svf, **kw)
    (o.float() * dout.float()).sum().backward()

    torch.save({
        "o": o.detach().float().cpu(),
        "dq": qa.grad.float().cpu(), "dk": ka.grad.float().cpu(), "dv": va.grad.float().cpu(),
        "dg": ga.grad.float().cpu(), "dbeta": ba.grad.float().cpu(),
    }, os.path.join(out_dir, f"bwd_{world}_{rank}.pt"))
    dist.barrier(device_ids=[dev])
    dist.destroy_process_group()


# --------------------------------------------------------------------------------------
# one CP-vs-nonCP comparison
# --------------------------------------------------------------------------------------
def run_case(lens, world, C, tag, out, device, fused=False, svf=False, seed=0,
             hv=None, verbose=True):
    import torch.multiprocessing as mp
    inp = make_cp_inputs(lens, C, hv=hv, seed=seed)
    ref = run_noncp(inp, lens, C, device, fused=fused, svf=svf)

    try:
        mp.spawn(_cp_worker, args=(world, lens, C, inp, fused, svf, out), nprocs=world, join=True)
    except Exception as ex:  # noqa: BLE001
        if verbose:
            print(f"    [{tag}] lens={lens} world={world} C={C}: ERROR "
                  f"{type(ex).__name__}: {str(ex)[:150]}")
        return None

    total = sum(lens)
    part = total // world
    aligned = (total % world == 0) and (part % C == 0)

    og, gg = [], {k: [] for k in ("dq", "dk", "dv", "dg", "dbeta")}
    for r in range(world):
        d = torch.load(os.path.join(out, f"bwd_{world}_{r}.pt"))
        og.append(d["o"])
        for k in gg:
            gg[k].append(d[k])
    o_cp = torch.cat(og, dim=1)
    n = min(o_cp.shape[1], ref["o"].shape[1])

    o_m, o_r = stats(ref["o"][:, :n], o_cp[:, :n])
    worst = o_r
    gs, worst_name = "", "o"
    for k in ("dq", "dk", "dv", "dg", "dbeta"):
        cp_k = torch.cat(gg[k], dim=1)[:, :n]
        m, rr = stats(ref[k][:, :n], cp_k)
        if rr > worst:
            worst, worst_name = rr, k
        gs += f" {k} {rr:.1e}"
    if verbose:
        note = "" if aligned else "  <-- UNALIGNED (expected to break)"
        print(f"    [{tag}] lens={lens} world={world} C={C:>2} part={part} "
              f"aligned={aligned}{note}")
        print(f"        o {o_r:.2e}  |{gs}   (worst={worst_name} {worst:.1e})")
    return worst, aligned


def part_a(device, out):
    print("\n" + "=" * 96)
    print("[Part A] CP backward, single sequence, equal split: gathered grads vs non-CP")
    print("=" * 96)
    for world, lens, C in [(1, [256], 64), (2, [256], 64), (3, [192], 64),
                           (4, [256], 64), (2, [256], 32), (4, [256], 32)]:
        run_case(lens, world, C, "eq", out, device)


def part_b(device, out):
    print("\n" + "=" * 96)
    print("[Part B] varlen x CP: several sequences flattened, token-partitioned across ranks")
    print("=" * 96)
    for world, lens, C in [(2, [128, 128], 64), (4, [64, 64, 64, 64], 64),
                           (2, [128, 128], 32), (4, [128, 64, 64], 32),
                           (2, [192, 64], 64)]:
        run_case(lens, world, C, "varlen", out, device)


def part_c(device, out):
    print("\n" + "=" * 96)
    print("[Part C] the two CP hard constraints for the backward (expect them to break)")
    print("=" * 96)
    # part % C != 0  (global chunk boundary cut inside a rank)
    run_case([250], 2, 64, "part%C!=0", out, device)
    # total % world != 0  (tail tokens unclaimed -> gathered length short)
    run_case([130], 4, 32, "total%world!=0", out, device)


def part_d(device, out):
    print("\n" + "=" * 96)
    print("[Part D] CP backward with the fused in-kernel gate (logits entry)")
    print("=" * 96)
    run_case([256], 2, 64, "fused", out, device, fused=True, seed=11)
    run_case([128, 128], 2, 64, "fused varlen", out, device, fused=True, seed=12)


def part_e(device, out):
    print("\n" + "=" * 96)
    print("[Part E] GVA (H_V > H) x CP backward: v/g/beta have 2x heads of q/k")
    print("=" * 96)
    run_case([256], 2, 64, "GVA x2", out, device, hv=4, seed=21)
    run_case([128, 128], 4, 64, "GVA x2 varlen", out, device, hv=4, seed=22)


def main():
    torch.backends.cuda.matmul.allow_tf32 = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 96)
    print("megatron-port M8-2: CP backward anchor (gathered CP grads vs non-CP) vs FLA chunk_kda")
    print(f"torch={torch.__version__} device={torch.cuda.get_device_name(0)} "
          f"visible={torch.cuda.device_count()}")
    print("=" * 96)
    if not torch.cuda.is_available():
        print("no CUDA, abort")
        return
    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_m8btmp")
    os.makedirs(out, exist_ok=True)
    part0()
    part_a(device, out)
    part_b(device, out)
    part_c(device, out)
    part_d(device, out)
    part_e(device, out)
    print("\n" + "=" * 96)
    print("Done. CP==non-CP up to bf16 kernel noise; aligned partitions should be bit-exact.")
    print("=" * 96)


if __name__ == "__main__":
    main()
