"""megatron-port M8-1: anchor the port reference against FLA ``chunk_kda`` for the
**chunk-size / padding / varlen / context-parallel** contract -- the three axes M6/M7
never touched:

* **chunk_size** sweep.  NOTE the roadmap wrote ``{16,32,64}`` but FLA *rejects* 16 for
  KDA: ``chunk.py`` raises ``chunk_size must be either 32 or 64``.  So the sweep is
  ``{32,64}`` x ``T`` with ``T`` hitting: exact multiples, a non-multiple, ``T<chunk``,
  and a large ``T``.  The reference pads ``T`` to ``ceil`` with zeros; FLA masks the
  tail chunk.  Both must agree (the padded tokens carry ``beta=0`` -> no write).
* **varlen** (``cu_seqlens``): B must be 1 and inputs flattened; ``initial_state``/
  ``final_state`` are per-sequence ``[N,HV,K,V]``.  Reference runs the per-seq loop.
* **CP** (``cp_context``): token-partitioned across ranks; ``initial_state``/
  ``output_final_state`` are forbidden and ``cu_seqlens`` is overridden by the context.
  We spawn real NCCL ranks (single node, 8 GPUs) and compare the gathered output against
  the non-CP full-sequence result.  Requires ``NCCL_NVLS_ENABLE=0`` on this node.

Methodology identical to M6/M7: both sides get the *same* bf16 operands, the port
reference runs in fp64, and the residual is read as the FLA bf16-kernel error.

Run (inside kernel_lab, GPU):
    FLA_DISABLE_BACKEND_DISPATCH=1 python3 \
        ~/proj/tech_record/code/flash-kda/megatron-port/harness/check_port_fla_m8.py
"""

import math
import os
import sys

import torch
import torch.nn.functional as F

os.environ.setdefault("FLA_DISABLE_BACKEND_DISPATCH", "1")

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(os.path.dirname(_HERE), "src")
sys.path.insert(0, _SRC)
from kda_ref import kda_chunked_fwd, kda_token_recurrent, kda_gate, kda_l2norm  # noqa: E402

LN2 = math.log(2.0)


def stats(gold, pred):
    g = gold.float()
    p = pred.float()
    d = (g - p).abs()
    return d.max().item(), (d.max() / (g.abs().max() + 1e-12)).item()


def gate_fp64(g_logit, A_log, dt_bias):
    z = g_logit + dt_bias.view(g_logit.shape[-2], -1)
    return -A_log.view(-1, 1).exp() * F.softplus(z)


def make_inputs(B, lens, H, K, V, seed, device, hv=None, raw_logits=False):
    """l2-normalized q/k + realistic KDA gate; returns bf16 operands.

    ``lens`` is a list of per-sequence lengths (one entry -> equal-length, B equal to
    the number of batches? we always use B slices concatenated; here B is implicit).
    """
    T = sum(lens)
    HV = hv or H
    torch.manual_seed(seed)
    q = torch.randn(1, T, H, K, device=device, dtype=torch.float64)
    k = torch.randn(1, T, H, K, device=device, dtype=torch.float64)
    v = torch.randn(1, T, HV, V, device=device, dtype=torch.float64)
    g_logit = 0.4 * torch.randn(1, T, HV, K, device=device, dtype=torch.float64)
    A_log = torch.log(torch.empty(HV, device=device, dtype=torch.float64).uniform_(1.0, 16.0))
    dt_bias = 0.1 * torch.randn(HV * K, device=device, dtype=torch.float64)
    beta = torch.rand(1, T, HV, device=device, dtype=torch.float64)
    h0 = torch.randn(len(lens), HV, K, V, device=device, dtype=torch.float64)

    qn = kda_l2norm(q).to(torch.bfloat16)
    kn = kda_l2norm(k).to(torch.bfloat16)
    vb = v.to(torch.bfloat16)
    beta_b = beta.to(torch.bfloat16)
    if raw_logits:
        return qn, kn, vb, g_logit.to(torch.bfloat16), beta_b, h0, A_log, dt_bias
    g_bf = kda_gate(g_logit, A_log, dt_bias).to(torch.bfloat16)
    return qn, kn, vb, g_bf, beta_b, h0, A_log, dt_bias


def cu_of(lens):
    c = [0]
    for L in lens:
        c.append(c[-1] + L)
    return c


# --------------------------------------------------------------------------------------
# Part 0: reference-internal identities (fp64 CPU)
# --------------------------------------------------------------------------------------
def part0():
    print("\n" + "=" * 96)
    print("[Part 0] reference-internal identities (fp64 CPU): pad semantics + varlen")
    print("=" * 96)
    torch.manual_seed(7)
    # (a) chunked (with zero padding) == token recurrence for T not divisible by chunk
    print("  (a) chunked(zero-pad) vs token recurrence, non-divisible T:")
    for C in (32, 64):
        for T in (33, 130, 250):
            B, H, K, V = 1, 2, 16, 20
            q = torch.randn(B, T, H, K, dtype=torch.float64)
            k = torch.randn(B, T, H, K, dtype=torch.float64)
            v = torch.randn(B, T, H, V, dtype=torch.float64)
            g = -torch.rand(B, T, H, K, dtype=torch.float64)
            beta = torch.rand(B, T, H, dtype=torch.float64)
            o_t, s_t = kda_token_recurrent(q, k, v, g, beta, output_final_state=True)
            o_c, s_c = kda_chunked_fwd(q, k, v, g, beta, output_final_state=True, chunk_size=C)
            eo = (o_t - o_c).abs().max().item() / (o_t.abs().max().item() + 1e-30)
            es = (s_t - s_c).abs().max().item() / (s_t.abs().max().item() + 1e-30)
            print(f"      C={C:>2} T={T:>3} pad={(-T) % C:>3}  rel|do|={eo:.2e} rel|dS|={es:.2e}")

    # (b) per-sequence loop == token recurrence per sequence (varlen reference)
    print("  (b) varlen (per-seq loop) vs token recurrence per sequence:")
    lens = [250, 130, 64]
    B, H, K, V = 1, 2, 16, 20
    T = sum(lens)
    q = torch.randn(B, T, H, K, dtype=torch.float64)
    k = torch.randn(B, T, H, K, dtype=torch.float64)
    v = torch.randn(B, T, H, V, dtype=torch.float64)
    g = -torch.rand(B, T, H, K, dtype=torch.float64)
    beta = torch.rand(B, T, H, dtype=torch.float64)
    h0 = torch.randn(len(lens), H, K, V, dtype=torch.float64)
    cu = cu_of(lens)
    o_parts, s_parts = [], []
    for i, (s, e) in enumerate(zip(cu[:-1], cu[1:])):
        oi, si = kda_token_recurrent(q[:, s:e], k[:, s:e], v[:, s:e], g[:, s:e], beta[:, s:e],
                                     initial_state=h0[i:i + 1], output_final_state=True)
        o_parts.append(oi)
        s_parts.append(si)
    o_t = torch.cat(o_parts, 1)
    s_t = torch.stack(s_parts, 0)
    o_c = torch.cat([kda_chunked_fwd(q[:, s:e], k[:, s:e], v[:, s:e], g[:, s:e], beta[:, s:e],
                                     initial_state=h0[i:i + 1], output_final_state=True, chunk_size=64)[0]
                     for i, (s, e) in enumerate(zip(cu[:-1], cu[1:]))], 1)
    s_c = torch.stack([kda_chunked_fwd(q[:, s:e], k[:, s:e], v[:, s:e], g[:, s:e], beta[:, s:e],
                                       initial_state=h0[i:i + 1], output_final_state=True, chunk_size=64)[1]
                       for i, (s, e) in enumerate(zip(cu[:-1], cu[1:]))], 0)
    eo = (o_t - o_c).abs().max().item() / (o_t.abs().max().item() + 1e-30)
    es = (s_t - s_c).abs().max().item() / (s_t.abs().max().item() + 1e-30)
    print(f"      lens={lens} chunk=64  rel|do|={eo:.2e} rel|dS|={es:.2e}")


# --------------------------------------------------------------------------------------
# Part A: chunk_size x T pad-semantics sweep (equal-length, explicit gate)
# --------------------------------------------------------------------------------------
def run_equal(C, T, seed, device, H=2, K=128, V=128, hv=None, tag=""):
    from fla.ops.kda import chunk_kda
    HV = hv or H
    qn, kn, vb, g_bf, beta_b, h0, _, _ = make_inputs(1, [T], H, K, V, seed, device, hv=HV)
    scale = K ** -0.5
    dout = (torch.randn(1, T, HV, V, device=device) * 0.1).to(torch.bfloat16)
    dfin = (torch.randn(1, HV, K, V, device=device) * 0.01).float()

    qa = qn.clone().requires_grad_(True)
    ka = kn.clone().requires_grad_(True)
    va = vb.clone().requires_grad_(True)
    ga = g_bf.clone().requires_grad_(True)
    ba = beta_b.clone().requires_grad_(True)
    ha = h0.to(torch.float32).clone().requires_grad_(True)
    o_f, s_f = chunk_kda(qa, ka, va, ga, ba, scale=scale, initial_state=ha,
                         output_final_state=True, use_gate_in_kernel=False,
                         use_beta_sigmoid_in_kernel=False, chunk_size=C)
    (o_f.float() * dout.float()).sum().add((s_f.float() * dfin).sum()).backward()

    qr = qn.double().requires_grad_(True)
    kr = kn.double().requires_grad_(True)
    vr = vb.double().requires_grad_(True)
    gr = g_bf.double().requires_grad_(True)
    br = beta_b.double().requires_grad_(True)
    hr = h0.double().clone().requires_grad_(True)
    o_r, s_r = kda_chunked_fwd(qr, kr, vr, gr, br, scale=scale, initial_state=hr,
                               output_final_state=True, chunk_size=C)
    (o_r * dout.double()).sum().add((s_r * dfin.double()).sum()).backward()

    om, orr = stats(o_r, o_f)
    sm, srr = stats(s_r, s_f)
    worst = max(orr, srr)
    gs = ""
    for (nm, rr), (_, rf) in zip([("dq", qr), ("dk", kr), ("dv", vr), ("dg", gr),
                                  ("dbeta", br), ("dh0", hr)],
                                 [("dq", qa), ("dk", ka), ("dv", va), ("dg", ga),
                                  ("dbeta", ba), ("dh0", ha)]):
        gm, grr = stats(rr.grad, rf.grad)
        worst = max(worst, grr)
        gs += f" {nm} {grr:.1e}"
    print(f"    [{tag}] C={C:>2} T={T:>3} pad={(-T) % C:>3}  o {orr:.2e}  S {srr:.2e}  |{gs}")
    return worst


def part_a(device):
    print("\n" + "=" * 96)
    print("[Part A] chunk_size {32,64} x T pad-semantics sweep (explicit gate) vs FLA")
    print("  FLA rejects chunk_size=16 for KDA -> the roadmap {16,32,64} is really {32,64}")
    print("=" * 96)
    try:
        from fla.ops.kda import chunk_kda
        q = torch.randn(1, 64, 2, 128, device=device, dtype=torch.bfloat16)
        k = torch.randn_like(q)
        v = torch.randn(1, 64, 2, 128, device=device, dtype=torch.bfloat16)
        g = -torch.rand(1, 64, 2, 128, device=device, dtype=torch.bfloat16)
        b = torch.rand(1, 64, 2, device=device, dtype=torch.bfloat16)
        chunk_kda(q, k, v, g, b, chunk_size=16)
        print("    chunk_size=16 unexpectedly accepted")
    except ValueError as e:
        print(f"    chunk_size=16 -> ValueError: {e}")

    for C in (32, 64):
        for T in (16, 33, 64, 130, 250, 256):
            run_equal(C, T, seed=100 + T, device=device, tag="eq")


# --------------------------------------------------------------------------------------
# Part B: varlen (cu_seqlens)
# --------------------------------------------------------------------------------------
def _ref_varlen(q, k, v, g, beta, cu_cpu, chunk_size, h0=None, scale=None):
    outs, fins = [], []
    for i in range(len(cu_cpu) - 1):
        s, e = cu_cpu[i], cu_cpu[i + 1]
        hh = None if h0 is None else h0[i:i + 1]
        oi, si = kda_chunked_fwd(q[:, s:e], k[:, s:e], v[:, s:e], g[:, s:e], beta[:, s:e],
                                 scale=scale, initial_state=hh, output_final_state=True,
                                 chunk_size=chunk_size)
        outs.append(oi)
        fins.append(si[0])  # drop the per-seq B=1 dim -> [HV,K,V]
    return torch.cat(outs, 1), torch.stack(fins, 0)  # S: [N,HV,K,V]


def run_varlen(lens, C, seed, device, H=2, K=128, V=128, hv=None, fused=False, tag=""):
    from fla.ops.kda import chunk_kda
    HV = hv or H
    N = len(lens)
    cu_cpu = cu_of(lens)
    if fused:
        qn, kn, vb, g_raw, beta_logit, h0, A_log, dt_bias = make_inputs(
            1, lens, H, K, V, seed, device, hv=HV, raw_logits=True)
    else:
        qn, kn, vb, g_bf, beta_b, h0, _, _ = make_inputs(1, lens, H, K, V, seed, device, hv=HV)
    T = sum(lens)
    scale = K ** -0.5
    cu = torch.tensor(cu_cpu, dtype=torch.int32, device=device)
    dout = (torch.randn(1, T, HV, V, device=device) * 0.1).to(torch.bfloat16)
    dfin = (torch.randn(N, HV, K, V, device=device) * 0.01).float()

    qa = qn.clone().requires_grad_(True)
    ka = kn.clone().requires_grad_(True)
    va = vb.clone().requires_grad_(True)
    ha = h0.to(torch.float32).clone().requires_grad_(True)

    if fused:
        ga = g_raw.clone().requires_grad_(True)
        ba = beta_logit.clone().requires_grad_(True)
        A_f = A_log.float().clone().requires_grad_(True)
        dt_f = dt_bias.float().clone().requires_grad_(True)
        o_f, s_f = chunk_kda(qa, ka, va, ga, ba, scale=scale, initial_state=ha,
                             output_final_state=True, use_gate_in_kernel=True,
                             use_beta_sigmoid_in_kernel=True, A_log=A_f, dt_bias=dt_f,
                             cu_seqlens=cu, chunk_size=C)
        gnames_f = [("dq", qa), ("dk", ka), ("dv", va), ("dg_logit", ga),
                    ("dbeta_logit", ba), ("dh0", ha), ("dA_log", A_f), ("ddt_bias", dt_f)]
    else:
        ga = g_bf.clone().requires_grad_(True)
        ba = beta_b.clone().requires_grad_(True)
        o_f, s_f = chunk_kda(qa, ka, va, ga, ba, scale=scale, initial_state=ha,
                             output_final_state=True, use_gate_in_kernel=False,
                             use_beta_sigmoid_in_kernel=False, cu_seqlens=cu, chunk_size=C)
        gnames_f = [("dq", qa), ("dk", ka), ("dv", va), ("dg", ga), ("dbeta", ba), ("dh0", ha)]
    (o_f.float() * dout.float()).sum().add((s_f.float() * dfin).sum()).backward()

    qr = qn.double().requires_grad_(True)
    kr = kn.double().requires_grad_(True)
    vr = vb.double().requires_grad_(True)
    hr = h0.double().clone().requires_grad_(True)
    if fused:
        gr = g_raw.double().requires_grad_(True)
        br = beta_logit.double().requires_grad_(True)
        Ar = A_log.double().requires_grad_(True)
        dtr = dt_bias.double().requires_grad_(True)
        g_act = gate_fp64(gr, Ar, dtr)
        beta_act = torch.sigmoid(br)
        o_r, s_r = _ref_varlen(qr, kr, vr, g_act, beta_act, cu_cpu, C, h0=hr, scale=scale)
        gnames_r = [("dq", qr), ("dk", kr), ("dv", vr), ("dg_logit", gr),
                    ("dbeta_logit", br), ("dh0", hr), ("dA_log", Ar), ("ddt_bias", dtr)]
    else:
        gr = g_bf.double().requires_grad_(True)
        br = beta_b.double().requires_grad_(True)
        o_r, s_r = _ref_varlen(qr, kr, vr, gr, br, cu_cpu, C, h0=hr, scale=scale)
        gnames_r = [("dq", qr), ("dk", kr), ("dv", vr), ("dg", gr), ("dbeta", br), ("dh0", hr)]
    (o_r * dout.double()).sum().add((s_r * dfin.double()).sum()).backward()

    om, orr = stats(o_r, o_f)
    sm, srr = stats(s_r, s_f)
    worst = max(orr, srr)
    gs = ""
    for (nm, rr), (_, rf) in zip(gnames_r, gnames_f):
        gm, grr = stats(rr.grad, rf.grad)
        worst = max(worst, grr)
        gs += f" {nm} {grr:.1e}"
    mode = "fused" if fused else "explicit"
    print(f"    [{tag}] lens={lens} C={C:>2} {mode:8s}  o {orr:.2e}  S {srr:.2e}  |{gs}")
    return worst


def part_b(device):
    print("\n" + "=" * 96)
    print("[Part B] varlen (cu_seqlens), B=1 flattened; initial/final state per sequence")
    print("=" * 96)
    run_varlen([250, 130, 64], 64, 201, device, tag="3seq")
    run_varlen([128, 64, 96, 64], 32, 202, device, tag="4seq C32")
    run_varlen([64, 250, 250], 64, 203, device, hv=4, tag="GVA x2")
    run_varlen([250, 130, 64], 64, 204, device, fused=True, tag="fused gate")


# --------------------------------------------------------------------------------------
# Part C: context parallel (real NCCL ranks, single node)
# --------------------------------------------------------------------------------------
def _cp_make(lens, H=2, K=128, V=128, seed=0):
    T = sum(lens)
    torch.manual_seed(seed)
    q = torch.randn(1, T, H, K, device="cuda")
    k = torch.randn(1, T, H, K, device="cuda")
    v = torch.randn(1, T, H, V, device="cuda")
    qn = (q / q.pow(2).sum(-1, keepdim=True).add(1e-6).sqrt()).to(torch.bfloat16)
    kn = (k / k.pow(2).sum(-1, keepdim=True).add(1e-6).sqrt()).to(torch.bfloat16)
    vb = v.to(torch.bfloat16)
    A = torch.log(torch.empty(H, device="cuda").uniform_(1, 16))
    dt = 0.1 * torch.randn(H * K, device="cuda")
    gl = torch.randn(1, T, H, K, device="cuda")
    g = (-A.view(-1, 1).exp() * F.softplus(gl + dt.view(H, -1))).to(torch.bfloat16)
    beta = torch.rand(1, T, H, device="cuda").to(torch.bfloat16)
    return qn, kn, vb, g, beta


def _cp_worker(rank, world, lens, C, out_dir):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29677"
    os.environ["NCCL_NVLS_ENABLE"] = "0"
    import torch.distributed as dist
    dist.init_process_group("nccl", rank=rank, world_size=world)
    torch.cuda.set_device(rank % torch.cuda.device_count())
    from fla.ops.cp import build_cp_context
    from fla.ops.kda import chunk_kda
    q, k, v, g, beta = _cp_make(lens)
    cu = torch.tensor(cu_of(lens), dtype=torch.int32, device="cuda")
    ctx = build_cp_context(cu, group=dist.group.WORLD)
    part = sum(lens) // world
    s, e = rank * part, (rank + 1) * part
    o, _ = chunk_kda(q[:, s:e].contiguous(), k[:, s:e].contiguous(), v[:, s:e].contiguous(),
                     g[:, s:e].contiguous(), beta[:, s:e].contiguous(),
                     cu_seqlens=cu, chunk_size=C, cp_context=ctx)
    torch.save(o.float().cpu(), os.path.join(out_dir, f"cp_{world}_{rank}.pt"))
    dist.barrier()
    dist.destroy_process_group()


def part_c(device):
    import torch.multiprocessing as mp
    print("\n" + "=" * 96)
    print("[Part C] context parallel: gathered CP output vs non-CP full sequence (NCCL)")
    print("  CP forbids initial/final state; cu_seqlens overridden by context.")
    print("=" * 96)
    if torch.cuda.device_count() < 1:
        print("    no CUDA visible, skip")
        return
    mp.set_start_method("spawn", force=True)
    from fla.ops.kda import chunk_kda
    out = os.path.join(_HERE, "_cptmp")
    os.makedirs(out, exist_ok=True)
    for world, lens, C in [(1, [256], 64), (2, [256], 64), (4, [256], 64),
                           (2, [128, 128], 64), (2, [250], 64), (4, [130], 32)]:
        q, k, v, g, beta = _cp_make(lens)
        cu = torch.tensor(cu_of(lens), dtype=torch.int32, device=device)
        o_ref, _ = chunk_kda(q, k, v, g, beta, cu_seqlens=cu, chunk_size=C)
        o_ref = o_ref.float().cpu()
        del q, k, v, g, beta, cu
        torch.cuda.empty_cache()
        try:
            mp.spawn(_cp_worker, args=(world, lens, C, out), nprocs=world, join=True)
            parts = [torch.load(os.path.join(out, f"cp_{world}_{r}.pt")) for r in range(world)]
            o_cp = torch.cat(parts, dim=1)
            n = min(o_cp.shape[1], o_ref.shape[1])
            d = (o_cp[:, :n] - o_ref[:, :n]).abs().max().item()
            rel = d / (o_ref.abs().max().item() + 1e-12)
            part = sum(lens) // world
            aligned = (part % C == 0) and (sum(lens) % world == 0)
            print(f"    lens={lens} world={world} C={C:>2} part={part} aligned={aligned}  "
                  f"got_T={o_cp.shape[1]} ref_T={o_ref.shape[1]}  max_abs={d:.3e} max_rel={rel:.3e}")
        except Exception as ex:  # noqa: BLE001
            print(f"    lens={lens} world={world} C={C}: ERROR {type(ex).__name__}: {str(ex)[:160]}")


def main():
    torch.backends.cuda.matmul.allow_tf32 = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 96)
    print("megatron-port M8-1: chunk_size / pad / varlen / CP anchor vs FLA chunk_kda")
    print(f"torch={torch.__version__} device={torch.cuda.get_device_name(0) if device == 'cuda' else 'cpu'}")
    print("=" * 96)
    part0()
    part_a(device)
    part_b(device)
    part_c(device)
    print("\n" + "=" * 96)
    print("Done. Residual = FLA bf16-kernel error (reference is fp64).")
    print("=" * 96)


if __name__ == "__main__":
    main()
