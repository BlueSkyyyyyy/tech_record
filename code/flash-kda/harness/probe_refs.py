"""P0-3: probe FlashKDA forward vs FLA chunk_kda (and an fp64 gold).

Runs on the same inputs:
  * FlashKDA  `flash_kda.fwd`      (built CUDA kernel, bf16, CHUNK=16)
  * FLA       `chunk_kda`          (triton, bf16, chunk_size=64)
  * FLA       `fused_recurrent_kda`(fp64 gold, token-recurrent)

Reports max abs / max rel / RMSE-ratio for outputs and final states.
"""

import math
import sys

import torch


def get_err_ratio(x, y):
    err = (x.detach().float() - y.detach().float()).flatten().square().mean().sqrt().item()
    base = x.detach().float().flatten().square().mean().sqrt().item()
    return err / (base + 1e-8)


def stats(label, gold, pred):
    d = (gold.float() - pred.float()).abs()
    ref = gold.float().abs()
    max_abs = d.max().item()
    max_rel = (d.max() / (ref.max() + 1e-8)).item()
    mean_abs = d.mean().item()
    ratio = get_err_ratio(gold, pred)
    print(f"  {label:28s} max_abs={max_abs:.4e}  max_rel={max_rel:.4e}  "
          f"mean_abs={mean_abs:.4e}  rmse_ratio={ratio:.4e}")
    return dict(max_abs=max_abs, max_rel=max_rel, mean_abs=mean_abs, ratio=ratio)


@torch.inference_mode()
def run_case(name, q, k, v, g, beta, h0, A_log, dt_bias, scale, lower_bound):
    from fla.ops.kda import chunk_kda, fused_recurrent_kda
    import flash_kda

    B, T, H, D = q.shape
    device = q.device
    print(f"\n{'='*78}\nCase [{name}]  B={B} T={T} H={H} D={D}")

    g_aug = g.float() + dt_bias.unsqueeze(0).unsqueeze(0)
    g_act = lower_bound * torch.sigmoid(torch.exp(A_log.view(1, 1, H, 1)) * g_aug)
    beta_act = torch.sigmoid(beta.float())

    tri, tri_ht = fused_recurrent_kda(
        q=q.to(torch.float64), k=k.to(torch.float64), v=v.to(torch.float64),
        g=g_act.to(torch.float64), beta=beta_act.to(torch.float64),
        A_log=None, dt_bias=None, scale=scale,
        initial_state=h0.to(torch.float64), output_final_state=True,
        use_qk_l2norm_in_kernel=True, use_gate_in_kernel=False,
        lower_bound=None, transpose_state_layout=True,
    )
    tri = tri.float()
    tri_ht = tri_ht.float()

    chunk_o, chunk_ht = chunk_kda(
        q=q.clone().to(torch.bfloat16), k=k.clone().to(torch.bfloat16), v=v.clone(),
        g=g.clone(), beta=beta_act.to(torch.bfloat16), scale=scale,
        initial_state=h0.clone(), output_final_state=True,
        use_gate_in_kernel=True, use_qk_l2norm_in_kernel=True,
        A_log=A_log.clone(), dt_bias=dt_bias.clone(), lower_bound=lower_bound,
        transpose_state_layout=True,
    )
    chunk_o = chunk_o.float()
    chunk_ht = chunk_ht.float()

    out_fk = torch.zeros_like(q)
    final_fk = torch.zeros_like(h0.to(torch.bfloat16))
    flash_kda.fwd(
        q.to(torch.bfloat16), k.to(torch.bfloat16), v.clone(), g.to(torch.bfloat16),
        beta.to(torch.bfloat16), scale, out_fk,
        A_log=A_log.clone(), dt_bias=dt_bias.clone(), lower_bound=lower_bound,
        initial_state=h0.clone().to(torch.bfloat16), final_state=final_fk,
    )
    torch.cuda.synchronize()
    out_fk = out_fk.float()
    final_fk = final_fk.float()

    print(f"  scale(o): gold={tri.abs().max().item():.3f} "
          f"chunk={chunk_o.abs().max().item():.3f} flash={out_fk.abs().max().item():.3f}")
    print(f"  scale(ht): gold={tri_ht.abs().max().item():.3f} "
          f"chunk={chunk_ht.abs().max().item():.3f} flash={final_fk.abs().max().item():.3f}")

    print(" output o vs fp64 gold:")
    r_chunk = stats("chunk_kda", tri, chunk_o)
    r_flash = stats("flash_kda", tri, out_fk)
    stats("flash_kda vs chunk_kda", chunk_o, out_fk)

    print(" final state ht vs fp64 gold:")
    s_chunk = stats("chunk_kda", tri_ht, chunk_ht)
    s_flash = stats("flash_kda", tri_ht, final_fk)
    stats("flash_kda vs chunk_kda", chunk_ht, final_fk)
    return dict(o_flash=r_flash, ht_flash=s_flash, o_chunk=r_chunk, ht_chunk=s_chunk)


def main():
    torch.manual_seed(42)
    D = 128
    H = 4
    B = 1
    T = 512
    lower_bound = -5.0
    scale = 1.0 / math.sqrt(D)
    device = "cuda"

    q = torch.rand(B, T, H, D, dtype=torch.bfloat16, device=device)
    k = torch.rand(B, T, H, D, dtype=torch.bfloat16, device=device)
    v = torch.rand(B, T, H, D, dtype=torch.bfloat16, device=device)
    beta = torch.randn(B, T, H, dtype=torch.bfloat16, device=device)
    h0 = torch.randn(B, H, D, D, dtype=torch.float32, device=device)
    A_log = torch.rand(H, dtype=torch.float32, device=device)
    dt_bias = torch.rand(H, D, dtype=torch.float32, device=device)

    g_rand = torch.randn(B, T, H, D, dtype=torch.bfloat16, device=device) * 4
    g_zero = torch.zeros(B, T, H, D, dtype=torch.bfloat16, device=device)

    run_case("g=N(0,4)", q, k, v, g_rand, beta, h0, A_log, dt_bias, scale, lower_bound)
    run_case("g=0", q, k, v, g_zero, beta, h0, A_log, dt_bias, scale, lower_bound)
    print("\nDONE")


if __name__ == "__main__":
    main()
