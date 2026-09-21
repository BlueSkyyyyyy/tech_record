#!/usr/bin/env python3
"""38 篇 SOTA 对标：paged KV-cache decode 的 torch / flash-attention 参考。

本项目的手写 kernel 用分页布局 [num_blocks, P, Hkv, D]；
这里把它 gather 成 dense 后，跑：
  * torch SDPA（enable_gqa）
  * flash_attn_func（q_len=1 的 decode 路径）
  * 纯 KV 读取带宽（sum(k)+sum(v)）作为 HBM 可达上限
给出 GB/s（按每 token 读一遍 KV 的口径），与手写 kernel 对比。

运行：docker exec kernel_lab python3 paged_ref.py [B] [L]
"""
import sys
import time

import torch
import torch.nn.functional as F

Hq, Hkv, G, D, P = 40, 8, 5, 128, 16


def bench(fn, warmup=10, iters=30):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / iters * 1e3


def main():
    B = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    L = int(sys.argv[2]) if len(sys.argv) > 2 else 32768
    dev = "cuda"
    torch.manual_seed(0)
    print(f"torch {torch.__version__}  B={B} L={L}  Hq={Hq} Hkv={Hkv} D={D} P={P}")

    nblk = (L + P - 1) // P
    nb_total = B * nblk
    kc = torch.randn(nb_total, P, Hkv, D, dtype=torch.bfloat16, device=dev) * 0.1
    vc = torch.randn_like(kc)
    q = torch.randn(B, Hq, D, dtype=torch.bfloat16, device=dev) * 0.1
    # block table：request b 用连续页 [b*nblk, (b+1)*nblk)
    bt = torch.arange(nb_total, device=dev).view(B, nblk)

    kb = kv_bytes = 2 * B * Hkv * L * D * 2
    print(f"KV bytes = {kv_bytes/1e9:.3f} GB")

    # ---- HBM 可达上限：纯读 ----
    def raw_read():
        return kc.view(-1)[::1].sum() + vc.view(-1)[::1].sum()
    ms = bench(raw_read)
    print(f"raw KV read            {ms:8.3f} ms  {kv_bytes/ms/1e6:8.1f} GB/s")

    # ---- gather 成 dense ----
    t0 = time.time()
    kd = kc.view(B, nblk, P, Hkv, D).reshape(B, nblk * P, Hkv, D)[:, :L]
    vd = vc.view(B, nblk, P, Hkv, D).reshape(B, nblk * P, Hkv, D)[:, :L]
    torch.cuda.synchronize()
    print(f"gather (view/reshape)  {(time.time()-t0)*1e3:8.3f} ms  "
          f"{kv_bytes/((time.time()-t0)*1e3)/1e6:8.1f} GB/s")

    # ---- torch SDPA（GQA）----
    qq = q.view(B, Hq, 1, D)
    kk = kd.permute(0, 2, 1, 3)  # [B,Hkv,L,D]
    vv = vd.permute(0, 2, 1, 3)

    def sdpa():
        return F.scaled_dot_product_attention(qq, kk, vv, enable_gqa=True)
    try:
        out = sdpa()
        torch.cuda.synchronize()
        ms = bench(sdpa)
        print(f"torch SDPA (gqa)       {ms:8.3f} ms  {kv_bytes/ms/1e6:8.1f} GB/s")
    except Exception as e:
        print("torch SDPA failed:", e)

    # ---- flash_attn decode ----
    try:
        from flash_attn import flash_attn_func
        qf = q.view(B, 1, Hq, D)
        kf = kd.contiguous()
        vf = vd.contiguous()

        def fa():
            return flash_attn_func(qf, kf, vf)
        out2 = fa()
        torch.cuda.synchronize()
        ms = bench(fa)
        print(f"flash_attn decode      {ms:8.3f} ms  {kv_bytes/ms/1e6:8.1f} GB/s")
        # 与 SDPA 对拍
        ref = F.scaled_dot_product_attention(
            qq, kk, vv, enable_gqa=True).view(B, Hq, D)
        err = (out2.view(B, Hq, D) - ref).abs().max().item()
        print(f"  flash_attn vs SDPA max_abs_err = {err:.3e}")
    except Exception as e:
        print("flash_attn failed:", e)


if __name__ == "__main__":
    main()
