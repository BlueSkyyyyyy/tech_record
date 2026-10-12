#!/usr/bin/env python3
# P8-1 实验：DeepGEMM einsum `bmk,bnk->mn`（批归约 GEMM）在 H100 上的语义/数值/性能实测。
#
# 数学：D[m,n] = sum_b sum_k A[b,m,k] * B[b,n,k]    （A:[s,m,k], B:[s,n,k], D:[m,n]）
# 实现：SM90 `sm90_bmk_bnk_mn.cuh`（block 128x128x64，WGMMA，split-K + atomicAdd float2）。
#
# 关键论点：
#   1) 它是「把 batch 维并入收缩轴」的 GEMM：D = A.view(m, s*k) @ B.view(n, s*k).T；
#   2) 它是 MoE wgrad 里「按 rank/按 chunk 求部分和后归约」的融合版本（省掉 [s,m,n] 中间量）；
#   3) 因为 split-K 用 atomicAdd，fp32 累积是非确定性的（可复现地观测到 bit 级抖动）。
#
# 运行：bash harness/exp_einsum_bmk_bnk.sh > harness/exp_einsum_bmk_bnk.out.txt
import torch
import deep_gemm

torch.manual_seed(0)
DEV = "cuda"
SMS = torch.cuda.get_device_properties(0).multi_processor_count


def reldiff(x, ref):
    denom = ref.abs().max().item()
    return (x - ref).abs().max().item() / (denom if denom > 0 else 1.0)


def bench(fn, iters=50, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters * 1e3  # us


def split_info(s, m, n, k, bm=128, bn=128, bk=64):
    num_mn = (m + bm - 1) // bm * ((n + bn - 1) // bn)
    num_sk = s * (k // bk)
    denom = max(SMS // num_mn, 1)
    split = (num_sk + denom - 1) // denom
    grid = num_mn * ((num_sk + split - 1) // split)
    return num_mn, num_sk, split, grid


print("=" * 78)
print(f"H100 sm{torch.cuda.get_device_capability()[0]}{torch.cuda.get_device_capability()[1]}  "
      f"num_sms={SMS}  torch={torch.__version__}")
print("=" * 78)

# ---------------------------------------------------------------------------
# A. 语义与 dtype 覆盖：ref = sum_b (A_b @ B_b^T)
# ---------------------------------------------------------------------------
print("\n[A] 语义: D[m,n] = sum_b sum_k A[b,m,k] B[b,n,k]  vs torch.bmm(...).sum(0)")
print(f"{'s':>6} {'m':>5} {'n':>5} {'k':>5} {'dtype':>8} {'reldiff':>12} "
      f"{'num_mn':>6} {'num_sk':>7} {'split':>5} {'grid':>6}")
for s in (129, 4096, 8192):
    for (m, n, k) in ((128, 384, 128), (256, 256, 256), (384, 128, 384)):
        for dtype in (torch.float, torch.bfloat16):
            a = torch.randn((s, m, k), dtype=torch.bfloat16, device=DEV)
            b = torch.randn((s, n, k), dtype=torch.bfloat16, device=DEV)
            if dtype == torch.float:
                d = torch.zeros((m, n), dtype=dtype, device=DEV)
                c = d
            else:
                d = torch.empty((m, n), dtype=dtype, device=DEV)
                c = None
            deep_gemm.einsum("bmk,bnk->mn", a, b, d, c=c)
            ref = torch.bmm(a.float(), b.float().mT).sum(0)
            nm, ns, sp, g = split_info(s, m, n, k)
            tag = "FP32" if dtype == torch.float else "BF16"
            print(f"{s:>6} {m:>5} {n:>5} {k:>5} {tag:>8} {reldiff(d.float(), ref):>12.3e} "
                  f"{nm:>6} {ns:>7} {sp:>5} {g:>6}")

# ---------------------------------------------------------------------------
# B. 「batch 并入收缩轴」等价性： D == A.view(m, s*k) @ B.view(n, s*k).T
# ---------------------------------------------------------------------------
print("\n[B] 等价于把 (b,k) 展平成收缩轴的普通 GEMM: D == A.view(m,s*k) @ B.view(n,s*k).T")
for (s, m, n, k) in ((512, 256, 256, 256), (4096, 384, 128, 384), (129, 128, 384, 128)):
    a = torch.randn((s, m, k), dtype=torch.bfloat16, device=DEV)
    b = torch.randn((s, n, k), dtype=torch.bfloat16, device=DEV)
    d = torch.zeros((m, n), dtype=torch.float, device=DEV)
    deep_gemm.einsum("bmk,bnk->mn", a, b, d, c=d)
    # 收缩轴 = (b,k)（k 内层）：先把 b 维 permute 到 k 之后，再展平
    af = a.permute(1, 0, 2).reshape(m, s * k).float()
    bf = b.permute(1, 0, 2).reshape(n, s * k).float()
    flat = af @ bf.mT
    print(f"  s={s:>5} m={m} n={n} k={k}: reldiff(einsum, flattened-gemm) = {reldiff(d, flat):.3e}")

# ---------------------------------------------------------------------------
# C. 与 MoE wgrad 的「按 rank 部分和再归约」模型等价
#    P_r = A_r @ B_r^T ; sum_r P_r == einsum(bmk,bnk->mn)
# ---------------------------------------------------------------------------
print("\n[C] 等价于「s 个 rank 各算部分 wgrad P_r=A_r@B_r^T 再求和」（MoE EP 归约模型）")
s, m, n, k = 8, 384, 128, 512
a = torch.randn((s, m, k), dtype=torch.bfloat16, device=DEV)
b = torch.randn((s, n, k), dtype=torch.bfloat16, device=DEV)
d = torch.zeros((m, n), dtype=torch.float, device=DEV)
deep_gemm.einsum("bmk,bnk->mn", a, b, d, c=d)
parts = sum((a[r].float() @ b[r].float().mT) for r in range(s))
print(f"  s={s} ranks: reldiff(einsum, sum_r P_r) = {reldiff(d, parts):.3e}")
mem_naive = s * m * n * 4 / 1e6
print(f"  朴素 [s,m,n] fp32 中间量 = {mem_naive:.1f} MB；融合输出 D = {m*n*4/1e6:.3f} MB "
      f"（省 {mem_naive/(m*n*4/1e6):.0f}x 显存）")

# ---------------------------------------------------------------------------
# D. split-K atomicAdd -> fp32 结果非确定（bit 级抖动）
# ---------------------------------------------------------------------------
print("\n[D] 非确定性：split-K 用 atomicAdd(float2)，fp32 累积顺序不固定")
s, m, n, k = 8192, 256, 256, 256
print(f"  shape s={s} m={m} n={n} k={k} -> (num_mn,num_sk,split,grid)={split_info(s,m,n,k)}")
a = torch.randn((s, m, k), dtype=torch.bfloat16, device=DEV)
b = torch.randn((s, n, k), dtype=torch.bfloat16, device=DEV)
ref = torch.zeros((m, n), dtype=torch.float, device=DEV)
deep_gemm.einsum("bmk,bnk->mn", a, b, ref, c=ref)
runs = []
for _ in range(20):
    d = torch.zeros((m, n), dtype=torch.float, device=DEV)
    deep_gemm.einsum("bmk,bnk->mn", a, b, d, c=d)
    runs.append(d)
bit_mismatch = [int((r.view(torch.int32) != ref.view(torch.int32)).sum().item()) for r in runs]
print(f"  20 次运行 vs 首次: bit 级不同元素数 min/median/max = "
      f"{min(bit_mismatch)}/{sorted(bit_mismatch)[len(bit_mismatch)//2]}/{max(bit_mismatch)} "
      f"(总 {m*n})")
max_rel = max(reldiff(r.float(), ref) for r in runs)
print(f"  非确定性幅度 max rel = {max_rel:.3e}（数值上可忽略，但逐位复现不可保证）")
# 对比：普通 torch 求和（同输入、无 atomic 竞争）逐位恒定
tsum = a.float().sum([0, 2])
tsum2 = a.float().sum([0, 2])
print(f"  对比 torch 求和（无 atomic 竞争）两次运行 bit 不同 = "
      f"{int((tsum.view(torch.int32) != tsum2.view(torch.int32)).sum().item())}（恒为 0）")

# ---------------------------------------------------------------------------
# E. 性能：einsum vs torch 朴素 (bmm+sum) vs 展平 GEMM
# ---------------------------------------------------------------------------
print("\n[E] 性能（FLOPs = 2*s*m*n*k）")
print(f"{'s':>6} {'m':>5} {'n':>5} {'k':>5} | {'einsum us':>10} {'TFLOPS':>8} {'GB/s':>8} | "
      f"{'torch us':>10} {'x':>6} | {'flatgemm us':>11}")
for (s, m, n, k) in ((129, 128, 384, 128), (4096, 256, 256, 256),
                     (8192, 384, 128, 384), (8192, 256, 256, 256),
                     (4096, 512, 512, 128), (4096, 1024, 1024, 64)):
    a = torch.randn((s, m, k), dtype=torch.bfloat16, device=DEV)
    b = torch.randn((s, n, k), dtype=torch.bfloat16, device=DEV)
    d = torch.zeros((m, n), dtype=torch.float, device=DEV)

    def run_einsum():
        deep_gemm.einsum("bmk,bnk->mn", a, b, d, c=d)

    def run_torch():
        out = torch.bmm(a.float(), b.float().mT)
        d.copy_(out.sum(0))

    af = a.permute(1, 0, 2).reshape(m, s * k).float()
    bf = b.permute(1, 0, 2).reshape(n, s * k).float()

    def run_flat():
        torch.mm(af, bf.mT, out=d)

    t_e = bench(run_einsum)
    t_t = bench(run_torch)
    t_f = bench(run_flat)
    flops = 2 * s * m * n * k
    bytes_io = (a.numel() * 2 + b.numel() * 2 + d.numel() * 4)
    print(f"{s:>6} {m:>5} {n:>5} {k:>5} | {t_e:>10.1f} {flops/t_e/1e6:>8.0f} "
          f"{bytes_io/t_e/1e3:>8.0f} | {t_t:>10.1f} {t_t/t_e:>6.2f} | {t_f:>11.1f}")

# ---------------------------------------------------------------------------
# F. 参数契约
# ---------------------------------------------------------------------------
print("\n[F] 参数契约（host 端 DG_HOST_ASSERT）")
def try_call(label, sa, sb, dd, fp32=True):
    a = torch.randn(sa, dtype=torch.bfloat16, device=DEV)
    b = torch.randn(sb, dtype=torch.bfloat16, device=DEV)
    d = torch.randn(dd, dtype=torch.float if fp32 else torch.bfloat16, device=DEV)
    c = d if fp32 else None
    try:
        deep_gemm.einsum("bmk,bnk->mn", a, b, d, c=c)
        print(f"  {label}: 未报错（意外）")
    except Exception as ex:
        print(f"  {label}: 被拒 -> {str(ex).splitlines()[0][:80]}")

# k 非 64 对齐（k=96）
try_call("k=96 (k%64!=0)", (4, 128, 96), (4, 128, 96), (128, 128))
# m 非 64 对齐（a 的 m=96，k=128 对齐）
try_call("m=96 (m%64!=0)", (4, 96, 128), (4, 128, 128), (96, 128))
# n 非 64 对齐（b 的 n=96，k=128 对齐）
try_call("n=96 (n%64!=0)", (4, 128, 128), (4, 96, 128), (128, 96))
# fp32 d 必须与 c 同张量
a = torch.randn((4, 128, 128), dtype=torch.bfloat16, device=DEV)
b = torch.randn((4, 128, 128), dtype=torch.bfloat16, device=DEV)
d1 = torch.zeros((128, 128), dtype=torch.float, device=DEV)
d2 = torch.zeros((128, 128), dtype=torch.float, device=DEV)
try:
    deep_gemm.einsum("bmk,bnk->mn", a, b, d1, c=d2)
    print("  fp32 c!=d: 未报错（意外）")
except Exception as ex:
    print(f"  fp32 c!=d: 被拒 -> {str(ex).splitlines()[0][:80]}")
# bf16 d 不允许 c
db = torch.zeros((128, 128), dtype=torch.bfloat16, device=DEV)
cb = torch.zeros((128, 128), dtype=torch.bfloat16, device=DEV)
try:
    deep_gemm.einsum("bmk,bnk->mn", a, b, db, c=cb)
    print("  bf16 with c: 未报错（意外）")
except Exception as ex:
    print(f"  bf16 with c: 被拒 -> {str(ex).splitlines()[0][:80]}")

# 观察：m,n,k 来自 a/b，d 只校验 dtype——d 形状偏大时多出区域不被写
a = torch.randn((4, 64, 64), dtype=torch.bfloat16, device=DEV)
b = torch.randn((4, 64, 64), dtype=torch.bfloat16, device=DEV)
d = torch.zeros((128, 128), dtype=torch.float, device=DEV)
deep_gemm.einsum("bmk,bnk->mn", a, b, d, c=d)
print(f"  d 形状 (128,128) 但实际 (m,n)=(64,64): d[64:,:] 全零 = "
      f"{bool((d[64:, :] == 0).all().item())}（m,n 取自 a/b，d 形状不校验）")

# ---------------------------------------------------------------------------
# G. PDL（Programmatic Dependent Launch）：set_pdl 后仍正确
# ---------------------------------------------------------------------------
print("\n[G] PDL：deep_gemm.set_pdl(True) 前后语义/性能")
s, m, n, k = 4096, 256, 256, 256
a = torch.randn((s, m, k), dtype=torch.bfloat16, device=DEV)
b = torch.randn((s, n, k), dtype=torch.bfloat16, device=DEV)
ref = torch.bmm(a.float(), b.float().mT).sum(0)
for pdl in (False, True):
    deep_gemm.set_pdl(pdl)
    d = torch.zeros((m, n), dtype=torch.float, device=DEV)
    deep_gemm.einsum("bmk,bnk->mn", a, b, d, c=d)
    err = reldiff(d, ref)
    t = bench(lambda: deep_gemm.einsum("bmk,bnk->mn", a, b, d, c=d))
    print(f"  set_pdl({pdl}): reldiff={err:.3e}  {t:.1f} us")
deep_gemm.set_pdl(False)

print("\n完成。")
