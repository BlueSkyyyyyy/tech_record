#!/usr/bin/env python3
"""P148-1 实验：分页 KV 的 MQA logits（lightning indexer）训练反向。

第 157 篇（P147-1）补了 MQA logits 的**连续版**反向；但生产推理/long-context 训练用的是
**分页 KV cache**（page/block table 间接寻址 + 每 query 行自己的 context 长度），
前向是 DeepGEMM 的 `fp8_fp4_paged_mqa_logits`（SM90/SM100 两条实现）。
本篇补分页版反向，核心是「**梯度沿页表逆映射散播回物理 KV**」，且一个物理槽会被**多行**引用
（next_n 前缀重叠 / 跨序列共享页）→ 必须 scatter-accumulate。

纯 PyTorch（float64 对拍 autograd + 中心差分）+ 一个 CUDA 反向后端（架构无关），H100 真机可跑。

分页前向（`tests/test_attention.py:225-253` 的 `ref_paged_mqa_logits`，已由 AST 抽取复用）：
    q[S,N,H,D]；每行 r=(i,j) 的 context 长度 cl[r]；
    逻辑位置 s 的物理槽 phys = block_table[i, s//BS]*BS + s%BS；
    khat = kv_phys * scale；z_{r,h,s}=<q_{r,h}, khat>；I_{r,s}=sum_h w_{r,h} relu(z)；s>=cl 记 -inf。

反向（本实验要验的）：
    dI_{r,s}=dL/dI_{r,s}（无效位置 0）
    dL/dw_{r,h}       = sum_s dI_{r,s} relu(z_{r,s,h})
    dL/dz_{r,s,h}     = dI_{r,s} w_{r,h} 1[z>0]
    dL/dq_{r,h,d}     = sum_s dL/dz_{r,s,h} khat_{phys(s),d}
    dL/dkv_phys[phys] += sum_{r,h,s : phys(s)=phys} dL/dz_{r,s,h} q_{r,h,d}   （scatter-add，别名安全）
"""

import ast
import math
import os
import torch
import torch.nn.functional as F

torch.manual_seed(0)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
DT = torch.float64
# NGC 容器默认开 TF32（TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1），会让 fp32 matmul 与 fp64 差 ~1e-3；
# 与官方 float32 参考对拍必须显式关掉，否则把 TF32 舍入当成实现误差（见专题「已知坑」）。
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

PASS = 0
FAIL = 0


def check(name, ok, detail=""):
    global PASS, FAIL
    tag = "PASS" if ok else "FAIL"
    if ok:
        PASS += 1
    else:
        FAIL += 1
    print(f"  [{tag}] {name:<62} {detail}")


def relerr(a, b):
    a = a.double(); b = b.double()
    return (a - b).abs().sum() / (a.abs().sum() + b.abs().sum() + 1e-30)


# ---------------------------------------------------------------------------
# 抽取 DeepGEMM 官方测试里纯 torch 的参考函数（真正核对过的锚）
# ---------------------------------------------------------------------------
DG = None
for _c in (os.environ.get("DEEPGEMM_DIR", ""), "~/github/DeepGEMM",
           os.path.expanduser("~/github/DeepGEMM"), "~/github/DeepGEMM"):
    if _c and os.path.isdir(_c):
        DG = _c
        break
TEST_ATTN = os.path.join(DG or "~/github/DeepGEMM", "tests/test_attention.py")


def ceil_div(a, b):
    return (a + b - 1) // b


def load_official_funcs(names):
    """从 tests/test_attention.py 用 AST 抽出指定顶层函数原样 exec（避开模块级 deep_gemm import）。"""
    src = open(TEST_ATTN).read()
    tree = ast.parse(src)
    out = {}
    ns = {"torch": torch, "ceil_div": ceil_div, "math": math,
          "Tuple": __import__("typing").Tuple, "List": __import__("typing").List}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            seg = ast.get_source_segment(src, node)
            exec(compile(seg, TEST_ATTN, "exec"), ns)
            out[node.name] = ns[node.name]
    missing = [n for n in names if n not in out]
    assert not missing, f"missing from test_attention.py: {missing}"
    return out


OFF = load_official_funcs(["ref_paged_mqa_logits", "kv_cache_cast_to_fp8", "to_mqa_weights"])


# ---------------------------------------------------------------------------
# 可微分分页前向（对拍官方 ref）
# ---------------------------------------------------------------------------
def page_map(block_table, BS, max_len):
    """返回 [S, max_len] 的逻辑->物理槽映射。"""
    dev = block_table.device
    pos = torch.arange(max_len, device=dev)
    return block_table[:, pos // BS] * BS + (pos % BS)


def paged_logits_diff(q, kv_flat, kvs, block_table, context_lens, BS, max_len, weights):
    S, N, H, D = q.shape
    phys = page_map(block_table, BS, max_len)
    kv_rows = kv_flat[phys] * kvs[phys][..., None]        # [S, max_len, D]
    z = torch.einsum("snhd,smd->snhm", q, kv_rows)        # [S,N,H,max_len]
    I = torch.einsum("snhm,snh->snm", z.relu(), weights)  # [S,N,max_len]
    pos = torch.arange(max_len, device=q.device)
    mask = pos[None, None, :] < context_lens[:, :, None]  # [S,N,max_len]
    I = I.masked_fill(~mask, -1e30)
    return I, z, kv_rows, phys, mask


# ---------------------------------------------------------------------------
# 构造分页用例
# ---------------------------------------------------------------------------
def build_case(S, N, H, D, BS, avg_kv, seed=0, share=False):
    g = torch.Generator(device="cpu").manual_seed(seed)
    context_lens = torch.randint(int(0.7 * avg_kv), int(1.3 * avg_kv), (S, N), generator=g).to(DEV).int()
    context_lens[:, -1] = context_lens.max(dim=1).values  # last slot sees full prefix
    max_cl = context_lens.max().item()
    max_len = ceil_div(max_cl, BS) * BS
    bps = max_len // BS
    if share:
        # sequence 1 aliases sequence 0's blocks (prefix sharing / beam), others private
        num_priv = S - 2
        total = bps + bps + num_priv * bps
        pool = torch.arange(total)
        block_table = torch.zeros((S, bps), dtype=torch.int)
        block_table[0] = pool[:bps]
        block_table[1] = pool[:bps]              # alias of seq 0
        for i in range(2, S):
            block_table[i] = pool[2 * bps + (i - 2) * bps: 2 * bps + (i - 1) * bps]
        NB = total
    else:
        block_table = torch.arange(S * bps).view(S, bps)
        NB = S * bps
    block_table = block_table.to(DEV).int()
    q = torch.randn(S, N, H, D, generator=g).to(DEV).to(DT) * 0.3
    kv_flat = torch.randn(NB * BS, D, generator=g).to(DEV).to(DT) * 0.3
    kvs = torch.rand(NB * BS, generator=g).to(DEV).to(DT) * 0.7 + 0.5
    w = torch.randn(S, N, H, generator=g).to(DEV).to(DT) * 0.3
    dI = torch.randn(S, N, max_len, generator=g).to(DEV).to(DT)
    pos = torch.arange(max_len, device=DEV)
    mask = pos[None, None, :] < context_lens[:, :, None]
    dI = dI * mask                                        # invalid positions carry zero grad
    return dict(S=S, N=N, H=H, D=D, BS=BS, max_len=max_len, bps=bps, NB=NB,
                q=q, kv_flat=kv_flat, kvs=kvs, w=w, dI=dI, block_table=block_table,
                context_lens=context_lens, mask=mask)


def analytic_backward(c, need=("dw", "dq", "dkv")):
    q, kv_flat, kvs, w = c["q"], c["kv_flat"], c["kvs"], c["w"]
    bt, cl, BS, max_len = c["block_table"], c["context_lens"], c["BS"], c["max_len"]
    S, N, H, D = c["S"], c["N"], c["H"], c["D"]
    I, z, kv_rows, phys, mask = c["I"], c["z"], c["kv_rows"], c["phys"], c["mask"]
    dI = c["dI"]
    r = z.relu()
    dw = torch.einsum("snm,snhm->snh", dI, r)                     # [S,N,H]
    dz = dI[:, :, None, :] * w[:, :, :, None] * (z > 0).to(z.dtype)  # [S,N,H,max_len]
    dq = torch.einsum("snhm,smd->snhd", dz, kv_rows)              # [S,N,H,D]
    dkv_rows = torch.einsum("snhm,snhd->smd", dz, q)              # [S,max_len,D] sum over n,h
    # kv_rows = kv_flat[phys] * kvs[phys]  =>  dL/dkv_flat[slot] 要乘回 scale
    dkv = torch.zeros_like(kv_flat)
    dkv.index_add_(0, phys.reshape(-1), (dkv_rows * kvs[phys][..., None]).reshape(-1, D))
    return {"dw": dw, "dq": dq, "dkv": dkv}


# ---------------------------------------------------------------------------
# Part A：可微分分页前向 == 官方 ref_paged_mqa_logits
# ---------------------------------------------------------------------------
def part_a():
    print("Part A  分页前向的官方锚（可微分实现 vs DeepGEMM ref_paged_mqa_logits）")
    S, N, H, D, BS, avg = 4, 2, 32, 64, 64, 512
    c = build_case(S, N, H, D, BS, avg, seed=1)
    I = paged_logits_diff(c["q"], c["kv_flat"], c["kvs"], c["block_table"],
                          c["context_lens"], BS, c["max_len"], c["w"])[0]

    # 官方 ref：q[S,N,H,D]、kv_cache[num_blocks,BS,1,D]、weights[S*N,H]、context_lens[S,N]
    kv_cache = c["kv_flat"].view(c["NB"], BS, 1, D)
    # fold per-slot scale into the physical KV (ref expects already-scaled k)
    kv_cache = kv_cache * c["kvs"].view(c["NB"], BS, 1, 1)
    weights = c["w"].reshape(S * N, H)
    ref = OFF["ref_paged_mqa_logits"](c["q"].float(), kv_cache.float(), weights.float(),
                                      c["context_lens"], c["block_table"], c["max_len"])
    ref = ref.view(S, N, c["max_len"])
    ours = I.float()
    valid = c["mask"]
    d = relerr(ours[valid], ref[valid])
    check("可微分前向(fp64) == ref_paged_mqa_logits(fp32, 关 TF32)（有效区）", d < 1e-5, f"relerr={d:.3e}")
    check("无效位置两者都是 -inf 语义（我们 -1e30）",
          bool((ref[~valid] == float("-inf")).all() or (ref[~valid] < -1e29).all()),
          f"ref 无效最小={float(ref[~valid].min()):.3e}")
    c["I"] = I
    return c


# ---------------------------------------------------------------------------
# Part B：解析反向 vs autograd + 中心差分
# ---------------------------------------------------------------------------
def part_b():
    print("Part B  分页反向解析式 vs autograd fp64（+ 中心差分）")
    S, N, H, D, BS, avg = 4, 2, 32, 64, 64, 768
    c = build_case(S, N, H, D, BS, avg, seed=2)

    q = c["q"].clone().requires_grad_(True)
    kv_flat = c["kv_flat"].clone().requires_grad_(True)
    kvs = c["kvs"].clone().requires_grad_(True)
    w = c["w"].clone().requires_grad_(True)
    I, z, kv_rows, phys, mask = paged_logits_diff(q, kv_flat, kvs, c["block_table"],
                                                  c["context_lens"], BS, c["max_len"], w)
    c2 = dict(c, q=q, kv_flat=kv_flat, kvs=kvs, w=w, I=I, z=z, kv_rows=kv_rows, phys=phys, mask=mask)
    g = analytic_backward(c2)
    # autograd on the same scalar L = sum(dI * I) ( dI is the upstream gradient )
    L = (c["dI"] * I).sum()
    L.backward()
    check("dq  解析 vs autograd", relerr(g["dq"], q.grad) < 1e-12, f"relerr={relerr(g['dq'], q.grad):.3e}")
    check("dw  解析 vs autograd", relerr(g["dw"], w.grad) < 1e-12, f"relerr={relerr(g['dw'], w.grad):.3e}")
    check("dkv_phys 解析 vs autograd（含页别名累加）", relerr(g["dkv"], kv_flat.grad) < 1e-12,
          f"relerr={relerr(g['dkv'], kv_flat.grad):.3e}")

    # 中心差分：对若干物理槽条目扰动
    worst = 0.0
    idx = [(0, 0), (5, 3), (c["NB"] * BS - 1, D - 1)]
    eps = 1e-6
    base = c["kv_flat"]
    for (slot, dd) in idx:
        kp = base.clone(); kp[slot, dd] += eps
        km = base.clone(); km[slot, dd] -= eps
        Lp = (c["dI"] * paged_logits_diff(c["q"], kp, c["kvs"], c["block_table"], c["context_lens"],
                                          BS, c["max_len"], c["w"])[0]).sum()
        Lm = (c["dI"] * paged_logits_diff(c["q"], km, c["kvs"], c["block_table"], c["context_lens"],
                                          BS, c["max_len"], c["w"])[0]).sum()
        fd = (Lp - Lm) / (2 * eps)
        worst = max(worst, relerr(fd.reshape(1), g["dkv"][slot, dd].reshape(1)))
    check("dkv_phys 中心差分（3 个物理槽条目）", worst < 5e-7, f"最差 relerr={worst:.3e}")
    return c


# ---------------------------------------------------------------------------
# Part C：别名累加 —— 分页反向的正确性硬要求
# ---------------------------------------------------------------------------
def part_c():
    print("Part C  物理页别名：scatter-accumulate vs 「写」（last-writer-wins）")
    S, N, H, D, BS, avg = 4, 3, 32, 64, 64, 512
    c = build_case(S, N, H, D, BS, avg, seed=3, share=True)
    I, z, kv_rows, phys, mask = paged_logits_diff(c["q"], c["kv_flat"], c["kvs"], c["block_table"],
                                                  c["context_lens"], BS, c["max_len"], c["w"])
    c["I"], c["z"], c["kv_rows"], c["phys"], c["mask"] = I, z, kv_rows, phys, mask
    g = analytic_backward(c)

    # 有多少物理槽被多个 (S,max_len) 逻辑位引用？
    flat = phys.reshape(-1)
    uniq, counts = torch.unique(flat, return_counts=True)
    max_share = int(counts.max().item())
    n_shared = int((counts > 1).sum().item())
    print(f"    页映射：{phys.numel()} 个逻辑引用 -> {uniq.numel()} 个物理槽；"
          f"被多引用的槽 {n_shared} 个、最大共享度 {max_share}（next_n 前缀重叠 + 跨序列共享页）")

    # naive「写」：每行 (i,n) 对自己引用的物理槽直接覆盖（last-writer-wins）
    dkv_naive = torch.zeros_like(c["kv_flat"])
    qq, kv_flat, kvs, w = c["q"], c["kv_flat"], c["kvs"], c["w"]
    r = z.relu()
    dz = c["dI"][:, :, None, :] * w[:, :, :, None] * (z > 0).to(z.dtype)   # [S,N,H,L]
    for i in range(c["S"]):
        for n in range(c["N"]):
            for s in range(c["max_len"]):
                if not bool(mask[i, n, s].item()):
                    continue
                p = int(phys[i, s].item())
                local = torch.einsum("h,hd->d", dz[i, n, :, s], qq[i, n]) * kvs[p]   # [D]
                dkv_naive[p] = local                                               # overwrite (bug)
    d_bad = relerr(dkv_naive, g["dkv"])
    # 只看被共享的槽，naive 完全不等于正确累加
    shared_ids = uniq[counts > 1]
    check("naive「写」在被引用槽上丢贡献（确有差异）",
          float((dkv_naive[shared_ids] - g["dkv"][shared_ids]).abs().max()) > 0,
          f"共享槽最大绝对差={float((dkv_naive[shared_ids]-g['dkv'][shared_ids]).abs().max()):.3e}")
    check("naive「写」相对正确累加的全局 relerr 显著 >0", d_bad > 1e-2, f"relerr={d_bad:.3e}")

    # 端到端 autograd 再验一次（带 share 的图）
    q = c["q"].clone().requires_grad_(True)
    kv_flat = c["kv_flat"].clone().requires_grad_(True)
    I2 = paged_logits_diff(q, kv_flat, c["kvs"], c["block_table"], c["context_lens"], BS,
                           c["max_len"], c["w"])[0]
    (c["dI"] * I2).sum().backward()
    check("share 用例：dkv 解析 == autograd", relerr(g["dkv"], kv_flat.grad) < 1e-12,
          f"relerr={relerr(g['dkv'], kv_flat.grad):.3e}")
    return c


# ---------------------------------------------------------------------------
# Part D：分页 logits 上的 KL 蒸馏梯度（有效集内 softmax）
# ---------------------------------------------------------------------------
def part_d():
    print("Part D  分页 logits 的 KL 蒸馏梯度 dL/dI = softmax(I) - p（有效集内）")
    S, N, H, D, BS, avg = 3, 2, 32, 64, 64, 512
    c = build_case(S, N, H, D, BS, avg, seed=4)
    I = paged_logits_diff(c["q"], c["kv_flat"], c["kvs"], c["block_table"], c["context_lens"],
                          BS, c["max_len"], c["w"])[0].clone().requires_grad_(True)
    mask = c["mask"]
    # target p: normalized over valid positions (detached)
    rnd = torch.rand_like(I)
    p = (rnd * mask).masked_fill(~mask, 0.0)
    p = p / p.sum(-1, keepdim=True).clamp_min(1e-30)
    # KL on valid positions: use masked softmax
    neg = torch.where(mask, I, torch.full_like(I, -1e30))
    logq = F.log_softmax(neg, dim=-1)
    loss = -(p * logq).sum(-1).sum()
    loss.backward()
    q_prob = torch.softmax(neg, dim=-1)
    analytic = (q_prob - p) * mask
    check("dL/dI 解析(softmax-p) vs autograd", relerr(analytic, I.grad) < 1e-12,
          f"relerr={relerr(analytic, I.grad):.3e}")
    a_sum = analytic.detach().sum(-1).abs().max()
    check("有效集内行和 = 0", float(a_sum) < 1e-12,
          f"max={float(a_sum):.3e}")
    av = analytic.detach()[~mask].abs().max()
    check("无效位置（s>=cl）梯度恒 0", float(av) == 0.0,
          f"max={float(av):.3e}")


# ---------------------------------------------------------------------------
# Part E：真实 DeepGEMM SM90 分页前向锚
# ---------------------------------------------------------------------------
def part_e():
    print("Part E  真实 DeepGEMM SM90 fp8_fp4_paged_mqa_logits 前向锚")
    try:
        import deep_gemm
        from deep_gemm.testing import get_arch_major, calc_diff
    except Exception as e:  # pragma: no cover
        print(f"    [SKIP] deep_gemm 不可用：{e}")
        return
    if get_arch_major() != 9:
        print(f"    [SKIP] 本机 arch_major={get_arch_major()}（分页 fp8 仅 SM90/SM100，本例走 SM90）")
        return
    S, N, H, D, BS, avg = 8, 2, 32, 64, 64, 1024
    c = build_case(S, N, H, D, BS, avg, seed=5)
    max_len = c["max_len"]
    # q fp8 (SM90: pure E4M3, no SF); kv fused layout by official caster
    q_fp = c["q"].float().to(torch.float8_e4m3fn).to(DEV)
    q_sim = q_fp.to(torch.bfloat16)
    kv_cache = c["kv_flat"].view(c["NB"], BS, 1, D).float()
    fused, kv_sim = OFF["kv_cache_cast_to_fp8"](kv_cache)       # [NB,BS,1,D+4] uint8 + dequant
    fused = fused.to(DEV).contiguous()
    kv_sim = kv_sim.to(DEV)
    weights = c["w"].reshape(S * N, H).float()
    kernel_weights = OFF["to_mqa_weights"](weights).to(DEV).contiguous()
    context_lens = c["context_lens"].contiguous()
    block_table = c["block_table"].contiguous()
    try:
        meta = deep_gemm.get_paged_mqa_logits_metadata(context_lens, BS, deep_gemm.get_num_sms(), None)
        logits = deep_gemm.fp8_fp4_paged_mqa_logits(
            (q_fp, None), fused, kernel_weights, context_lens, block_table, meta,
            max_len, None)
    except Exception as e:
        print(f"    [SKIP] 真实分页 kernel 调用失败：{type(e).__name__}: {e}")
        return
    logits = logits.float().view(S, N, max_len)
    # 模拟参考（用反量化后的 q/kv 跑官方 ref）
    ref_sim = OFF["ref_paged_mqa_logits"](q_sim.float(), kv_sim.float(), kernel_weights.float(),
                                          context_lens, block_table, max_len).view(S, N, max_len)
    m = c["mask"]
    d_kernel_sim = calc_diff(logits[m], ref_sim[m])
    check("真实 kernel vs 模拟参考（fp8 反量化）在容差内", d_kernel_sim < 1e-3,
          f"calc_diff={d_kernel_sim:.3e} (tol 1e-3)")
    check("kernel 不对无效区写 -inf（-inf 只由 ref 提供，kernel 侧比较须 masked）",
          not bool((logits[~m] < -1e29).all()),
          f"无效区最小={float(logits[~m].min()):.3e} 最大={float(logits[~m].max()):.3e}")
    print(f"    真实分页 kernel 有效区样本：{logits[m][:3].tolist()} vs ref {ref_sim[m][:3].tolist()}")


if __name__ == "__main__":
    print("=" * 78)
    print("P148-1 分页 KV 的 MQA indexer 反向（H100 真机）")
    print("=" * 78)
    part_a()
    part_b()
    part_c()
    part_d()
    part_e()
    print("-" * 78)
    print(f"PASS={PASS}  FAIL={FAIL}")
    raise SystemExit(0 if FAIL == 0 else 1)
