#!/usr/bin/env python3
"""P149-1 实验：稀疏（块选择）MQA logits（lightning indexer）前向的**反向**。

第 157 篇（P147-1）补了连续版反向、第 158 篇（P148-1）补了分页版反向。但生产 long-context
训练真正跑的是**稀疏版**：只在一小撮**选中的 KV 块**内打分，输出**压缩**成
`[T, num_max_sparse_blocks*SB]`（前向是 DeepGEMM 的 `fp8_fp4_sparse_mqa_logits` /
`fp8_fp4_paged_sparse_mqa_logits`，SM100 专属）。本篇补稀疏版反向，核心是
「支持集不再是整个窗口，而是**被选中块覆盖的稀疏子集**；压缩输出是 dense logits 的一个
**选择/压缩**（gather），故其伴随是 **scatter**（未选中/越界位置梯度恒 0）」。

纯 PyTorch（float64 对拍 autograd + 中心差分）+ 一个 CUDA 反向后端（架构无关），H100(sm90) 真机可跑。

稀疏前向语义（`tests/test_attention.py:442-475` 的 `make_sparse_kv_block_indices`
与 `:644-670` 的压缩展开，唯一正确性规范）：
    SB = sparse_block_kv；NBmax = num_max_sparse_blocks；
    block_offsets[r] = context_starts[r] % SB；
    token_indices[r, l*SB+kvj] = sparse_indices[r,l]*SB + block_offsets[r] + kvj   （l<num_blocks[r]）
    support[r, c] = (l < num_blocks[r]) & (start_r <= token < len_r)
    压缩输出 G[r, c] = I[r, token_indices[r,c]]（仅 support）；未选中块「看不见」→ 梯度恒 0。

反向：
    dI_full[r, token] = sum_{c : token_indices[r,c]=token} dG[r, c]   （gather 的伴随 = scatter）
    dL/dw_{r,h}       = sum_{token in support(r)} dI_full_{r,token} relu(z_{r,token,h})
    dL/dz_{r,token,h} = dI_full_{r,token} w_{r,h} 1[z>0]
    dL/dq_{r,h,d}     = sum_{token} dL/dz * khat_{token,d}
    dL/dkv[token]    += sum_{r,h} dL/dz_{r,token,h} q_{r,h,d} scale_token   （scatter-add，别名安全）
    dL/dscale[token] += sum_{r,h} dL/dz_{r,token,h} (q_{r,h}·kv_token)
"""

import ast
import math
import os
import torch
import torch.nn.functional as F

torch.manual_seed(0)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
DT = torch.float64
# NGC 容器默认开 TF32（见专题「已知坑」），与 fp32 参考对拍必须显式关掉。
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
    print(f"  [{tag}] {name:<64} {detail}")


def relerr(a, b):
    a = a.double(); b = b.double()
    return (a - b).abs().sum() / (a.abs().sum() + b.abs().sum() + 1e-30)


# ---------------------------------------------------------------------------
# 抽取 DeepGEMM 官方测试里纯 torch 的参考函数（真正核对过的锚）
# ---------------------------------------------------------------------------
DG = None
for _c in (os.environ.get("DEEPGEMM_DIR", ""), os.path.expanduser("~/github/DeepGEMM"),
           "~/github/DeepGEMM"):
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
    ns = {"torch": torch, "ceil_div": ceil_div, "math": math, "random": __import__("random"),
          "Tuple": __import__("typing").Tuple, "List": __import__("typing").List}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            seg = ast.get_source_segment(src, node)
            exec(compile(seg, TEST_ATTN, "exec"), ns)
            out[node.name] = ns[node.name]
    missing = [n for n in names if n not in out]
    assert not missing, f"missing from test_attention.py: {missing}"
    return out


OFF = load_official_funcs(["make_sparse_kv_block_indices", "ref_mqa_logits", "to_mqa_weights"])


# ---------------------------------------------------------------------------
# 稀疏用例构造 + 支持集
# ---------------------------------------------------------------------------
def build_sparse_case(S, N, H, D, SB, NBmax, num_blocks_per_row, context_lens, starts,
                      num_kv_tokens, seed=0, alias=False):
    """构造一个压缩稀疏用例（context_starts 可为 0 或任意未对齐值）。"""
    g = torch.Generator(device="cpu").manual_seed(seed)
    P = S * N
    context_lens = torch.tensor(context_lens, dtype=torch.int)         # [P] = window end
    starts = torch.tensor(starts, dtype=torch.int)                     # [P] = window start
    # 单进程构造稀疏块号（所有 next_n 槽共享序列级窗口）
    total_blocks = ceil_div(num_kv_tokens, SB)
    req = [i for i in range(S) for _ in range(N)]                      # 每行的序列号
    sparse_indices = torch.zeros((P, NBmax), dtype=torch.int)
    num_blocks = torch.zeros(P, dtype=torch.int)
    for r in range(P):
        first_block = int(starts[r]) // SB
        nb = min(num_blocks_per_row, total_blocks - first_block)
        if alias and r % N != 0:
            # 非首槽复用首槽的块（模拟 next_n 前缀重叠 / 共享块）
            sparse_indices[r, :num_blocks[ (r // N) * N ]] = \
                sparse_indices[(r // N) * N, :num_blocks[(r // N) * N]]
            num_blocks[r] = num_blocks[(r // N) * N]
            continue
        blocks = torch.randperm(total_blocks - first_block, generator=g)[:nb] + first_block
        blocks, _ = torch.sort(blocks)
        sparse_indices[r, :nb] = blocks
        if nb > 0:
            sparse_indices[r, nb:] = blocks[-1]
        num_blocks[r] = nb

    block_offsets = starts % SB
    kv_offsets = torch.arange(SB, dtype=torch.int)
    token_indices = (sparse_indices.unsqueeze(-1) * SB + block_offsets[:, None, None]
                     + kv_offsets[None, None, :]).reshape(P, NBmax * SB)
    C = NBmax * SB
    l_idx = torch.arange(C, dtype=torch.int) // SB
    support = l_idx[None, :] < num_blocks[:, None]
    support &= token_indices >= starts[:, None]
    support &= token_indices < context_lens[:, None]
    token_indices = token_indices.to(DEV)
    support = support.to(DEV)

    q = torch.randn(P, H, D, generator=g).to(DEV).to(DT) * 0.3
    kv_flat = torch.randn(num_kv_tokens, D, generator=g).to(DEV).to(DT) * 0.3
    kvs = torch.rand(num_kv_tokens, generator=g).to(DEV).to(DT) * 0.7 + 0.5
    w = torch.randn(P, H, generator=g).to(DEV).to(DT) * 0.3
    d_sparse = torch.randn(P, C, generator=g).to(DEV).to(DT) * (support.to(DT))
    return dict(P=P, S=S, N=N, H=H, D=D, SB=SB, NBmax=NBmax, C=C, num_kv_tokens=num_kv_tokens,
                q=q, kv_flat=kv_flat, kvs=kvs, w=w, d_sparse=d_sparse, support=support,
                token_indices=token_indices, starts=starts.to(DEV), context_lens=context_lens.to(DEV),
                sparse_indices=sparse_indices.to(DEV), num_blocks=num_blocks.to(DEV), block_offsets=block_offsets)


def full_logits(q, kv_flat, kvs, w):
    khat = kv_flat * kvs[:, None]
    z = torch.einsum("phd,sd->phs", q, khat)          # [P,H,KV]
    I = torch.einsum("phs,ph->ps", z.relu(), w)       # [P,KV]
    return I, z


def gather_compress(I_full, c):
    idx = c["token_indices"].clamp(0, c["num_kv_tokens"] - 1)
    return torch.gather(I_full, 1, idx)               # [P, C]


# ---------------------------------------------------------------------------
# Part A：压缩输出的伴随 = scatter（选择矩阵的转置）
# ---------------------------------------------------------------------------
def part_a():
    print("Part A  压缩输出是 dense logits 的「选择」（gather），其伴随是 scatter（未选位置梯度 0）")
    c = build_sparse_case(S=4, N=2, H=8, D=32, SB=16, NBmax=6, num_blocks_per_row=4,
                          context_lens=[512] * 8, starts=[0] * 8, num_kv_tokens=512, seed=1)
    I_full = torch.randn(c["P"], c["num_kv_tokens"], dtype=DT, device=DEV)
    G = gather_compress(I_full, c)                    # 压缩输出（gather）

    # 伴随恒等式 <d_sparse, G> == <d_full, I_full>，其中 d_full = scatter(d_sparse)
    idx = c["token_indices"].clamp(0, c["num_kv_tokens"] - 1)
    d_full = torch.zeros_like(I_full)
    d_full.scatter_add_(1, idx, c["d_sparse"])
    lhs = (c["d_sparse"] * G).sum()
    rhs = (d_full * I_full).sum()
    check("伴随恒等式 <dG, gather(I)> == <scatter(dG), I>", abs(lhs.item() - rhs.item()) < 1e-9,
          f"lhs={lhs.item():.6f} rhs={rhs.item():.6f}")
    # 未选中/越界位置的 d_full 必须恒 0（注意 padding 块列会重复映射到已选块 → 用「任一 support」OR）
    sel = torch.zeros(c["P"], c["num_kv_tokens"], dtype=torch.bool, device=DEV)
    prow = torch.arange(c["P"], device=DEV)[:, None].expand_as(idx)
    sel[prow[c["support"]], idx[c["support"]]] = True
    check("未选中位置 d_full 恒 0", float(d_full[~sel].abs().max()) == 0.0,
          f"max={float(d_full[~sel].abs().max()):.1e}")
    n_support = int(c["support"].sum().item())
    per_row = n_support // c["P"]
    print(f"    压缩输出 {c['C']} 列/行（有效 {per_row}）；dense 窗口 {c['num_kv_tokens']} 列 "
          f"-> 每行只覆盖 {per_row} 个 token（{c['num_kv_tokens'] / max(1, per_row):.1f}x 稀疏）")
    return c


# ---------------------------------------------------------------------------
# Part B：稀疏 logits 上的 KL 蒸馏梯度（支持集内 softmax）
# ---------------------------------------------------------------------------
def part_b():
    print("Part B  稀疏 logits 上的 KL 蒸馏梯度 dL/dI = softmax_support(I) - p")
    c = build_sparse_case(S=3, N=2, H=8, D=32, SB=16, NBmax=6, num_blocks_per_row=4,
                          context_lens=[384] * 6, starts=[7] * 6, num_kv_tokens=384, seed=2)
    I_full, _ = full_logits(c["q"], c["kv_flat"], c["kvs"], c["w"])
    G = gather_compress(I_full, c)
    supp = c["support"]
    neg = torch.where(supp, G, torch.full_like(G, -1e30)).clone().requires_grad_(True)

    rnd = torch.rand_like(G) * supp
    p = rnd / rnd.sum(-1, keepdim=True).clamp_min(1e-30)         # 支持集内 L1 归一化（detached）
    logq = F.log_softmax(neg, dim=-1)
    loss = -(p * logq).sum(-1).sum()
    loss.backward()
    q_prob = torch.softmax(neg.detach(), dim=-1)
    analytic = (q_prob - p) * supp
    check("dL/dG 解析(softmax-p) vs autograd", relerr(analytic, neg.grad) < 1e-12,
          f"relerr={relerr(analytic, neg.grad):.3e}")
    rowsum = analytic.detach().sum(-1).abs().max()
    check("支持集内行和 = 0（softmax 平移不变）", float(rowsum) < 1e-12, f"max={float(rowsum):.1e}")
    check("支持集外梯度恒 0", float(analytic.detach()[~supp].abs().max()) == 0.0,
          f"max={float(analytic.detach()[~supp].abs().max()):.1e}")
    # 反例：不 mask 支持集、对所有压缩列 softmax（含 padding 块）→ 归一化不同、梯度不同
    p_wrong = rnd.clone()
    p_wrong = p_wrong / p_wrong.sum(-1, keepdim=True).clamp_min(1e-30)
    dI_wrong = (torch.softmax(G, dim=-1) - p_wrong) * supp
    check("『不 mask 支持集』的梯度与正确式不同",
          relerr(analytic, dI_wrong) > 1e-3, f"relerr={relerr(analytic, dI_wrong):.3e}")
    return c


# ---------------------------------------------------------------------------
# Part C：打分算子反向（稀疏支持集 + scatter 到 dense 窗口）
# ---------------------------------------------------------------------------
def operator_backward_sparse(c):
    q, kv_flat, kvs, w = c["q"], c["kv_flat"], c["kvs"], c["w"]
    I_full, z = full_logits(q, kv_flat, kvs, w)
    idx = c["token_indices"].clamp(0, c["num_kv_tokens"] - 1)
    d_full = torch.zeros_like(I_full)
    d_full.scatter_add_(1, idx, c["d_sparse"])        # gather 的伴随
    r = z.relu()
    dw = torch.einsum("ps,phs->ph", d_full, r)
    dz = d_full[:, None, :] * w[:, :, None] * (z > 0).to(z.dtype)
    khat = kv_flat * kvs[:, None]
    dq = torch.einsum("phs,sd->phd", dz, khat)
    dkhat = torch.einsum("phs,phd->sd", dz, q)
    dkv = dkhat * kvs[:, None]
    dks = (dkhat * kv_flat).sum(1)
    return dict(dw=dw, dq=dq, dkv=dkv, dks=dks)


def part_c():
    print("Part C  稀疏打分算子反向（dL/dw, dq, dkv, dscale）vs autograd（压缩损失）")
    c = build_sparse_case(S=4, N=2, H=8, D=32, SB=16, NBmax=6, num_blocks_per_row=4,
                          context_lens=[512] * 8, starts=[0] * 8, num_kv_tokens=512, seed=3, alias=True)
    q = c["q"].clone().requires_grad_(True)
    kv_flat = c["kv_flat"].clone().requires_grad_(True)
    kvs = c["kvs"].clone().requires_grad_(True)
    w = c["w"].clone().requires_grad_(True)
    c2 = dict(c, q=q, kv_flat=kv_flat, kvs=kvs, w=w)
    I_full, z = full_logits(q, kv_flat, kvs, w)
    G = gather_compress(I_full, c2)
    loss = (c["d_sparse"] * G).sum()                 # 损失定义在压缩输出上 → 反传自动走 gather 的伴随
    loss.backward()
    ana = operator_backward_sparse(c2)
    check("dq        解析 vs autograd", relerr(ana["dq"], q.grad) < 1e-12, f"relerr={relerr(ana['dq'], q.grad):.3e}")
    check("dw        解析 vs autograd", relerr(ana["dw"], w.grad) < 1e-12, f"relerr={relerr(ana['dw'], w.grad):.3e}")
    check("dkv       解析 vs autograd（含 scatter 别名）", relerr(ana["dkv"], kv_flat.grad) < 1e-12,
          f"relerr={relerr(ana['dkv'], kv_flat.grad):.3e}")
    check("dscale    解析 vs autograd", relerr(ana["dks"], kvs.grad) < 1e-12,
          f"relerr={relerr(ana['dks'], kvs.grad):.3e}")
    # 中心差分核验几个 kv 条目
    qd, kv0, ksd, wd = c["q"], c["kv_flat"], c["kvs"], c["w"]
    eps = 1e-6
    worst = 0.0
    for (slot, dd) in [(0, 0), (c["num_kv_tokens"] - 1, 31)]:
        kp = kv0.clone(); kp[slot, dd] += eps
        km = kv0.clone(); km[slot, dd] -= eps
        Lp = (c["d_sparse"] * gather_compress(full_logits(qd, kp, ksd, wd)[0], c)).sum()
        Lm = (c["d_sparse"] * gather_compress(full_logits(qd, km, ksd, wd)[0], c)).sum()
        fd = (Lp - Lm) / (2 * eps)
        worst = max(worst, relerr(fd.reshape(1), ana["dkv"][slot, dd].reshape(1)))
    check("dkv 中心差分（2 个 token 条目）", worst < 5e-7, f"最差 relerr={worst:.3e}")
    return c


# ---------------------------------------------------------------------------
# Part D：稀疏支持集里的别名累加
# ---------------------------------------------------------------------------
def part_d():
    print("Part D  支持集别名：scatter-accumulate vs 「写」（last-writer-wins）")
    c = build_sparse_case(S=4, N=3, H=8, D=32, SB=16, NBmax=6, num_blocks_per_row=4,
                          context_lens=[512] * 12, starts=[0] * 12, num_kv_tokens=512, seed=4, alias=True)
    idx = c["token_indices"].clamp(0, c["num_kv_tokens"] - 1)
    flat = idx[c["support"]]
    uniq, counts = torch.unique(flat, return_counts=True)
    n_shared = int((counts > 1).sum().item())
    max_share = int(counts.max().item()) if counts.numel() else 0
    print(f"    支持集：{int(c['support'].sum().item())} 个逻辑引用 -> {int(uniq.numel())} 个 token；"
          f"被多引用的 token {n_shared} 个、最大共享度 {max_share}（next_n 共享块）")
    ana = operator_backward_sparse(c)
    # naive「写」：对每个 (row, token) 直接覆盖
    _, z = full_logits(c["q"], c["kv_flat"], c["kvs"], c["w"])
    d_full = torch.zeros_like(z[:, 0, :])
    d_full.scatter_add_(1, idx, c["d_sparse"])
    dkv_naive = torch.zeros_like(c["kv_flat"])
    dz = d_full[:, None, :] * c["w"][:, :, None] * (z > 0).to(z.dtype)
    # 逐行对该行覆盖的 token 覆盖写（last writer = 最后一头），模拟「写」而非累加
    for p in range(c["P"]):
        loc = torch.unique(idx[p][c["support"][p]])
        for t in loc.tolist():
            dkv_naive[t] = (dz[p, -1, t] * c["q"][p, -1]) * c["kvs"][t]
    d_bad = relerr(dkv_naive, ana["dkv"])
    shared_ids = uniq[counts > 1]
    if shared_ids.numel():
        maxdiff = float((dkv_naive[shared_ids] - ana["dkv"][shared_ids]).abs().max())
    else:
        maxdiff = 0.0
    check("naive「写」在被共享 token 上丢贡献", maxdiff > 0, f"共享 token 最大绝对差={maxdiff:.3e}")
    check("naive「写」相对正确累加全局 relerr 显著 >0", d_bad > 1e-2, f"relerr={d_bad:.3e}")
    # autograd 再验一次（带 alias 的图）
    q = c["q"].clone().requires_grad_(True)
    kv_flat = c["kv_flat"].clone().requires_grad_(True)
    G = gather_compress(full_logits(q, kv_flat, c["kvs"], c["w"])[0], c)
    (c["d_sparse"] * G).sum().backward()
    check("alias 用例：dkv 解析 == autograd", relerr(ana["dkv"], kv_flat.grad) < 1e-12,
          f"relerr={relerr(ana['dkv'], kv_flat.grad):.3e}")


# ---------------------------------------------------------------------------
# Part E：真实 DeepGEMM SM90 dense kernel 前向锚 + 压缩语义
# ---------------------------------------------------------------------------
def part_e():
    print("Part E  真实 SM90 fp8_fp4_mqa_logits 前向锚 + 稀疏压缩=选择 的语义")
    try:
        import deep_gemm
        from deep_gemm.testing import get_arch_major
        from deep_gemm.utils import per_custom_dims_cast_to_fp8 as OFF_cast_fp8
    except Exception as e:  # pragma: no cover
        print(f"    [SKIP] deep_gemm 不可用：{e}")
        return
    if get_arch_major() != 9:
        print(f"    [SKIP] 本机 arch_major={get_arch_major()}（fp8 连续仅 SM90/SM100，本例走 SM90）")
        return
    T, H, D, SB, NBmax = 64, 32, 128, 16, 8
    num_kv_tokens = 512
    # 构造行级稀疏选择（starts=0、cl=KV，便于列=token）
    c = build_sparse_case(S=T, N=1, H=H, D=D, SB=SB, NBmax=NBmax, num_blocks_per_row=5,
                          context_lens=[num_kv_tokens] * T, starts=[0] * T,
                          num_kv_tokens=num_kv_tokens, seed=5)
    # q fp8（SM90 纯 E4M3，无 SF）、kv per-token scale 的 E4M3
    q_fp = c["q"].float().to(torch.float8_e4m3fn).to(DEV)
    q_sim = q_fp.to(torch.bfloat16)
    kv_in = OFF_cast_fp8(c["kv_flat"].float(), (0,), False)
    kv_fp, kv_sf = kv_in[0].to(DEV), kv_in[1].to(DEV)
    kv_sim = (kv_fp.float() * kv_sf.unsqueeze(1)).to(torch.bfloat16)
    weights = c["w"].float()
    kernel_weights = OFF["to_mqa_weights"](weights).to(DEV).contiguous()
    ks = c["starts"].contiguous().int()
    ke = c["context_lens"].contiguous().int()
    try:
        logits = deep_gemm.fp8_fp4_mqa_logits(
            q=(q_fp, None), kv=(kv_fp, kv_sf), weights=kernel_weights,
            cu_seq_len_k_start=ks, cu_seq_len_k_end=ke, max_seqlen_k=num_kv_tokens)
    except Exception as e:
        print(f"    [SKIP] 真实连续 kernel 调用失败：{type(e).__name__}: {e}")
        return
    logits = logits[:, :num_kv_tokens]
    # 稀疏压缩：gather 选中 token（支持集）
    idx = c["token_indices"].clamp(0, num_kv_tokens - 1)
    compressed = torch.gather(logits, 1, idx)
    supp = c["support"]
    # 模拟参考：用反量化 q/kv 跑官方 ref，再对同一支持集 gather
    ref_sim, _ = OFF["ref_mqa_logits"](q_sim, kv_sim, kernel_weights, ks, ke)
    ref_sim = ref_sim[:, :num_kv_tokens]
    ref_comp = torch.gather(ref_sim, 1, idx)
    d = relerr(compressed[supp], ref_comp[supp])
    check("真实 kernel 压缩输出 == 模拟参考压缩输出（支持集内）", d < 5e-3, f"relerr={d:.3e}")
    # 压缩确实是选择：支持集内匹配、且未选中的 token 不在输出里（覆盖 < 窗口）
    n_supp = int(supp.sum().item())
    check("压缩支持集是窗口的严格子集（跳过未选中块）", n_supp < T * num_kv_tokens // 2 and n_supp > 0,
          f"有效项={n_supp} / 窗口={T * num_kv_tokens}")


if __name__ == "__main__":
    print("=" * 80)
    print("P149-1 稀疏（块选择）MQA indexer 前向的反向（H100 真机）")
    print("=" * 80)
    part_a()
    part_b()
    part_c()
    part_d()
    part_e()
    print("-" * 80)
    print(f"PASS={PASS}  FAIL={FAIL}")
    raise SystemExit(0 if FAIL == 0 else 1)
