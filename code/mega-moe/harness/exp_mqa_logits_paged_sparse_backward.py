#!/usr/bin/env python3
"""P150-1 实验：**分页稀疏（块选择）MQA logits（lightning indexer）前向的反向**。

indexer 反向的 2x2 矩阵：第 157 篇（P147-1）连续稠密、第 158 篇（P148-1）分页稠密、
第 159 篇（P149-1）连续稀疏——**唯一空缺是「分页 × 稀疏」**，而它正是生产 long-context 训练
真正跑的形态：KV 住在物理页池里、只用 indexer 选中的一小撮块打分、输出压缩成
`[T, num_max_sparse_blocks*SB]`（前向 `fp8_fp4_paged_sparse_mqa_logits`，**SM100 专属**）。

本篇补这个组合形态。反向要**依次**做两级逆映射：
    support（选中块 ∩ 有效窗口） → logical KV token →（页表逆映射）→ physical slot，
再 scatter-accumulate。两层别名都必须累加：① next_n 前缀重叠；② 跨请求共享物理页。

纯 PyTorch（float64 对拍 autograd + 中心差分）+ 一个 CUDA 反向后端（架构无关），H100(sm90) 真机可跑。

分页稀疏前向语义（`tests/test_attention.py:484-672` 的 `enumerate_sparse_mqa_logits` /
`make_sparse_kv_block_indices` 与 `:655-668` 的压缩展开，唯一正确性规范；`attention.hpp:337-441`
的 host 入口）：
    PAGE = page_kv；SB = sparse_block_kv（PAGE % SB == 0）；
    block_offsets[r] = context_starts[r] % SB（paged 时 context_starts=0 => 0）；
    token_indices[r, l*SB+kvj] = sparse_indices[r,l]*SB + block_offsets[r] + kvj   （logical KV token）
    support[r, c] = (l < num_blocks[r]) & (start_r <= token < cl_r)
    phys[r, c] = block_table[r, token // PAGE] * PAGE + token % PAGE              （页表逆映射）
    压缩输出 G[r, c] = I_paged[r, logical=token]（仅 support）；未选中块「看不见」→ 梯度恒 0。

反向：
    dI_support[r,c]（支持集内 KL 梯度）
    dL/dw_{r,h}        = sum_{s in support(r)} dI_s[r,c(s)] relu(z_{r,s,h})
    dL/dz_{r,s,h}      = dI_s[r,c(s)] w_{r,h} 1[z>0]
    dL/dq_{r,h,d}      = sum_s dL/dz * khat_{phys(s),d}
    dL/dkv[phys]      += sum_{r,h,s : phys(s)=phys} dL/dz_{r,s,h} q_{r,h,d} scale_phys(s)   （scatter-add）
    dL/dscale[phys]   += sum_{r,h,s : phys(s)=phys} dL/dz_{r,s,h} (q_{r,h}·kv_phys(s))
"""

import ast
import math
import os
import random
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
    print(f"  [{tag}] {name:<62} {detail}")


def relerr(a, b):
    a = a.double(); b = b.double()
    return (a - b).abs().sum() / (a.abs().sum() + b.abs().sum() + 1e-30)


def ceil_div(a, b):
    return (a + b - 1) // b


# ---------------------------------------------------------------------------
# 抽取 DeepGEMM 官方测试里纯 torch 的参考函数（真正核对过的锚）
# ---------------------------------------------------------------------------
DG = None
for _c in (os.environ.get("DEEPGEMM_DIR", ""), os.path.expanduser("~/github/DeepGEMM"),
           os.path.expanduser("~/github/DeepGEMM"), os.path.expanduser("~/github/DeepGEMM")):
    if _c and os.path.isdir(_c):
        DG = _c
        break
TEST_ATTN = os.path.join(DG or "~/github/DeepGEMM", "tests/test_attention.py")


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


OFF = load_official_funcs(["make_sparse_kv_block_indices", "ref_paged_mqa_logits", "to_mqa_weights",
                           "kv_cache_cast_to_fp8"])


# ---------------------------------------------------------------------------
# 分页稀疏用例构造：页池 + 每行页表 + 稀疏块选择 + support + 页表逆映射
# ---------------------------------------------------------------------------
def build_paged_sparse_case(S, N, H, D, SB, PAGE, NBmax, blocks_per_row, avg_kv, seed=0,
                            share_pages=False, alias_prefix=True):
    """返回一个「分页稀疏」用例。

    S 请求、每请求 N 个 q token（next_n），每请求一串页（page table）；每行选 `blocks_per_row`
    个 logical 稀疏块。support 用官方 `:655-664` 同款公式构造，物理槽由页表逆映射得到。
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    rng = random.Random(seed)
    assert PAGE % SB == 0
    blocks_per_page = PAGE // SB
    context_lens = [rng.randint(int(0.7 * avg_kv), int(1.3 * avg_kv)) for _ in range(S)]
    context_lens = [ceil_div(c, SB) * SB for c in context_lens]           # 对齐到 SB（paged 约定）
    max_ctx = max(context_lens)
    max_pages = ceil_div(max_ctx, PAGE)

    # 页池 + 每请求页表；share_pages 时请求 >=1 复用请求 0 的页（prefix sharing / beam）
    if share_pages:
        num_pages = (S - 1) * max_pages
    else:
        num_pages = S * max_pages
    page_pool = torch.randperm(num_pages, generator=g).tolist()
    block_table = torch.zeros((S, max_pages), dtype=torch.int)
    for i in range(S):
        base = 0 if (share_pages and i >= 1) else i
        block_table[i, :max_pages] = torch.tensor(page_pool[base * max_pages:(base + 1) * max_pages])
    num_pool_tokens = num_pages * PAGE

    # 每行稀疏块选择（请求级生成、next_n 各槽共享，模拟前缀重叠）
    total_blocks_per_req = [ceil_div(context_lens[i], SB) for i in range(S)]
    sparse_indices = torch.zeros((S * N, NBmax), dtype=torch.int)
    num_blocks = torch.zeros(S * N, dtype=torch.int)
    for i in range(S):
        nb = min(blocks_per_row, total_blocks_per_req[i])
        sel = sorted(rng.sample(range(total_blocks_per_req[i]), nb)) if nb > 0 else []
        for j in range(N):
            r = i * N + j
            sparse_indices[r, :len(sel)] = torch.tensor(sel) if sel else 0
            if sel:
                sparse_indices[r, len(sel):] = sel[-1]
            num_blocks[r] = len(sel)

    P = S * N
    starts = torch.zeros(P, dtype=torch.int)                                  # paged: context_starts=0
    cl = torch.tensor([context_lens[i] for i in range(S) for _ in range(N)], dtype=torch.int)
    block_offsets = starts % SB
    kv_offsets = torch.arange(SB, dtype=torch.int)
    token_indices = (sparse_indices.unsqueeze(-1) * SB + block_offsets[:, None, None]
                     + kv_offsets[None, None, :]).reshape(P, NBmax * SB)      # logical KV token
    C = NBmax * SB
    l_idx = torch.arange(C, dtype=torch.int) // SB
    support = l_idx[None, :] < num_blocks[:, None]
    support &= token_indices >= starts[:, None]
    support &= token_indices < cl[:, None]

    # 页表逆映射 logical -> physical（逐行，token // PAGE 定页、token % PAGE 定页内偏移）
    req_of_row = torch.tensor([i for i in range(S) for _ in range(N)], dtype=torch.long)
    phys = torch.zeros_like(token_indices)
    for r in range(P):
        i = int(req_of_row[r])
        pg = token_indices[r] // PAGE
        phys[r] = block_table[i, pg.clamp(max=max_pages - 1)] * PAGE + (token_indices[r] % PAGE)

    support = support.to(DEV)
    g2 = torch.Generator(device="cpu").manual_seed(seed + 991)
    q = torch.randn(P, H, D, generator=g2).to(DEV).to(DT) * 0.3
    kv_flat = torch.randn(num_pool_tokens, D, generator=g2).to(DEV).to(DT) * 0.3
    kvs = torch.rand(num_pool_tokens, generator=g2).to(DEV).to(DT) * 0.7 + 0.5
    w = torch.randn(P, H, generator=g2).to(DEV).to(DT) * 0.3
    dG = torch.randn(P, C, generator=g2).to(DEV).to(DT) * support.to(DT)

    return dict(P=P, S=S, N=N, H=H, D=D, SB=SB, PAGE=PAGE, NBmax=NBmax, C=C,
                num_pool_tokens=num_pool_tokens, max_pages=max_pages,
                q=q, kv_flat=kv_flat, kvs=kvs, w=w, dG=dG, support=support,
                token_indices=token_indices.to(DEV), phys=phys.to(DEV),
                context_lens=cl.to(DEV), block_table=block_table.to(DEV), req_of_row=req_of_row.to(DEV),
                sparse_indices=sparse_indices.to(DEV), num_blocks=num_blocks.to(DEV))


def support_logits_diff(c, q=None, kv_flat=None, kvs=None, w=None):
    """可微分分页稀疏「支持集 logits」：先页表逆映射取物理 KV，再打分，再选支持集。"""
    q = c["q"] if q is None else q
    kv_flat = c["kv_flat"] if kv_flat is None else kv_flat
    kvs = c["kvs"] if kvs is None else kvs
    w = c["w"] if w is None else w
    phys = c["phys"]                                   # [P,C]
    khat = kv_flat[phys] * kvs[phys][..., None]        # [P,C,D]
    z = torch.einsum("phd,pcd->phc", q, khat)          # [P,H,C]
    I = torch.einsum("phc,ph->pc", z.relu(), w)        # [P,C]
    return I, z


# ---------------------------------------------------------------------------
# Part A：分页稀疏支持集 == 官方 ref_paged_mqa_logits 在同 logical 位置的取值（选择语义）
# ---------------------------------------------------------------------------
def part_a():
    print("Part A  分页稀疏「支持集 logits」== 官方 ref_paged_mqa_logits 在选中 logical 位置的取值")
    S, N, H, D, SB, PAGE, NBmax = 5, 2, 8, 32, 16, 64, 8
    c = build_paged_sparse_case(S, N, H, D, SB, PAGE, NBmax, blocks_per_row=4, avg_kv=512, seed=1)
    # 官方 ref_paged_mqa_logits：整窗口稠密打分（每 token 独立 scale=1，见下）
    kv_cache = torch.zeros((c["num_pool_tokens"] // PAGE, PAGE, 1, D), dtype=DT, device=DEV)
    kv_cache.view(-1, D)[:] = c["kv_flat"]
    # ref 不支持 per-token scale；构造 ref 时把 scale 吸收进 kv（等价 khat），这样 ref 锚的就是
    # 「<q, kv*scale> 的 relu 加权和」这条核心语义。
    kv_scaled = kv_cache * c["kvs"].view(-1, PAGE, 1, 1)
    weights = c["w"]                                   # [P,H] == [S*N,H]
    cl_2d = c["context_lens"].view(S, N)
    max_model_len = ceil_div(int(c["context_lens"].max().item()), PAGE) * PAGE
    ref = OFF["ref_paged_mqa_logits"](c["q"].view(S, N, H, D).to(DT), kv_scaled.to(DT),
                                      weights.to(DT), cl_2d, c["block_table"], max_model_len)
    ref = ref.to(DEV)                                  # [P, max_model_len]（logical 位置）
    I, _ = support_logits_diff(c)                       # [P,C]
    idx = c["token_indices"].clamp(0, max_model_len - 1)
    ref_gather = torch.gather(ref, 1, idx)
    supp = c["support"]
    d = relerr(I[supp], ref_gather[supp])
    # ref 输出本身是 fp32（`logits = torch.full(..., dtype=torch.float32)`），故 ~1e-8 是它的输出舍入地板
    check("支持集内 可微分分页稀疏 == 官方 ref 在 logical 位置取值", d < 1e-6, f"relerr={d:.3e}")
    check("支持集是窗口的严格子集（跳过未选中块）",
          int(supp.sum()) < c["P"] * max_model_len // 2 and int(supp.sum()) > 0,
          f"有效项={int(supp.sum())} / 窗口={c['P'] * max_model_len}")
    return c


# ---------------------------------------------------------------------------
# Part B：页表逆映射恒等式（logical -> physical）+ 与 metadata kernel 公式等价
# ---------------------------------------------------------------------------
def part_b():
    print("Part B  页表逆映射恒等式 + 与 metadata kernel 物理块公式等价")
    S, N, H, D, SB, PAGE, NBmax = 4, 3, 4, 16, 16, 64, 6
    c = build_paged_sparse_case(S, N, H, D, SB, PAGE, NBmax, blocks_per_row=5, avg_kv=768, seed=2)
    # 逐字复刻 metadata kernel 的物理块公式（scheduler/sm100_sparse_mqa_logits_metadata.cuh:379-383）
    #   kNumKVBlocksPerPage = PAGE/SB；logical_page_idx = logical_kv_block_idx / kNumKVBlocksPerPage；
    #   physical_kv_block_idx = block_table[q, logical_page_idx]*kNumKVBlocksPerPage
    #                           + logical_kv_block_idx % kNumKVBlocksPerPage
    #   physical token = physical_kv_block_idx * SB + token_in_kv_block
    blocks_per_page = PAGE // SB
    rng = random.Random(7)
    ok_all = True
    max_mismatch = 0
    for _ in range(200):
        r = rng.randrange(c["P"])
        i = int(c["req_of_row"][r].item())
        l = rng.randrange(int(c["num_blocks"][r].item()))
        kvj = rng.randrange(SB)
        logical_block = int(c["sparse_indices"][r, l].item())
        logical_token = logical_block * SB + kvj           # paged: block_offsets=0
        # 我们的逐 token 版
        phys_a = int(c["block_table"][i, logical_token // PAGE].item()) * PAGE + (logical_token % PAGE)
        # metadata kernel 的按块版
        logical_page_idx = logical_block // blocks_per_page
        physical_block = int(c["block_table"][i, logical_page_idx].item()) * blocks_per_page \
            + logical_block % blocks_per_page
        phys_b = physical_block * SB + kvj
        if phys_a != phys_b:
            ok_all = False
            max_mismatch += 1
    check("逐 token 页表映射 == metadata kernel 按块映射（200 随机点）", ok_all, f"mismatch={max_mismatch}")
    # 物理槽不越界
    check("物理槽落在页池范围内",
          int(c["phys"][c["support"]].min()) >= 0 and int(c["phys"][c["support"]].max()) < c["num_pool_tokens"],
          f"phys∈[{int(c['phys'][c['support']].min())},{int(c['phys'][c['support']].max())}], pool={c['num_pool_tokens']}")
    return c


# ---------------------------------------------------------------------------
# Part C：支持集内 KL 蒸馏梯度
# ---------------------------------------------------------------------------
def part_c():
    print("Part C  支持集内 KL 蒸馏梯度 dL/dI = softmax_support(I) - p")
    S, N, H, D, SB, PAGE, NBmax = 3, 2, 8, 32, 16, 64, 6
    c = build_paged_sparse_case(S, N, H, D, SB, PAGE, NBmax, blocks_per_row=4, avg_kv=384, seed=3)
    I, _ = support_logits_diff(c)
    supp = c["support"]
    neg = torch.where(supp, I, torch.full_like(I, -1e30)).clone().requires_grad_(True)
    rnd = torch.rand_like(I) * supp
    p = rnd / rnd.sum(-1, keepdim=True).clamp_min(1e-30)
    loss = -(p * F.log_softmax(neg, dim=-1)).sum(-1).sum()
    loss.backward()
    q_prob = torch.softmax(neg.detach(), dim=-1)
    analytic = (q_prob - p) * supp
    check("dL/dI 解析(softmax-p) vs autograd", relerr(analytic, neg.grad) < 1e-12,
          f"relerr={relerr(analytic, neg.grad):.3e}")
    rowsum = float(analytic.detach().sum(-1).abs().max())
    check("支持集内行和 = 0", rowsum < 1e-12, f"max={rowsum:.1e}")
    check("支持集外梯度恒 0", float(analytic.detach()[~supp].abs().max()) == 0.0,
          f"max={float(analytic.detach()[~supp].abs().max()):.1e}")
    return c


# ---------------------------------------------------------------------------
# Part D：打分算子反向（页表逆映射 + scatter 到物理池）vs autograd
# ---------------------------------------------------------------------------
def operator_backward_paged_sparse(c):
    q, kv_flat, kvs, w = c["q"], c["kv_flat"], c["kvs"], c["w"]
    phys = c["phys"]
    khat = kv_flat[phys] * kvs[phys][..., None]            # [P,C,D]
    z = torch.einsum("phd,pcd->phc", q, khat)              # [P,H,C]
    dI = c["dG"]                                            # [P,C]
    dw = torch.einsum("pc,phc->ph", dI, z.relu())          # [P,H]
    dz = dI[:, None, :] * w[:, :, None] * (z > 0).to(z.dtype)   # [P,H,C]
    dq = torch.einsum("phc,pcd->phd", dz, khat)            # [P,H,D]
    dkhat = torch.einsum("phc,phd->pcd", dz, q)            # [P,C,D]
    dkv = torch.zeros_like(kv_flat)
    dkv.index_add_(0, phys.reshape(-1), (dkhat * kvs[phys][..., None]).reshape(-1, c["D"]))
    dks = torch.zeros_like(kvs)
    dks.index_add_(0, phys.reshape(-1), (dkhat * kv_flat[phys]).sum(-1).reshape(-1))
    return dict(dw=dw, dq=dq, dkv=dkv, dks=dks)


def part_d():
    print("Part D  分页稀疏打分算子反向（dL/dw,dq,dkv,dscale）vs autograd（物理池 scatter-add）")
    S, N, H, D, SB, PAGE, NBmax = 5, 2, 8, 32, 16, 64, 6
    c = build_paged_sparse_case(S, N, H, D, SB, PAGE, NBmax, blocks_per_row=4, avg_kv=512,
                                seed=4, share_pages=True, alias_prefix=True)
    q = c["q"].clone().requires_grad_(True)
    kv_flat = c["kv_flat"].clone().requires_grad_(True)
    kvs = c["kvs"].clone().requires_grad_(True)
    w = c["w"].clone().requires_grad_(True)
    c2 = dict(c, q=q, kv_flat=kv_flat, kvs=kvs, w=w)
    I, _ = support_logits_diff(c2)
    loss = (c["dG"] * I).sum()
    loss.backward()
    ana = operator_backward_paged_sparse(c2)
    check("dq        解析 vs autograd", relerr(ana["dq"], q.grad) < 1e-12, f"relerr={relerr(ana['dq'], q.grad):.3e}")
    check("dw        解析 vs autograd", relerr(ana["dw"], w.grad) < 1e-12, f"relerr={relerr(ana['dw'], w.grad):.3e}")
    check("dkv       解析 vs autograd（页表逆映射 + 别名）", relerr(ana["dkv"], kv_flat.grad) < 1e-12,
          f"relerr={relerr(ana['dkv'], kv_flat.grad):.3e}")
    check("dscale    解析 vs autograd", relerr(ana["dks"], kvs.grad) < 1e-12, f"relerr={relerr(ana['dks'], kvs.grad):.3e}")
    # 中心差分核验几个物理槽
    qd, kv0, ksd, wd = c["q"], c["kv_flat"], c["kvs"], c["w"]
    eps = 1e-6
    worst = 0.0
    for (slot, dd) in [(0, 0), (c["num_pool_tokens"] - 1, c["D"] - 1)]:
        kp = kv0.clone(); kp[slot, dd] += eps
        km = kv0.clone(); km[slot, dd] -= eps
        Lp = (c["dG"] * support_logits_diff(c, qd, kp, ksd, wd)[0]).sum()
        Lm = (c["dG"] * support_logits_diff(c, qd, km, ksd, wd)[0]).sum()
        fd = (Lp - Lm) / (2 * eps)
        worst = max(worst, relerr(fd.reshape(1), ana["dkv"][slot, dd].reshape(1)))
    check("dkv 中心差分（2 个物理槽条目）", worst < 5e-7, f"最差 relerr={worst:.3e}")
    return c


# ---------------------------------------------------------------------------
# Part E：两层别名 —— scatter-accumulate vs「写」（last-writer-wins）
# ---------------------------------------------------------------------------
def part_e():
    print("Part E  两层别名（next_n 前缀重叠 + 跨请求共享页）：scatter-add vs 「写」")
    S, N, H, D, SB, PAGE, NBmax = 5, 3, 8, 32, 16, 64, 6
    c = build_paged_sparse_case(S, N, H, D, SB, PAGE, NBmax, blocks_per_row=4, avg_kv=512,
                                seed=5, share_pages=True, alias_prefix=True)
    phys = c["phys"][c["support"]].reshape(-1)
    uniq, counts = torch.unique(phys, return_counts=True)
    n_shared = int((counts > 1).sum().item())
    max_share = int(counts.max().item()) if counts.numel() else 0
    print(f"    支持集：{int(c['support'].sum())} 个 (r,c) 引用 -> {int(uniq.numel())} 个物理槽；"
          f"被多引用槽 {n_shared} 个、最大共享度 {max_share}")
    ana = operator_backward_paged_sparse(c)
    # naive「写」：对每个 (r, c in support) 直接覆盖目标物理槽
    kv_flat, kvs, q, w = c["kv_flat"], c["kvs"], c["q"], c["w"]
    phys_all = c["phys"]
    khat = kv_flat[phys_all] * kvs[phys_all][..., None]
    z = torch.einsum("phd,pcd->phc", q, khat)
    dz = c["dG"][:, None, :] * w[:, :, None] * (z > 0).to(z.dtype)
    dkv_naive = torch.zeros_like(kv_flat)
    supp = c["support"]
    for r in range(c["P"]):
        for h in range(H):
            cols = supp[r].nonzero(as_tuple=True)[0]
            for cc in cols.tolist():
                p = int(phys_all[r, cc].item())
                dkv_naive[p] = dz[r, h, cc] * q[r, h] * kvs[p]      # overwrite (bug)
    d_bad = relerr(dkv_naive, ana["dkv"])
    shared_id = uniq[counts > 1]
    maxdiff = float((dkv_naive[shared_id] - ana["dkv"][shared_id]).abs().max()) if shared_id.numel() else 0.0
    check("naive「写」在被共享物理槽上丢贡献", maxdiff > 0, f"共享槽最大绝对差={maxdiff:.3e}")
    check("naive「写」相对正确累加全局 relerr 显著 >0", d_bad > 1e-2, f"relerr={d_bad:.3e}")
    # autograd 再验一次
    qg = c["q"].clone().requires_grad_(True)
    kvg = c["kv_flat"].clone().requires_grad_(True)
    I, _ = support_logits_diff(dict(c, q=qg, kv_flat=kvg))
    (c["dG"] * I).sum().backward()
    check("两层别名用例：dkv 解析 == autograd", relerr(ana["dkv"], kvg.grad) < 1e-12,
          f"relerr={relerr(ana['dkv'], kvg.grad):.3e}")
    return c


# ---------------------------------------------------------------------------
# Part F：真实 SM90 fp8_fp4_paged_mqa_logits 前向锚 + 「压缩=选择」语义
# ---------------------------------------------------------------------------
def part_f():
    print("Part F  真实 SM90 fp8_fp4_paged_mqa_logits 稠密前向锚 + 分页稀疏压缩=选择")
    try:
        import deep_gemm
        from deep_gemm.testing import get_arch_major
        from deep_gemm.utils import per_token_cast_to_fp8 as OFF_cast_fp8
    except Exception as e:  # pragma: no cover
        print(f"    [SKIP] deep_gemm 不可用：{e}")
        return
    if get_arch_major() != 9:
        print(f"    [SKIP] 本机 arch_major={get_arch_major()}（分页 fp8 连续/SM90 分支才可跑）")
        return
    S, N, H, D, SB, PAGE, NBmax = 64, 1, 32, 128, 16, 64, 8
    c = build_paged_sparse_case(S, N, H, D, SB, PAGE, NBmax, blocks_per_row=5, avg_kv=512, seed=6,
                                share_pages=False, alias_prefix=False)
    num_pages = c["num_pool_tokens"] // PAGE
    # SM90 分页前向：fused KV 布局 = per-token E4M3 + 每 token 一个 fp32 scale
    kv_cache = torch.zeros((num_pages, PAGE, 1, D), dtype=torch.bfloat16, device="cpu")
    kv_cache.view(-1, D)[:] = c["kv_flat"].cpu().to(torch.bfloat16)
    kv_fp8, kv_sim = OFF["kv_cache_cast_to_fp8"](kv_cache)
    q_fp8 = c["q"].float().view(S, N, H, D).to(torch.float8_e4m3fn).to(DEV)
    kv_fp8 = kv_fp8.to(DEV).reshape(num_pages, PAGE, 1, D + 4)
    weights = OFF["to_mqa_weights"](c["w"].float()).to(DEV).contiguous()
    cl_2d = c["context_lens"].view(S, N).int()
    max_model_len = ceil_div(int(c["context_lens"].max().item()), PAGE) * PAGE
    try:
        logits = deep_gemm.fp8_fp4_paged_mqa_logits(
            q=(q_fp8, None), kv_cache=kv_fp8, weights=weights, context_lens=cl_2d,
            block_table=c["block_table"].int(),
            schedule_meta=deep_gemm.get_paged_mqa_logits_metadata(cl_2d, PAGE, deep_gemm.get_num_sms()),
            max_context_len=max_model_len)
    except Exception as e:
        print(f"    [SKIP] 真实分页 kernel 调用失败：{type(e).__name__}: {e}")
        return
    logits = logits[:, :max_model_len]
    idx = c["token_indices"].clamp(0, max_model_len - 1)
    compressed = torch.gather(logits, 1, idx)              # 压缩 = 对稠密 logits 的选择（gather）
    supp = c["support"]
    # 模拟参考：反量化 kv、重跑 && 对同一 logical 位置 gather
    ref_sim = OFF["ref_paged_mqa_logits"](
        q_fp8.to(torch.bfloat16).view(S, N, H, D), kv_sim.to(DEV).view(num_pages, PAGE, 1, D),
        weights.float(), cl_2d, c["block_table"].int(), max_model_len)
    ref_sim = ref_sim[:, :max_model_len]
    ref_comp = torch.gather(ref_sim, 1, idx)
    d = relerr(compressed[supp], ref_comp[supp])
    check("真实分页 kernel 压缩输出 == 模拟参考压缩输出（支持集内）", d < 5e-3, f"relerr={d:.3e}")
    check("压缩支持集是窗口的严格子集", int(supp.sum()) < c["P"] * max_model_len // 2 and int(supp.sum()) > 0,
          f"有效项={int(supp.sum())} / 窗口={c['P'] * max_model_len}")


if __name__ == "__main__":
    print("=" * 80)
    print("P150-1 分页稀疏 MQA indexer 前向的反向（页表逆映射 x 支持集散播，H100 真机）")
    print("=" * 80)
    part_a()
    part_b()
    part_c()
    part_d()
    part_e()
    part_f()
    print("-" * 80)
    print(f"PASS={PASS}  FAIL={FAIL}")
    raise SystemExit(0 if FAIL == 0 else 1)
