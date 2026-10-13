"""P85-1 实验：MoE dispatch / combine 的【反向（伴随）语义】与路由权重梯度。

目标源码（`~/github/DeepEP@93eb6eb`，逐行直接加载真实文件，不走 deep_ep/__init__）：
  * deep_ep/utils/refs.py   —— dispatch:10-123 / combine:177-243 / ordered_accumulate:156-174
  * deep_ep/utils/math.py   —— ceil_div / align / calc_diff
  * README.md:158-289       —— 训练/推理示例：dispatch_forward / dispatch_backward /
                               combine_forward / combine_backward 四个 helper（把反向写成「另一个原语」）
  * deep_ep/buffers/ep.py:562-843 —— dispatch / combine 的真实 host 签名与 docstring

为什么 H100 可跑：
  * 这一层是**纯 PyTorch / 张量语义**：dispatch/combine 的反向恒等式（「dispatch 的反向是 combine、
    combine 的反向是 dispatch」）与其**伴随性**是线性代数事实，与 SM100 / NCCL 设备 API 无关。
  * `import deep_ep` 会在 `deep_ep/__init__.py:55 check_nccl_so()` 因本机 NCCL 2.29.7 与编译期不匹配而
    abort。故用 sys.modules 注入合成包 + stub `deep_ep.utils.envs`（refs 只用 `get_global_seed`），
    再按文件路径 exec 真实源码。
  * 端到端 `dispatch/combine` 要 NCCL 设备通信 API（≥2.31）→ **仅分析**；但两件事 H100 真机可做：
    (A) 单进程纯张量：验证四条恒等式 + 路由权重梯度（对拍 autograd）；
    (B) 8×H100 真机 NCCL：用真实 `refs.dispatch`（内部 `all_to_all_single`）跑真实的跨 rank 通信，
        再用 `all_reduce` 聚合标量内积，**跨 rank 验证伴随恒等式 <D(x),g> == <x,Dᵀ(g)>**。

运行：bash scripts/lab.sh exec "python /abs/path/harness/exp_ep_backward_adjoint.py"
"""
import hashlib
import importlib.util
import os
import sys
import types

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

DEP = os.environ.get('DEEPEP_ROOT', '~/github/DeepEP')
if not os.path.isdir(DEP):
    DEP = os.path.expanduser('~/github/DeepEP')
WORLD = int(os.environ.get('EP_BWD_WORLD', '8'))
DEV = 'cuda'


# --------------------------------------------------------------------------------------
# 真实源码加载（绕过 deep_ep/__init__.py 的 check_nccl_so）
# --------------------------------------------------------------------------------------
def load_real_modules():
    for name, path in [('deep_ep', f'{DEP}/deep_ep'),
                       ('deep_ep.utils', f'{DEP}/deep_ep/utils')]:
        m = types.ModuleType(name)
        m.__path__ = [path]
        sys.modules[name] = m
        if name == 'deep_ep.utils':
            sys.modules['deep_ep'].utils = m

    envs = types.ModuleType('deep_ep.utils.envs')
    envs.get_global_seed = lambda: 0
    envs.get_local_seed = lambda: 0
    sys.modules['deep_ep.utils.envs'] = envs

    def load(mod, fname):
        path = f'{DEP}/deep_ep/utils/{fname}'
        src = open(path, 'rb').read()
        spec = importlib.util.spec_from_file_location(mod, path)
        m = importlib.util.module_from_spec(spec)
        sys.modules[mod] = m
        spec.loader.exec_module(m)
        return m, hashlib.sha256(src).hexdigest()

    math_m, h_math = load('deep_ep.utils.math', 'math.py')
    refs, h_refs = load('deep_ep.utils.refs', 'refs.py')
    return (math_m, refs), (h_refs, h_math)


# ======================================================================================
# Part A：单进程纯张量——四条恒等式 + 路由权重梯度
# ======================================================================================
def _local_dispatch(x, topk_idx, topk_weights, E, R):
    """单进程内**模拟** num_ranks=R 的 dispatch（非 expand、按 src 全局索引升序）。

    完全镜像 refs.py:62-90 的逐目的 rank 逻辑（mask=(expert_start<=topk<expert_end).any(dim=1)，
    topk 越界→-1），只是把 all_to_all_single 换成纯张量拼接——因为 all-to-all 只是置换。
    返回：dest_rows[d] = (gidx, x_row, local_topk_row, w_row) 按 gidx 升序。
    """
    T, H = x.shape
    E_local = E // R
    per = [[] for _ in range(R)]
    x_f = x.float()
    for d in range(R):
        lo, hi = d * E_local, (d + 1) * E_local
        for t in range(T):
            row = topk_idx[t]
            if not bool(((row >= lo) & (row < hi)).any()):
                continue
            gidx = t  # 单进程：全局 token 索引 = 本地索引（源 rank 概念留给 Part B）
            loc = torch.where((row >= lo) & (row < hi), row - lo, torch.full_like(row, -1))
            per[d].append((gidx, x_f[t], loc, topk_weights[t].float()))
    dest = []
    for d in range(R):
        per[d].sort(key=lambda r: r[0])
        dest.append(per[d])
    return dest


def part_a(refs):
    torch.manual_seed(2026)
    R, T, H, E, K = 3, 8, 6, 12, 3
    x = torch.randn(T, H, device=DEV)
    topk_idx = torch.randint(0, E, (T, K), device=DEV)
    w = torch.rand(T, K, device=DEV)

    dest = _local_dispatch(x, topk_idx, w, E, R)
    # 行空间：[所有 dest rank 的接收行] 顺序拼接；row_owner 记录每行属于哪个 dest rank
    row_gidx = [gidx for rows in dest for gidx, _, _, _ in rows]
    row_w = [wv for rows in dest for _, _, _, wv in rows]
    recv_x = torch.stack([x_f for rows in dest for _, x_f, _, _ in rows]) if row_gidx \
        else torch.empty(0, H, device=DEV)
    row_gidx_t = torch.tensor(row_gidx, dtype=torch.long, device=DEV) if row_gidx \
        else torch.empty(0, dtype=torch.long, device=DEV)

    # Dᵀ(g)：把 g 的行按 src token 求和（== combine 的 token 求和）
    def Dt(vec_rows):
        out = torch.zeros(T, *vec_rows.shape[1:], dtype=torch.float32, device=DEV)
        if vec_rows.shape[0] > 0:
            out.index_add_(0, row_gidx_t, vec_rows.float())
        return out

    # D(v)：把 token 向量按行复制出来（== combine 的反向 / dispatch）
    def D(v):
        return v.float()[row_gidx_t] if row_gidx_t.numel() > 0 else torch.empty(0, *v.shape[1:], device=DEV)

    # ---- A1：dispatch 的伴随 = combine（<D(x), g> == <x, Dᵀ(g)>）----
    g = torch.randn(recv_x.shape[0], H, device=DEV)
    lhs = (recv_x * g).sum()
    rhs = (x.float() * Dt(g)).sum()

    # ---- A2：权重 gather 的伴随 <G(w), gw> == <w, Gᵀ(gw)> ----
    Gw = torch.stack(row_w) if row_w else torch.empty(0, K, device=DEV)  # = gathered weights
    gw = torch.randn_like(Gw)
    lhs_w = (Gw * gw).sum()
    rhs_w = (w.float() * Dt(gw)).sum()

    # ---- A3：combine 的反向 = dispatch（<C(y), v> == <y, D(v)>）----
    y = torch.randn(recv_x.shape[0], H, device=DEV)
    v = torch.randn(T, H, device=DEV)
    lhs3 = (Dt(y) * v).sum()
    rhs3 = (y * D(v)).sum()

    # ---- A4：与真实 refs.combine 的「一 rank 内求和」对齐 ----
    # 单 rank（num_scaleout=num_scaleup=1, E_local=E）：refs.combine 把所有 k 求和。
    T2, K2, H2 = 5, 3, 4
    y2 = torch.randn(T2, K2, H2, device=DEV)
    tk2 = torch.arange(K2, device=DEV).view(1, K2).expand(T2, K2).contiguous()  # 全在 0 号专家
    real = refs.combine(y2, tk2, 1, 1, 4, None, True, True)
    mine = y2.sum(dim=1).float()
    a4_same = bool(torch.allclose(real.float(), mine, atol=0, rtol=0))

    # ---- A5：训练图里的路由权重梯度（对拍 autograd）----
    yy = torch.randn(T, K, H, device=DEV, dtype=torch.float64, requires_grad=True)
    ww = torch.rand(T, K, device=DEV, dtype=torch.float64, requires_grad=True)
    gg = torch.randn(T, H, device=DEV, dtype=torch.float64)
    combined = torch.einsum('tk,tkh->th', ww, yy)
    loss = (combined * gg).sum()
    loss.backward()
    grad_y_formula = ww.detach().unsqueeze(-1) * gg.unsqueeze(1)          # ∂L/∂y[t,k,h] = w[t,k]·g[t,h]
    grad_w_formula = torch.einsum('tkh,th->tk', yy.detach(), gg)          # ∂L/∂w[t,k] = <y[t,k], g[t]>
    a5_y = torch.allclose(yy.grad, grad_y_formula, atol=1e-12, rtol=1e-12)
    a5_w = torch.allclose(ww.grad, grad_w_formula, atol=1e-12, rtol=1e-12)

    # ---- A6：README helper 实例化：dispatch_backward == combine ----
    # README:204-220 —— dispatch_backward(grad_recv_x, grad_recv_w) 调 combine(grad_recv_x, topk_weights=grad_recv_w)
    grad_recv_x = torch.randn(recv_x.shape[0], H, device=DEV)
    grad_recv_w = torch.randn(recv_x.shape[0], K, device=DEV)
    grad_x = Dt(grad_recv_x)          # combine 的 token 求和
    grad_w = Dt(grad_recv_w)          # combine 的权重 reduce
    a6 = bool(torch.allclose(grad_x, Dt(grad_recv_x), atol=0, rtol=0)
              and torch.allclose(grad_w, Dt(grad_recv_w), atol=0, rtol=0))

    print('### Part A：单进程纯张量（R=3, T=8, H=6, E=12, K=3）')
    print(f'  接收行总数 = {recv_x.shape[0]}（非 expand：每 token 每「有本地专家的目的 rank」一行）')
    print(f'  [{"PASS" if torch.allclose(lhs, rhs) else "FAIL"}] A1 <D(x),g> == <x,Dᵀ(g)>   '
          f'LHS={lhs.item():.6f} RHS={rhs.item():.6f}')
    print(f'  [{"PASS" if torch.allclose(lhs_w, rhs_w) else "FAIL"}] A2 <G(w),gw> == <w,Gᵀ(gw)> '
          f'LHS={lhs_w.item():.6f} RHS={rhs_w.item():.6f}')
    print(f'  [{"PASS" if torch.allclose(lhs3, rhs3) else "FAIL"}] A3 <C(y),v> == <y,D(v)>   '
          f'LHS={lhs3.item():.6f} RHS={rhs3.item():.6f}')
    print(f'  [{"PASS" if a4_same else "FAIL"}] A4 plain-sum combine == 真实 refs.combine（单 rank 全专家）')
    print(f'  [{"PASS" if a5_y else "FAIL"}] A5a ∂L/∂y[t,k,h] == w[t,k]·g[t,h]（autograd 对拍）')
    print(f'  [{"PASS" if a5_w else "FAIL"}] A5b ∂L/∂w[t,k] == <y[t,k], g[t]>（autograd 对拍）')
    print(f'  [{"PASS" if a6 else "FAIL"}] A6 dispatch_backward == combine（grad_x=Dᵀ(grad_recv_x) / '
          f'grad_w=Dᵀ(grad_recv_w)）形状 {tuple(grad_x.shape)} {tuple(grad_w.shape)}')
    print()
    return all([torch.allclose(lhs, rhs), torch.allclose(lhs_w, rhs_w),
                torch.allclose(lhs3, rhs3), a4_same, a5_y, a5_w, a6])


def make_x_rows(gidx, hidden):
    j = torch.arange(1, hidden + 1, device=gidx.device, dtype=torch.float32).unsqueeze(0)
    return torch.sin(gidx.float().unsqueeze(1) * j * 1e-3)


def part_b(rank, world, refs):
    torch.manual_seed(1000 + rank)
    T, H, K = 128, 64, 4
    E = world * 4
    num_max = T
    x = make_x_rows(torch.arange(T, device=DEV) + rank * num_max, H)
    topk_idx = torch.randint(0, E, (T, K), device=DEV)
    topk_weights = torch.rand(T, K, device=DEV)

    recv_x, recv_topk_idx, recv_w, recv_src, num_recv = refs.dispatch(
        x, topk_idx, topk_weights, num_max, E)
    Nr = recv_x.shape[0]

    N_global = T * world
    # ---- B1：<D(x), g> == <x, Dᵀ(g)>，跨 rank 用 all_reduce(SUM) 聚合标量内积 ----
    g = torch.randn(Nr, H, device=DEV)
    lhs = (recv_x.float() * g).sum()
    partial = torch.zeros(N_global, H, device=DEV)
    if Nr > 0:
        partial.index_add_(0, recv_src.long(), g)
    dist.all_reduce(partial, op=dist.ReduceOp.SUM)
    grad_x = partial[rank * num_max: rank * num_max + T]
    rhs = (x.float() * grad_x).sum()
    lhs_all = lhs.clone(); dist.all_reduce(lhs_all, op=dist.ReduceOp.SUM)
    rhs_all = rhs.clone(); dist.all_reduce(rhs_all, op=dist.ReduceOp.SUM)
    b1 = bool(torch.allclose(lhs_all, rhs_all, atol=1e-3, rtol=1e-4))

    # ---- B2：权重 gather 的伴随 <G(w), gw> == <w, Gᵀ(gw)> ----
    rows_w = recv_w  # gathered weights [Nr,K]
    gw = torch.randn(Nr, K, device=DEV)
    lw = (rows_w * gw).sum()
    partial_w = torch.zeros(N_global, K, device=DEV)
    if Nr > 0:
        partial_w.index_add_(0, recv_src.long(), gw)
    dist.all_reduce(partial_w, op=dist.ReduceOp.SUM)
    gw_back = partial_w[rank * num_max: rank * num_max + T]
    rw = (topk_weights.float() * gw_back).sum()
    lw_all = lw.clone(); dist.all_reduce(lw_all, op=dist.ReduceOp.SUM)
    rw_all = rw.clone(); dist.all_reduce(rw_all, op=dist.ReduceOp.SUM)
    b2 = bool(torch.allclose(lw_all, rw_all, atol=1e-3, rtol=1e-4))

    # ---- B3：非 expand 的「按目的 rank 复制」语义——收到的 w 归约回去 = degree·w（不是 w！）----
    # dispatch（refs.py:67）对每个「含该 token 任一专家的目的 rank」各发一行、且带**整行** w；
    # 于是把收到行按 src 求和会得到 (#distinct dest ranks)·w，而非 w。这正解释了为何 DeepEP 的
    # combine 要把 combined_topk_weights 还原成 topk_weights（test_ep.py:514）——它不能是朴素求和。
    e_local = E // world
    expert_dst = topk_idx // e_local                       # [T,K] 每个槽的目的 rank（-1 槽忽略）
    valid = topk_idx >= 0
    deg = torch.zeros(T, dtype=torch.float32, device=DEV)
    for d in range(world):
        deg += ((expert_dst == d) & valid).any(dim=1).float()
    rec = torch.zeros(N_global, K, device=DEV)
    if Nr > 0:
        rec.index_add_(0, recv_src.long(), recv_w)
    dist.all_reduce(rec, op=dist.ReduceOp.SUM)
    rec_w = rec[rank * num_max: rank * num_max + T]
    expected = topk_weights.float() * deg.unsqueeze(1)
    b3 = bool(torch.allclose(rec_w, expected, atol=1e-5, rtol=1e-4))
    b3_deg = float(deg.mean().item())

    if rank == 0:
        print(f'### Part B：{world}×H100 真机 NCCL（真实 refs.dispatch，T={T}, H={H}, E={E}, K={K}）')
        print(f'  本 rank 收到 {Nr} 行（各来源 rank 合计）')
        print(f'  [{"PASS" if b1 else "FAIL"}] B1 跨 rank <D(x),g> == <x,Dᵀ(g)>  '
              f'LHS={lhs_all.item():.4f} RHS={rhs_all.item():.4f}')
        print(f'  [{"PASS" if b2 else "FAIL"}] B2 跨 rank <G(w),gw> == <w,Gᵀ(gw)>  '
              f'LHS={lw_all.item():.4f} RHS={rw_all.item():.4f}')
        print(f'  [{"PASS" if b3 else "FAIL"}] B3 非 expand 按目的 rank 复制：Σ recv_w == degree·w '
              f'（平均 degree={b3_deg:.3f}）')
        print()
    return b1 and b2 and b3


# --------------------------------------------------------------------------------------
def worker(rank, world):
    os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl', rank=rank, world_size=world,
                            timeout=__import__('datetime').timedelta(seconds=180))
    (math_m, refs), (h_refs, h_math) = load_real_modules()
    if rank == 0:
        print('### Part 0：真实源码加载（sha256）')
        print(f'  refs.py sha256={h_refs[:16]}…  dispatch:10-123 / combine:177-243 / ordered_accumulate:156-174')
        print(f'  math.py sha256={h_math[:16]}…')
        print(f'  RefEP@93eb6eb  README.md:158-289（训练/推理示例）+ ep.py:562-843（dispatch/combine 签名）')
        print()
        ok_a = part_a(refs)
    else:
        ok_a = True
    ok_b = part_b(rank, world, refs)
    dist.barrier()
    if rank == 0:
        print(f'### 总判定：Part A {"PASS" if ok_a else "FAIL"} · Part B {"PASS" if ok_b else "FAIL"} · '
              f'ALL {"PASS" if (ok_a and ok_b) else "FAIL"}')
    dist.destroy_process_group()


if __name__ == '__main__':
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = str(20000 + (os.getpid() % 20000))
    os.environ.setdefault('NCCL_NVLS_ENABLE', '0')
    os.environ.setdefault('NCCL_DEBUG', 'WARN')
    print(f'torch {torch.__version__}  world={WORLD}')
    print('MoE dispatch/combine 的反向（伴随）语义与路由权重梯度 —— H100 真机')
    print('=' * 96)
    if WORLD > 1:
        mp.spawn(worker, args=(WORLD,), nprocs=WORLD, join=True)
    else:
        worker(0, 1)
