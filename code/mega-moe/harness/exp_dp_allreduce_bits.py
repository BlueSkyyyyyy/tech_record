#!/usr/bin/env python3
"""P175-1: 数据并行（DP）梯度 all-reduce 的「逐位一致性」——真实 NCCL、H100 多卡实测。

问题：第 184 篇（P174-1，docs/184）把 MoE 反向梯度累积的「切分 × 结合序」量到「wgrad 对结合序
逐位敏感、dgrad 免疫」，并在最后写下「DP 的 ring-reduce（顺序）与 tree-reduce（树形）在 fp32 下
给出不同 bit」——但那只是**纯 torch 把 8 份偏梯度手动相加**的模拟，不是真实集合通信。第 170 篇
（docs/180）也留了口子：把结论推到真实多卡。

这一篇把这一步补上：在 H100 上用 **真实 NCCL**（torch.distributed）做 DP 的梯度归约，回答三个问题：

  Part A  同一配置重跑 N 次，all-reduce 结果逐位一致吗？（NCCL 的 run-to-run 确定性）
  Part B  换 NCCL 算法（Ring / Tree / NVLS）bit 变吗？（算法 = 归约结合序）
  Part C  world size（2/4）变、同一逻辑和，bit 变吗？与 fp64 真值差多少？
  Part D  all_reduce vs ZeRO 的 reduce_scatter + all_gather，bit 一致吗？
  Part E  真实 NCCL 落在「手动顺序链（sequential）/ 两两树（pairwise-tree）/ 逐 rank 旋转」
          的哪一个上？——把第 184 篇的规律在真实通信上复现。

来源事实：DeepEP 两条跨-rank 归约通路正好是两个极端——
  * EP combine 的 token 归约在 kernel 内按 top-k 槽位**固定顺序**做、且以 **bf16** 累加
    （`deep_ep/include/deep_ep/impls/ep/combine_utils.cuh:107/133-151`、`combine_reduce_epilogue.cuh:101`）；
  * EPLB 冗余专家梯度归约用硬件 `tma_store_reduce_add_f32`（`.../impls/ep/reduce_grads.cuh:82`），
    即 **red.add**——顺序由硬件/到达时序决定。
  而 DP 的稠密梯度 all-reduce 走 NCCL（本篇）。

运行：bash scripts/lab.sh exec "python /abs/path/harness/exp_dp_allreduce_bits.py"
输出重定向到 exp_dp_allreduce_bits.out.txt。
"""
import os
import sys
import socket

import torch
import torch.distributed as dist


# --------------------------------------------------------------------------- helpers
def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]


def relerr(a, b):
    a, b = a.double(), b.double()
    return ((a - b).abs().mean() / (b.abs().mean() + 1e-30)).item()


def bitdiff(x, y):
    return int((x != y).sum().item())


def make_partial(rank, n, gen):
    """每个 rank 的『本地梯度偏和』——同一个全局逻辑张量按 DP 切给的 rank 片段。"""
    g = torch.Generator(device='cpu').manual_seed(1000 * gen + rank)
    return torch.randn(n, generator=g, dtype=torch.float32)


# --------------------------------------------------------------------------- worker
def _worker(rank, world, algo, proto, size, out_dir, deterministic):
    os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
    os.environ.setdefault('MASTER_PORT', str(find_free_port()))
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world)
    if algo:
        os.environ['NCCL_ALGO'] = algo
    if proto:
        os.environ['NCCL_PROTO'] = proto
    if deterministic:
        os.environ['NCCL_DETERMINISTIC'] = '1'

    torch.cuda.set_device(rank)
    dist.init_process_group('nccl', rank=rank, world_size=world)
    dev = torch.device('cuda', rank)

    # 固定输入：rank r 的偏和由全局种子决定，与 world 无关（同一逻辑数据切成 world 份）
    part = make_partial(rank, size, 0).to(dev)

    results = {}

    # ---- Part A: 重跑确定性（同一输入、同一算法，20 次）
    base = None
    max_diff = 0
    for _ in range(20):
        buf = part.clone()
        dist.all_reduce(buf, op=dist.ReduceOp.SUM)
        if base is None:
            base = buf.clone()
        else:
            max_diff = max(max_diff, bitdiff(base, buf))
    results['rerun_20_max_bitdiff'] = max_diff
    results['reduced'] = base.cpu()

    # reduce_scatter + all_gather（ZeRO 梯度归约的等价结构）——**集合操作，所有 rank 都要调**。
    # 注意：`NCCL_ALGO=Tree` 不支持 AllGather（会 `ncclInvalidUsage`），故 Tree 配置跳过此项。
    rs_ag = None
    if algo != 'Tree':
        chunk = size // world
        rs_in = part.clone().reshape(world, chunk)
        rs_out = torch.empty(chunk, device=dev, dtype=torch.float32)
        dist.reduce_scatter_tensor(rs_out, rs_in.contiguous(), op=dist.ReduceOp.SUM)
        ag = [torch.empty(chunk, device=dev, dtype=torch.float32) for _ in range(world)]
        dist.all_gather(ag, rs_out)
        rs_ag = torch.cat(ag).cpu()

    if rank == 0:
        # 每个 rank 的偏和由确定性种子给出，rank0 本地重建即可（无需 all_gather）
        stack = torch.stack([make_partial(r, size, 0).double() for r in range(world)])
        results['partials'] = stack.float()
        results['fp64_sum'] = stack.sum(0)

        # Part E 对照：手动顺序链 / 两两树 / 逐 rank 旋转
        def seq_sum(order):
            acc = results['partials'][order[0]].clone()
            for i in order[1:]:
                acc = acc + results['partials'][i]
            return acc

        def tree_sum(ps):
            cur = [p.clone() for p in ps]
            while len(cur) > 1:
                nxt = [cur[i] + cur[i + 1] for i in range(0, len(cur) - 1, 2)]
                if len(cur) % 2:
                    nxt.append(cur[-1])
                cur = nxt
            return cur[0]

        results['manual_seq'] = seq_sum(list(range(world)))
        results['manual_rot1'] = seq_sum([(1 + i) % world for i in range(world)])
        results['manual_tree'] = tree_sum([results['partials'][i] for i in range(world)])
        if rs_ag is not None:
            results['reduce_scatter_allgather'] = rs_ag

    dist.barrier()
    if rank == 0:
        torch.save(results, os.path.join(out_dir, f'algo={algo}_proto={proto}_w={world}_det={int(deterministic)}.pt'))
    dist.destroy_process_group()


def launch(world, algo, proto, size, out_dir, deterministic=False):
    port = find_free_port()
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = str(port)
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '1,2,3,4')
    # 本机 NVSwitch fabric manager 未启用 → NVLS multicast 建不起来；
    # 关掉 NVLS 让默认路径退到 Ring/Tree（正文 Part B 会说明这一环境事实）。
    os.environ.setdefault('NCCL_NVLS_ENABLE', '0')
    tag = f'algo={algo}_proto={proto}_w={world}_det={int(deterministic)}.pt'
    path = os.path.join(out_dir, tag)
    if os.path.exists(path):
        os.remove(path)
    import torch.multiprocessing as mp
    mp.spawn(_worker, nprocs=world,
             args=(world, algo, proto, size, out_dir, deterministic), join=True)
    return torch.load(path, weights_only=False)


# --------------------------------------------------------------------------- parent
def main():
    out_dir = '/tmp/dp_allreduce_bits'
    os.makedirs(out_dir, exist_ok=True)
    size = 1 << 20  # 1M fp32 = 4 MB，模拟一个展平的梯度桶

    print(f'torch={torch.__version__} cuda={torch.version.cuda} nccl={torch.cuda.nccl.version()}')
    print(f'n_gpus={torch.cuda.device_count()}  size={size} (~{size*4/2**20:.0f} MB)')
    print('CUDA_VISIBLE_DEVICES=', os.environ.get('CUDA_VISIBLE_DEVICES'))

    configs = [
        (2, None, None),           # 默认算法
        (2, 'Ring', 'Simple'),
        (2, 'Tree', 'Simple'),
        (2, 'Ring', 'LL'),
        (4, None, None),
        (4, 'Ring', 'Simple'),
        (4, 'Tree', 'Simple'),
    ]
    # 注：本机 NVSwitch fabric manager 未启用 → NVLS multicast 建不起来（NCCL_NVLS_ENABLE=0），
    # 故不测 NVLS，只对比 Ring / Tree 两条算法（正文 §2 会说明该环境事实）。
    results = {}
    for w, algo, proto in configs:
        try:
            r = launch(w, algo, proto, size, out_dir)
            results[(w, algo or 'default', proto or 'default')] = r
            print(f'  ran w={w} algo={algo} proto={proto}')
        except Exception as e:
            print(f'  SKIP w={w} algo={algo}: {type(e).__name__}: {str(e)[:120]}')

    print('\n=== Part A: 重跑确定性（同输入同算法 all_reduce 20 次）===')
    for (w, algo, proto), r in results.items():
        rd = r['reduced']
        print(f'  w={w} algo={algo:8s} proto={proto:8s}: 20 次之间最大逐位差 = '
              f'{r["rerun_20_max_bitdiff"]}  (0 = run-to-run 确定)')

    # 参考：fp64 真值（用 w=2 的 partials 当作全量逻辑向量；不同 world 的 partials 不同切片，
    #   这里各配置单独和自己的 fp64 比）。
    print('\n=== Part B/C: 与 fp64 真值 / 各算法之间 ===')
    for (w, algo, proto), r in results.items():
        ref = r['fp64_sum']
        n = r['reduced'].numel()
        d = bitdiff(r['reduced'], ref.float())
        print(f'  w={w} algo={algo:8s}: vs fp64 逐位差 = {d}/{n}, relerr = {relerr(r["reduced"], ref):.3e}')

    # 同 world 内不同算法两两对比
    print('\n  -- 同 world、不同算法/协议之间逐位对比 --')
    for w in (2, 4):
        keys = [k for k in results if k[0] == w]
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                a = results[keys[i]]['reduced']
                b = results[keys[j]]['reduced']
                print(f'  w={w} {keys[i][1]:8s} vs {keys[j][1]:8s}: '
                      f'逐位差={bitdiff(a, b):7d}/{a.numel()} ({100.0*bitdiff(a,b)/a.numel():5.1f}%), '
                      f'relerr={relerr(a, b):.3e}')

    print('\n=== Part D: all_reduce vs reduce_scatter+all_gather（ZeRO 结构）===')
    for (w, algo, proto), r in results.items():
        if 'reduce_scatter_allgather' not in r:
            continue
        a, b = r['reduced'], r['reduce_scatter_allgather']
        print(f'  w={w} algo={algo:8s}: all_reduce vs rs+ag: '
              f'逐位差={bitdiff(a, b):7d}/{a.numel()} ({100.0*bitdiff(a,b)/a.numel():5.1f}%), '
              f'relerr={relerr(a, b):.3e}')

    print('\n=== Part E: 真实 NCCL 落在哪种手动归约序上（复现第 184 篇规律）===')
    for (w, algo, proto), r in results.items():
        if 'manual_seq' not in r:
            continue
        for name in ('manual_seq', 'manual_rot1', 'manual_tree'):
            ref = r[name]
            d = bitdiff(r['reduced'], ref)
            print(f'  w={w} algo={algo:8s}: all_reduce vs {name:20s}: '
                  f'逐位差={d}/{ref.numel()} ({100.0*d/ref.numel():5.1f}%), '
                  f'relerr={relerr(r["reduced"], ref):.3e}')
        # 手动序之间也互比，给出「结合序本身改 bit」的幅度
        print(f'  w={w}: manual_seq vs manual_tree: 逐位差='
              f'{bitdiff(r["manual_seq"], r["manual_tree"])}/{r["manual_seq"].numel()}, '
              f'relerr={relerr(r["manual_seq"], r["manual_tree"]):.3e}')
        print(f'  w={w}: manual_seq vs manual_rot1: 逐位差='
              f'{bitdiff(r["manual_seq"], r["manual_rot1"])}/{r["manual_seq"].numel()}, '
              f'relerr={relerr(r["manual_seq"], r["manual_rot1"]):.3e}')

    print('\n=== Part F: NCCL_DETERMINISTIC=1 是否改变结果 ===')
    for w in (2, 4):
        try:
            rd = launch(w, 'Ring', 'Simple', size, out_dir, deterministic=True)
            rn = results.get((w, 'Ring', 'Simple'))
            if rn is not None:
                a, b = rn['reduced'], rd['reduced']
                print(f'  w={w} Ring/Simple: DET=0 vs DET=1 逐位差='
                      f'{bitdiff(a, b)}/{a.numel()}, relerr={relerr(a, b):.3e}, '
                      f'(DET=1 重跑逐位差={rd["rerun_20_max_bitdiff"]})')
        except Exception as e:
            print(f'  w={w} DET=1 SKIP: {type(e).__name__}: {str(e)[:100]}')

    print('\n=== 判定 ===')
    print('  A: NCCL all_reduce 对固定输入 run-to-run 是否逐位确定，见上。')
    print('  B/C: 换算法（Ring/Tree）或 world size 是否改 bit。')
    print('  D: all_reduce 与 reduce_scatter+all_gather 是否同 bit（ZeRO 混合并行可复现性）。')
    print('  E: 真实 NCCL 命中哪种手动序——是否复现「结合序改 bit」。')


if __name__ == '__main__':
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
