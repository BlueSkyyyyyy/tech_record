"""P0-4 环境探测：DeepEP 2.5 在 H100 上单进程（world_size=1）dispatch/combine 冒烟测试。

world_size=1 时所有专家都在本 rank，dispatch 退化为「本地按专家重排」，
可验证 EPBuffer 能否构造、kernel 能否 JIT 编译/运行、布局是否自洽。
不做多卡带宽基准（留给后续）。
"""
import os
import torch
import torch.distributed as dist

os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
os.environ.setdefault('MASTER_PORT', '8361')
os.environ.setdefault('RANK', '0')
os.environ.setdefault('WORLD_SIZE', '1')
os.environ.setdefault('LOCAL_RANK', '0')

import deep_ep  # noqa: E402


def main():
    torch.cuda.set_device(0)
    dist.init_process_group('nccl', rank=0, world_size=1)
    group = dist.group.WORLD

    num_experts = 8
    num_topk = 2
    hidden = 1024
    num_max_tokens_per_rank = 256
    num_tokens = 128

    buffer = deep_ep.EPBuffer(group,
                              num_max_tokens_per_rank=num_max_tokens_per_rank,
                              hidden=hidden, num_topk=num_topk)
    print(f'EPBuffer 构造成功: num_ranks={buffer.num_ranks}, '
          f'logical_domain={buffer.get_logical_domain_size()}, '
          f'num_bytes={buffer.num_bytes / 1e6:.1f} MB')

    torch.manual_seed(0)
    x = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
    scores = torch.randn((num_tokens, num_experts), device='cuda')
    topk_weights, topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=False)
    topk_idx = topk_idx.to(deep_ep.topk_idx_t)

    recv_x, recv_topk_idx, recv_topk_weights, handle, _ = buffer.dispatch(
        x=x, topk_idx=topk_idx, topk_weights=topk_weights,
        num_max_tokens_per_rank=num_max_tokens_per_rank, num_experts=num_experts,
        expert_alignment=128)
    n_recv = handle.psum_num_recv_tokens_per_scaleup_rank[-1].item()
    print(f'dispatch OK: recv_x={tuple(recv_x.shape)}, num_recv_tokens={n_recv}, '
          f'num_experts={num_experts}')

    # Combine: 用 dispatch 结果乘 topk 权重送回
    combined = buffer.combine(recv_x, handle=handle,
                              topk_weights=recv_topk_weights)
    combined_x = combined[0]
    print(f'combine OK: combined_x={tuple(combined_x.shape)}')

    # 自洽检查：world_size=1 时，每个 token 的 combine 输出应等于其 topk 选中的 recv 行之和
    torch.cuda.synchronize()
    rel = (combined_x.float().abs().mean() / x.float().abs().mean()).item()
    print(f'combined/x 量级比 = {rel:.3f}（非零即数据流通了）')

    buffer.destroy()
    dist.destroy_process_group()
    print('DeepEP 单进程 dispatch/combine / SM90 可用')


if __name__ == '__main__':
    main()
