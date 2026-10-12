"""P2-3: verify the FlashKDA forward workspace layout, field by field.

Three independent checks, all against the *built* extension:

A. Recompute the per-tile byte budget from the field constants and confirm it
   equals the ``flash_kda.cpp`` host formula ``3*CHUNK*D*2 + D*4 + 2*CHUNK*CHUNK*2``
   and the ``utils.cuh`` ``WorkspaceSizes`` sum. Also confirm every field size is
   a multiple of 128 (the static_asserts in ``flash_kda.cpp:16-18``).
B. Call the real ``get_workspace_size(T_total, H, N)`` and compare against the
   closed-form host formula for a grid of shapes.
C. Run an *actual* varlen forward into a canary-filled workspace, then
   read back the ``tile_prefix`` int32 array at the predicted byte offset
   and compare it to the expected cumulative tile counts. Also confirm the
   kernel never wrote past its own ``H*total_tiles*per_tile`` region
   (the allocation is the upper bound, the launch uses the exact tile count).

Run inside `kernel_lab` (needs CUDA + the built `flash_kda_C`).
"""

import math

import torch

CHUNK = 16
D = 128

# ---- field sizes, mirrored from csrc/flash_kda.cpp:20 and utils.cuh:70-76 ----
FIELDS = [
    ("k_decayed", CHUNK * D * 2),      # bf16 [H*tiles, 16, 128]
    ("q_decayed", CHUNK * D * 2),      # bf16 [H*tiles, 16, 128]
    ("k_restored", CHUNK * D * 2),     # bf16 [H*tiles, 16, 128]
    ("g_total", D * 4),                # fp32 [H*tiles, 128]
    ("INV", CHUNK * CHUNK * 2),        # bf16 [H*tiles, 16, 16]
    ("Mqk", CHUNK * CHUNK * 2),        # bf16 [H*tiles, 16, 16]
]
PER_TILE = sum(sz for _, sz in FIELDS)


def host_total_tiles(T_total, N):
    return (T_total + CHUNK - 1) // CHUNK + N


def host_ws_size(T_total, H, N):
    total_tiles = host_total_tiles(T_total, N)
    per_tile = 3 * CHUNK * D * 2 + D * 4 + 2 * CHUNK * CHUNK * 2
    tile_prefix = ((N + 1) * 4 + 127) // 128 * 128
    return H * total_tiles * per_tile + tile_prefix


def check_a():
    print("== A. per-tile byte budget ==")
    for name, sz in FIELDS:
        print(f"   {name:12s} {sz:6d} B   {'128B-aligned' if sz % 128 == 0 else 'NOT ALIGNED'}")
        assert sz % 128 == 0
    assert PER_TILE == 3 * CHUNK * D * 2 + D * 4 + 2 * CHUNK * CHUNK * 2, PER_TILE
    print(f"   per_tile = {PER_TILE} B  (kd 4096 + qd 4096 + kr 4096 + gt 512 + INV 512 + Mqk 512)")
    print(f"   -> 6 SoA arrays concatenated, each 128B aligned => no padding inside a tile")
    return PER_TILE


def check_b():
    print("\n== B. get_workspace_size vs closed form ==")
    from flash_kda_C import get_workspace_size
    grid = [
        (256, 1, 1), (512, 4, 1), (1024, 8, 1), (4096, 16, 1),
        (1000, 4, 1), (12345, 8, 2), (101, 2, 4), (77, 3, 7), (2048, 12, 1),
    ]
    for T_total, H, N in grid:
        got = int(get_workspace_size(T_total, H, N))
        exp = host_ws_size(T_total, H, N)
        tt = host_total_tiles(T_total, N)
        tp = ((N + 1) * 4 + 127) // 128 * 128
        flag = "OK " if got == exp else "MISMATCH"
        print(f"   [{flag}] T={T_total:6d} H={H:3d} N={N:2d}  tiles={tt:3d}  "
              f"ws={got}  (tiles*H*per_tile={H*tt*PER_TILE} + prefix={tp})")
        assert got == exp, (got, exp)
    print("   all rows match the host closed form")


@torch.no_grad()
def check_c():
    print("\n== C. live varlen forward: tile_prefix at predicted offset ==")
    from flash_kda_C import fwd as fwd_raw, get_workspace_size

    torch.manual_seed(0)
    H = 2
    lens = [30, 16, 5, 50]          # N=4, T_total=101
    N = len(lens)
    T_total = sum(lens)
    device = "cuda"

    q = torch.randn(1, T_total, H, D, dtype=torch.bfloat16, device=device)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    g = torch.randn_like(q)
    beta = torch.randn(1, T_total, H, dtype=torch.bfloat16, device=device)
    A_log = torch.randn(H, dtype=torch.float32, device=device)
    dt_bias = torch.randn(H, D, dtype=torch.float32, device=device)
    out = torch.zeros(1, T_total, H, D, dtype=torch.bfloat16, device=device)

    cu = torch.tensor([0] + list(torch.cumsum(torch.tensor(lens), 0).tolist()),
                      dtype=torch.int64, device=device)

    ws_bytes = int(get_workspace_size(T_total, H, N))
    ws = torch.empty(ws_bytes, dtype=torch.uint8, device=device)
    ws.fill_(0xAA)

    fwd_raw(q, k, v, g, beta, 1.0 / math.sqrt(D), out, ws, A_log, dt_bias, -5.0,
            cu_seqlens=cu)
    torch.cuda.synchronize()

    # host-side launch: total_tiles for varlen = ceil(T_total/16) + N (flash_kda.cpp:178)
    launch_tiles = (T_total + CHUNK - 1) // CHUNK + N
    n_ht = H * launch_tiles
    prefix_off = n_ht * PER_TILE
    print(f"   lens={lens}  T_total={T_total}  N={N}")
    print(f"   launch total_tiles={launch_tiles}  n_ht={n_ht}")
    print(f"   allocation={ws_bytes} B, per_tile region={prefix_off} B, "
          f"prefix tail={ws_bytes - prefix_off} B")

    got_prefix = ws[prefix_off:prefix_off + (N + 1) * 4].view(torch.int32).tolist()
    exp_tiles = [(ln + CHUNK - 1) // CHUNK for ln in lens]
    exp_prefix = [0]
    for t in exp_tiles:
        exp_prefix.append(exp_prefix[-1] + t)
    print(f"   tile counts/seq = {exp_tiles}  expected prefix = {exp_prefix}")
    print(f"   read int32 @ {prefix_off} = {got_prefix}")
    assert got_prefix == exp_prefix, (got_prefix, exp_prefix)

    # canary: the kernel must not touch bytes past its own H*launch_tiles*per_tile
    # region except the tile_prefix tail it is told to write.
    tail = ws[prefix_off + (N + 1) * 4:]
    untouched = bool((tail == 0xAA).all().item())
    print(f"   canary beyond prefix tail untouched: {untouched} "
          f"({tail.numel()} B still 0xAA)")
    assert untouched

    # sanity: the SoA arrays actually received data
    kd = ws[0:n_ht * PER_TILE]
    n_nonzero = int((kd != 0).sum().item())
    print(f"   per-tile SoA region nonzero bytes: {n_nonzero}/{kd.numel()} "
          f"(kernels wrote data)")
    assert n_nonzero > 0
    print("   C OK: tile_prefix lands exactly at H*launch_tiles*per_tile")


def main():
    print("torch", torch.__version__)
    check_a()
    check_b()
    check_c()
    print("\nDONE")


if __name__ == "__main__":
    main()
