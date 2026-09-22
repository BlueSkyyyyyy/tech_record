#!/usr/bin/env python3
# 从真实的 DeepSeek-V4-Pro checkpoint 里取出 routed expert 的 FP4 权重，落成裸
# 二进制供 CUDA 程序读取。同时验证：e2m1 权重 + E8M0(2 的幂) block-32 scale
# 折进 e4m3 是**逐位无损**的。
#
#   python3 extract.py [layer] [expert]
#
# 产物（放在本目录）：
#   w1_fp4.bin   [I, H/2]      uint8  两 nibble/byte（e2m1）
#   w1_scale.bin [I, H/32]     uint8  E8M0 指数（value = 2^(b-127)）
#   w1_fp8.bin   [I, H]        uint8  精确折叠后的 e4m3
#   w1_bf16.bin  [I, H]        bf16   折叠后的 bf16（对照读带宽用）
#   w2/w3 同上（w2 形状 [H, I]，w3 [I, H]）
import sys, os, numpy as np, torch
from safetensors import safe_open

CKPT = "/ssd/models/DeepSeek-V4-Pro/model-00002-of-00064.safetensors"
H = 7168
I = 3072
OUT = os.path.dirname(os.path.abspath(__file__))

# e2m1 code -> value
def e2m1_table():
    t = []
    for n in range(16):
        s = (n >> 3) & 1; e = (n >> 1) & 3; m = n & 1
        v = (m * 0.5) if e == 0 else (1.0 + m * 0.5) * (2.0 ** (e - 1))
        t.append(-v if s else v)
    return np.array(t, dtype=np.float32)

def unpack_fp4(packed):
    # packed: [R, C] uint8 -> [R, 2C] float32
    lo = (packed & 0x0F).astype(np.int64)
    hi = ((packed >> 4) & 0x0F).astype(np.int64)
    t = torch.from_numpy(e2m1_table())
    lof = t[torch.from_numpy(lo)].numpy()
    hif = t[torch.from_numpy(hi)].numpy()
    out = np.empty((packed.shape[0], packed.shape[1] * 2), np.float32)
    out[:, 0::2] = lof
    out[:, 1::2] = hif
    return out

def fold_fp8(recon):
    # recon 已经含 scale（= e2m1 值 × 2^e）。e2m1 ⊂ e4m3，且 ×2^e 只是指数平移
    # （e<0 时可能落到 e4m3 的次正规），所以 fp8(recon) 逐位无损。
    return torch.from_numpy(recon).to(torch.float8_e4m3fn).view(torch.uint8).numpy()

def process(name, layer, expert, out_shape_hw):
    key = f"layers.{layer}.ffn.experts.{expert}.{name}"
    with safe_open(CKPT, framework="pt", device="cpu") as f:
        w = f.get_tensor(key + ".weight").numpy()          # int8 packed
        s = f.get_tensor(key + ".scale").view(torch.uint8).numpy()   # e8m0 raw bytes
    deq = unpack_fp4(w)                                    # [R, 2C]
    # expand scale per-32
    sexp = np.repeat(s.astype(np.int32), 32, axis=1)
    assert sexp.shape[1] == deq.shape[1], (sexp.shape, deq.shape)
    recon = deq * np.exp2(sexp - 127.0)
    fp8 = fold_fp8(recon)
    bf16 = torch.from_numpy(recon).to(torch.bfloat16).view(torch.uint8).numpy()
    for suf, arr in [("fp4", w.astype(np.uint8)), ("scale", s.astype(np.uint8)),
                     ("fp8", fp8), ("bf16", bf16)]:
        fn = os.path.join(OUT, f"{name}_{suf}.bin")
        arr.tofile(fn)
    return recon, w, s

if __name__ == "__main__":
    layer = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    expert = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    tbl = e2m1_table()
    print("e2m1 codes:", [f"{v:g}" for v in tbl])
    for name in ["w1", "w3", "w2"]:
        recon, w, s = process(name, layer, expert, None)
        # 无损性检查：折叠后的 fp8 反量化回 fp32 == recon？
        fn = os.path.join(OUT, f"{name}_fp8.bin")
        fp8 = np.fromfile(fn, dtype=np.uint8)
        # 反量化 fp8
        back = torch.from_numpy(fp8).view(torch.float8_e4m3fn).to(torch.float32).numpy()
        rel = np.sqrt(((back - recon.reshape(-1)) ** 2).mean()) / np.sqrt((recon ** 2).mean())
        mx = np.abs(back - recon.reshape(-1)).max()
        print(f"{name}: packed{w.shape} scale{s.shape} recon_rms={np.sqrt((recon**2).mean()):.4e} "
              f"fp8_fold_relRMS={rel:.2e} maxabs={mx:.2e}")
