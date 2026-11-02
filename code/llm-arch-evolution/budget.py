#!/usr/bin/env python3
"""LLM 结构账本：把架构选择折算成三个数字。

    参数量 P（容量）、每 token FLOPs（算力）、每 token KV 字节数（带宽/容量）

这三个数字决定了训练和推理的成本形状。模型结构的每一次「搬家」——
GQA、MLA、稀疏注意力、线性注意力、MoE、Engram——
本质上都是在三个数字之间重新分配预算。

用法：
    python budget.py                 # 打印全部模型的账本表
    python budget.py --ctx 1048576   # 指定上下文长度（默认 1M）
    python budget.py --json          # 输出 JSON
    python budget.py --test          # 自测：与论文公布参数量对拍

所有公式与本文件末尾的 MODEL_* 注释里的出处一一对应。
纯标准库，CPU 可跑。
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# 1. 结构描述
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Attn:
    """一层注意力的结构。

    kind:
      - "gqa" : 经典 MHA/MQA/GQA，K/V 按 head 存
      - "mla" : DeepSeek-V2/V3 式低秩联合压缩 + decoupled RoPE
      - "mqa" : V4 式共享 KV（K=V=一条 latent，MQA 方式广播到所有 query head）
    """

    kind: str
    n_heads: int
    head_dim: int = 128          # GQA: q/k 每头维度；MLA/MQA: KV 条目维度
    n_kv_heads: int = 8          # 仅 gqa
    q_lora: int = 0              # MLA/MQA: query 低秩压缩维度
    kv_lora: int = 0             # MLA: KV 联合压缩维度
    qk_nope: int = 0             # MLA: 不带 RoPE 的 k 维度
    qk_rope: int = 0             # MLA/MQA: 带 RoPE 的维度
    v_head: int = 0              # MLA: 每头 v 维度
    o_lora: int = 0              # 分组输出投影的中间维度（0 = 普通 o_proj）
    o_groups: int = 1
    nope: bool = False           # NoPE：不存 RoPE key（Kimi-K3 的 MLA 全 NoPE）

    # --- 参数量（每层） ---
    def params(self, d: int) -> float:
        if self.kind == "gqa":
            q = d * self.n_heads * self.head_dim
            kv = 2 * d * self.n_kv_heads * self.head_dim
            o = self.n_heads * self.head_dim * d
            return q + kv + o
        if self.kind == "mla":
            q_a = d * self.q_lora
            q_b = self.q_lora * self.n_heads * (self.qk_nope + self.qk_rope)
            kv_a = d * (self.kv_lora + self.qk_rope)
            kv_b = self.kv_lora * self.n_heads * (self.qk_nope + self.v_head)
            return q_a + q_b + kv_a + kv_b + self._o_params(d)
        if self.kind == "mqa":
            # K = V = 一条 head_dim 的 latent；RoPE 作用在最后 qk_rope 维上
            q_a = d * self.q_lora
            q_b = self.q_lora * self.n_heads * self.head_dim
            kv_a = d * self.head_dim
            return q_a + q_b + kv_a + self._o_params(d)
        raise ValueError(self.kind)

    def _o_params(self, d: int) -> float:
        v_dim = self.v_head if self.kind == "mla" else self.head_dim
        if self.o_lora:
            # 组内先降到 o_lora，再拼起来升回 d（V4 的 grouped output projection）
            return self.n_heads * v_dim * self.o_lora + self.o_groups * self.o_lora * d
        return self.n_heads * v_dim * d

    # --- 注意力核心 FLOPs（每 token，单层；含 QK^T 与 PV） ---
    def core_flops(self, n_entries: float) -> float:
        """QK^T + PV：每个 query 对 n_entries 个条目各做一次点积。"""
        v_dim = self.v_head if self.kind == "mla" else self.head_dim
        return 2 * self.n_heads * (self.head_dim + v_dim) * n_entries

    @property
    def kv_entry_dim(self) -> int:
        if self.kind == "gqa":
            return 2 * self.n_kv_heads * self.head_dim
        if self.kind == "mla":
            return self.kv_lora + (0 if self.nope else self.qk_rope)
        return self.head_dim


@dataclass(frozen=True)
class MoE:
    n_routed: int
    top_k: int
    expert_inter: int
    n_shared: int = 0
    latent_dim: int = 0          # LatentMoE：routed expert 在低维空间工作（0 = 全宽）

    def params(self, d: int) -> float:
        w = self.latent_dim or d
        routed = self.n_routed * 3 * w * self.expert_inter
        shared = self.n_shared * 3 * d * self.expert_inter
        return routed + shared

    def active(self, d: int) -> float:
        w = self.latent_dim or d
        routed = self.top_k * 3 * w * self.expert_inter
        shared = self.n_shared * 3 * d * self.expert_inter
        return routed + shared


@dataclass(frozen=True)
class LinearAttn:
    """线性注意力层（KDA / GDN / Mamba-2）：状态恒定，不存 per-token KV。"""

    n_layers: int
    n_heads: int = 96
    head_dim: int = 128
    decay_rank: int = 128
    gate_full_rank: bool = True
    conv_kernel: int = 4

    def params_per_layer(self, d: int) -> float:
        h = self.n_heads * self.head_dim
        qkv = 3 * d * h
        o = h * d
        decay = d * self.decay_rank + self.decay_rank * h
        beta = d * self.n_heads
        conv = 3 * d * self.conv_kernel
        gate = h * d if self.gate_full_rank else d * self.decay_rank
        return qkv + o + decay + beta + conv + gate

    def params(self, d: int) -> float:
        return self.n_layers * self.params_per_layer(d)

    def flops_per_token(self, d: int) -> float:
        # 状态更新 + 读出：每个 head 一个 dk×dv 的状态矩阵读写
        return 2 * self.n_heads * self.head_dim * self.head_dim


@dataclass(frozen=True)
class Indexer:
    """稀疏注意力的索引器：每个 query 都要给可见条目打分。"""

    n_layers: int
    n_heads: int = 32
    head_dim: int = 128
    frac_ctx: float = 1.0        # 层次化索引器只扫候选池时为 <1

    def flops(self, ctx: float) -> float:
        return self.n_layers * 2 * self.n_heads * self.head_dim * ctx * self.frac_ctx


@dataclass(frozen=True)
class KVSource:
    """一组共享同一份 KV 的层（CSA2 的 Full 层）。

    n_sources: 这一档有几份独立的 KV 缓存（一份被若干层共享）
    compress : 每多少个 token 压成一个条目（1 = 不压）
    """

    n_sources: int
    compress: float = 1.0
    layers: int = 0              # 使用这份 KV 的层数（0 = 等于 n_sources）
    indexer_k_bytes: float = 0.0 # 索引器 K 的每条目字节（V4.1: 128 维 FP4 + E8M0 scale = 68 B）


@dataclass
class Spec:
    name: str
    d: int
    layers: int
    vocab: int
    attn: Attn
    moe: Optional[MoE] = None
    n_dense_layers: int = 0
    dense_inter: int = 0
    tied: bool = True
    hc_mult: int = 1             # mHC 残差流条数（1 = 普通残差）
    extra_params: float = 0      # Engram 等外挂参数
    kv_sources: tuple[KVSource, ...] = ()
    kv_bits: float = 16.0        # 全局 KV 的位宽
    kv_scale_bits_per_16: float = 0.0  # 每 16 个元素一个 scale 的额外位宽
    sparse_topk: int = 0         # >0：每个 query 只看 top-k（+ 滑窗）个条目
    swa_window: int = 0
    linear: Optional[LinearAttn] = None
    kv_bytes_override: Optional[float] = None   # 官方口径的 KV 字节数（含索引器等）
    indexer: Optional[Indexer] = None
    active_override: Optional[float] = None
    active_prefill_override: Optional[float] = None

    # --- 参数量 ---
    def params_ffn(self) -> float:
        p = 0.0
        if self.n_dense_layers:
            p += self.n_dense_layers * 3 * self.d * self.dense_inter
        if self.moe:
            p += (self.layers - self.n_dense_layers) * self.moe.params(self.d)
        return p

    def params_attn(self) -> float:
        p = (self.layers - (self.linear.n_layers if self.linear else 0)) * self.attn.params(self.d)
        if self.linear:
            p += self.linear.params(self.d)
        return p

    def params_hc(self) -> float:
        if self.hc_mult == 1:
            return 0.0
        # 每个 block 两个 HyperConnection（attn / mlp 各一），外加末尾一个 HyperHead。
        # fn: [(2+hc)*hc, hc*d]；base: (2+hc)*hc；scale: 3
        hc = self.hc_mult
        per = (2 + hc) * hc * hc * self.d + (2 + hc) * hc + 3
        head = hc * hc * self.d + hc + 1
        return self.layers * 2 * per + head

    def params_total(self) -> float:
        emb = self.vocab * self.d * (1 if self.tied else 2)
        return emb + self.params_attn() + self.params_ffn() + self.params_hc() + self.extra_params

    def params_active(self) -> float:
        if self.active_override is not None:
            return self.active_override
        emb = self.vocab * self.d  # 输入 embedding 每 token 只查一行
        attn = self.params_attn()
        ffn = 0.0
        if self.n_dense_layers:
            ffn += self.n_dense_layers * 3 * self.d * self.dense_inter
        if self.moe:
            ffn += (self.layers - self.n_dense_layers) * self.moe.active(self.d)
        # 输出 head 每 token 都要算：vocab*d
        head = 0 if self.tied else self.vocab * self.d
        return emb + attn + ffn + head

    # --- KV ---
    @property
    def kv_entry_bytes(self) -> float:
        bits_per_entry = self.attn.kv_entry_dim * self.kv_bits
        # scale 开销：每 16 个元素一个 fp8 scale
        bits_per_entry += (self.attn.kv_entry_dim / 16) * self.kv_scale_bits_per_16
        return bits_per_entry / 8

    def kv_bytes_per_token(self) -> float:
        """KV 常驻容量：每个 token 摊到的字节数（压缩后）。"""
        if not self.kv_sources:
            return 0.0
        return sum(src.n_sources * (self.kv_entry_bytes + src.indexer_k_bytes) / src.compress
                   for src in self.kv_sources)

    def kv_read_per_token(self, ctx: float) -> float:
        """decode 一步每个 token 实际从 HBM 读的 KV 字节。

        稀疏注意力把「容量」和「带宽」解耦：容量仍是 O(ctx)，但每步只读 top-k 个条目。
        """
        if not self.kv_sources:
            return 0.0
        if self.sparse_topk:
            per_entry = min(self.sparse_topk + self.swa_window, ctx)
        else:
            per_entry = ctx
        total = 0.0
        for src in self.kv_sources:
            n_layers = src.layers or src.n_sources
            # 压缩后可见条目数 = ctx / compress，与 top-k 取小
            n = min(per_entry, ctx / src.compress) if self.sparse_topk else ctx
            total += n_layers * n * self.kv_entry_bytes
        return total

    # --- FLOPs（前向，每 token）---
    def flops_linear(self, prefill: bool = False) -> float:
        """非注意力矩阵乘：2 × 激活参数量（不含 embedding 查表）。"""
        active = self.params_active()
        if prefill and self.active_prefill_override is not None:
            active = self.active_prefill_override
        return 2 * (active - self.vocab * self.d - (0 if self.tied else self.vocab * self.d))

    def eff_entries(self, ctx: float) -> float:
        """每个 query 实际打分的 KV 条目数：稠密=ctx，稀疏=top-k+滑窗，线性层=0。"""
        if self.sparse_topk:
            return min(ctx, self.sparse_topk + self.swa_window)
        return ctx

    def flops_attn(self, ctx: float) -> float:
        n_dense = self.layers - (self.linear.n_layers if self.linear else 0)
        f = n_dense * self.attn.core_flops(self.eff_entries(ctx))
        if self.linear:
            f += self.linear.n_layers * self.linear.flops_per_token(self.d)
        return f

    def flops_prefill(self, ctx: float) -> float:
        """prefill：线性部分 + 注意力 + 索引器。"""
        f = self.flops_linear(prefill=True) + self.flops_attn(ctx)
        if self.indexer:
            f += self.indexer.flops(ctx)
        return f

    def flops_decode(self, ctx: float) -> float:
        return self.flops_prefill(ctx)

    def decode_bytes(self, ctx: float) -> float:
        """decode 每步从 HBM 读的字节：权重（激活部分）+ 实际读到的 KV。"""
        return self.params_active() * 2 + self.kv_read_per_token(ctx)

    def arithmetic_intensity(self, ctx: float) -> float:
        b = self.decode_bytes(ctx)
        return self.flops_decode(ctx) / b if b else 0.0


# ---------------------------------------------------------------------------
# 2. 真实模型配置（数字来源见每个条目的注释）
# ---------------------------------------------------------------------------

MODEL_SPECS: list[Spec] = []


def add(spec: Spec) -> Spec:
    MODEL_SPECS.append(spec)
    return spec


# Llama-3.1-405B：Meta 官方模型卡 / 技术报告
add(Spec(
    name="Llama-3.1-405B", d=16384, layers=126, vocab=128256,
    attn=Attn("gqa", n_heads=128, head_dim=128, n_kv_heads=8),
    n_dense_layers=126, dense_inter=53248, tied=False,
    kv_sources=(KVSource(126, 1.0),), kv_bits=16,
))

# Qwen3-32B：本地 config.json（~/models/Qwen3-32B）
add(Spec(
    name="Qwen3-32B", d=5120, layers=64, vocab=151936,
    attn=Attn("gqa", n_heads=64, head_dim=128, n_kv_heads=8),
    n_dense_layers=64, dense_inter=25600, tied=False,
    kv_sources=(KVSource(64, 1.0),), kv_bits=16,
))

# Qwen3-30B-A3B：128 专家 top-8，expert inter 768
add(Spec(
    name="Qwen3-30B-A3B", d=2048, layers=48, vocab=151936,
    attn=Attn("gqa", n_heads=32, head_dim=128, n_kv_heads=4),
    moe=MoE(n_routed=128, top_k=8, expert_inter=768, n_shared=0),
    tied=False,
    kv_sources=(KVSource(48, 1.0),), kv_bits=16,
))

# DeepSeek-V3：671B / 37B 激活，MLA(1536,512,128+64,128)，256 专家 top-8 + 1 共享
add(Spec(
    name="DeepSeek-V3", d=7168, layers=61, vocab=129280,
    attn=Attn("mla", n_heads=128, kv_lora=512, q_lora=1536,
              qk_nope=128, qk_rope=64, v_head=128),
    moe=MoE(n_routed=256, top_k=8, expert_inter=2048, n_shared=1),
    n_dense_layers=3, dense_inter=18432, tied=False,
    kv_sources=(KVSource(61, 1.0),), kv_bits=16,
))

# DeepSeek-V4-Flash：284B / 13B，head_dim=512 的共享 KV（MQA），256 专家 top-6
add(Spec(
    name="DeepSeek-V4-Flash", d=4096, layers=43, vocab=129280,
    attn=Attn("mqa", n_heads=64, head_dim=512, q_lora=1024,
              qk_rope=64, o_lora=1024, o_groups=8),
    moe=MoE(n_routed=256, top_k=6, expert_inter=2048, n_shared=1),
    tied=False, hc_mult=4,
    kv_sources=(KVSource(20, 4.0, layers=20, indexer_k_bytes=68),
                KVSource(20, 128.0, layers=20)),
    kv_bits=9,   # 64 维 RoPE 存 BF16 + 448 维存 FP8 ≈ 9 bit/维 的等效位宽
    sparse_topk=512, swa_window=128,
    indexer=Indexer(n_layers=20, n_heads=64, head_dim=128),
))

# DeepSeek-V4.1-Flash：552B 骨干 + 196B Engram，4 个 KV source（3 个 m=2 + 1 个 m=1）
# 890 B/token 为报告数字；这里按 512 维 FP4 + 16 通道 scale 复算主 KV，差额是索引器 K。
add(Spec(
    name="DeepSeek-V4.1-Flash", d=5120, layers=40, vocab=129280,
    attn=Attn("mqa", n_heads=64, head_dim=512, q_lora=1280,
              qk_rope=64, o_lora=1024, o_groups=8),
    moe=MoE(n_routed=384, top_k=6, expert_inter=2304, n_shared=1),
    tied=False, hc_mult=4, extra_params=196e9,
    kv_sources=(KVSource(3, 2.0, layers=18, indexer_k_bytes=68),
                KVSource(1, 1.0, layers=20, indexer_k_bytes=68)),
    kv_bits=4, kv_scale_bits_per_16=8,
    sparse_topk=512, swa_window=128,
    indexer=Indexer(n_layers=8, n_heads=32, head_dim=128, frac_ctx=0.14),
    active_override=16e9, active_prefill_override=8e9,
))

# Kimi-K2.6：1T 级，MLA(1536,512,128+64,128)，384 专家 top-8
add(Spec(
    name="Kimi-K2.6", d=7168, layers=61, vocab=163840,
    attn=Attn("mla", n_heads=64, kv_lora=512, q_lora=1536,
              qk_nope=128, qk_rope=64, v_head=128),
    moe=MoE(n_routed=384, top_k=8, expert_inter=2048, n_shared=1),
    n_dense_layers=1, dense_inter=18432, tied=False,
    kv_sources=(KVSource(61, 1.0),), kv_bits=16,
))

# Kimi-K3：2.78T / 104.2B，69 层 KDA + 24 层 NoPE MLA（3:1），896 专家 top-16，latent 3584
add(Spec(
    name="Kimi-K3", d=7168, layers=93, vocab=163840,
    attn=Attn("mla", n_heads=96, kv_lora=512, q_lora=1536,
              qk_nope=128, qk_rope=64, v_head=128, nope=True),
    moe=MoE(n_routed=896, top_k=16, expert_inter=3072, n_shared=2, latent_dim=3584),
    n_dense_layers=1, dense_inter=33792, tied=False,
    kv_sources=(KVSource(24, 1.0, layers=24),), kv_bits=16,
    linear=LinearAttn(n_layers=69, n_heads=96, head_dim=128),
))

# GLM-5.2：78 层，MLA(2048,512,192+64,256) + DSA(top-2048)，256 专家 top-8
add(Spec(
    name="GLM-5.2", d=6144, layers=78, vocab=154880,
    attn=Attn("mla", n_heads=64, kv_lora=512, q_lora=2048,
              qk_nope=192, qk_rope=64, v_head=256),
    moe=MoE(n_routed=256, top_k=8, expert_inter=2048, n_shared=1),
    n_dense_layers=3, dense_inter=12288, tied=False,
    kv_sources=(KVSource(78, 1.0),), kv_bits=16,
    sparse_topk=2048, indexer=Indexer(n_layers=20, n_heads=32, head_dim=128),
))

# gpt-oss-120b：36 层，GQA(64/8, head 64)，128 专家 top-4，MXFP4 权重
add(Spec(
    name="gpt-oss-120b", d=2880, layers=36, vocab=201088,
    attn=Attn("gqa", n_heads=64, head_dim=64, n_kv_heads=8),
    moe=MoE(n_routed=128, top_k=4, expert_inter=2880, n_shared=0),
    tied=False,
    kv_sources=(KVSource(36, 1.0),), kv_bits=16,
))

# ERNIE-4.5-300B-A47B：54 层，GQA(64/8,128)，64 专家 top-8，无共享专家
add(Spec(
    name="ERNIE-4.5-300B-A47B", d=8192, layers=54, vocab=103424,
    attn=Attn("gqa", n_heads=64, head_dim=128, n_kv_heads=8),
    moe=MoE(n_routed=64, top_k=8, expert_inter=3584, n_shared=0),
    n_dense_layers=3, dense_inter=28672, tied=False,
    kv_sources=(KVSource(54, 1.0),), kv_bits=16,
))


# ---------------------------------------------------------------------------
# 3. 报告
# ---------------------------------------------------------------------------

def fmt(x: float) -> str:
    if x >= 1e12:
        return f"{x/1e12:.2f}T"
    if x >= 1e9:
        return f"{x/1e9:.1f}B"
    if x >= 1e6:
        return f"{x/1e6:.1f}M"
    if x >= 1e3:
        return f"{x/1e3:.1f}K"
    return f"{x:.0f}"


def fmt_bytes(x: float) -> str:
    for unit, scale in (("TB", 1e12), ("GB", 1e9), ("MB", 1e6), ("KB", 1e3)):
        if x >= scale:
            return f"{x/scale:.1f}{unit}"
    return f"{x:.0f}B"


def report(specs: list[Spec], ctx: int) -> str:
    rows = []
    for s in specs:
        rows.append({
            "model": s.name,
            "total": s.params_total(),
            "active": s.params_active(),
            "kv_per_token": s.kv_bytes_per_token(),
            "kv_at_ctx": s.kv_bytes_per_token() * ctx,
            "flops_prefill": s.flops_prefill(ctx),
            "flops_decode": s.flops_decode(ctx),
            "bytes_decode": s.decode_bytes(ctx),
            "ai": s.arithmetic_intensity(ctx),
        })

    head = (
        "| 模型 | 总参数 | 激活参数 | KV B/token | KV@ctx | prefill FLOPs/token | decode 读字节/token | 算术强度 |\n"
        "|---|---|---|---|---|---|---|---|"
    )
    lines = [head]
    for r in rows:
        lines.append(
            f"| {r['model']} | {fmt(r['total'])} | {fmt(r['active'])} | "
            f"{r['kv_per_token']:.0f} | {fmt_bytes(r['kv_at_ctx'])} | "
            f"{fmt(r['flops_prefill'])} | {fmt_bytes(r['bytes_decode'])} | {r['ai']:.2f} |"
        )
    return "\n".join(lines)


def print_detail(spec: Spec, ctx: int) -> None:
    p = spec.params_total()
    print(f"\n== {spec.name} ==")
    print(f"  总参数       {fmt(p)}   (embed {fmt(spec.vocab*spec.d*(1 if spec.tied else 2))}, "
          f"attn {fmt(spec.params_attn())}, ffn {fmt(spec.params_ffn())}, "
          f"hc {fmt(spec.params_hc())}, extra {fmt(spec.extra_params)})")
    print(f"  激活参数     {fmt(spec.params_active())}")
    print(f"  KV/token     {spec.kv_bytes_per_token():.0f} B")
    print(f"  权重字节     {fmt_bytes(spec.params_active()*2)} (bf16)")
    print(f"  @ctx={ctx}: KV {fmt_bytes(spec.kv_bytes_per_token()*ctx)}, "
          f"decode 读 {fmt_bytes(spec.decode_bytes(ctx))}, 算术强度 {spec.arithmetic_intensity(ctx):.2f}")


# ---------------------------------------------------------------------------
# 4. 自测：与论文/模型卡公布的参数量对拍
# ---------------------------------------------------------------------------

# (name, 公布总参, 相对容差) —— 容差来自结构简化（未计 bias、norm、router 等）
CHECKS = [
    ("Llama-3.1-405B", 405e9, 0.02),
    ("Qwen3-32B", 32.8e9, 0.02),
    ("Qwen3-30B-A3B", 30.5e9, 0.03),
    ("DeepSeek-V3", 671e9, 0.02),
    ("DeepSeek-V4-Flash", 284e9, 0.02),
    ("DeepSeek-V4.1-Flash", 552e9, 0.02),   # 只比骨干，Engram 另算
    ("Kimi-K2.6", 1.04e12, 0.03),
    ("Kimi-K3", 2.78e12, 0.03),
    ("gpt-oss-120b", 117e9, 0.03),
]


def test() -> None:
    by_name = {s.name: s for s in MODEL_SPECS}
    ok = True
    for name, published, tol in CHECKS:
        s = by_name[name]
        # 只比「骨干」：Engram 等外挂参数单独记账（V4.1 报告口径 552B + 196B）
        p = s.params_total() - s.extra_params
        rel = abs(p - published) / published
        mark = "PASS" if rel <= tol else "FAIL"
        if rel > tol:
            ok = False
        print(f"[{mark}] {name:22s} 复算 {fmt(p):>8s} vs 公布 {fmt(published):>8s} "
              f"({rel*100:5.1f}% ≤ {tol*100:.0f}%)")

    # 结构性断言：KV 预算的排序必须与已知事实一致
    kv = {s.name: s.kv_bytes_per_token() for s in MODEL_SPECS}
    assert kv["DeepSeek-V4.1-Flash"] < kv["DeepSeek-V3"] / 10, "V4.1 的 KV 应比 V3 小一个量级"
    assert kv["Llama-3.1-405B"] > kv["Qwen3-32B"], "405B 的 KV 应大于 32B"
    assert by_name["Kimi-K3"].params_active() < by_name["Kimi-K3"].params_total() / 20, \
        "K3 的激活比应低于 1/20"
    # V4.1 主 KV 复算（报告 890 B/token，含索引器 K 与 scale）
    v41 = by_name["DeepSeek-V4.1-Flash"].kv_bytes_per_token()
    assert abs(v41 - 890) < 1, f"V4.1 KV 复算 {v41:.0f}B ≠ 报告 890B"
    print(f"[PASS] 结构断言全部通过（V4.1 KV 复算 {v41:.0f} B/token = 报告 890 B/token）")
    print("\n" + ("ALL PASS" if ok else "SOME FAILED"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, default=1 << 20, help="上下文长度，默认 1M")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--detail", action="store_true")
    ap.add_argument("--test", action="store_true")
    args = ap.parse_args()

    if args.test:
        test()
        return

    if args.json:
        out = [{
            "model": s.name,
            "params_total": s.params_total(),
            "params_active": s.params_active(),
            "kv_bytes_per_token": s.kv_bytes_per_token(),
            "flops_prefill": s.flops_prefill(args.ctx),
            "bytes_decode": s.decode_bytes(args.ctx),
        } for s in MODEL_SPECS]
        print(json.dumps(out, indent=2, ensure_ascii=False))
        return

    print(f"# 上下文 {args.ctx:,} tokens\n")
    print(report(MODEL_SPECS, args.ctx))
    if args.detail:
        for s in MODEL_SPECS:
            print_detail(s, args.ctx)


if __name__ == "__main__":
    main()
