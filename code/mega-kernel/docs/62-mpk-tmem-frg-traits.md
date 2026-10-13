# SM100 TMEM fragment trait 家族的「其余两面」走读：稀疏元数据 E（`tmem_e_frg`）与 weight-stationary C/D（`tmem_frg_ws`）

> 快照：`mirage-project/mirage` 分支 `mpk`，commit `f9eb70c`（与 `docs/06`–`docs/61` 同快照）；
> CUTLASS 侧走的是 MPK 该快照钉死的 submodule `deps/cutlass`，commit `f3fde583`
> （`/ssd/mirage_mpk_build/deps/cutlass`，这条 checkout 有完整 submodule）。
> `docs/60`（P146-1）把 `mma_traits_sm100.hpp` 里的**累加器 `tmem_frg`** 与 **block-scaled
> scale-factor `tmem_sf_frg`** 求出来了；本文补上 `docs/60` §9 与 ROADMAP「下一步候选」都点名的
> **其余两面**——**稀疏 MMA 的元数据 E**（`tmem_e_frg` / `tmem_e_frg_ws`）与 **weight-stationary 的
> C/D**（`tmem_frg_ws`），并把「这个 trait 家族谁有消费者、谁没有」一并审计出来。
> **纯源码走读 + 1 个 host 侧结构探测**：SM100 指令/TMEM 本体在本机（目标卡 H100/sm90）不可执行，
> 但所有 `UMMA::tmem_*_frg::make` 都是 `CUTE_HOST_DEVICE`，整套布局代数可在 **host** 上实例化/求值
> ——本文用快照钉死的**真实 CUTLASS 头文件**把四个 trait 的形状钉死（`src/tmem_frg_traits.cu` +
> `harness/tmem_frg_traits.sh` + `harness/tmem_frg_traits.out.txt`）。所有行号为快照静态事实。

---

## 0. 三条 headline

**(A)（headline）`tmem_e_frg` 把稀疏 MMA 的「元数据 E」的 TMEM 布局写成一个 CuTe Layout，其原子宽度
恰等于 `512 / sizeof_bits<A>`。** 本文用真实 CUTLASS 头在 host 上求值：对 A/B 元素 32b（TF32）→ 原子
`128×16`、16b（FP16）→ `128×32`、8b → `128×64`、4b → `128×128`；返回的 tensor 元素类型是
`sparse_elem<N,uint8_t>`（TF32 `N=4`、FP16 `N=8`、8b/F4 `N=16`）。这与 `MMA_Traits` 里稀疏指令的
**K = `512/sizeof_bits`（「SparseMma consume double mma-k bits」）**正是同一个数——E 在 M 轴的
128 行上、K 轴每一列一个元数据原子。E 经 `MMA_Traits::with(E)` 存进 `sparse_args_`，执行时被
`mma_unpack` 拆成「低 1 位 `id2` + 其余地址」，前者填指令描述符的 `sparse_id2_` 位域、后者作为
**独立的 TMEM 操作数 `[%9]`** 传给 `tcgen05.mma.sp`（不是描述符的低 32 位——源码注释与实现有落差）。

**(B)（headline）weight-stationary 的两个 trait（`tmem_frg_ws`、`tmem_e_frg_ws`）在这份 CUTLASS 快照里
是「孤儿」——定义完整、却一个消费者都没有，因为快照里根本没有 WS MMA 指令。** `tmem_frg_ws` 把
C/D 的 WS 布局按 `M∈{32,64,128}` 写成 **1x4 / 2x2 / 4x1 数据通路**（本文实测：M=64 时 N 的后半段整体
落到 `dp+64`、M=32 时 N 的四段各落到 `dp+32k`）；`tmem_e_frg_ws` 是它的稀疏元数据版（多一个 `_2`/`_4`
的 M 子块维）。但 consumer audit 显示 `FrgTypeC/D = UMMA::tmem_frg_ws<...>`、`FrgTypeE = UMMA::tmem_e_frg_ws<...>`
**均为 0**，且全仓 **0 个 `SM100_MMA_*_WS*` 原子**——只有指令描述符的 `max_shift_ : 2` 位（无法线偏移）
为 WS 预留了比特。即「WS 的 TMEM 布局先写好了，WS 指令还没来」。

**(C)（headline）MPK 源码对这两个 trait 也是零引用（0 次），它们纯属 CUTLASS 侧。** 本文与 `docs/60`
合起来把 `mma_traits_sm100.hpp` 的 TMEM fragment 家族**六件套**走完：`tmem_frg`（累加器/A，`docs/60`）、
`tmem_sf_frg`（scale-factor，`docs/60`）、`tmem_e_frg`（稀疏元数据 E，本文）、`tmem_e_frg_ws`（WS 稀疏
E，本文）、`tmem_frg_ws`（WS C/D，本文），以及 `MakeTensor` 定制点。收口结论：MPK 的 SM100 device
身体是**dense block-scaled linear + MLA** 路线，稀疏与 weight-stationary 两条 TMEM 契约在 MPK 里
**未被使用**——这与本专题反复出现的「CUTLASS 提供了、MPK 只借一部分」同向（`docs/52`/`docs/61`）。

---

## 1. `mma_traits_sm100.hpp` 的 TMEM fragment 家族地图

文件里以 `UMMA::tmem_frg_base` 为基类的一族 trait（行号见 `docs/60` §1 与本文）：

| trait | 行 | 职责 | 消费者（本文 audit） |
|---|---|---|---|
| `tmem_frg` | `:461-583` | 累加器 / A 值的 TMEM 布局（dense） | 全部 dense MMA（`docs/60` 已走） |
| `tmem_e_frg` | `:593-669` | **稀疏 MMA 的元数据 E** | **13 个 sparse `MMA_Traits`** |
| `tmem_e_frg_ws` | `:671-811` | **WS 稀疏 MMA 的元数据 E** | **0** |
| `tmem_sf_frg` | `:813-873` | block-scaled 的 scale-factor | dense block-scaled（`docs/60` 已走） |
| `tmem_frg_ws` | `:876-959` | **WS 的 C/D** | **0** |
| `MakeTensor<...>` | `:967-1017` | 把上述 trait 接到 `make_tensor` | — |

`docs/60` 走了上表第 1、4 行；本文走第 2、3、5 行。三个 trait 都遵循同一套「虚拟 TMEM 寻址」
（`DP_STRIDE=1`、`COL_STRIDE=128`，再 `composition` 回逻辑地址）与同一族 `MakeTensor` 定制点。

---

## 2. `tmem_e_frg`：稀疏元数据 E 的布局（`:593-669`）

### 2.1 按 A/B 元素宽度分支的原子（`:613-665`）

`make` 先断言 `rank<0>==2`、`M_MMA==128`（`:600/:605`），再按 `sizeof_bits<ValueType>` 分四支：

| `sizeof_bits<ValueType>` | 断言 | `tmem_atom` 形状 / 步长 | 返回元素 |
|---|---|---|---|
| 32（TF32） | `N_MMA==16`（`:615`） | `((8,2,8),(8,2))` / `((DP_b,1024,16),(128,8))` + `upcast<2>` | `sparse_elem<4,uint8_t>` |
| 16（FP16） | `N_MMA==32`（`:627`） | `((8,2,8),(16,2))` / `((DP_b,2048,16),(128,8))` | `sparse_elem<8,uint8_t>` |
| 8（i8/f8） | `N_MMA==64`（`:640`） | `((128,64))` / `((DP_b,128))` | `sparse_elem<16,uint8_t>` |
| 4（F4） | `N_MMA==128`（`:656`） | `((128,128))` / `((2<<22,128))` | `sparse_elem<16,uint8_t>` |

注意三点：**(i)** 第二个原子维（本文记作 `N_MMA`，实为 E 覆盖的 K 原子）恰为 `512/sizeof_bits`
——与稀疏 `MMA_Traits` 的 `K = 512/sizeof_bits`（`:1454`/`:1542` 等）同源；**(ii)** 8b 与 4b 用**扁平**
原子（`128×64`/`128×128`），32b/16b 用**嵌套**原子（把 M 拆成 `(8,2,8)`、把 E 列拆成 `(W,2)`）；
**(iii)** 源码注释 `:591-592` 明写「TMEM fragment addresses are assumed to be **COL-4 aligned**」
（E 的 TMEM 地址低 2 位隐含对齐约束，见 §3.2）。

### 2.2 实验 A 证据（`harness/tmem_frg_traits.out.txt` §A）

对 `tmem_e_frg<half_t>`、M=128、E 列=32 逐点求值（输出第 8–30 行）：

```
[E fp16 (16b, 128x32 atom)]
  shape : ((_128,_32),_1,_1)
  layout: (((_8,_2,_8),(_16,_2)),_1,_1):(((_2097152,_16,_33554432),(_1,_16777216)),_0,_0)
    r= 0 n=0 : off=        0
    r= 1 n=0 : off=  2097152      (DP_b = 1<<21)
    r= 8 n=0 : off=       16      (M 的 (8,2,8) 第 2 子维 -> 16 字节偏移)
    r=16 n=0 : off= 33554432      (第 3 子维 -> 16*DP_b)
    r=0 n=16 : off= 16777216      (E 列的第 2 子维 -> 8*DP_b)
```

即 E 原子把 M 拆成 `(8,2,8)`（步长 `{DP_b, 16, 16·DP_b}`）、把 E 列拆成 `(16,2)`（步长 `{1, 8·DP_b}`）。
对照 A/B 元素变宽时 `N_MMA` 翻倍（32b→16、16b→32、8b→64、4b→128）、而 M 原子宽度恒 128——**E 的
门控维在 K（列）上、M 恒 128**。

---

## 3. E 怎么被稀疏 MMA 消费（`docs/60` §4 的「另一半」）

### 3.1 `.with(E)`：E 作为 `sparse_args_` 进入 traits（`:1507-1517`）

所有稀疏 trait 都带一组 `sparse_args...` 模板参数，`MMA_Traits` 里 `FrgTypeE = UMMA::tmem_e_frg<a_type>`
（`:1449` 等 13 处）。`with(E)` 把 E tensor 的 TMEM 地址拷进 `sparse_args_`（`:1513-1516`）：

```cpp
with(Tensor<TE, ELayout> const& E) const {
  uint32_t tmem_e_addr = raw_pointer_cast(E.data());
  return {accumulate_, {tmem_e_addr}, idesc_};
}
```

### 3.2 `mma_unpack`：低 1 位 → `sparse_id2_`，其余 → 独立 TMEM 操作数（`:1496-1504`）

```cpp
uint32_t tmem_e = get<0>(traits.sparse_args_);
uint32_t id2    = tmem_e &  0x00000001;   // 低 1 位 = sparse_id2
tmem_e          = tmem_e & ~0x00000001;
uint64_t idesc = UMMA::make_runtime_instr_desc<true>(traits.idesc_, (uint16_t)id2, tmem_e);
...::fma(desc_a, desc_b, tmem_c, accumulate, idesc, tmem_e);
```

`make_runtime_instr_desc<true>`（`mma_sm100_desc.hpp:532-550`）只把 `sparse_id2` 写进描述符的
`sparse_id2_ : 2` 位域（`mma_sm100_desc.hpp:418`，`bit [0,2)`），**并没有**把 `tmem_e` 写进 idesc 的
低 32 位——尽管函数注释 `:543-546` 写着「Lower 32b URe: Specifies the tmem address that stores the
sparse metadata」。真正送达硬件的是 `fma` 的**独立操作数**：`SM100_MMA_F16BF16_SS_SPARSE::fma`
（`mma_sm100_umma.hpp:341-370`）里 `tcgen05.mma.sp.cta_group::1.kind::f16 [%0], %1, %2, [%9], %3,
{%5,%6,%7,%8}, p;`，其中 `[%9]` 就是 `tmem_e`。**这是又一处「按注释推断行为会错」**（与 `docs/60`
§5.2 的 `sf_id`、`docs/58`/`docs/60` 的 XOR 转置同族）：注释描述的「idescE 低 32 位 = E 地址」在
1SM 稀疏 asm 里并未实现，E 走的是单独的 TMEM 地址操作数。

要点对照 `docs/60` §4 的 block-scaled SF：两者都从 TMEM 地址的低位「挤出」一个 id 填进指令描述符
（SF 取高 2 位 `>>30`、E 取低 1 位 `&1`），都把真正的 TMEM 地址作为**独立操作数**给指令。

---

## 4. weight-stationary 的两个 trait（`tmem_e_frg_ws` `:671-811`、`tmem_frg_ws` `:876-959`）

### 4.1 `tmem_e_frg_ws`：WS 稀疏 E，多一个 M 子块维（`:683-807`）

与 `tmem_e_frg` 的差别是**允许 M_MMA ∈ {32,64,128}**（`:683`），于是原子在 M 上多出一维：
M=128 与 `tmem_e_frg` 完全同形；M=64 的 FP16 原子是 `((8,2,4),(16,2),2)`（步长 `((DP_b,2048,16),(128,8),64)`）；
M=32 是 `((8,2,2),(16,2),4)`（末维步长 `32`）。实验 B 输出（第 36–44 行）逐条对上。语义与
`docs/60` §2.2 的 1SM M=64「半子分区」同构，只是作用在 E 上。

### 4.2 `tmem_frg_ws`：WS C/D 的 1x4 / 2x2 / 4x1 数据通路（`:912-955`）

`make` 断言 `N_SM==1`、`M_MMA∈{32,64,128}`、`N_MMA∈{64,128,256}`（`:912-917`），分支：

- **M=32 → 1x4**：`Shape<32, Shape<N/4,4>>` / `Stride<1, Stride<128,32>>`（`:922-923`）；
- **M=64 → 2x2**：`Shape<64, Shape<N/2,2>>` / `Stride<1, Stride<128,64>>`（`:935-936`）；
- **M=128 → 4x1**：`Shape<128,N>` / `Stride<1,128>`（`:949-950`）。

实验 C 逐点映射（第 60–82 行）：

```
ws M=64 (2x2):  r 0..63 -> dp 0..63 (col 0..63)；n 0..63 -> dp 0；n 64..127 -> dp 64
ws M=32 (1x4):  n 0..31 -> dp 0；n 32..63 -> dp 32；n 64..95 -> dp 64；n 96..127 -> dp 96
```

即 WS 把 N 切成 4 段（M=32）/2 段（M=64），每段落到**相差 32（或 64）个数据通路**的子分区上——
与 `docs/60` §2.3 的 `tmem_frg`「半子分区交织」是同一套 DP 复用思想，只是按 WS 的 datapath 形态重排。

### 4.3 但这两个 trait 没有消费者（consumer audit）

`harness/tmem_frg_traits.out.txt` 第 84–89 行（对 `deps/cutlass` 与 MPK 双 grep）：

```
FrgTypeE = UMMA::tmem_e_frg<...>    : 13
FrgTypeE = UMMA::tmem_e_frg_ws<...> : 0
FrgTypeC/D = UMMA::tmem_frg_ws<...> : 0
any SM100_MMA_*_WS* atom struct     : 0
MPK (f9eb70c) refs tmem_e_frg/frg_ws: 0
```

13 处 `tmem_e_frg` 消费者全是稀疏 `MMA_Traits`（`SM100_MMA_{TF32,F16BF16,S8,F8F6F4,MXF8F6F4}_[2x1SM_]SS_SPARSE`，
行号见 `:91-103` 输出；其中 `:3767` 是被注释掉的一行、有效 12 处）。而 `tmem_frg_ws`/`tmem_e_frg_ws`
的消费者 **0**，且**全仓没有任何 `SM100_MMA_*_WS*` 原子结构**——只有指令描述符的
`max_shift_ : 2`（`mma_sm100_desc.hpp:433`，注释「Maximum shift for WS instruction」）为 WS 预留了比特。
即：**「WS 的 TMEM 布局契约已写好、WS 指令尚未落地」**，这两个 trait 在本快照里是**死代码**（同
`docs/49`/`docs/50`/`docs/61` 的「有实现未接线」族）。

---

## 5. 与相邻文档的关系 + 判据映射

- `docs/60`（scale-factor TMEM trait）：本文是它的**同一家族续篇**——`docs/60` 走了累加器与 scale，
  本文走稀疏 E 与 WS C/D，并把 `docs/60` §1 表里的两行「未展开」补上；`docs/60` §9 明留的
  「`tmem_frg_ws` 未展开」即 §4。两文合起来 = `mma_traits_sm100.hpp` 的 TMEM fragment 家族全覆盖。
- `docs/57`/`docs/59`（SM100 原语 / TMEM 列与 UTCCP）：本文补上它们未点名的两个 TMEM fragment
  契约（稀疏 E、WS），与「TMEM 是 per-SM、须 alloc、列数 2 的幂」同层，但更细到**指令怎么找到 E**。
- `docs/52`/`docs/61`（CUTLASS 只被借一部分）：本文给「MPK 只用了 CUTLASS 的 dense + block-scaled
  两支、稀疏/WS 两支零引用」以 grep 级证据。
- **判据映射**：本文只贡献 **C3（多算子范围 / 设备体内布局契约）**——它是「一个 task 的 device body
  内部、TMEM 的布局/寻址契约」，不涉及 launch/常驻（C1/C2）、不涉及去中心化调度（C5）。

---

## 6. 局限（诚实标注）

- **SM100 指令/TMEM 本体未执行**：`tmem_e_frg`/`tmem_frg_ws` 的布局是**在 host 上对真实 CUTLASS 头
  求值**得到的（`src/tmem_frg_traits.cu`），属「trait 规定的形状」，**不是**在 SM100 上观测到的硬件
  行为；E 的 `sparse_id2`/COL-4 对齐、`tcgen05.mma.sp` 的 `[%9]` 元数据寻址语义均**未验证**。
- **未构建/未跑任何稀疏或 WS 内核**：本机无 SM100；本文是纯源码走读 + host 侧布局求值。
- **稀疏元数据 E 的**硬件**编码（2:4 索引位序、每 atom 的 bit 打包）未展开**：本文只到 trait 的形状/
  步长与「低 1 位是 id2」这一级，未追 `tcgen05.mma.sp` 对 E 内容的解释。
- **WS 的 1x4/2x2/4x1 仅为 trait 形状**：因快照无 WS 指令，本文无法（也无法在别处）确认它与真实 WS
  硬件的一致性；「max_shift_ 是唯一 WS 痕迹」这一判断只覆盖本快照。
- **consumer audit 是 `grep` 级**：`FrgType` 别名计数可能受宏/注释影响（已标注 `:3767` 注释行）；
  CUTLASS 更高版本的命名/实现可能变化。

---

## 7. 参考坐标（可核对）

- CUTLASS（`deps/cutlass` `f3fde583`）：
  - `include/cute/atom/mma_traits_sm100.hpp`：
    `tmem_frg_base:457`、`tmem_frg:461-583`、`tmem_e_frg:593-669`（32b `:613-623`、16b `:625-635`、
    8b `:637-649`、4b `:650-665`）、`tmem_e_frg_ws:671-811`、`tmem_sf_frg:813-873`、
    `tmem_frg_ws:876-959`（M=32 `:919-931`、M=64 `:932-944`、M=128 `:945-955`）、
    `MakeTensor:967-1017`；
  - 稀疏 `MMA_Traits`：`SM100_MMA_TF32_SS_SPARSE:1436-1518`（`FrgTypeE:1449`、`K=512/bits:1454`、
    `mma_unpack:1477-1505`、`with(E):1507-1517`）、`F16BF16_SS_SPARSE:1520+`、`S8_SS_SPARSE:2326+`、
    `MXF8F6F4_SS_SPARSE:2790+`、`F8F6F4_SS_SPARSE:2961+`、2x1SM 变体 `:2020/2108/2543/3186/3364`；
  - `include/cute/arch/mma_sm100_desc.hpp`：`InstrDescriptor` 位域（`sparse_id2_ : 2` `:418`、
    `sparse_flag_ : 1` `:419`、`max_shift_ : 2` `:433`）、`make_instr_desc:480-502`、
    `make_runtime_instr_desc:504-550`（注释 `:543-546` 与实现落差）；
  - `include/cute/arch/mma_sm100_umma.hpp`：`SM100_MMA_TF32_SS_SPARSE:300-339`、
    `SM100_MMA_F16BF16_SS_SPARSE:341-370`（`tcgen05.mma.sp... [%9]`）；
  - `include/cute/arch/tmem_allocator_sm100.hpp:45-53`（`DP_b = 1<<21`）；
  - `include/cute/pointer_sparse.hpp:42-89`（`sparse_elem` 与 `sizeof_bits` 定义）。
- MPK（`f9eb70c`）：对 `tmem_e_frg`/`tmem_e_frg_ws`/`tmem_frg_ws` **零引用**（`grep` 0 次）。
- 本文实验：`src/tmem_frg_traits.cu` + `harness/tmem_frg_traits.sh` + `harness/tmem_frg_traits.out.txt`
  （目标卡 H100/sm90a，host 侧，未 launch kernel）。
