# CUTLASS SM100 稀疏 UMMA 的 A 操作数通路走读：`sparse_elem<2,T>` / 稀疏 SMEM 布局 / `sparse_smem_desc`

> 快照：`mirage-project/mirage` 分支 `mpk`，commit `f9eb70c`（与 `docs/06`–`docs/62` 同快照）；
> CUTLASS 侧走 MPK 该快照钉死的 submodule `deps/cutlass`，commit `f3fde583`
> （`~/mirage_mpk_build/deps/cutlass`，这条 checkout 有完整 submodule）。
> `docs/62`（P148-1）把稀疏 MMA 的**元数据 E** 侧（`tmem_e_frg`）走完了，但只走到「E 的形状 + 低 1 位
> `id2`」；本文补上 `docs/62` §6 明留的两条尾巴之一（「E 的硬件编码……未追 `tcgen05.mma.sp` 对 E 内容
> 的解释」的**相邻面**——`tcgen05.mma.sp` 的另一半输入、**A 操作数**），并把稀疏 SMEM 布局原子、
> `sparse_smem_desc` 与指令描述符的 `sparse_flag_` 一并钉死。
> **纯源码走读 + 1 个 host 侧结构探测**：所有 `cute::Layout`、`sparse_ptr` 算术与 `InstrDescriptor`
> 都是 `CUTE_HOST_DEVICE`，整套代数在 host 上可实例化/求值；SM100 指令本体在本机（目标卡 H100/sm90）
> 不可执行（`src/sparse_mma_a.cu` + `harness/sparse_mma_a.sh` + `harness/sparse_mma_a.out.txt`）。
> 所有行号为快照静态事实。

---

## 0. 三条 headline

**(A)（headline）稀疏 MMA 把「A 的 2:4 压缩」全部折叠进一个 `sparse_elem<2,T>` 类型里，剩下的
描述符与 dense 逐位相同。** `MMA_Traits` 的 A 值类型是 `sparse_elem<2, a_type>`（`ValTypeA`，
`mma_traits_sm100.hpp:1442` 等，12 处中 10 处是 `sparse_elem<2,...>`），其 `sizeof_bits` 是
`ratio(sizeof_bits<T>, 2)`（`pointer_sparse.hpp:80-88`）——**一个物理元素代表 2 个逻辑元素**；而 A 的
fragment 类型 `FrgTypeA = UMMA::sparse_smem_desc<a_major>`（`:1448` 等 12 处）在实现上**就是**
`UMMA::smem_desc`（`MakeTensor<sparse_smem_desc>` 的函数体与 dense 逐字相同，只多两条 `is_sparse` /
`is_sparse_ptr` 静态断言，`:379-397`）。**「稀疏」不在 A 描述符里**，而在两处：指令描述符的
`sparse_flag_` 位（`mma_sm100_desc.hpp:419`，实测把 dense `0x08200010` 变成 sparse `0x08200014`，
即 bit[2] 置 1）与那份 metadata E（`docs/62`）。

**(B)（headline）稀疏 SMEM 布局原子是「在 dense 原子的 MN 轴上叠一层 S 倍的 blocked_product」，且
换用 `smem_sparse_ptr_flag_bits` 门控指针类型。** `Layout_MN_SW128_SpAtom<half_t,2>` 与 dense
`Layout_MN_SW128_Atom<half_t>` 共用同一 `Sw<3,4,3>` swizzle，差别只在：**(i)** 指针 flag 从
`smem_ptr_flag` 换成 `smem_sparse_ptr_flag_bits<2,16>`（`pointer_flagged.hpp:128`，一个「未设置的
sparse 指针」占位类型，真正接指针时用带 flag 的 `make_tensor` 校验 `S==sparsity` 与 `B==bits`，
`:137-148`）；**(ii)** 布局体多一维 `blocked_product(Layout<Shape<_1,Int<S>>>, dense_atom)`，host 实测
dense 形状 `(_64,_8)` → sparse 形状 `((_1,_64),(_2,_8))`——**逻辑 MN 轴被声明为 S 倍**，而物理步长
不变（故同一块 SMEM 覆盖 S 倍逻辑元素）。这就是 2:4 稀疏「从物理布局合同上就承认 A 减半」的落点。

**(C)（headline）稀疏的 `sparse_ptr` 是「逻辑索引 ÷ S → 物理槽」，且 CUTLASS 有完整的 SM100 稀疏
builder、MPK 零引用。** `sparse_ptr<S,Iter>::op+` 要求偏移是 S 的倍数、并把 `i` 换成 `i/S` 物理槽，
`op[]` 允许任意索引、但 `i` 与 `i+1` 会**别名同一物理槽**（`pointer_sparse.hpp:101-125`；host 实测
`(ps+2)-buf=1`、`ps[0].elem_==ps[1].elem_`）。consumer audit（`harness/sparse_mma_a.out.txt`）：
CUTLASS 有 **7 个 sparse collective builder**（`sm100_sparse_umma_builder.inl` 等）、12 处
`FrgTypeA = sparse_smem_desc`、8 处 `K = 512/sizeof_bits`；而 **MPK 对
`sparse_smem_desc`/`sparse_elem`/`SpAtom`/`make_sparse_ptr` 零引用**（0 次），与 `docs/62` 的
「稀疏/WS 两支 CUTLASS 契约 MPK 未用」完全一致。

---

## 1. 定位：`tcgen05.mma.sp` 的两半输入

一条稀疏 SM100 UMMA 指令 `tcgen05.mma.sp.cta_group::1.kind::f16 [%0], %1, %2, [%9], %3, {%5..%8}, p;`
（`mma_sm100_umma.hpp:341-370` 的 `SM100_MMA_F16BF16_SS_SPARSE::fma`）的操作数：

| asm | 含义 | 由谁提供 |
|---|---|---|
| `[%0]` | 累加器 D（TMEM） | `tmem_c` |
| `%1` | A 的 SMEM 描述符 | `desc_a`（= `sparse_smem_desc`，**与 dense 同构**） |
| `%2` | B 的 SMEM 描述符 | `desc_b` |
| `[%9]` | **metadata E**（TMEM 地址） | `tmem_e`（`docs/62` §3.2 走完） |
| `%3` | 指令描述符（`idescE>>32`） | `sparse_flag_`/`sparse_id2_` 等位域 |
| `{%5..%8}` | input-D mask（这里恒 0） | 稀疏路径不用 |

`docs/62` 只走了 `[%9]`（E）；本文走 `%1`（A）与 `%3` 里的 `sparse_flag_`。**关键观察**：稀疏与 dense
的 `fma` 只是「A 描述符类型 + E 操作数 + 一个描述符位」的差别，A 描述符的**构造代码完全复用**
dense 的（§3）。

---

## 2. A 的 2:4 压缩：`sparse_elem<2,T>` 与 `sparse_ptr`

### 2.1 `sparse_elem<S,T>`：一个物理元素 = S 个逻辑元素（`pointer_sparse.hpp:45-55`）

```cpp
template <int Sparsity, class T>
struct sparse_elem { static constexpr int sparsity = Sparsity; using raw_type = T; T elem_; ... };
```

`sizeof_bits<sparse_elem<S,T>>` 被特化为 `cute::ratio(sizeof_bits<T>, S)`（`:80-88`）——不是 0、
也不是整数除法截断，而是一个 **integral ratio**，故 `N * sizeof_bits_v<E>` 这类表达式仍成立。
host 实测（`harness/sparse_mma_a.out.txt` §i）：

```
sparse_elem<2,half_t>  raw=16 bits, S=2 -> sizeof_bits(sparse_elem)= 8
sparse_elem<2,float>   raw=32 bits, S=2 -> sizeof_bits(sparse_elem)=16
sparse_elem<2,uint8_t> raw= 8 bits, S=2 -> sizeof_bits(sparse_elem)= 4
```

即 **fp16 的 A 有效位宽从 16 变 8、tf32 从 32 变 16**——「2:4 压缩后 A 少一半」，与 `docs/62` §2.1
的 E 原子宽度 `512/sizeof_bits<A>` 同源（K 翻倍、A 减半是同一枚硬币的两面）。

> 注意：E 的 metadata 也复用 `sparse_elem`，但那里的 `Sparsity` 是 **E 的打包因子**（`sparse_elem<4,uint8_t>`
> tf32、`sparse_elem<8,uint8_t>` fp16，`docs/62` §2.1），不是 A 的 2:4 因子——同一个类型承载两种语义，
> 只看 `Sparsity` 参数无法区分「A 的 2:4」与「E 的打包」。

### 2.2 `sparse_ptr<S,Iter>`：逻辑索引 ÷ S 的物理契约（`:101-125`）

```cpp
template <int Sparsity, class Iterator>
struct sparse_ptr : iter_adaptor<Iterator, sparse_ptr<Sparsity,Iterator>> {
  template <class Index> ... sparse_ptr operator+(Index const& i) const {
    assert(i % Sparsity == 0);                 // 偏移必须是 S 的倍数
    return {this->get() + i / Sparsity};       // 物理步进 = i/S
  }
  template <class Index> ... reference operator[](Index const& i) const {
    return *(this->get() + i / Sparsity);      // 任意索引，但 i 与 i+1 别名
  }
};
```

三处静态约束（`:108-110`）：`value_type` 本身必须 `is_sparse`（故 `sparse_ptr<S, half_t*>` 编不过，
必须是 `sparse_ptr<S, sparse_elem<S,T>*>`——host 首次编译即撞上，见 `src/sparse_mma_a.cu` 注释）、
`Sparsity == value_type::sparsity`、且不能嵌套 `sparse_ptr`。`make_sparse_ptr<S>`（`:137-146`）在
`S==1` 时直接返回原迭代器（**dense 是稀疏的特例**）。host 实测（§ii）：

```
(ps+0)-buf=0  (ps+2)-buf=1  (ps+4)-buf=2      # op+ 物理步进 i/S
ps[0].elem_=10 ps[1].elem_=10 ps[2].elem_=11  # op[] 的 0/1 别名同一物理槽
```

**含义**：稀疏 A 的迭代器把「A 的逻辑形状翻倍」与「物理存储减半」在类型层焊在一起——上层按逻辑
元素索引，底层按物理槽寻址，**错配（不按 S 对齐地 `+`）由 `assert` 拦下**。

---

## 3. 稀疏 SMEM 布局原子与 `sparse_smem_desc`

### 3.1 稀疏布局原子：dense 原子 MN 轴 × S（`mma_traits_sm90_gmma_sparse.hpp:54-77`）

```cpp
template <class Type, int S>
using Layout_MN_SW128_SpAtom = ComposedLayout<Swizzle<3,4,3>, smem_sparse_ptr_flag_bits<S,sizeof_bits_v<Type>>,
    decltype(blocked_product(Layout<Shape<_1,Int<S>>>{}, Layout_MN_SW128_Atom<Type>{}.layout_b()))>;
```

八种原子（`{MN,K} × {INTER,SW32,SW64,SW128}`，`:54-77`）与 dense 一一对应，只有两个改动：指针 flag
换成 `smem_sparse_ptr_flag_bits<S,bits>`、布局体 `blocked_product(<1,S>, dense)`。`mma_traits_sm100.hpp`
把 `GMMA::Layout_*_SpAtom` 全部 `using` 进 `UMMA`（`:81-89`），并另给一个 **32B 原子**
`Layout_MN_SW128_32B_SpAtom<Type,S>`（`:90-92`，`Sw<2,5,2>` + 同样的 S 扩展）。host 实测（§iii）：

```
dense  Layout_MN_SW128_Atom<half_t>       Sw<3,4,3> o smem_ptr[16b]        o (_64,_8):(_1,_64)
sparse Layout_MN_SW128_SpAtom<half_t,2>   Sw<3,4,3> o smem_sparse<2>_ptr[16b] o ((_1,_64),(_2,_8)):((_0,_2),(_1,_128))
sparse Layout_MN_SW128_SpAtom<half_t,4>   Sw<3,4,3> o smem_sparse<4>_ptr[16b] o ((_1,_64),(_4,_8)):((_0,_4),(_1,_256))
```

——**swizzle 一模一样**（所以打拍/银行冲突行为不变），变的只是逻辑 MN 轴被声明成 `(_1,_S)` 积（S 倍），
即同一物理 SMEM 段可被稀疏 A 迭代器按 S 倍逻辑元素覆盖。`smem_sparse_ptr_flag_bits`（`:128`）注释自称
「a model of a nullptr sparse_ptr with B == sizeof_bits<T>」，带 flag 的 `make_tensor`（`:137-148`）在
真正接上 `smem_ptr` 时校验 `S`/`B` 与布局声明一致——**错配是编译期错，不是运行期静默错**（与
`docs/62` 的 E、`docs/58` 的 `smem_tma` 的「静默错」形成对照）。

### 3.2 `sparse_smem_desc`：与 dense `smem_desc` 逐字相同（`mma_traits_sm100.hpp:361/379-397`）

```cpp
template <UMMA::Major> struct smem_desc        : DescriptorIterator {};
template <UMMA::Major> struct sparse_smem_desc : DescriptorIterator {};   // :361

template <UMMA::Major MajorMode> struct MakeTensor<UMMA::sparse_smem_desc<MajorMode>> {
  // Note that this is the exact same as UMMA::smem_desc above.
  // Only the interface validates that we are passed a sparse_ptr, which is recast away ...
  operator()(Tensor<TEngine,TLayout> const& smem_tensor) {
    static_assert(is_smem<TEngine>::value, ...);
    static_assert(is_sparse<value_type>::value, ...);       // 多出的两条
    static_assert(is_sparse_ptr<TEngine>::value, ...);
    return make_tensor(DescriptorIterator{make_umma_desc<MajorMode>(tensor<0>(smem_tensor))},
                       replace<0>(recast<uint128_t const>(smem_tensor).layout(), Layout<_1,_0>{}));
  }
};
```

——`make_umma_desc` 与 dense 完全同一条（连 `recast<uint128_t const>` 也一样），**稀疏描述符就是
dense 描述符**，唯一作用是「在接口层用 `is_sparse` 挡住误传 dense 指针」。这与 `docs/60` §5.2 的
`sf_id`、`docs/62` §3.2 的 E「独立 TMEM 操作数」同族：**稀疏的语义不在描述符里，而在
指令描述符的 flag 位 + 那份 E**。

---

## 4. 指令描述符：`sparse_flag_` 一位定稀疏（`mma_sm100_desc.hpp`）

`InstrDescriptor`（`:412-434`）与 `InstrDescriptorBlockScaled`（`:441-463`）都以 `sparse_id2_ : 2`
（bit[0,2)）+ `sparse_flag_ : 1`（bit[2,3)）打头：

```
sparse_id2_    : 2,  // bit [ 0, 2) : Sparse meta data id2
sparse_flag_   : 1,  // bit [ 2, 3) : 0 = dense. 1 = sparse. 1 value valid only for F32F16/S8/MXF8F6F4
```

`make_instr_desc<...>`（`:466-502`）里 `desc_i.sparse_flag_ = is_sparse; desc_i.sparse_id2_ = 0;`（`:496-497`）；
`sparse_id2` 由 `make_runtime_instr_desc<true>(desc_i, sparse_id2, tmem_e)`（`:532-550`）在运行期填（
`desc_i.sparse_id2_ = sparse_id2`，`:538`）——注意 `tmem_e` 参数被**忽略**（`docs/62` §3.2 已指出：
注释说 idescE 低 32b 存 E 地址，实现却不填）。host 实测（§v）：

```
dense  desc = 0x08200010  (sparse_flag_=0)
sparse desc = 0x08200014  (sparse_flag_=1)     # 差 0x4 = bit[2]
```

——**dense↔sparse 的指令描述符只差一个 bit**。同时 `InstrDescriptor` 尾部 `max_shift_ : 2`（`:433`，
「Maximum shift for WS instruction」）是 WS 指令的唯一痕迹（`docs/62` §4.3）。

### 4.1 `K = 512/sizeof_bits`：「稀疏 MMA 消耗双倍 mma-k 位」

12 个稀疏 `MMA_Traits`（`mma_traits_sm100.hpp:1436/1524/2020/2108/2326/2543/2790/2961/3186/3364/3556/3747`）
都写着 `static constexpr int K = 512 / cute::sizeof_bits<a_type>::value;`（`:1454`/`:1542`/…，共 8 处
按 A 元素宽度算），注释「SparseMma consume double mma-k bits」。host 实测：

```
half(16b)->K=32  tf32(32b)->K=16  i8(8b)->K=64     # dense 的一半
```

——逻辑 K 翻倍正是「A 物理减半 × E 补回压缩率」的直接结果。**4-bit A 的特例**：`MXF4NVF4` 两支
（`:3556`/`:3747`）的 A 是 `sparse_elem<4,uint8_t>`（一个 uint8 装 4 个 4-bit nibble）、`K` **硬编码 128**
（`// Logical shape-K is always 256bits`，`:3759-3761`），因为 `sizeof_bits<uint8_t>` 无法表达 4-bit——
这是「按 `512/sizeof_bits` 通式推断 4-bit”会错」的一处（同族于本专题反复出现的「按注释/通式推断会错」）。

---

## 5. Consumer audit：CUTLASS 全套、MPK 零引用

`harness/sparse_mma_a.out.txt` 末尾：

```
sparse MMA_Traits FrgTypeA = sparse_smem_desc : 12
sparse MMA_Traits ValTypeA = sparse_elem<2,...> : 10
sparse K = 512/sizeof_bits<A> sites           : 8
SM100 sparse collective builders (CUTLASS)    : 7
  -> sm100_blockscaled_sparse_umma_builder.inl sm100_sparse_umma_builder.inl
     sm120_blockscaled_sparse_mma_builder.inl sm120_sparse_mma_builder.inl
     sm1xx_sparse_config.inl sm90_sparse_config.inl sm90_sparse_gmma_builder.inl
MPK (f9eb70c) refs sparse_smem_desc/SpAtom/   : 0
MPK (f9eb70c) refs sparse anywhere (runtime)  : 0
```

即：CUTLASS 从**布局原子**（SM90 GMMA 稀疏 `:54-77`）、**trait**（SM100 12 支）、**指令原子**
（`SM100_MMA_*_SS_SPARSE` 12 个，`mma_sm100_umma.hpp:300/341/630/671/793/916/1117/1160/1236/1317/1429/1570`）、
**collective builder**（`sm90_sparse_gmma_builder.inl`/`sm100_sparse_umma_builder.inl`/…7 个）到
**布局选择器**（`sm90_common.inl:340-380`、`sm120_common.inl:126+` 按 `BLK_MN/BLK_K % size(SpAtom)` 挑 swizzle）
是**一条完整稀疏 GEMM 栈**；而 **MPK 对它零引用**。这是 `docs/52`/`docs/61`/`docs/62` 反复出现的
「CUTLASS 提供了、MPK 只借一部分」的又一次量化：**MPK 的 SM100 device 身体是 dense + block-scaled
路线（`docs/33`、`docs/32`、`docs/60`），稀疏与 WS 两支都没接**。

---

## 6. 与相邻文档的关系 + 判据映射

- `docs/62`（TMEM fragment 家族）：本文是它的**操作性对面**——`docs/62` 走稀疏的 **E**（`tmem_e_frg`），
  本文走稀疏的 **A**（`sparse_smem_desc` + `SpAtom` + `sparse_ptr`）与指令 `sparse_flag_`；两文合起来 =
  `tcgen05.mma.sp` 的完整操作数清单（A 描述符 / E / 描述符 flag）。
- `docs/60`（scale-factor TMEM trait）：同样是「CUTLASS 契约 vs MPK 手工路径」的并置——`docs/60` 是
  **block-scaled**（MPK **在用**），本文是**稀疏**（MPK **零引用**），恰是「MPK 借了哪支、没借哪支」的
  正反两面。
- `docs/58`/`docs/59`（SMEM/TMEM 布局即 ABI）：本文补上稀疏布局这一支的 ABI——`SpAtom` 与 dense
  `Atom` **共用 swizzle、只改 flag 与逻辑轴**，把「布局类型即 ABI」延伸到稀疏。
- **判据映射**：本文只贡献 **C3（多算子范围 / 设备体内布局与操作数契约）**——它是「一个 task 的
  device body 内部、稀疏 A 的布局/寻址契约」，不涉及 launch/常驻（C1/C2）、不涉及去中心化调度（C5）。

---

## 7. 局限（诚实标注）

- **SM100 指令/硬件本体未执行**：`sparse_elem`/`SpAtom`/`sparse_smem_desc` 的布局是**在 host 上对真实
  CUTLASS 头求值**得到的（`src/sparse_mma_a.cu`），属「trait 规定的形状」，**不是** SM100 上观测到的
  硬件行为；`tcgen05.mma.sp` 对 A 的物理排布（哪些元素被 2:4 丢弃、E 的 4-bit 索引如何映射到 A 的
  偶数/奇数槽）均**未验证**。
- **未构建/未跑任何稀疏内核**：本机无 SM100；本文是纯源码走读 + host 侧结构探测。
- **`sparse_flag_` 的语义只到「是稀疏」这一位**：`sparse_flag_` 的注释「1 value valid only for
  F32F16/S8/MXF8F6F4」在源码里未逐格式核对；`sparse_id2_` 具体如何被硬件用来选 A 的哪一半（E 的
  `id2`）只引用 `docs/62` §3.2，未新验证。
- **4-bit A 的两支（`MXF4NVF4`）只记不判**：`K=128` 硬编码、`ValTypeA = sparse_elem<4,uint8_t>` 与
  通式不一致，本文只记该静态事实，不解释硬件为何如此。
- **consumer audit 是 `grep` 级**：计数可能受宏/注释影响（`FrgTypeA` 12 处均有效、`ValTypeA
  = sparse_elem<2,...>` 10 处 + 2 处 `sparse_elem<4,uint8_t>`），CUTLASS 更高版本的命名/实现可能变化。

---

## 8. 参考坐标（可核对）

- CUTLASS（`deps/cutlass` `f3fde583`）：
  - `include/cute/pointer_sparse.hpp`：`sparse_elem:45-55`、`is_sparse:63-67`、
    `sizeof_bits<sparse_elem>:80-88`、`is_sparse_ptr:96-98`、`sparse_ptr:101-125`（`operator+ :113-120`、
    `operator[] :122-125`）、`is_sparse_ptr<sparse_ptr>:132`、`make_sparse_ptr:137-146`、`recast_ptr:148-153`；
  - `include/cute/pointer_flagged.hpp`：`smem_sparse_ptr_flag_bits:128`、`smem_sparse_ptr_flag:131`、
    带 flag 的 `make_tensor:137-148`、`upcast/downcast:152-166`（安全起见 `static_assert` 未实现）、
    `print:176-180`；
  - `include/cute/atom/mma_traits_sm90_gmma_sparse.hpp`：`Layout_MN_{INTER,SW32,SW64,SW128}_SpAtom:54-63`、
    `Layout_K_*_SpAtom:68-77`、`Major` 分派 `:82-96`；
  - `include/cute/atom/mma_traits_sm100.hpp`：UMMA 引入 SpAtom `:81-89`、`Layout_MN_SW128_32B_SpAtom:90-92`、
    `smem_desc:359`、`sparse_smem_desc:361`、`MakeTensor<sparse_smem_desc>:379-397`；
    稀疏 `MMA_Traits`：`TF32_SS_SPARSE:1436-1518`（`ValTypeA:1442`、`FrgTypeA:1448`、`K:1454`）、
    `F16BF16_SS_SPARSE:1524`、`TF32_2x1SM_SS_SPARSE:2020`、`F16BF16_2x1SM_SS_SPARSE:2108`、
    `S8_SS_SPARSE:2326`、`S8_2x1SM_SS_SPARSE:2543`、`MXF8F6F4_SS_SPARSE:2790`、
    `F8F6F4_SS_SPARSE:2961`、`F8F6F4_2x1SM_SS_SPARSE:3186`、`MXF8F6F4_2x1SM_SS_SPARSE:3364`、
    `MXF4NVF4_SS_SPARSE:3556`、`MXF4NVF4_2x1SM_SS_SPARSE:3747`；
  - `include/cute/arch/mma_sm100_desc.hpp`：`InstrDescriptor.sparse_id2_:418`、`sparse_flag_:419`、
    `max_shift_:433`、`InstrDescriptorBlockScaled.sparse_*:447-448`、`make_instr_desc:466-502`
    （`sparse_flag_:496`）、`make_runtime_instr_desc:504-550`（`sparse_id2_:538`、注释 `:524`）；
  - `include/cute/arch/mma_sm100_umma.hpp`：稀疏原子 `:300/341/630/671/793/916/1117/1160/1236/1317/1429/1570`；
    1SM asm `tcgen05.mma.sp... [%9]` `:326`（tf32）、`:367`（f16）、`:819`（i8）；2SM `:655/696/941/…`；
  - `include/cutlass/gemm/collective/builders/`：`sm90_sparse_gmma_builder.inl`、`sm100_sparse_umma_builder.inl`、
    `sm100_blockscaled_sparse_umma_builder.inl`、`sm120_sparse_mma_builder.inl`、
    `sm120_blockscaled_sparse_mma_builder.inl`、`sm90_sparse_config.inl`、`sm1xx_sparse_config.inl`；
    swizzle 选择 `sm90_common.inl:340-380`、`sm120_common.inl:126-136`。
- MPK（`f9eb70c`）：对 `sparse_smem_desc`/`sparse_elem`/`SpAtom`/`make_sparse_ptr` **零引用**（`grep` 0 次）。
- 本文实验：`src/sparse_mma_a.cu` + `harness/sparse_mma_a.sh` + `harness/sparse_mma_a.out.txt`
  （目标卡 H100/sm90a，host 侧，未 launch kernel）。
