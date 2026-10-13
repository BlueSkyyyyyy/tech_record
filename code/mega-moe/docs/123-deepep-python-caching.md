# 123 · DeepEP 的 Python 缓存基座：方法级 `weak_lru` 弱引用缓存 vs 模块/静态级 `lru_cache`

> 前面 122 篇把 DeepEP 的 **kernel**（dispatch/combine/PP/Engram/bucket 集合通信）、**host 编排**
> （对称内存、NCCL 域绑定、JIT 配置、环境变量、进程引导）和**测试基座**都拆遍了。但有 27 行
> Python 被反复引用却从没单独讲过——`deep_ep/utils/semantic.py` 的 `weak_lru`。它是全仓库
> **唯一一个「方法级」缓存**：`functools.lru_cache` 直接修饰实例方法会把 `self` 钉在缓存里、
> 让 buffer **永远不被回收**，于是 DeepEP 先用 `weakref.ref(self)` 当 key 再缓存。
>
> 本篇把这 18 行装饰器逐行拆透：**为什么普通 `lru_cache` 会泄漏**、**`weakref.ref` 当 key 的
> 相等性会委托给 referent**（因此「值相等但不同」的实例会撞成同一条——一个真实但 DeepEP 恰好
> 踩不到的陷阱）、**死引用条目不会自动清只受 `maxsize` 约束**、以及**为什么模块级/静态方法的
> `lru_cache` 反而不需要 `weak_lru`**。最后用真实源码在 H100 上跑 39 项验证。

本篇目标源码（`~/github/DeepEP@93eb6eb`，`git describe` = `v1.2.1-41-g93eb6eb`，`__version__ = 2.5.0`）：

- `deep_ep/utils/semantic.py`（27 行，sha256[:16] `8115b73e1ecaf1f5`）：`value_or`（`:5-6`）
  与 `weak_lru`（`:9-26`）。
- 两个 `weak_lru` 消费点（`deep_ep/buffers/ep.py:18` 导入）：
  - `get_theoretical_num_sms`（`ep.py:430-431`，`@weak_lru(maxsize=None)`）；
  - `lb_get_theoretical_num_sms`（`ep.py:845-846`，`@weak_lru(maxsize=None)`）。
- 对照——库内**其它所有**缓存都用普通 `functools.lru_cache`：
  - 模块级：`find_pkgs.py:58`（`find_nccl_root`）、`envs.py:193/200/207/237/260`（5 个带宽/NIC 探测）；
  - 静态方法级：`engram.py:129-130`（`@staticmethod` + `functools.lru_cache(maxsize=None)`）。

**可跑边界**：`semantic.py` 只 `import weakref / functools / typing`，`find_pkgs.py` 只
`import functools / os / sys / importlib.metadata / pathlib / typing`，`envs.py` 用到 torch 但本实验
只碰它的纯 host 探测函数。`import deep_ep` 会触发 `check_nccl_so()`（本机 NCCL 2.29.7 与编译期
不符 → abort，见第 86/117 篇），故实验**按文件路径直接加载真实源码**（`sys.modules` 注入合成
`deep_ep._C` / `deep_ep.comm`，与第 118 篇同法）。这是**纯 Python、架构无关**的一层，H100(sm90)
真机 **39 PASS / 0 FAIL**（`harness/exp_deepep_python_caching.{sh,py,out.txt}`）。

---

## 1. 为什么需要一个「弱引用」的 LRU：先看它服务谁

`weak_lru` 的两个消费者都在 `EPBuffer` 上，都是「**参数固定、被反复调用、结果只取决于实例
自身配置**」的估算函数：

```python
# ep.py:430-431
@weak_lru(maxsize=None)
def get_theoretical_num_sms(self, num_experts, num_topk, num_scaleout_topk=0,
                            rdma_gbs=0, nvlink_gbs=0, sm_read_gbs=0, sm_write_gbs=0) -> int:
    ...
    sm_read_gbs  = get_sm_read_gbs()  if sm_read_gbs  == 0 else sm_read_gbs
    sm_write_gbs = get_sm_write_gbs() if sm_write_gbs == 0 else sm_write_gbs
    if rdma_gbs == 0 and self.num_rdma_ranks > 1: rdma_gbs = get_rdma_gbs()
    if nvlink_gbs == 0:                           nvlink_gbs = get_nvlink_gbs()
    ...  # 带宽建模，返回偶数 SM 数
```

```python
# ep.py:845-846
@weak_lru(maxsize=None)
def lb_get_theoretical_num_sms(self) -> int:
    """Estimate the LB SM count from bandwidths, rounded up to a multiple of 2."""
    nvlink_gbs = get_nvlink_gbs()
    return align(math.ceil(max(nvlink_gbs / get_sm_read_gbs(),
                               nvlink_gbs / get_sm_write_gbs())), 2)
```

它们在 `ep.py:643`（`num_sms == 0` 时自动选 K 数）与 `:882/904`（LB 版）被消费，也在
`bucket.py:324/352/385` 有**同名但未缓存**的兄弟（见 §6）。为什么值得缓存：`get_theoretical_num_sms`
的默认分支会层层调用 `get_nvlink_gbs()`——而那个函数**每次都会 `subprocess.run(['nvidia-smi',
'nvlink', '-s'])`**（`envs.py:207-235`）。训练循环里 dispatch/combine 每步都问一次「用几个 SM」，
参数却完全一样，缓存把子进程调用与建模压成一次。

那为什么不直接 `functools.lru_cache`？那句话就写在装饰器自己的 docstring 里：

```python
# semantic.py:9-13
def weak_lru(maxsize: Optional[int] = 128, typed: bool = False):
    """
    LRU Cache decorator that keeps a weak reference to `self`
    Useful for caching methods in classes that may cause memory leaks if `functools.lru_cache` is used directly.
    From https://stackoverflow.com/a/68052994/16569836
    """
```

---

## 2. `value_or`：只有 `None` 才回退（`:5-6`）

```python
def value_or(value: Optional[Any], default: Any) -> Any:
    return default if value is None else value
```

5 行，语义是「**只把 `None` 当缺省**」。`ep.py:656-676` 用它把「用户显式传的」与「handle/
buffer 里存的」合并。边界在于 `0` / `False` / `''` **都是合法值**，不会被默认值盖掉——这正是
`or` 做不到的（`value or default` 会把合法的 `0` 也换成默认）。实验 Part A 逐条验证：
`value_or(None,7)=7`、`value_or(0,7)=0`、`value_or(False,7) is False`、`value_or('',7)==''`、
`value_or(None,None) is None`。

---

## 3. `weak_lru` 逐行：把 `self` 换成 `weakref.ref(self)` 再喂给 `lru_cache`

```python
# semantic.py:15-26
def wrapper(func):
    @functools.lru_cache(maxsize, typed)          # :17  真正的缓存
    def _func(_self, *args, **kwargs):
        return func(_self(), *args, **kwargs)      # :19  调用前把弱引用「解引用」
    @functools.wraps(func)                         # :21  保留原函数元数据
    def inner(self, *args, **kwargs):
        return _func(weakref.ref(self), *args, **kwargs)   # :23  key 是 weakref，不是 self
    return inner
return wrapper
```

关键在 `:23`：传给缓存的第一个实参是 `weakref.ref(self)` 而不是 `self`。于是缓存里键存的是
**弱引用**——它不会增加 `self` 的引用计数，实例用完即可被 GC 回收；而当 `_func` 真要算值时，
`:19` 的 `_self()` 再解出对象传给原函数。`functools.wraps`（`:21`）保证 `inner` 的
`__name__` / `__doc__` / `__wrapped__` 与被装饰方法一致。

> 注意 `functools.lru_cache` 的第一个位置参数就是 `maxsize`（`:17`），所以 `typed` 也被透传；
> `weak_lru` 自己默认 `maxsize=128, typed=False`（`:9`），而 `ep.py` 两处都显式传 `maxsize=None`
> （**无界**：实例少、参数固定，偏向「永不驱逐」）。

---

## 4. 复现泄漏：普通 `lru_cache` 为什么钉死实例

`functools.lru_cache` 把**调用参数整体**当缓存键，实例方法调用时第一个参数就是 `self`——于是
`self` 被缓存强引用，只要缓存不驱逐，实例永不回收。实验 Part B 用标准库最小复现：

```
class PlainCached:
    @functools.lru_cache(maxsize=None)
    def value(self, x): return x + 1

p = PlainCached(); rp = weakref.ref(p); p.value(1); del p; gc.collect()
assert rp() is not None          # ← 实例还活着（被缓存钉住）
PlainCached.value.cache_info()   # CacheInfo(hits=0, misses=1, maxsize=None, currsize=1)
```

对照 `weak_lru`：同样的 `del + gc.collect()` 后 `weakref.ref` 立即变 `None`（实例已回收）。
实测 **`plain` 泄漏 = True / `weak_lru` 可回收 = True / `currsize` 永久占位 = 1**。这就是
DeepEP 在 EP buffer 上必须用 `weak_lru` 的理由——一个 buffer 可能持有几百 MB 对称显存与
NCCL 通信域，被一个缓存钉住是灾难性泄漏（呼应第 111 篇的生命周期契约）。

---

## 5. `weak_lru` 的缓存键语义与两个真实陷阱

既然 key 是 `weakref.ref`，它的行为就决定了缓存的语义。Python 规定弱引用
**支持相等性测试，且当 referent 仍存活时，两个弱引用拥有与 referent 相同的相等关系与哈希**
（referent 已死后才退化为「仅同一对象相等」）。实验 Part C 证明：

- **同实例同参 → 命中**；同实例不同参 → 新条；**不同（身份相等的）实例同参 → 各占一条**
  （`misses==3`）。这对 `EPBuffer` 是正确的，因为它用默认的「身份」`__eq__`。
- **陷阱一：值相等但不同的实例会撞成同一条。** 定义一个 `__eq__/__hash__` 按值比较的类再套
  `weak_lru`，两个 `v==1` 的不同实例的 `weakref.ref` 相等、哈希相同，于是**第二个实例拿到的是
  第一个实例缓存的结果**（实测 `e1.probe() == e2.probe() == id(e1)`）。DeepEP 的 buffer 没有
  值语义的 `__eq__`，所以踩不到；但这是「用 `weakref.ref` 当 key」的一般性代价，值得记住。
- **陷阱二：死引用条目不会立刻清。** `del` 掉实例后，缓存里那条键指向的 weakref 已死，但**条目
  仍在**（不自动过期），只受 `maxsize` 约束：`weak_lru(maxsize=4)` 跑 20 个临时实例，`currsize`
  稳定在 4、`misses==20`（每实例都算一次，键按实例不同）。换句话说弱引用**只解决「不钉住
  实例」**，条目数仍靠 `maxsize` 控。
- `typed=True` 时 `1` 与 `1.0` 是不同键；`typed=False` 时被当作等价（短路于先到的那个）。
- **不可弱引用对象直接报错**：`__slots__` 且未加 `__weakref__` 的类调 `weak_lru` 会抛
  `TypeError: cannot create weak reference to 'NoWeak' object`（`weakref.ref(self)` 在 `:23`
  就失败）。这决定了 `weak_lru` 只能用于普通可弱引用实例。

---

## 6. 全库普查：三种缓存落点，以及一个「该缓存却没缓存」的对照

用脚本解析 `deep_ep/**/*.py` 的所有 `lru_cache` / `weak_lru` 装饰器（实验 Part D），结论干净：

| 落点 | 位置 | key 里有没有 `self` | 用的装饰器 |
|---|---|---|---|
| 模块级函数 | `find_pkgs.py:58`、`envs.py:193/200/207/237/260` | 无 | `functools.lru_cache()` |
| 静态方法 | `engram.py:130` | 无（`@staticmethod` 无 `self`） | `functools.lru_cache(maxsize=None)` |
| **实例方法** | `ep.py:430`、`ep.py:845` | **有** | **`weak_lru`** |

一条判据贯穿：**只有 key 里含 `self` 的实例方法才需要 `weak_lru`**。静态方法（`@staticmethod`
不绑定 `self`）与模块级函数都没有这个问题，直接用标准库即可——所以 `engram.py:129-131` 的
`get_theoretical_config` 用普通 `lru_cache` 是**刻意且正确**的，不是漏改。普查同时确认：全仓库
**没有任何一处「实例方法直接用 `functools.lru_cache`」**（`semantic.py:17` 那一行是 `weak_lru`
内部「怎么建缓存」的定义，不是一次使用）。

一个有意思的对照：`BucketBuffer.get_theoretical_num_sms`（`bucket.py:236`）与 EP 的同名方法
职责相似、也被 `:324/352/385` 每类集合通信调用，却**没有加任何缓存**。也就是说缓存不是无差别
铺开的——EP 的 dispatch/combine 在训练循环里高频且参数恒定，`maxsize=None` 无界缓存最划算；
bucket 的调用路径（每类 op 一次）则没必要。

---

## 7. 模块级缓存也会咬人：它吞掉「环境变化」

模块级 `lru_cache` 看着无害，但它的键是**函数参数**，不看函数体读到的**外部状态**。实验 Part E
用真实 `envs.py` 演示这个副作用：`get_nvlink_gbs`（`envs.py:207`）会 `subprocess.run` 解析
`nvidia-smi nvlink -s`。我们把 `envs.subprocess.run` 换成假输出：

1. 首次喂「2×100 GB/s」→ 返回 **200.0**；
2. 改成「2×10 GB/s」再调 → **仍返回 200.0**（缓存命中，未重跑子进程）；
3. `get_nvlink_gbs.cache_clear()` 之后再调 → 才读到 **20.0**。

同理 `find_nccl_root`（`find_pkgs.py:58`）带 `lru_cache`，第二次调用 `hits==1`、返回稳定路径
`/usr/local/lib/python3.12/dist-packages/nvidia/nccl`，`get_nccl_lib_name` 给出 `libnccl.so.2`。
实践含义（与第 117 篇的环境变量面相接）：**运行期改 `EP_*` / 换 NIC / 动态加载库后，这些探测结果
不会自动刷新**，需要显式 `cache_clear()` 或重启进程——这也是为什么这些探测函数大多在进程早期
一次性定型。

---

## 8. 用真实装饰器复刻 `lb_get_theoretical_num_sms`

实验 Part F 直接 import 真实 `weak_lru`，把 `lb_get_theoretical_num_sms` 的算术复刻成一个方法：
`align(ceil(max(nvlink/read, nvlink/write)), 2)`。同参第二次调用**只计算一次**（`calls==1`），
结果 `430.3/…` 下为偶数 10；改一个参（写带宽 45→30）后 `calls==2`、`ceil(430.3/30)=15→align 到 16`。
这坐实了装饰器对真实消费者的行为：**同参命中、异参各自成条、结果只算一次**。

---

## 9. H100 实测（`harness/exp_deepep_python_caching.{sh,py,out.txt}`）

实验按文件路径加载真实 `semantic.py` / `find_pkgs.py` / `envs.py`（`sys.modules` 注入合成
`deep_ep._C` / `deep_ep.comm` 绕开 `check_nccl_so`），并解析 `ep.py` / `engram.py` 源码。
源文件 sha256 见输出（`semantic.py 8115b73e1ecaf1f5`、`find_pkgs.py 5464c1a307f36b5b`、
`envs.py 23843e168a6f5d43`、`ep.py a8f72c3975bbf043`、`engram.py 92d39d989af238dc`）。
**39 PASS / 0 FAIL**：

- **Part A（5）** `value_or` 的五条语义全对（`None`/`0`/`False`/`''`/`None`）。
- **Part B（5）** 复现泄漏：普通 `lru_cache` 下实例 `del+gc` 后仍存活且 `currsize==1`；
  `weak_lru` 下实例可回收；`weak_lru` 保留 `__name__` 与 `__wrapped__`。
- **Part C（11）** 缓存键语义：同实例命中、异参新条、不同实例各占一条（`misses==3`）、
  `weakref.ref` 相等性委托成立、**值相等实例撞同一条（陷阱）**、死引用条目 `currsize<=maxsize`、
  20 实例 20 miss、`typed` 开关、`__slots__` 抛 `TypeError`。
- **Part D（4）** 全库普查：`weak_lru` 恰 2 处（`ep.py` 两个方法）、模块级 `lru_cache` 恰 6 处
  （`envs` 5 + `find_pkgs` 1）、静态方法级恰 1 处（`engram.py:130`）、无「实例方法直接
  `functools.lru_cache`」。
- **Part E（10）** 真实模块级缓存：`find_nccl_root` 命中与路径稳定、`get_nccl_lib_name`、
  带宽常量 180/45、`get_nvlink_gbs` 的「吞掉环境变化 → `cache_clear()` 才更新」（200→200→20）。
- **Part F（4）** 真实 `weak_lru` 复刻 `lb_get_theoretical_num_sms`：同参只算一次、异参再算、
  结果偶数。

---

## 10. 小结

- `weak_lru` 是 DeepEP 唯一的**方法级**缓存：把 key 从 `self` 换成 `weakref.ref(self)`，让实例
  可被 GC，从而不做缓存泄漏；代价是 `weakref.ref` 的相等性会委托给 referent。
- 判据只有一句：**key 里含 `self` 才需要 `weak_lru`**。静态方法 / 模块级函数用标准库
  `lru_cache` 即可，且 DeepEP 全仓库遵循了这一分工。
- 两类真实陷阱：**值相等但不同的实例会撞同一条缓存**（DeepEP 因 buffer 无值语义 `__eq__`
  而恰好安全）；**死引用条目不会自动清，只受 `maxsize` 约束**。
- 模块级缓存会**吞掉环境变化**——改 `EP_*` / 换 NIC 后需 `cache_clear()` 或重启，这与第 117 篇
  的环境变量面互为脚注。
- 这是纯 Python、架构无关的一层：H100(sm90) 真机 39 / 39 全过，与目标卡是否 SM100 无关。
