# MPK 依赖分析器源码走读：从 KNGraph 到 tGraph

> 本文是 `docs/06-mpk-source.md`（运行层）与 `docs/07-mpk-tasks.md`（单 task device 实现）的**上游补齐**：
> MPK 论文 §4.1 用五个步骤描述「算子图 → tGraph」（算子分解 / 依赖分析 / event fusion / normalization /
> linearization），但 `docs/06` 是**从 tGraph 已经被构造好之后**讲起（`generate_task_graph` → worker/
> scheduler），并没有走这一段。本文落到**真实可核对的源码**上：主文件是
> `src/kernel/annotated_graph.cc`（1014 行）与其头文件 `include/mirage/kernel/annotated_graph.h`（198 行）。
>
> 主参考仓库 `https://github.com/mirage-project/mirage`，分支 `mpk`，本机快照
> **commit `f9eb70c`（与 `docs/06`/`docs/07` 同一快照）**，所有行号都对着这次快照核对过。
> 论文侧证据见 `~/github/ai-infra-book/experiments/ch05/05-08/sources/mpk-v2.txt`（§4.1 行 449–1146）。
>
> 定位：`docs/01` §3 解决「论文的算法说了什么」，`docs/06`/`docs/07` 解决「tGraph 之后的运行层/device
> 层怎么跑」，本文解决「**tGraph 本身是怎么被分析、融合、归一、线性化出来的**」。与 `docs/01` 的算法
> 推导、`docs/06` 的运行层不重复。
>
> 本文为**纯源码走读（资料分析）**：mirage 是完整编译器/运行时，在目标卡 H100/sm90 容器里从零编译
> 运行成本过高，本轮不做构建/运行实验，所有结论均标注到文件与行号。

## 先记住一个画面

把 tGraph 想成一张**排课表**。

- 每门课是一个 task（一块输出 tile 的计算）；
- 「先修课都上完了」才准开课——这张放行条就是 event；
- 依赖分析器干的事，就是**从一堆算子调用里推出谁必须在谁之前**，再把「同一批先修课」
  合并成尽量少的放行条。

难点在于：算子是按 tile 切的，一个算子写的是**一大片**输出，下游只读其中**一块**。
所以「最近写者」不能简单按语句顺序连边，得看**写覆盖了谁**；连完之后还要检查有没有环、
把冗余边剥掉，最后才能定一个全局顺序。

> **类比的边界**：排课表里「先修」是人对课程内容的判断；这里必须由**地址区间**推出来——
> 覆盖判断一旦有遗漏，就会少一条边，运行期表现为随机读到旧数据，而不是编译报错。
> 另外 event 不是「越少越好」：合并过头会把本可并行的 task 串起来（形状/角色约束会拦住一部分，
> 但拦不全）。

读完这一篇，你应该能回答：

1. 「最近写者 SSA」在 tile 粒度上为什么会失效？`bbox` 阴影覆盖补的是什么洞？
2. 环是怎么检出来的、残余边为什么可以剥？
3. event fusion 的 GCD / LCM 各自合并的是哪种结构？合并的边界在哪？

## 0. 定位与源码地图

`generate_task_graph`（`src/kernel/runtime.cc:1915`，`docs/06` §1.1 走读）内部的**第一步**就是调用
`build_annotated_graph(graph, task_configs)`（`src/kernel/runtime.cc:243`）：

```cpp
// runtime.cc:236-243
// Build the AnnotatedGraph. This replaces the old chain-only matching ...
AnnotatedGraph ag = build_annotated_graph(graph, task_configs);
```

`AnnotatedGraph`（头文件 `:149-156`）是编译期的**中间表示（IR）**：把 KNGraph 的层面板（`LayerInfo`）、
数据流边（`EdgeInfo`）、fork/join 组（`ForkGroupInfo`/`JoinGroupInfo`）和**线性化后的层序**
（`ordered_layers`）全部冻结下来。`runtime.cc:642` 之后就是**遍历 `ordered_layers`** 去 emit
`FullTaskDesc` + `EventDesc`——也就是说 **tGraph 的任务表与事件表是在这一步由 `AnnotatedGraph` 展开出来的**。

| 文件 | 行数 | 角色 |
|---|---|---|
| `include/mirage/kernel/annotated_graph.h` | 198 | IR 定义：`TaskView` / `EdgeInfo` / `ForkGroupInfo` / `JoinGroupInfo` / `LayerInfo` / `AnnotatedGraph` |
| `src/kernel/annotated_graph.cc` | 1014 | **依赖分析器主体**：`build_annotated_graph`（119–975）与 dump（977–1011） |
| `src/kernel/runtime.cc` | 1972 | 消费者：`build_annotated_graph` 调用（243）、按 `ordered_layers` emit task/event（642+） |
| `include/mirage/kernel/device_tensor.h` | — | `is_virtual()` / `resolve_base_guid()`：view 语义（`:122-133`） |

`build_annotated_graph` 的流水线在头文件注释里就列全了（`annotated_graph.h:162-179`），**十个步骤 (a)–(j)**：

```
(a) DAG construction with most-recent-writer   (b) cycle check (Kahn)
(c) residual stripping (transitive reduction)  (d) role classification
(e) case 2/3 rejection                         (f) topological ordering
(g) per-edge GCD event_dim + per-side views    (h) fork LCM
(i) join LCM                                   (j) tag fork-consumer bundles
```

这十步与论文 §4.1 的映射是本文的主线：

| 论文 §4.1 步骤 | 快照行号 | 实现位置（本文节） |
|---|---|---|
| 算子分解 operator decomposition | 462–468 | 不在本文件：`task_configs`（每算子的 `(num_inputs,num_outputs,TaskType,variant)`）与 bgraph 的 `grid_dim`（§1） |
| 依赖分析 dependency analysis | 469–519 | Step (a) 最近写者 SSA + bbox 阴影覆盖（§2）、Step (b) 环检测 / (c) 残余边剥离（§3） |
| event fusion | 520–779 | Step (g) 逐边 GCD + (h) fork LCM + (i) join LCM（§6） |
| normalization | 780–951 | Step (e) case 2/3 拒绝 + barrier edge 坍缩（§4、§6.2） |
| linearization | 952–1098 | Step (f) 两趟 Kahn 拓扑排序 → `ordered_layers`（§5） |

> 一句话：**论文把这五步画成 Figure 5 的流程图；实现把它们压进一个 1014 行的函数，用「最近写者
> + bounding box 阴影覆盖」做依赖分析、用「GCD 逐边 + LCM 汇聚」做 event fusion、用「拒绝非法角色
> 组合 + barrier 坍缩」替代 normalization 的重写、用「按 depth 的 Kahn」做 linearization。**

## 1. IR 的数据结构：event_dim（张量维）与 last3（网格轴）

理解本文件的关键是分清两个坐标系（头文件 `TaskView`，`annotated_graph.h:52-57`）：

```cpp
struct TaskView {
  std::array<int, MAX_TENSOR_DIMS> event_dim;  // 每个张量维上有几个 event
  std::array<int, 3> last3;                    // 每个网格轴上一个 event 覆盖几个 task
  dim3 grid_dim;                               // 本层 bgraph 的网格
  int3 axis_map;                               // 网格轴 -> 张量维（-1 表示复制轴）
};
```

- **`event_dim` 活在「桥接张量」的维空间**：它描述「这条边在张量的每个维上被切成几段 event」，
  因此对两侧网格怎么映射到这个张量是**不变的**。
- **`last3` 活在「拥有该张量那一层的 task 网格轴空间」**：一个 event 对应网格上的一个**子立方体**
  （连续的一批 task），用 `last3[g]` 表示它在每条轴上的边长。它由
  `derive_last3`（`annotated_graph.cc:42-58`）算：

```
last3[g] = axis_map[g] >= 0 ? grid[g] / event_dim[axis_map[g]] : grid[g]
```

即「把网格轴映射到某个带 event 的张量维就除，映射到复制轴（`-1`）就整条轴算一个 task block」
（`annotated_graph.cc:37-41` 的注释）。**`event_dim` 的连乘 = 这条边界上的 event 总数**；
`last3` 的连乘 = 一个 event 覆盖的 task 数。事件总数 × 每事件 task block = 两侧网格规模。

`EdgeInfo`（`annotated_graph.h:75-94`）是**一条数据流边**。注意两个刻意的设计：

1. **边的身份是 `(prod_layer, out_slot, cons_layer, in_slot)`，不是张量 guid**
   （`:62-67` 注释）。qwen3 会大量**复用物理 buffer**（`x = attn_out; x = mlp_out`），同一个 guid 在
   程序里会被多个不同的生产者写；用 `(prod, out_slot)` 把每次读**钉死到「读注册时刻的当前写者」**，
   天然避免伪环。
2. 每条边存**两侧各自的 `TaskView`**（producer_side_view / consumer_side_view）。头文件 `:30-37`
   解释：一个 task 的 `trigger_event`（完成时**发**的事件）与 `dependent_event`（**等**的事件）分别由
   producer 侧 / consumer 侧视图驱动，两者独立，因此同时扮演两类角色的层也不会冲突。

`ForkGroupInfo` / `JoinGroupInfo`（`:106-122`）分别聚合「一个 fork 生产者的所有出边」与
「一个 join 消费者的所有入边」，各自记录 LCM 后的统一 `lcm_last3`；`LayerInfo`（`:124-147`）记录每层
的 `in_edges`/`out_edges`（**已剔除残余边**）、fork/join 标志、bundle 归属与 `task_type`。

## 2. Step (a) 依赖分析：最近写者 SSA + bbox 阴影覆盖

Step (a)（`annotated_graph.cc:119-425`）的核心注释（`:123-136`）说得很直白：

> qwen3（及大多数真实 MPK 模型）在层间复用 DTensor buffer，同一 guid 会被多个 customized op 写。
> 纯按 guid 匹配会造出伪环。规则：**层 L 读张量 g 时，把边绑到 L 之前「最近」写过 g 的那个生产者**；
> L 处理完后，L 的输出更新 `last_writer`。这就是 SSA 式的 def-use，隐式重命名了被复用的 buffer。

实现上：

- `last_writers`（`:154`）以 `resolve_base_guid()`（view 归一到根存储，见 `device_tensor.h:132`）为键，
  每个条目是一个 `WriterEntry{layer, out_slot, bbox, is_virtual_writer}`（`:148-153`）。
- `compute_bbox`（`:170-228`）把 DTensor 映射成**父存储坐标里的包围盒**：非虚张量就是 `[0,dim)`；
  同秩 narrow view 用 `view_offset / stride` divmod 逐轴还原起点；reshape（秩不同）或非行主可分解的
  view 走**保守回退**——当成整个父张量（宁可多报重叠，绝不漏报，`:166-169`）。
- **阴影覆盖（shadow-cover）** 是 (a) 最有意思的部分（`:331-369`）：对一次读，**逆序**遍历该存储的写者
  列表，维护「读者 bbox 中尚未被更晚写者覆盖的部分」这一组 N 维超矩形（`uncovered`）。每个写者只
  对它**最近写过且未被后续写者盖住**的子区域贡献一条边（`bbox_intersect` `:234-253`、
  `bbox_subtract` `:258-276`）。这样「同一 buffer 被多路写、再被整体读」会被拆成**按区域精确的少数边**，
  而不是把读者错误地连到每一个历史写者。
- **view 引发的 barrier 边**（`:391-400`）：当生产者/消费者触碰的是**同一个父存储的不同窗口**
  （部分重叠 / view-vs-base / 形状不一致）时，逐 tile 的对应关系不成立，于是把这条边标成
  `is_barrier_edge`，后续会坍缩成一个粗粒度 event（§6.2）。若两侧引用**同一个 view 窗口**，则保留
  细粒度 GCD 分析。
- 读到输入后**更新写者**（`:409-424`）：非虚写（全存储覆盖）**清空**该存储所有历史写者再登记一条全窗口；
  虚写（写 view）**追加**一条部分窗口记录，让多个写不相交切片的生产者共存。

对照论文 §4.1 的「依赖分析」：论文说「对共享张量的任意两算子，枚举所有 task 对 `(t1,t2)`，当且仅当
`t1` 的输出区域与 `t2` 的输入区域重叠时引入一个 event `e`」。实现里，「区域重叠」正是 bbox 的
`intersect/subtract`，「最近写者」则额外解决了**buffer 复用**（论文表述里没提，但真实模型必须有）。

## 3. Step (b)(c) 环检测与残余边剥离

**Step (b)（`:429-465`）**：Kahn 拓扑排序做环检测，有环就把 offending layer 列表打进异常
（`:455-464`）。在最近写者规则下，良构程序本不可能成环，这是**防御性 guard**。

**Step (c)（`:467-551`）残余边剥离 = 传递规约（transitive reduction）**。注释（`:470-482`）解释：

> 一条「残余边」`u->v` 是 transformer 残差直连：`u` 既分叉出计算路径 `u->w->...->v`，又直连 `v`
> （`v` 做残差相加）。**直连边要剥掉**，因为长路径的事件链已经传递地保证 `u` 完成前 `v` 不启动，
> 直连边不提供任何调度信息，留着只会把 `v` 误判成 join-consumer 并可能触发 case-2 违规。

实现先对**原边集**做一次全源 BFS 求可达性 `reachable`（`:488-507`，复杂度 `O(V(V+E))`，注释说
qwen3 的 ~300 层 / ~400 边亚毫秒级，不值得上更聪明的算法），然后**一次性**标记所有要剥的边
（`exists w: u->w 且 w⇝v`，`:509-536`），最后从每层的 `in_edges`/`out_edges` 里原子地移除
（`:537-550`）——边记录仍留在 `ag.edges` 里，只是打了 `is_residual_stripped=true`。
「一次性」很关键：若迭代剥离，可能先剥掉长路径再处理直连边，模式就检测不到了（`:478-481`）。

这一步骤把 transformer 的「跳跃连接」结构**规约成纯链**，直接减轻后续 fork/join 分类的负担。

## 4. Step (d)(e) 角色分类与 case 2/3 校验（normalization 的约束面）

**Step (d)（`:553-581`）**：对每层按**不同**下游/上游层计数，`>=2` 才标 fork-producer /
join-consumer。注释（`:556-562`）记了一个真实踩坑：一个层可以产出多个输出张量喂给**同一个**消费者
（融合 attention 的两个输出都喂下一 block），按**原始出边数**会误判成 fork、造出假 case-3；所以要用
「distinct 层数」。

**Step (e)（`:583-662`）case 2/3 拒绝**——这是实现与论文 normalization 的**关键差异点**。
头文件 `:181-189` 与源码 `:583-601` 都点明：`FullTaskDesc` **只有一个 `trigger_event` 槽和一个
`dependent_event` 槽**，因此两类组合不可表示：

- **case 2**（join-consumer + fork-consumer）：X 的 task 需要被**两个** event 触发
  （上游 fork event 与本层 join event），槽位不够；
- **case 3**（join-producer + fork-producer）：L 的 task 需要**发两个** event，槽位不够。

两者总是成对出现（`:598-601`），实现按层序**先命中先拒绝**，异常信息里把相关边逐条打印
（`:621-662`）。

论文 §4.1 把 normalization 描述为**重写**：当 `T0` 触发多个 event `e1..ek` 时，引入新 event `e'`
（Figure 6，快照行 780–951）。而线上实现采用的是**另一条等价路线**：不是「用空 task 把扇入/扇出压到 1」，
而是（1）**编译期直接拒绝**需要两个槽位的非法角色组合；（2）对合法的分歧/汇聚用 **fork/join LCM**
把一组分支的 event 语义统一成单一 event（§6），从而天然满足「每 task 至多一个 trigger、一个 dependent」
（见 `:564-570` 的注释：barrier 边经 step (g) 取 `event_dim=1` 后，fork/join LCM 会把混合 bundle
统一降成一个 event，正是 barrier 语义）。**结论：论文的 normalization 在实现里是「拒绝 + LCM 降级」
的组合，而不是 Figure 6 的显式空-task 重写。**

## 5. Step (f) 拓扑线性化：两趟 Kahn + (depth, index) 定序

**Step (f)（`:664-745`）就是论文 Algorithm 1 的 linearization**，输出 `ag.ordered_layers`
（`runtime.cc:644` 直接按它 emit task/event）。它跑**两趟 Kahn**：

1. 第一趟（`:687-710`）用任意合法拓扑序算出 `depth[v] = max_{(u,v)∈E}(depth[u]+1)`；
2. 第二趟（`:711-745`）用**最小堆**按 `(depth, index)` 排序——`depth` 小者优先，同 `depth` 按原始
   插入序。

为什么用 depth 而不只用层序？注释（`:665-684`）给了反例：对称 fork（`A->B->C->D` 与
`A->E->F->G`）若只按 index，会输出 `A,B,C,D,E,F,G`（alpha 链整条先出），导致 C/F 的 chain event
落在 prelaunch 队列两端、虽同 depth 却不同步推进；按 depth-first-ascending + index-second 则输出
`A,B,E,C,F,D,G`——**同一 depth 层内仍保持插入序，跨层则两条链锁步前进**。对纯链图（qwen3 剥掉残余后
`depth == 层序`）就退化成原来的插入序。

论文 Algorithm 1（快照行 952–1098）保证「每个 task 入队一次、每个 event 入队一次」且「**一个 event
触发的所有 task 在 `T` 中连续**」。实现的连续性不靠排序本身，而靠 **`runtime.cc` 的 emit 顺序**配合：
fork bundle 由 branch 0「头」**交错** emit 所有分支的 task（`runtime.cc:748-795`，`bundle_emitted`
跳过其它分支 `:269-273`、`:663-665`），使「一个 fork event 的消费者范围」在 `all_tasks` 里连续；
join 消费者只有一层，天然连续（`runtime.cc:810-872`）。§7 展开。

## 6. Step (g)(h)(i) event fusion：逐边 GCD + fork/join LCM

这是实现的 event fusion，对应论文的 **successor-set fusion / predecessor-set fusion**
（Definition 4.1 / 4.2，快照行 520–779）。

### 6.1 逐边 GCD（Step (g)，`:747-784`）

对每条**非残余**边，若它不是 barrier 边：

```
event_dim[d] = gcd( producer_partition[d], consumer_partition[d] )
```

其中 `build_partition`（`:60-77`）把一层的 `grid_dim` 按 `axis_map` 铺到张量维上
（被映射的维 = 该轴的网格大小，未映射维 = 1）。**即：两侧沿某张量维各切成多少段，取它们的最大公约数
作为 event 数**——把两侧的分块边界对齐，只保留公共的切分点（这正是「对任意 task 对枚举重叠」的
压缩形式）。然后据此为两侧各算一套 `TaskView`（`:773-783`）。

barrier 边（§2）特殊处理（`:756-765`）：`event_dim` 全设为 1 → `last3` = 整条网格 → **一条边一个
event，横跨全部生产者 task、触发全部消费者 task**，就是粗粒度屏障语义。

### 6.2 fork LCM（Step (h)，`:786-885`）= successor-set fusion

对一个 fork 生产者 F，把**所有出边的 producer 侧 `last3` 取逐轴 LCM**
（`lcm_last3[g]`，`:820-829`），再把「增益因子」`scale = lcm_last3[g] / branch_last3[g]` 通过
`event_dim[d] /= scale`（`d = axis_map`）**吸收掉**（`:849-871`）。效果：**event 数量变少、每个 event
覆盖更大的 task block**——即「若干条只被同一组消费者依赖的 event 被合成了一个」，正是
successor-set fusion 的语义。安全性检查（`:830-842`）：`lcm_last3` 必须整除 F 的 `grid_dim`
（因为每条分支的 `last3` 都是 `grid_dim` 的因子，因子们的 LCM 仍是因子）。注释 `:786-804` 解释了为何
在**网格轴空间**而非张量维空间取 LCM：F 的网格被所有分支共享，而每条分支的桥接张量/消费者网格可以
不同，网格轴是唯一可比的公共坐标系。最后重算两侧视图（`:872-883`）。

### 6.3 join LCM（Step (i)，`:887-958`）= predecessor-set fusion

与 fork 完全对称：对一个 join 消费者，把**所有入边的 consumer 侧 `last3` 逐轴 LCM**
（`:905-913`）、按同样方式缩 `event_dim`（`:931-947`）、重算视图（`:948-956`）。每个 join event 对应
消费者网格的一个子立方体，收集来自 N 条入边的完成信号。同样有「LCM 整除 `grid_dim`」的安全检查
（`:914-923`）。

### 6.4 Step (j)（`:960-972`）

把 fork 组的直接消费者层打上 `fork_parent_group` + `fork_branch_index`，供 `runtime.cc` 做「头分支
emit 全部、其余跳过」的交错发射。

> **实现的 event fusion 与论文的差异**：论文用 InTasks(e)/OutTasks(e) 的**集合相等**来定义
> successor/predecessor-set fusion（「合并那些要求同一组消费者的 event」），是**图级重写**；
> 实现则用**分块因子的 GCD/LCM** 等价地达到「事件数最小、每个事件覆盖最大的、共享同一组 task 的
> 连续块」。二者目标一致（消除冗余同步点、让 tGraph 规整），但实现的表述是**分块算术**而非集合论。

## 7. 与 runtime.cc 的接合：IR 如何变成 EventDesc/FullTaskDesc

`build_annotated_graph` 返回后，`runtime.cc:642` 起遍历 `ag.ordered_layers`，按层的角色 emit：

- **首层**（无入边、非 fork 消费者，`:679-727`）：按 bid 字典序 emit，并把 `first_tasks` 记上。
- **普通链层**（单一上游，`:878+`）：调用 `dfs_create_events_add_tasks`（`runtime.cc:76-219`）——它按
  `event_dims` **递归展开**（`depth` 走到 `MAX_TENSOR_DIMS=4`），在每个事件处切出消费者/生产者的 bid
  子范围（`:102-121` / `:124-153`），生成一个 `EventDesc`，把生产者的 `trigger_event` 指向它并累加
  `num_triggers`；事件类型按「消费者 task 数 ≥ 8」分成 `EVENT_LAUNCH_MASSIVE_TASKS` / `EVENT_LAUNCH_TASKS`
  （`:158-161`）。
- **fork 组**（`:735-807`）：只处理 branch 0，对每个 fork event **交错**推入各分支消费者 task
  （保证连续范围），再 `set_producer_triggers` 在生产者 `[lcm_last3]` 子范围上设置 trigger。
- **join 消费者**（`:810-872`）：按 join event 主序推入单层消费者 task，再从**每条入边**累积
  `num_triggers` 到同一个 `EventDesc`。

这样，论文 tGraph 的「task 触发 event、event 触发 task」交替结构，最终变成 `all_tasks` /
`all_events` 两张表，被 `docs/06` §1.1 的 `_init_persistent_kernel` 硬编码进生成的 C++、由
`launch_persistent_kernel` 单次常驻执行。

## 8. 与论文 §4.1 的逐条对照表

| 论文 §4.1（快照行号） | 线上实现（`f9eb70c`） | 备注 |
|---|---|---|
| 算子分解（462–468） | 不在 `annotated_graph.cc`；由 `task_configs` + bgraph `grid_dim` 提供（`runtime.cc:287-311`） | 分解策略本身在 `task_register.cc`/前端，`docs/07` §0 已走读 |
| 依赖分析：区域重叠 → event（469–519） | Step (a) 最近写者 SSA + bbox intersect/subtract 阴影覆盖（`:119-425`）；额外处理 **buffer 复用/view**（论文未提） | 行为等价，实现更保守（reshape view 回退整父张量） |
| — 环检测 | Step (b) Kahn（`:429-465`） | 论文未显式列，实现加了防御 guard |
| — 传递规约 | Step (c) 残余边剥离（`:467-551`） | 论文未单列，实现用它把残差规约成链 |
| event fusion（520–779） | Step (g) 逐边 GCD + (h) fork LCM + (i) join LCM（`:747-958`） | 论文按集合论重写；实现按分块因子 GCD/LCM，目标一致 |
| normalization（780–951） | Step (e) case 2/3 拒绝 + barrier 边 `event_dim=1` 坍缩（`:583-662`、`:756-765`） | 论文用 Figure 6 空-task 重写；实现在编译期拒绝非法组合 + 靠 LCM 降级，**不是显式重写** |
| linearization Algorithm 1（952–1098） | Step (f) 两趟 Kahn（`:664-745`）→ `ordered_layers` | 「每 event 的 task 连续」由 emit 顺序 + fork 交错保证（`runtime.cc:748-795`） |

## 9. 局限与复现要点

- **纯源码走读，未构建/运行**：以上行号对着 `mirage-project/mirage` 分支 `mpk` 的 commit `f9eb70c`
  核对；mirage 完整编译器/运行时在容器从零编译成本过高，未做本机实验。`docs/01` §4 的量化
  （每算子 32–47 task、fusion 37–118×、linearization 4.4–15×）来自论文，本文不复述。
- **版本差异提示**：头文件注释引用的设计文档 `docs/mpk/parallel_path_design.md` 在该快照里**并不存在**
  （`docs/mpk/` 下只有 `tma.md`），说明这些注释是随实现演进的**设计意图**，不能当作规范。
- **normalization 的实现口径**是本文最值得注意的一点：论文的「空 task 重写」在线上是
  「拒绝 case 2/3 + fork/join LCM 降级」。二者都保证「每 task 至多一个 trigger / 一个 dependent event」
  这一 `FullTaskDesc` 的硬约束，但**实现不会自动救活非法角色组合，而是编译期报错**——这对写新模型的
  人是个硬边界。
- **交叉引用**：运行层见 `docs/06-mpk-source.md`；单 task device 实现见 `docs/07-mpk-tasks.md`；
  论文算法推导见 `docs/01-mpk-paper.md` §3；生产案例的段级手工版本见 `docs/02-production-cases.md`
  与 `code/mega-moe/` 专题（DeepGEMM mega 家族没有这套自动依赖分析器，靠手工静态角色）。
- **复现命令**（供后续增量）：

```bash
git clone --depth 1 -b mpk https://github.com/mirage-project/mirage
# 打开编译期 dump（env 开关见 annotated_graph.cc:977-1011）：
MIRAGE_DUMP_ANNOTATED_GRAPH=1 <compile your model>
```
