# 链表

数组在内存里是连续的一整块，可以随意按下标跳着访问；链表恰恰相反，节点散落在各处，
只能顺着 `next` 指针一个一个走。这个「不连续」是链表所有特点的根源：**插入、删除只要改几根指针，
但代价是丢失了随机访问，且操作稍不留神就会断链**。

也正因为如此，链表题的难点不在算法思想，而在**指针操作的准确与边界**。好消息是，链表题的
套路极其集中——把本篇的三个基础模板（反转、虚拟头结点、快慢指针）练熟，后面绝大多数链表题
都是它们的组合与变形。写链表的代码，动手前先画图、想清楚「哪几个指针需要在改指向之前先备份」，
比背代码更重要。

本篇题目（由易到难）：

| 模式 | 题目 | 难度 |
|---|---|---|
| 原地反转（三指针） | 206. 反转链表 | 简单 |
| 虚拟头结点 + 尾插 | 21. 合并两个有序链表 | 简单 |
| 虚拟头结点 + 哨兵删除 | 203. 移除链表元素 | 简单 |
| 快慢指针（龟兔赛跑） | 141. 环形链表 | 简单 |
| 快慢指针（找中点） | 876. 链表的中间结点 | 简单 |
| 快慢指针（定距离） | 19. 删除链表的倒数第 N 个结点 | 中等 |
| 快慢指针（求入环口） | 142. 环形链表 II | 中等 |
| 双指针换头 | 160. 相交链表 | 简单 |
| 头插法区间反转 | 92. 反转链表 II | 中等 |
| K 个一组翻转 | 25. K 个一组翻转链表 | 困难 |
| 三模板组合 | 143. 重排链表 | 中等 |

---

## 模式一：原地反转（三指针）

**适用信号**：题目要求把一条链表的**方向整个掉头**，或者把某一段的反转嵌入更大的流程里。
关键词是「逆序」「翻转」「方向」。

核心动作：维护三个指针——

- `prev`：已经反转好的那一段的头；
- `cur`：当前正在处理的节点；
- `nxt`：`cur` 的原始后继。

每轮先把 `cur.next` 指向 `prev`（改方向），再把 `prev`、`cur` 整体向前挪一格。
**先存 `nxt`、再改指向**，是整个模式里最不能颠倒的一步。

### 206. 反转链表（简单）

**题目**：给你单链表的头节点 `head`，反转链表并返回新的头节点。例如 `1 -> 2 -> 3 -> 4 -> 5` 变为 `5 -> 4 -> 3 -> 2 -> 1`。

**思路（边走边掉头）**：
反转的本质就是「把每条边的方向掉个头」。我们从原链表头一路走到尾，经过每个节点时，
让它指向前一个节点。用 `prev` 记「上一个节点」（也就是反转后应该跟在当前节点后面的那个），
初始为 `None`（原头节点反转后就是尾节点，指针指向空）。

每轮做四件事：先用 `nxt = cur.next` 备份后继；把 `cur.next = prev` 完成掉头；
把 `prev = cur`（当前节点进入已反转段）；把 `cur = nxt` 继续向右推进。
走完时 `cur` 为 `None`，`prev` 停在原链表最后一个节点，它就是新的头。

为什么必须先备份 `nxt`：一旦执行 `cur.next = prev`，`cur` 与它原来后继的连接就断了，
如果之前没记住，后半条链表就永远找不回来了。所以顺序只能是「先存后改」。

为什么用迭代而不是递归：迭代是 O(1) 额外空间；递归时间同为 O(n)，但会消耗 O(n) 的调用栈，
链表很长时有栈溢出风险。本题只详展迭代版。

**代码**（完整可运行版见 `src/linked-list/reverse_linked_list.py` / `.cpp`）：

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


def reverse_list(head):
    prev = None
    cur = head
    while cur:
        nxt = cur.next
        cur.next = prev
        prev = cur
        cur = nxt
    return prev
```

```cpp
struct ListNode {
    int val;
    ListNode *next;
    ListNode(int x = 0, ListNode *n = nullptr) : val(x), next(n) {}
};

ListNode *reverseList(ListNode *head) {
    ListNode *prev = nullptr;
    ListNode *cur = head;
    while (cur) {
        ListNode *nxt = cur->next;
        cur->next = prev;
        prev = cur;
        cur = nxt;
    }
    return prev;
}
```

- **复杂度**：时间 O(n)，空间 O(1)。
- **易错点**：`nxt` 的备份必须放在改 `cur.next` 之前；循环结束后返回的是 `prev` 而不是 `cur`（`cur` 已经走空）；空链表返回 `None`，循环不进入、`prev` 恰为 `None`，无需特判；别用「新建节点依次头插」的做法，那样额外开了 O(n) 空间，也不满足「原地」的要求。
- **相似题**：92. 反转链表 II（只反转第 `left` 到第 `right` 个节点，先走到反转段前驱，套用本模板后接回两段）；25. K 个一组翻转链表（把本模板循环地作用在每 K 个节点上）；234. 回文链表（先用快慢指针找中点，再反转后半段比较）。这些题都会在本分类后续篇目中展开。

---

## 模式二：虚拟头结点 + 尾插

**适用信号**：结果链表需要**边比较边拼接**，而「第一个节点是谁」事先不确定；或者要删除/插入
节点而头结点本身也可能被改动。关键词是「构造新链表」「合并」「删除头结点」。

核心动作：先建一个不参与结果的哨兵节点 `dummy`，让 `tail` 指向它。
以后每加入一个节点，都做「`tail.next = 新节点`、再 `tail = tail.next`」这一对操作；
全部完成后返回 `dummy.next`。这样每个节点都走同一条代码路径，边界分支被消掉。

### 21. 合并两个有序链表（简单）

**题目**：将两个升序链表合并为一个新的升序链表并返回，新链表由拼接原有节点组成。例如 `1 -> 2 -> 4` 与 `1 -> 3 -> 4` 合并为 `1 -> 1 -> 2 -> 3 -> 4 -> 4`。

**思路（归并的链表版）**：
这就是归并排序里「合并」那一步，只不过在链表上做，不需要额外数组。
两条链表各自有序，于是每次比较两个当前头节点，谁小就把谁接到结果尾部，并让那条链表前进一格。
因为每一步取的都是「两条链表中当前最小的那个」，结果自然保持升序。

循环在任一条链表走空时结束。此时把未走空的那条**整段**接到末尾即可：
另一条已经有序，且它剩下的元素都不小于已接上的所有元素，不会破坏顺序。

为什么用虚拟头结点：结果链表的第一个节点由「谁更小」决定，若不用 `dummy`，
就得为「结果链表还是空的」这一种情况单独写分支。`dummy` 让所有节点统一地接在 `tail` 之后，
最后返回 `dummy.next` 即可，边界代码被彻底消掉——这是本模式最典型的收益。

**代码**（`src/linked-list/merge_two_sorted_lists.py` / `.cpp`）：

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


def merge_two_lists(list1, list2):
    dummy = ListNode()
    tail = dummy
    while list1 and list2:
        if list1.val <= list2.val:
            tail.next = list1
            list1 = list1.next
        else:
            tail.next = list2
            list2 = list2.next
        tail = tail.next
    tail.next = list1 if list1 else list2
    return dummy.next
```

```cpp
struct ListNode {
    int val;
    ListNode *next;
    ListNode(int x = 0, ListNode *n = nullptr) : val(x), next(n) {}
};

ListNode *mergeTwoLists(ListNode *list1, ListNode *list2) {
    ListNode dummy;
    ListNode *tail = &dummy;
    while (list1 && list2) {
        if (list1->val <= list2->val) {
            tail->next = list1;
            list1 = list1->next;
        } else {
            tail->next = list2;
            list2 = list2->next;
        }
        tail = tail->next;
    }
    tail->next = list1 ? list1 : list2;
    return dummy.next;
}
```

- **复杂度**：时间 O(m + n)（每条链表各走一遍），空间 O(1)（只重接指针，不新建节点）。
- **易错点**：`tail = tail.next` 别漏，否则所有节点都挂在同一个位置、链表根本没长；循环结束后必须补上「接上剩余段」，否则会丢掉一部分节点；比较用 `<=` 还是 `<` 不影响正确性，但都用 `<=` 能保持稳定；注意 `dummy` 是哨兵，返回值是 `dummy.next` 而不是 `dummy`。
- **相似题**：23. 合并 K 个升序链表（把两两合并扩展成多路，用堆或分治）、88. 合并两个有序数组（同一思想在数组上的版本，见 `01-array`）、148. 排序链表（自底向上归并排序，本题是其中的合并步骤）。这些会在 `heap` 与 `divide-conquer` 篇中再遇。

### 203. 移除链表元素（简单）

**题目**：给你一个链表的头节点 `head` 和一个整数 `val`，删除链表中所有满足 `Node.val == val` 的节点，并返回新的头节点。例如 `1 -> 2 -> 6 -> 3 -> 4 -> 5 -> 6` 删除 `6` 后为 `1 -> 2 -> 3 -> 4 -> 5`。

**思路（哨兵前驱 + 跳跃）**：
删除一个节点需要它的**前驱**，而头节点没有前驱，它自己也可能被删。解法是让
`dummy` 当所有节点的前驱：从 `dummy` 出发，`cur` 始终指向「当前节点的前驱」，
再看 `cur.next` 的值——等于 `val` 就跳过它（`cur.next = cur.next.next`），
否则 `cur` 前进一步。整个链表走完，返回 `dummy.next`。

为什么删除后 `cur` 不前进：跳过之后，新的 `cur.next` 还是一个没检查过的节点，
可能也要删，所以必须原地再判一次。只有确认保留 `cur.next` 时，`cur` 才往前走。

为什么不特判头节点：如果不用 `dummy`，删头节点就得写 `head = head.next` 的分支。
`dummy` 把「删头」变成「删中间」，代码只有一条路径，这正是哨兵的价值。

**代码**（`src/linked-list/remove_linked_list_elements.py` / `.cpp`）：

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


def remove_elements(head, val):
    dummy = ListNode(0, head)
    cur = dummy
    while cur.next:
        if cur.next.val == val:
            cur.next = cur.next.next
        else:
            cur = cur.next
    return dummy.next
```

```cpp
struct ListNode {
    int val;
    ListNode *next;
    ListNode(int x = 0, ListNode *n = nullptr) : val(x), next(n) {}
};

ListNode *removeElements(ListNode *head, int val) {
    ListNode dummy(0, head);
    ListNode *cur = &dummy;
    while (cur->next) {
        if (cur->next->val == val) {
            cur->next = cur->next->next;
        } else {
            cur = cur->next;
        }
    }
    return dummy.next;
}
```

- **复杂度**：时间 O(n)，空间 O(1)。
- **易错点**：删除后忘记「保持 `cur` 不动再判一次」，会漏删连续相同的节点（如 `6 -> 6 -> 6`）；循环条件是 `cur.next` 而不是 `cur`，因为我们要检查并操作的是下一个节点；返回值必须是 `dummy.next`，直接返回 `head` 时若头被删就错了。
- **相似题**：19. 删除链表的倒数第 N 个结点（同一哨兵删除思路，只是定位方式换成快慢指针）；82. 删除排序链表中的重复元素 II（重复的全删，保留的判据从「等于 val」变成「不等于后继」）；21. 合并、92. 反转都靠 `dummy` 消边界，见本模式上文与下文。

---

## 模式三：快慢指针（龟兔赛跑）

**适用信号**：要在**一次遍历**里判断链表有没有环、找中点、找倒数第 K 个节点等。
关键词是「环」「中点」「相对位置」。核心是让两个指针以**不同速度**前进。

核心动作：`slow` 每次走一步，`fast` 每次走两步。若链表无环，`fast` 会先到尽头；
若有环，两者进入环后必定相遇。不同问题里 `fast` 的速度和起始位置会稍有调整，
但「差速前进」这一内核不变。

### 141. 环形链表（简单）

**题目**：给你一个链表的头节点 `head`，判断链表中是否有环。为表示环，链表的尾节点会连到某个先前节点上。

**思路（快慢指针相遇即有环）**：
慢指针每次走一步，快指针每次走两步。

- 如果链表**没有环**：快指针会一路走到尽头（`None`），循环自然结束，返回 `False`；
- 如果链表**有环**：两个指针迟早都会进入环，之后快指针每轮比慢指针多走一步，
  相当于以「速度为 1」在追慢指针，两者在环内的距离每轮缩短 1，必定在某一刻重叠。

为什么「有环必相遇」值得说清：因为快慢指针的速度差固定为 1，距离是整数且每轮严格减 1，
不存在「总是跨过而永远不相遇」的可能——这正是用相对速度把「追及问题」变简单的妙处。

为什么不改节点、不用哈希表：哈希表记录访问过的节点能做到 O(n) 时间和 O(n) 空间；
快慢指针把额外空间压到 O(1)，是本题的标准最优解。

为什么循环条件是 `fast and fast.next`：快指针要连续走两步，必须保证 `fast` 和 `fast.next`
都不是 `None`，否则执行 `fast.next.next` 会访问空指针。写成 `while fast and fast.next` 恰好卡住这条边界。

**代码**（`src/linked-list/linked_list_cycle.py` / `.cpp`）：

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


def has_cycle(head):
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow is fast:
            return True
    return False
```

```cpp
struct ListNode {
    int val;
    ListNode *next;
    ListNode(int x = 0, ListNode *n = nullptr) : val(x), next(n) {}
};

bool hasCycle(ListNode *head) {
    ListNode *slow = head;
    ListNode *fast = head;
    while (fast && fast->next) {
        slow = slow->next;
        fast = fast->next->next;
        if (slow == fast) return true;
    }
    return false;
}
```

- **复杂度**：时间 O(n)，空间 O(1)。
- **易错点**：循环条件必须是 `fast and fast.next`（两个都要判），只写 `fast` 会在快指针走到最后一个节点时越界访问 `fast.next.next`；判断相遇要在移动**之后**做，且用「同一个节点（`is` / `==`）」比较而不是比较 `val`（值相等的不同节点不算相遇）；空链表返回 `False`；移动快指针时记得 `fast = fast.next.next`，写成两次 `fast.next` 容易写错。
- **相似题**：142. 环形链表 II（在判环基础上求入环点，相遇后让一个指针回到起点、同速再走，相遇处即入环口，见下文）；876. 链表的中间结点（快指针走到头时慢指针恰在中点，见下文）；19. 删除链表的倒数第 N 个结点（快指针先走 N 步，再同速前进，见下文）；143. 重排链表（用快慢指针找中点后，接 206 反转后半段，见篇末）。都用同一个「差速前进」的内核。

### 876. 链表的中间结点（简单）

**题目**：给你单链表的头节点 `head`，返回链表的中间结点。如果有两个中间结点，则返回第二个中间结点。

**思路（快指针是慢指针速度的两倍）**：
慢指针每次走一步，快指针每次走两步。相同时间里快指针走过的路程是慢指针的两倍，
当快指针走完整条链表，慢指针恰好走了一半，落在中间。

为什么偶数长度时返回第二个中间结点：循环条件 `fast and fast.next` 会在偶数长度下、
`fast` 走到最后一个节点时结束，此时 `slow` 停在两个中间结点的后一个，正好符合题目要求。

**代码**（`src/linked-list/middle_of_linked_list.py` / `.cpp`）：

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


def middle_node(head):
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    return slow
```

```cpp
struct ListNode {
    int val;
    ListNode *next;
    ListNode(int x = 0, ListNode *n = nullptr) : val(x), next(n) {}
};

ListNode *middleNode(ListNode *head) {
    ListNode *slow = head;
    ListNode *fast = head;
    while (fast && fast->next) {
        slow = slow->next;
        fast = fast->next->next;
    }
    return slow;
}
```

- **复杂度**：时间 O(n)，空间 O(1)。
- **易错点**：循环条件是两个都要判，`while fast and fast.next`；如果想让偶数长度返回**第一个**中间结点，把条件改成 `while fast.next and fast.next.next` 即可；返回的是节点而不是下标，别去数长度。
- **相似题**：143. 重排链表（本题是它的第一步）；141/142（同一套差速模板，见上、下）；234. 回文链表（找中点后反转后半段再比较）。

### 142. 环形链表 II（中等）

**题目**：给你一个链表的头节点 `head`，返回链表开始入环的第一个节点；如果链表无环，返回 `None`。

**思路（先判环，再求入环口）**：
第一阶段和 141 一样，用快慢指针判断有没有环。若相遇，进入第二阶段：
让一个指针回到 `head`，两个指针都改成每次走一步，它们再次相遇的位置就是入环的第一个节点。

为什么这样能找到入环口（推导）：设头到入环口距离为 `x`，入环口到相遇点为 `y`，环长为 `L`。
相遇时慢指针走了 `x + y`，快指针走了 `x + y + nL`（`n` 为快指针多绕的圈数）。
由速度两倍关系得 `x + y + nL = 2(x + y)`，即 `x = nL - y`。
这说明「从头走 `x` 步」与「从相遇点走 `nL - y` 步」会到达同一点，
而后者等于「从相遇点绕到入环口」，所以两指针同速对走必然在入环口相遇。

**代码**（`src/linked-list/linked_list_cycle_ii.py` / `.cpp`）：

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


def detect_cycle(head):
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow is fast:
            slow = head
            while slow is not fast:
                slow = slow.next
                fast = fast.next
            return slow
    return None
```

```cpp
struct ListNode {
    int val;
    ListNode *next;
    ListNode(int x = 0, ListNode *n = nullptr) : val(x), next(n) {}
};

ListNode *detectCycle(ListNode *head) {
    ListNode *slow = head;
    ListNode *fast = head;
    while (fast && fast->next) {
        slow = slow->next;
        fast = fast->next->next;
        if (slow == fast) {
            slow = head;
            while (slow != fast) {
                slow = slow->next;
                fast = fast->next;
            }
            return slow;
        }
    }
    return nullptr;
}
```

- **复杂度**：时间 O(n)，空间 O(1)。
- **易错点**：第二阶段的快慢都改成每次走一步，别让 `fast` 继续走两步；返回的是入环的那个节点，不是相遇点；无环时返回空；把「相遇点」和「入环口」混为一谈是常见错误——相遇点一般不在入环口。
- **相似题**：141. 环形链表（只判有无环的前半段）；287. 寻找重复数（把数组下标当 `next` 指针，本质就是求入环口）；876/19（同一差速模板的不同用法）。

### 19. 删除链表的倒数第 N 个结点（中等）

**题目**：给你一个链表的头节点 `head`，删除链表的倒数第 `n` 个结点，并返回头节点。

**思路（快指针先拉开 N 步）**：
快指针先走 `n` 步，拉开与慢指针 `n` 个身位的差距；然后快慢一起走，
当快指针到达最后一个结点时，慢指针正好停在待删结点的**前驱**上，
执行 `slow.next = slow.next.next` 就完成删除。

为什么快指针先走 `n` 步：领先 `n` 个结点后，当快指针走到尾部时，
慢指针所在位置之后恰好还有 `n` 个结点，所以它的下一个就是倒数第 `n` 个。

为什么用虚拟头结点：若不使用 `dummy`，当 `n` 等于链表长度（删除头结点）时，
慢指针会停在 `head` 之前的位置而无处可停。让快慢都从 `dummy` 出发，
「删头」和「删中间」被统一处理。

**代码**（`src/linked-list/remove_nth_from_end.py` / `.cpp`）：

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


def remove_nth_from_end(head, n):
    dummy = ListNode(0, head)
    fast = slow = dummy
    for _ in range(n):
        fast = fast.next
    while fast.next:
        fast = fast.next
        slow = slow.next
    slow.next = slow.next.next
    return dummy.next
```

```cpp
struct ListNode {
    int val;
    ListNode *next;
    ListNode(int x = 0, ListNode *n = nullptr) : val(x), next(n) {}
};

ListNode *removeNthFromEnd(ListNode *head, int n) {
    ListNode dummy(0, head);
    ListNode *fast = &dummy;
    ListNode *slow = &dummy;
    for (int i = 0; i < n; ++i) {
        fast = fast->next;
    }
    while (fast->next) {
        fast = fast->next;
        slow = slow->next;
    }
    slow->next = slow->next->next;
    return dummy.next;
}
```

- **复杂度**：时间 O(n)（只遍历一趟），空间 O(1)。
- **易错点**：循环条件是 `fast.next`（停在最后一个结点），不是 `fast`，否则慢指针会走到待删结点本身、失去前驱；删除头结点的情形靠 `dummy` 处理，别忘了返回 `dummy.next`；`n` 一定有效，无需再判越界。
- **相似题**：876（同样是差速定位，只是快指针走的路数不同）；203 移除链表元素（哨兵删除的纯版本）；143 重排链表（内部也用快慢指针找中点）。

---

## 模式四：双指针换头（相交链表）

**适用信号**：两条链表要「对齐后一起走」，但长度未知、不想先各算一遍长度。
关键词是「相交」「公共部分」「对齐」。核心是让两个指针各自走完自己再换到对方链子上，
用「总路程相同」代替「先测长度」。

### 160. 相交链表（简单）

**题目**：给你两个单链表的头节点 `headA` 和 `headB`，找出并返回两个链表相交的起始节点；如果不相交，返回 `None`。题目保证整个链式结构中没有环。

**思路（走完自己换到对方）**：
指针 `p` 从 `headA` 出发、`q` 从 `headB` 出发，每次各走一步。谁先走到自己链表的尽头，
就换到对方的头节点重新出发。若两链表相交，它们会在相交点相遇；
若不相交，两指针都会走完两条链表的全部结点，最后同时变成 `None`，循环结束返回 `None`。

为什么一定能相遇：设 A 独有段长 `a`、B 独有段长 `b`、公共段长 `c`。
`p` 走的路径是 `a + c + b`，`q` 走的是 `b + c + a`，长度都是 `a + b + c`。
由于总路程相等，它们必然同时到达同一个位置；公共段起点是两段路径唯一的重合点，
所以相遇处就是相交起始节点（若 `c = 0`，则最后一起到达空指针）。

**代码**（`src/linked-list/intersection_of_two_linked_lists.py` / `.cpp`）：

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


def get_intersection_node(headA, headB):
    p, q = headA, headB
    while p is not q:
        p = p.next if p else headB
        q = q.next if q else headA
    return p
```

```cpp
struct ListNode {
    int val;
    ListNode *next;
    ListNode(int x = 0, ListNode *n = nullptr) : val(x), next(n) {}
};

ListNode *getIntersectionNode(ListNode *headA, ListNode *headB) {
    ListNode *p = headA;
    ListNode *q = headB;
    while (p != q) {
        p = p ? p->next : headB;
        q = q ? q->next : headA;
    }
    return p;
}
```

- **复杂度**：时间 O(m + n)，空间 O(1)。
- **易错点**：换头判断的是「当前指针是否为空」，用 `p = p.next if p else headB` 的写法；比较要用节点身份（Python 的 `is`、C++ 的 `==`），不能比 `val`；无相交时两个指针最终同为 `None`，`while` 自然退出并返回 `None`，无需额外判断。
- **相似题**：141 判环（都靠「指针走两遍」的思想，但一个靠速度差、一个靠换头对齐）；876 找中点（把「相对位置」交给两个指针）；19 删除倒数第 N 个（同样是先造出位置差）。

---

## 模式五：区间反转（头插法）

**适用信号**：只反转链表的**某一段**，或**按固定长度分组**反复反转。
关键词是「第 left 到第 right 个」「每 K 个一组」。核心是**头插法**：
每轮把当前节点的后继「摘下来、插到区间头部」，区间内的节点就会依次被搬到最前面。

### 92. 反转链表 II（中等）

**题目**：给你单链表的头节点 `head` 和两个整数 `left`、`right`（`left <= right`），反转从位置 `left` 到位置 `right` 的节点（位置从 1 开始），返回反转后的链表。

**思路（局部头插）**：
先用 `dummy` 消掉「`left = 1`（从头反转）」的边界。走到第 `left` 个节点的前驱记为 `pre`，
`pre.next` 就是反转段的第一个节点 `cur`。然后做 `right - left` 轮头插：
每轮把 `cur` 的下一个节点 `nxt` 摘下来、插到 `pre` 之后。
这样反转段里的节点会依次被搬到最前面，最终整段反转完成，前后两段自然接回。

头插的四步（把 `nxt` 插到 `pre` 之后）：

```text
nxt = cur.next        # 摘下 cur 的后继
cur.next = nxt.next   # 让 cur 跨过 nxt
nxt.next = pre.next   # nxt 指向当前区间头
pre.next = nxt        # 区间头更新为 nxt
```

为什么用头插法而不是「整段摘下再反转再接回」：头插法不需要额外记录反转段的结尾，
每轮只改几个指针，就地完成，且只走一趟。

**代码**（`src/linked-list/reverse_linked_list_ii.py` / `.cpp`）：

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


def reverse_between(head, left, right):
    dummy = ListNode(0, head)
    pre = dummy
    for _ in range(left - 1):
        pre = pre.next
    cur = pre.next
    for _ in range(right - left):
        nxt = cur.next
        cur.next = nxt.next
        nxt.next = pre.next
        pre.next = nxt
    return dummy.next
```

```cpp
struct ListNode {
    int val;
    ListNode *next;
    ListNode(int x = 0, ListNode *n = nullptr) : val(x), next(n) {}
};

ListNode *reverseBetween(ListNode *head, int left, int right) {
    ListNode dummy(0, head);
    ListNode *pre = &dummy;
    for (int i = 0; i < left - 1; ++i) pre = pre->next;
    ListNode *cur = pre->next;
    for (int i = 0; i < right - left; ++i) {
        ListNode *nxt = cur->next;
        cur->next = nxt->next;
        nxt->next = pre->next;
        pre->next = nxt;
    }
    return dummy.next;
}
```

- **复杂度**：时间 O(n)，空间 O(1)。
- **易错点**：头插四步的顺序不能乱，尤其 `nxt.next = pre.next` 必须在 `pre.next` 被改之前执行；`cur` 在整段过程中始终是区间原来的第一个节点（也就是反转后的尾），不要跟着 `nxt` 走；`left = right` 时循环不进入，直接返回；`dummy` 让 `left = 1` 也有前驱。
- **相似题**：206 反转链表（区间等于整条链表时的特例）；25 K 个一组翻转（把区间长度固定为 K 并反复执行，见下）；143 重排链表（内部也反转后半段）。

### 25. K 个一组翻转链表（困难）

**题目**：给你一个链表的头节点 `head`，每 `k` 个节点一组进行翻转，返回翻转后的链表。如果节点总数不是 `k` 的整数倍，最后剩余的节点保持原有顺序。

**思路（先试探够不够，再整段反转）**：
用 `dummy` 加「组前驱」`group_prev`（初始为 `dummy`），循环做三件事：

1. **试探**：从 `group_prev` 向前数 `k` 个节点，如果凑不满，说明到了尾部，直接返回；
2. 记下这一组之后的第一个节点 `group_next`（即第 `k+1` 个节点），作为翻转后的挂接点；
3. 用「整段反转」把 `group_prev.next` 到第 `k` 个节点反转，再让 `group_prev.next` 指向新的组头，
   并把原来的组头（现在是组尾）接到 `group_next`。

为什么每组都要先试探能否凑满 `k` 个：题目规定不足 `k` 个的尾组保持原序，
所以必须提前知道「这一组够不够」，不够就整体退出，不能翻转。

为什么把 `prev` 初始化为 `group_next`：相当于给这段预置一个「后面的锚」，
循环结束时原组头自然就指向 `group_next`，省去单独拼接尾部的代码。

**代码**（`src/linked-list/reverse_nodes_in_k_group.py` / `.cpp`）：

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


def reverse_k_group(head, k):
    dummy = ListNode(0, head)
    group_prev = dummy
    while True:
        kth = group_prev
        for _ in range(k):
            kth = kth.next
            if not kth:
                return dummy.next
        group_next = kth.next
        prev, cur = group_next, group_prev.next
        while cur is not group_next:
            nxt = cur.next
            cur.next = prev
            prev = cur
            cur = nxt
        old_head = group_prev.next
        group_prev.next = kth
        group_prev = old_head
    return dummy.next
```

```cpp
struct ListNode {
    int val;
    ListNode *next;
    ListNode(int x = 0, ListNode *n = nullptr) : val(x), next(n) {}
};

ListNode *reverseKGroup(ListNode *head, int k) {
    ListNode dummy(0, head);
    ListNode *groupPrev = &dummy;
    while (true) {
        ListNode *kth = groupPrev;
        for (int i = 0; i < k; ++i) {
            kth = kth->next;
            if (!kth) return dummy.next;
        }
        ListNode *groupNext = kth->next;
        ListNode *prev = groupNext;
        ListNode *cur = groupPrev->next;
        while (cur != groupNext) {
            ListNode *nxt = cur->next;
            cur->next = prev;
            prev = cur;
            cur = nxt;
        }
        ListNode *oldHead = groupPrev->next;
        groupPrev->next = kth;
        groupPrev = oldHead;
    }
    return dummy.next;
}
```

- **复杂度**：时间 O(n)（每个节点恰好被访问常数次），空间 O(1)。
- **易错点**：试探必须逐节点走，走空就返回，不能提前假设长度是 `k` 的倍数；反转循环的终止条件是 `cur != group_next`，用 `group_next` 当锚点最稳妥；记录 `old_head` 必须在改 `group_prev.next` **之前**，否则找不到下一组的前驱；`k = 1` 时每组只含一个节点，等价于原样返回。
- **相似题**：92 反转链表 II（区间反转的单次版，先把那题做熟）；206 反转链表（反转内核）；24 两两交换链表中的节点（`k = 2` 的特例，可用本题解法）。

---

## 模式六：三模板组合（重排链表）

**适用信号**：一道题同时需要「定位中点」「反转一段」「交错拼接」几个动作。
应对策略不是发明新模板，而是**把问题拆成已经练熟的模板再串起来**。

### 143. 重排链表（中等）

**题目**：给定单链表 `L0 -> L1 -> ... -> Ln-1`，将其重排为 `L0 -> Ln-1 -> L1 -> Ln-2 -> L2 -> Ln-3 -> ...`。不能只是单纯改变节点内部的值。

**思路（找中点 + 反转后半段 + 交错合并）**：
重排的规律是「一根从前往后、一根从后往前」轮流取，所以把它拆成三步：

1. 快慢指针找中点，在 `slow.next` 处把链表切成前后两半；
2. 用 206 的反转模板把后半段整个反转；
3. 用 21 的合并思路，把前半段和反转后的后半段交替穿插起来。

为什么从 `slow.next` 断开：前半段长度 `>=` 后半段，断开后两段都非空，
拼接时从后半段取尽即可，边界自然统一。

为什么拼接前要先存两个 `next`：一旦改指向，两段的后续节点就会丢失，
所以每轮先把两条链的下一个节点各自备份，再交叉连接、整体后移。

**代码**（`src/linked-list/reorder_list.py` / `.cpp`）：

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


def reorder_list(head):
    if not head or not head.next:
        return
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next

    prev, cur = None, slow.next
    slow.next = None
    while cur:
        nxt = cur.next
        cur.next = prev
        prev = cur
        cur = nxt

    first, second = head, prev
    while second:
        next1 = first.next
        next2 = second.next
        first.next = second
        second.next = next1
        first, second = next1, next2
```

```cpp
struct ListNode {
    int val;
    ListNode *next;
    ListNode(int x = 0, ListNode *n = nullptr) : val(x), next(n) {}
};

void reorderList(ListNode *head) {
    if (!head || !head->next) return;
    ListNode *slow = head;
    ListNode *fast = head;
    while (fast && fast->next) {
        slow = slow->next;
        fast = fast->next->next;
    }

    ListNode *prev = nullptr;
    ListNode *cur = slow->next;
    slow->next = nullptr;
    while (cur) {
        ListNode *nxt = cur->next;
        cur->next = prev;
        prev = cur;
        cur = nxt;
    }

    ListNode *first = head;
    ListNode *second = prev;
    while (second) {
        ListNode *next1 = first->next;
        ListNode *next2 = second->next;
        first->next = second;
        second->next = next1;
        first = next1;
        second = next2;
    }
}
```

- **复杂度**：时间 O(n)（找中点、反转、合并各 O(n)），空间 O(1)。
- **易错点**：空链表或单节点要直接返回，否则 `slow.next` 可能为 `None`；断开必须做（`slow.next = None`），否则反转会和前半段缠在一起；拼接循环的条件是 `while second`，因为后半段较短、先用完；返回类型是 `void`，原地修改，不要误返回新头。
- **相似题**：206、876、21 分别是本题三步的原型；234. 回文链表（同样是「找中点 + 反转后半段」，只是最后做的是比较而非拼接）；25 K 个一组翻转（多段反转的组合）。

---

## 规律总结

1. **画图再写码，改指向先备份**。链表题几乎所有的错误都来自「指针改早了、把还没用的连接弄丢了」。动手前在纸上画出节点和箭头，给每个要用的指针标出它当前指向哪里、下一步该指向哪里；凡是「改某指针之前还要用到它原来的目标」的，就先把它备份到另一个变量里。206 里的 `nxt`、143 拼接前的 `next1`/`next2` 都是最典型的例子。

2. **能原地就原地，别新建节点**。反转、合并、删除这类题只改 `next` 指针就能完成，空间是 O(1)。新建节点依次拼接虽然也能出结果，但白白多花 O(n) 空间，且可能不满足题目对「复用原节点」的要求。判断标准：题目说「由拼接原有节点组成」时，就只能重接指针。

3. **虚拟头结点是消掉边界分支的万能钥匙**。凡是「第一个节点不确定」「头节点可能被删/被换」的题，都先建一个 `dummy`，让所有操作统一发生在其后，最后返回 `dummy.next`。它不参与结果，却能把「空链表」「删头」等特判全部消灭。21 合并、203 删元素、19 删倒数第 N 个、92 反转区间、25 分组翻转都用它收边界。

4. **循环结束后的「收尾」别忘**。合并类题目里，主循环通常在一条链表走空时结束，另一条的剩余部分必须显式接上（`tail.next = list1 if list1 else list2`），否则结果会缺一截；分组反转里则要提前「试探够不够 `k` 个」，不够的尾组保持原序。这类「循环后处理」是链表题与数组题很不一样的一点。

5. **快慢指针解决的是「相对位置」问题**。判环靠速度差（每轮距离减 1，必相遇）；找中点靠「快的到终点时慢的正好一半」；找倒数第 K 个靠「快的先走 K 步拉开差距」。遇到和「位置、距离、环」有关而只想遍历一遍的题，先想想能不能让两个指针差速走。

6. **求入环口要「一回起点、同速对走」**。142 的推导结论是 `x = nL - y`：头到入环口的距离等于相遇点绕到入环口的距离。记住这个结论就够用——相遇后把其中一个指针放回头节点，两个指针都改成每次走一步，再次相遇处就是入环口，不要把「相遇点」当成答案。

7. **反转区间用头插法**。206 的反转适合「整条掉头」，但只反转一段时，头插法更省事：固定 `pre` 为区间前驱，每轮把区间第一个节点 `cur` 的后继摘到 `pre` 之后（`nxt = cur.next; cur.next = nxt.next; nxt.next = pre.next; pre.next = nxt`）。92 用它单次反转，25 把它循环用于每组 `k` 个，都是同一套动作。

8. **两条链对齐可以靠「换头」**。160 不去预先算长度差，而是让两个指针各走完自己再换到对方，靠「总路程都是 `a + b + c`」保证同时到达相交点。这类「用等长路径代替显式对齐」的技巧，比先测长度再同步更简洁。

9. **复杂题先拆成熟模板**。143 重排链表看着难，拆开就是「876 找中点 + 206 反转后半段 + 21 交错合并」三件事的串联。遇到综合题，先问自己可以用哪几个已经练熟的模板拼出来，而不是从零硬写。

10. **两套边界要背准**。判环、找中点的循环是 `while fast and fast.next`（快指针要连走两步）；而反转、遍历时用 `while cur` 即可；删除的循环盯「下一个节点」，用 `while cur.next`。返回值的坑同样要记牢：反转返回 `prev`、合并返回 `dummy.next`、重排原地修改不返回——这些「循环结束时哪个指针才是答案」的地方，是链表题最后一道关。

11. **以 `src` 为准，docs 只做粘贴**。题解里的代码必须和 `code/leetcode/src/linked-list/` 下的实现**逐字一致**，避免文档与可运行代码脱节。C++ 自测时不要把 `{1, 2}` 这类初值列表直接写进 `assert` 实参——花括号里的逗号会被当成宏参数分隔符，先把期望值存进变量再比较。
