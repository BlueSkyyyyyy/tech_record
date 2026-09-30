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
| 快慢指针（判环） | 141. 环形链表 | 简单 |

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
- **相似题**：142. 环形链表 II（在判环基础上求入环点，相遇后让一个指针回到起点、同速再走，相遇处即入环口）；876. 链表的中间结点（快指针走到头时慢指针恰在中点）；19. 删除链表的倒数第 N 个结点（快指针先走 N 步，再同速前进）；143. 重排链表（用快慢指针找中点后，接 206 反转后半段）。都会在本分类后续篇目展开。

---

## 规律总结

1. **画图再写码，改指向先备份**。链表题几乎所有的错误都来自「指针改早了、把还没用的连接弄丢了」。动手前在纸上画出节点和箭头，给每个要用的指针标出它当前指向哪里、下一步该指向哪里；凡是「改某指针之前还要用到它原来的目标」的，就先把它备份到另一个变量里。206 里的 `nxt` 就是最典型的例子。

2. **能原地就原地，别新建节点**。反转、合并、删除这类题只改 `next` 指针就能完成，空间是 O(1)。新建节点依次拼接虽然也能出结果，但白白多花 O(n) 空间，且可能不满足题目对「复用原节点」的要求。判断标准：题目说「由拼接原有节点组成」时，就只能重接指针。

3. **虚拟头结点是消掉边界分支的万能钥匙**。凡是「第一个节点不确定」「头节点可能被删/被换」的题，都先建一个 `dummy`，让所有操作统一发生在 `tail.next` 上，最后返回 `dummy.next`。它不参与结果，却能把「空链表」「删头」等特判全部消灭。21 的合并、以及后续 203 删元素、92 反转区间都会反复用到它。

4. **循环结束后的「收尾」别忘**。合并类题目里，主循环通常在一条链表走空时结束，另一条的剩余部分必须显式接上（`tail.next = list1 if list1 else list2`），否则结果会缺一截。这类「循环后处理」是链表题与数组题很不一样的一点。

5. **快慢指针解决的是「相对位置」问题**。判环靠速度差（每轮距离减 1，必相遇）；找中点靠「快的到终点时慢的正好一半」；找倒数第 K 个靠「快的先走 K 步拉开差距」。遇到和「位置、距离、环」有关而只想遍历一遍的题，先想想能不能让两个指针差速走。

6. **两套边界要背准**。判环的循环条件是 `while fast and fast.next`，因为快指针要连走两步；而反转、遍历时用 `while cur` 即可。返回值的坑同样要记牢：反转返回 `prev`、合并返回 `dummy.next`、判环返回布尔值——这些「循环结束时哪个指针才是答案」的地方，是链表题最后一道关。

7. **以 `src` 为准，docs 只做粘贴**。题解里的代码必须和 `code/leetcode/src/linked-list/` 下的实现**逐字一致**，避免文档与可运行代码脱节。C++ 自测时不要把 `{1, 2}` 这类初值列表直接写进 `assert` 实参——花括号里的逗号会被当成宏参数分隔符，先把期望值存进变量再比较。
