"""143. 重排链表（Reorder List）

题目：给定单链表 L0 -> L1 -> ... -> Ln-1，将其重排为
      L0 -> Ln-1 -> L1 -> Ln-2 -> L2 -> Ln-3 -> ...
      注意不能只是单纯改变节点内部的值。

思路：分三步，把「交错拼接」拆成三个熟模板的组合：
      1. 快慢指针找中点，把链表切成前后两半；
      2. 用 206 的反转模板把后半段整个反转；
      3. 用 21 的合并思路，把前半段和反转后的后半段交替穿插起来。

      为什么是中点切开：重排的规律是「一根从前往后、一根从后往前」轮流取，
      所以后半段要逆序。找中点时快指针走两步、慢指针走一步，快到头时慢正好在中点，
      从 slow.next 处断开即可（前半段长度 >= 后半段）。

      为什么要在拼接里先存 next1、next2：一旦改指向，两段后续节点就会丢失，
      所以每轮先把两条链的下一个节点都备份下来再交叉连接。

复杂度：时间 O(n)，空间 O(1)。
"""


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


def build(values):
    dummy = ListNode()
    tail = dummy
    for v in values:
        tail.next = ListNode(v)
        tail = tail.next
    return dummy.next


def to_list(head):
    out = []
    while head:
        out.append(head.val)
        head = head.next
    return out


if __name__ == "__main__":
    h = build([1, 2, 3, 4])
    reorder_list(h)
    assert to_list(h) == [1, 4, 2, 3]

    h2 = build([1, 2, 3, 4, 5])
    reorder_list(h2)
    assert to_list(h2) == [1, 5, 2, 4, 3]

    h3 = build([1])
    reorder_list(h3)
    assert to_list(h3) == [1]

    h4 = build([1, 2])
    reorder_list(h4)
    assert to_list(h4) == [1, 2]
    print("reorder_list: all tests passed")
