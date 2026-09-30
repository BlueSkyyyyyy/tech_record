"""92. 反转链表 II（Reverse Linked List II）

题目：给你单链表的头节点 head 和两个整数 left、right（left <= right），
      反转从位置 left 到位置 right 的链表节点（位置从 1 开始计数），返回反转后的链表。

思路：先用虚拟头结点 dummy 消掉「left = 1（从头开始反转）」的边界。
      走到第 left 个节点的前驱记为 pre，pre.next 是反转段的第一个节点 cur。
      然后做 right - left 轮「头插法」：每轮把 cur 的下一个节点 nxt 择到反转段的头部。
      这样反转段里的节点会依次被搬到最前面，最终整段反转完成，且前后两段自然接回。

      为什么用头插法而不是整段摘下再反转再接回：头插法不需要额外记录反转段的结尾，
      每轮只改三个指针，就地完成，并且全程只走一趟。

      头插三步（把 nxt 插到 pre 之后）：
        nxt = cur.next
        cur.next = nxt.next
        nxt.next = pre.next
        pre.next = nxt

复杂度：时间 O(n)，空间 O(1)。
"""


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
    assert to_list(reverse_between(build([1, 2, 3, 4, 5]), 2, 4)) == [1, 4, 3, 2, 5]
    assert to_list(reverse_between(build([5]), 1, 1)) == [5]
    assert to_list(reverse_between(build([1, 2, 3]), 1, 3)) == [3, 2, 1]
    assert to_list(reverse_between(build([1, 2]), 2, 2)) == [1, 2]
    print("reverse_linked_list_ii: all tests passed")
