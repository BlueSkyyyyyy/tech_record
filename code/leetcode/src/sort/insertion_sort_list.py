"""147. 对链表进行插入排序（Insertion Sort List）

题目：给你单链表的头节点 head，请将它按升序排序，并返回排序后的链表头。
      要求使用插入排序。

思路（哨兵 + 逐节点插入有序前缀）：
    插入排序的链表版非常自然：维护一条「已经排好序」的链，依次把原链表的每个节点
    插到有序链中正确的位置。

    用虚拟头结点 dummy 作为有序链的哨兵，这样「插到最前面」和「插到中间」可以用同一
    段代码处理，不用为头结点写特例。每轮：
      1) 用 `nxt` 记住当前节点的下一个（插入会改指针，先存后改，与反转链表同理）；
      2) 从 dummy 出发，找第一个「值大于当前值」的节点，插在它前面；
      3) 把当前节点接上，游标移到 `nxt` 继续。

    为什么找的是「first greater」：这样相等元素会排在已有相等元素之后，保持稳定；
    链表插入排序本身是稳定的。

    复杂度：时间 O(n^2)（每个节点在最坏情况下要从头找插入点），空间 O(1)。
    若追求 O(n log n)，可用归并排序（见 `docs/12-divide-conquer.md` 的 148 题）。
"""


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


def insertion_sort_list(head):
    dummy = ListNode(0)
    cur = head
    while cur:
        nxt = cur.next
        prev = dummy
        while prev.next and prev.next.val <= cur.val:
            prev = prev.next
        cur.next = prev.next
        prev.next = cur
        cur = nxt
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
    assert to_list(insertion_sort_list(build([4, 2, 1, 3]))) == [1, 2, 3, 4]
    assert to_list(insertion_sort_list(build([-1, 5, 3, 4, 0]))) == [-1, 0, 3, 4, 5]
    assert to_list(insertion_sort_list(build([]))) == []
    assert to_list(insertion_sort_list(build([1]))) == [1]
    assert to_list(insertion_sort_list(build([3, 3, 1, 1]))) == [1, 1, 3, 3]
    print("insertion_sort_list: all tests passed")
