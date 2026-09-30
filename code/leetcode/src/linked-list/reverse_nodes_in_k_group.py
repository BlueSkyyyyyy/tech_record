"""25. K 个一组翻转链表（Reverse Nodes in k-Group）

题目：给你一个链表的头节点 head，每 k 个节点一组进行翻转，返回翻转后的链表。
      如果节点总数不是 k 的整数倍，最后剩余的节点保持原有顺序。

思路：用虚拟头结点 dummy，让「组前驱」group_prev 从 dummy 出发。循环做三件事：
      1. 试探：从 group_prev 向前数 k 个节点，如果凑不满，说明到了尾部，直接返回；
      2. 记下这一组之后的第一个节点 group_next（即第 k+1 个节点），作为翻转后的挂接点；
      3. 用「整段反转」的方式把 group_prev.next 到 kth 这 k 个节点反转，
         再把 group_prev.next 指向新的组头 kth，把原来的组头（现在是组尾）接到 group_next。

      为什么每组都要先试探能否凑满 k 个：题目规定不足 k 个的尾组保持原序，
      所以必须能提前知道「这一组够不够」，不够就整体退出，不能翻转。

      翻转时把 prev 初始化为 group_next，相当于给这段预置了一个「后面的锚」，
      循环结束时原组头自然指向 group_next，省去单独拼接尾部的代码。

复杂度：时间 O(n)（每个节点恰好被访问常数次），空间 O(1)。
"""


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
    assert to_list(reverse_k_group(build([1, 2, 3, 4, 5]), 2)) == [2, 1, 4, 3, 5]
    assert to_list(reverse_k_group(build([1, 2, 3, 4, 5]), 3)) == [3, 2, 1, 4, 5]
    assert to_list(reverse_k_group(build([1, 2, 3, 4, 5]), 1)) == [1, 2, 3, 4, 5]
    assert to_list(reverse_k_group(build([1]), 1)) == [1]
    assert to_list(reverse_k_group(build([1, 2]), 3)) == [1, 2]
    print("reverse_nodes_in_k_group: all tests passed")
