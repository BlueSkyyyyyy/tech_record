"""206. 反转链表（Reverse Linked List）

题目：给你单链表的头节点 head，反转链表并返回新的头节点。
      例如 1 -> 2 -> 3 -> 4 -> 5 变为 5 -> 4 -> 3 -> 2 -> 1。

思路：反转的本质是「把每条边的方向掉个头」。用三个指针一边走一边改：
        prev 指向已经反转好的那段的头（初始为空），
        cur 指向当前待处理的节点，
        每轮先用 nxt 记住 cur 的下一个节点（否则改完指针就丢了后半段），
        再把 cur.next 指向 prev，把 prev 和 cur 各往前挪一格。
      走完时 cur 为空、prev 落在原链表的最后一个节点，它就是新链表的头。

      为什么必须先用 nxt 暂存：一旦执行 cur.next = prev，cur 与后继的联系就断了。
      所以「先存后改」是把链表反转写对的关键顺序。

      为什么用迭代而不是递归：迭代是 O(1) 额外空间，递归虽然同样 O(n) 时间，但会占用
      O(n) 的调用栈，链表很长时有栈溢出风险。本题只详展迭代版。

复杂度：时间 O(n)，空间 O(1)。
"""


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
    assert to_list(reverse_list(build([1, 2, 3, 4, 5]))) == [5, 4, 3, 2, 1]
    assert to_list(reverse_list(build([1, 2]))) == [2, 1]
    assert to_list(reverse_list(build([1]))) == [1]
    assert to_list(reverse_list(build([]))) == []
    print("reverse_linked_list: all tests passed")
