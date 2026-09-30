"""203. 移除链表元素（Remove Linked List Elements）

题目：给你一个链表的头节点 head 和一个整数 val，删除链表中所有满足
      Node.val == val 的节点，并返回新的头节点。

思路：头节点本身也可能被删，所以先建一个虚拟头结点 dummy 指向 head，
      让「删除」这件事对所有节点（包括原来的头）都走同一条路径。
      从 dummy 开始遍历，每次看「下一个节点」的值：
      - 如果它的值等于 val，就跳过它（cur.next = cur.next.next），cur 不动，
        因为跳过之后新的 cur.next 还需要再判断；
      - 否则 cur 前进一步。
      最后返回 dummy.next。

      为什么要盯「下一个节点」而不是当前节点：单链表不能回头，想删掉某个节点
      必须有它的前驱。把 cur 当作前驱、检查 cur.next，删除时就能直接改指针。

复杂度：时间 O(n)，空间 O(1)。
"""


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
    assert to_list(remove_elements(build([1, 2, 6, 3, 4, 5, 6]), 6)) == [1, 2, 3, 4, 5]
    assert to_list(remove_elements(build([6, 6, 6, 1]), 6)) == [1]
    assert to_list(remove_elements(build([7, 7, 7, 7]), 7)) == []
    assert to_list(remove_elements(build([]), 1)) == []
    print("remove_linked_list_elements: all tests passed")
