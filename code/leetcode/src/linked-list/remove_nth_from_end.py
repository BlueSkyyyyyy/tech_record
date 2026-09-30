"""19. 删除链表的倒数第 N 个结点（Remove Nth Node From End of List）

题目：给你一个链表的头节点 head，删除链表的倒数第 n 个结点，并返回头节点。

思路：用快慢指针一次遍历完成。快指针先走 n 步，拉开与慢指针 n 个身位的差距；
      然后快慢一起走，当快指针到达最后一个结点时，慢指针正好停在待删结点的**前驱**上，
      于是 slow.next = slow.next.next 即可完成删除。

      为了统一处理「删除的是头结点」这种边界，仍然先建虚拟头结点 dummy，
      让快慢都从 dummy 出发：快指针先走 n 步后，慢指针停在倒数第 n 个的前驱，
      不会因为 n 等于链表长度而越界。返回 dummy.next。

      为什么快指针先走 n 步：快指针比慢指针领先 n 个结点，当快指针到尾部（最后一个结点）时，
      慢指针所在位置之后恰好有 n 个结点，慢指针的下一个就是倒数第 n 个。

复杂度：时间 O(n)（只走一趟），空间 O(1)。
"""


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
    assert to_list(remove_nth_from_end(build([1, 2, 3, 4, 5]), 2)) == [1, 2, 3, 5]
    assert to_list(remove_nth_from_end(build([1]), 1)) == []
    assert to_list(remove_nth_from_end(build([1, 2]), 1)) == [1]
    assert to_list(remove_nth_from_end(build([1, 2]), 2)) == [2]
    print("remove_nth_from_end: all tests passed")
