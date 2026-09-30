"""876. 链表的中间结点（Middle of the Linked List）

题目：给你单链表的头节点 head，返回链表的中间结点。如果有两个中间结点，
      则返回第二个中间结点。

思路：快慢指针。慢指针每次走一步，快指针每次走两步。
      快指针走到尽头时，慢指针恰好走了一半，正好停在中间。
      因为快指针的速度是慢指针的两倍，同样的时间里快指针走过的路程也是两倍，
      当快指针走完整条链表，慢指针自然落在中点。

      为什么「两个中间结点时返回第二个」：当链表长度为偶数，循环条件
      fast and fast.next 会在 fast 走到最后一个节点时结束，此时 slow 停在后半段的
      第一个节点，也就是两个中间结点中的第二个——正好符合题目要求。

复杂度：时间 O(n)，空间 O(1)。
"""


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


def build(values):
    dummy = ListNode()
    tail = dummy
    for v in values:
        tail.next = ListNode(v)
        tail = tail.next
    return dummy.next


if __name__ == "__main__":
    assert middle_node(build([1, 2, 3, 4, 5])).val == 3
    assert middle_node(build([1, 2, 3, 4, 5, 6])).val == 4
    assert middle_node(build([1])).val == 1
    assert middle_node(build([1, 2])).val == 2
    print("middle_of_linked_list: all tests passed")
