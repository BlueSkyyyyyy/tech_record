"""141. 环形链表（Linked List Cycle）

题目：给你一个链表的头节点 head，判断链表中是否有环。
      为表示环，链表的尾节点会连到某个先前节点上。

思路：用快慢指针（龟兔赛跑）。慢指针每次走一步，快指针每次走两步。
      如果链表没有环，快指针会先走到尽头（None），循环结束、返回 False；
      如果有环，两个指针最终都会进入环里，快的一直在慢慢逼近慢的，
      在环内它们每轮距离缩短 1，所以必定会在某一刻相遇，返回 True。

      为什么不改节点值 / 不用哈希表：哈希表记录访问过的节点能做到 O(n) 时间和 O(n) 空间；
      快慢指针把空间压到 O(1)，是本题的标准最优解。

      为什么「有环必相遇」：进入环后，快指针相对慢指针每轮多走一步，
      相当于以速度 1 追慢指针，二者距离每轮减少 1，迟早归零，不可能一直错开。

      为什么循环条件是 fast and fast.next：快指针要连续走两步，
      必须保证 fast 和 fast.next 都不是 None，否则访问 fast.next.next 会越界。

复杂度：时间 O(n)，空间 O(1)。
"""


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


def build_with_cycle(values, pos):
    if not values:
        return None
    nodes = [ListNode(v) for v in values]
    for i in range(len(nodes) - 1):
        nodes[i].next = nodes[i + 1]
    if pos != -1:
        nodes[-1].next = nodes[pos]
    return nodes[0]


if __name__ == "__main__":
    assert has_cycle(build_with_cycle([3, 2, 0, -4], 1)) is True
    assert has_cycle(build_with_cycle([1, 2], 0)) is True
    assert has_cycle(build_with_cycle([1], -1)) is False
    assert has_cycle(build_with_cycle([1, 2, 3, 4], -1)) is False
    assert has_cycle(build_with_cycle([], -1)) is False
    print("linked_list_cycle: all tests passed")
