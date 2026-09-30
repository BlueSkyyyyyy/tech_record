"""142. 环形链表 II（Linked List Cycle II）

题目：给你一个链表的头节点 head，返回链表开始入环的第一个节点；如果链表无环，返回 None。

思路：先用快慢指针判断有没有环；若有环，在相遇点再做一次「找入环口」。
      第一阶段与 141 相同：慢指针走一步、快指针走两步，若相遇则有环。
      第二阶段：让其中一个指针回到 head，两个指针都改成每次走一步，
      它们再次相遇的位置就是入环的第一个节点。

      为什么这样能找到入环口（推导）：设头到入环口距离为 x，入环口到相遇点为 y，
      环长为 L。相遇时慢指针走了 x + y，快指针走了 x + y + nL（n 是快指针多绕的圈数）。
      又因为快指针速度是慢的两倍，所以 x + y + nL = 2(x + y)，即 x = nL - y。
      这意味着「从头走 x 步」与「从相遇点走 nL - y 步」会到达同一个点——
      而后者恰好等于「从相遇点绕到入环口」，所以两个指针同速对走必然在入环口相遇。

      为什么快指针相对慢指针一定会相遇（而不是跳过）：进入环后快指针每轮比慢指针多走一步，
      两者在环内的距离每轮减少 1，必在有限轮内归零。

复杂度：时间 O(n)，空间 O(1)。
"""


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


def detect_cycle(head):
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow is fast:
            slow = head
            while slow is not fast:
                slow = slow.next
                fast = fast.next
            return slow
    return None


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
    head = build_with_cycle([3, 2, 0, -4], 1)
    assert detect_cycle(head) is head.next

    loop = build_with_cycle([1, 2], 0)
    assert detect_cycle(loop) is loop

    assert detect_cycle(build_with_cycle([1], -1)) is None
    assert detect_cycle(build_with_cycle([], -1)) is None
    print("linked_list_cycle_ii: all tests passed")
