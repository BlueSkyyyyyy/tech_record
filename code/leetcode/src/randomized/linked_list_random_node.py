"""382. 链表随机节点（Linked List Random Node）

题目：给定一个单链表的头结点，实现 getRandom()：等概率返回链表中某个节点的值。
    进阶要求：只能遍历一次，且额外空间 O(1)。

思路（水塘抽样 Reservoir Sampling）：
    从头到尾扫一遍。设当前是第 i 个节点（i 从 1 开始），以 1/i 的概率把答案替换成当前
    节点的值，否则保持不变（i=1 时必选，作为初始答案）。

    正确性：考察第 k 个节点最终留在答案里的概率——它要在第 k 步被选中（概率 1/k），
    且之后第 k+1…n 步都不能把它替换掉（第 i 步不被替换的概率是 1 - 1/i）：
        (1/k) · ∏_{i=k+1}^{n} (1 - 1/i) = (1/k) · ∏ (i-1)/i = (1/k) · (k/n) = 1/n。
    每个节点都恰好是 1/n，正是等概率。

复杂度：时间 O(n)，空间 O(1)。
"""

import random


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class Solution:
    def __init__(self, head):
        self.head = head

    def getRandom(self):
        res = self.head.val
        node = self.head.next
        i = 2
        while node:
            if random.randint(1, i) == 1:
                res = node.val
            node = node.next
            i += 1
        return res


if __name__ == "__main__":
    random.seed(0)
    head = ListNode(1, ListNode(2, ListNode(3, ListNode(4))))
    s = Solution(head)
    seen = set()
    for _ in range(4000):
        x = s.getRandom()
        assert x in (1, 2, 3, 4)
        seen.add(x)
    assert seen == {1, 2, 3, 4}
    assert Solution(ListNode(9)).getRandom() == 9
    print("linked_list_random_node: all tests passed")
