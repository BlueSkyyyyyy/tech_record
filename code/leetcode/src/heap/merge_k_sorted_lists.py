"""23. 合并 K 个升序链表（Merge k Sorted Lists）

题目：给定一个链表数组，每个链表都已经按升序排列。
      请将所有链表合并成一个升序链表并返回。

思路（小顶堆做多路归并）：
    这是「归并两个有序链表」的推广：当有 k 条链时，暴力做法是两两合并，
    每次都要重新比较 k 个头结点，复杂度会退化。
    用小顶堆把「当前所有链表的头结点」放进一个池子，堆顶就是全局最小的结点：
    每次弹出堆顶，接到结果链表尾部；然后把这个结点的下一个再压回堆。
    堆里始终只有至多 k 个候选，取最小值只要 O(log k)。

    为什么堆里要存一个额外下标：
    Python/C++ 的堆在比较元素时，如果值相等会继续比较后面的字段。
    如果直接放链表结点，相等时就会去比较结点对象，报错或行为未定义。
    所以打包成 (结点值, 唯一序号, 结点)：值相等时用递增序号区分，绝对不会比到结点本身。

    另一种做法是分治：把 k 条链两两配对合并，共 log k 轮，每轮 O(n)。
    两者都是 O(n log k)，堆解更直观，分治解在分治篇里再讲。

复杂度：时间 O(n log k)（n 为结点总数，k 为链表条数），空间 O(k)。
"""

import heapq


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


def merge_k_lists(lists):
    heap = []
    counter = 0
    for head in lists:
        if head is not None:
            heapq.heappush(heap, (head.val, counter, head))
            counter += 1

    dummy = ListNode()
    tail = dummy
    while heap:
        _, _, node = heapq.heappop(heap)
        tail.next = node
        tail = node
        if node.next is not None:
            heapq.heappush(heap, (node.next.val, counter, node.next))
            counter += 1
    tail.next = None
    return dummy.next


def build_list(values):
    dummy = ListNode()
    tail = dummy
    for value in values:
        tail.next = ListNode(value)
        tail = tail.next
    return dummy.next


def to_values(head):
    values = []
    while head is not None:
        values.append(head.val)
        head = head.next
    return values


if __name__ == "__main__":
    merged = merge_k_lists([
        build_list([1, 4, 5]),
        build_list([1, 3, 4]),
        build_list([2, 6]),
    ])
    assert to_values(merged) == [1, 1, 2, 3, 4, 4, 5, 6]

    assert merge_k_lists([]) is None
    assert to_values(merge_k_lists([build_list([])])) == []
    assert to_values(merge_k_lists([build_list([1])])) == [1]
    assert to_values(merge_k_lists([build_list([1, 2]), build_list([])])) == [1, 2]
    assert to_values(merge_k_lists([build_list([-2, 0]), build_list([-1, 3])])) == [-2, -1, 0, 3]
    print("merge_k_sorted_lists: all tests passed")
