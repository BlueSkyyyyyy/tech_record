"""23. 合并 K 个升序链表（Merge k Sorted Lists，分治解）

题目：给定一个链表数组，每个链表都已经按升序排列。
      请将所有链表合并成一个升序链表并返回。

思路（分治：两两配对合并）：
    如果先把第 1 条和第 2 条合并、结果再和第 3 条合并……那么第 1 条链会被
    反复扫描 k 次，最坏退化到 O(kn)。分治把 k 条链两两配对：第 1 与第 2 合、
    第 3 与第 4 合……一轮下来链表条数减半，每轮的合并总量都是 O(n)，
    共 log k 轮，于是总复杂度 O(n log k)。

    为什么配对合并更均衡：每条链在每一轮最多参与一次合并，随着轮数增加，
    每条链被扫描的次数是 log k 左右，而不是被「一条龙」串起来时的 k 次。
    这和归并排序「每层都两两合并」的均衡思想完全一致。

    为什么用迭代的分治而不是递归：把链表数组不断折半成「待合并的两个子数组」，
    迭代版只需一个 while 循环，把相邻两条合并后放回新数组，直到数组里只剩一条。

    另一种做法是用小顶堆做多路归并（见堆篇），复杂度同为 O(n log k)，
    但堆解需要额外的堆空间，分治解只靠「合并两个有序链表」这一个积木。

复杂度：时间 O(n log k)（n 为结点总数，k 为链表条数），
    空间 O(log k)（若按递归折半写；本实现为迭代，额外空间 O(k) 存放中间结果）。
"""


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


def merge_two(a, b):
    dummy = ListNode()
    tail = dummy
    while a and b:
        if a.val <= b.val:
            tail.next = a
            a = a.next
        else:
            tail.next = b
            b = b.next
        tail = tail.next
    tail.next = a if a else b
    return dummy.next


def merge_k_lists(lists):
    if not lists:
        return None
    while len(lists) > 1:
        merged = []
        for i in range(0, len(lists), 2):
            if i + 1 < len(lists):
                merged.append(merge_two(lists[i], lists[i + 1]))
            else:
                merged.append(lists[i])
        lists = merged
    return lists[0]


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
