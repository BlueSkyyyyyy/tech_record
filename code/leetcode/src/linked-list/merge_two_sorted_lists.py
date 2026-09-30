"""21. 合并两个有序链表（Merge Two Sorted Lists）

题目：将两个升序链表合并为一个新的升序链表并返回。新链表由拼接原有节点组成。
      例如 1 -> 2 -> 4 与 1 -> 3 -> 4 合并为 1 -> 1 -> 2 -> 3 -> 4 -> 4。

思路：这就是归并排序里「合并」的那一步，但在链表上做，不需要额外数组。
      用一个「虚拟头结点」dummy 简化边界：tail 始终指向结果链表的最后一个节点。
      比较两条链表当前的头，谁小就把谁接到 tail 后面，并让那条链表前进一格，
      然后 tail 也前进。循环到某一条走空为止。

      为什么最后可以直接接上剩下的整条：两条链表各自已经有序，走空那条之后，
      另一条剩下的部分必然都不小于已接上的所有元素，直接整段挂到 tail.next 即可。

      为什么用虚拟头结点：结果链表的第一个节点由「谁更小」决定，若不用 dummy，
      就要为「结果为空」单独写一个分支。dummy 让每个节点都统一地接在 tail 后面，
      最后返回 dummy.next 就行，边界代码被消掉了。

复杂度：时间 O(m + n)，空间 O(1)（只重接指针，不新建节点）。
"""


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


def merge_two_lists(list1, list2):
    dummy = ListNode()
    tail = dummy
    while list1 and list2:
        if list1.val <= list2.val:
            tail.next = list1
            list1 = list1.next
        else:
            tail.next = list2
            list2 = list2.next
        tail = tail.next
    tail.next = list1 if list1 else list2
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
    assert to_list(merge_two_lists(build([1, 2, 4]), build([1, 3, 4]))) == [1, 1, 2, 3, 4, 4]
    assert to_list(merge_two_lists(build([]), build([]))) == []
    assert to_list(merge_two_lists(build([]), build([0]))) == [0]
    assert to_list(merge_two_lists(build([5]), build([1, 2, 3]))) == [1, 2, 3, 5]
    print("merge_two_sorted_lists: all tests passed")
