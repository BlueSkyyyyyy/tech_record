"""148. 排序链表（Sort List）

题目：给你链表的头结点 head，请将其按升序排列并返回排序后的链表。

思路（分治：链表上的归并排序）：
    数组的归并排序需要「切一刀」把区间分开，链表没有下标，但可以用快慢指针
    找到中点，然后断开，得到两条子链；分别递归排好序后，再用「合并两个有序
    链表」的方式合并。整体还是分治三步：找中点（分解）→ 递归排序（解决）→
    合并两条有序链（合并）。

    为什么链表适合归并而不是快排：链表无法 O(1) 随机访问，快排的分区要来回
    跳跃，很别扭；而归并只需要顺序遍历，天然适配链表，还能做到「只改指针、
    不搬值」。

    为什么用快慢指针找中点：快指针每次走两步、慢指针走一步，快指针到尾部时
    慢指针正好在中点。这里让慢指针停在中点的前一个位置，方便用 slow.next = None
    把链断成两半，避免递归时互相纠缠。注意快指针从 head.next 起步，这样偶数
    长度时会取到偏左的中点。

    为什么合并要用虚拟头结点：合并结果的头结点不确定是来自左链还是右链，
    用 dummy 结点「先挂着」，最后返回 dummy.next，能省去大量判空分支。

复杂度：时间 O(n log n)（每层找中点 + 合并共 O(n)，共 log n 层），
    空间 O(log n)（递归栈；指针原地调整，不额外分配结点）。
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


def sort_list(head):
    if head is None or head.next is None:
        return head

    slow, fast = head, head.next
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    mid = slow.next
    slow.next = None

    left = sort_list(head)
    right = sort_list(mid)
    return merge_two(left, right)


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
    assert to_list(sort_list(build([4, 2, 1, 3]))) == [1, 2, 3, 4]
    assert to_list(sort_list(build([-1, 5, 3, 4, 0]))) == [-1, 0, 3, 4, 5]
    assert to_list(sort_list(build([]))) == []
    assert to_list(sort_list(build([1]))) == [1]
    assert to_list(sort_list(build([2, 2, 1, 1]))) == [1, 1, 2, 2]
    print("sort_list: all tests passed")
