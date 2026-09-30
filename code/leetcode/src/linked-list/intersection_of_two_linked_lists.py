"""160. 相交链表（Intersection of Two Linked Lists）

题目：给你两个单链表的头节点 headA 和 headB，找出并返回两个链表相交的起始节点；
      如果不相交，返回 None。题目保证整个链式结构中没有环。

思路：双指针「走完自己换到对方」。指针 p 从 headA 出发、q 从 headB 出发，每次各走一步。
      当 p 走到自己链表的尽头时，让它从 headB 重新出发；q 同理，走到尽头就换到 headA。
      如果两链表相交，它们会在相交点相遇；如果不相交，两个指针都会走完两条链表的全部结点，
      最后同时变成 None，循环自然结束并返回 None。

      为什么这样能相遇：设 A 独有段长 a、B 独有段长 b、公共段长 c。
      p 走的路径是 a + c + b，q 走的是 b + c + a，长度都是 a+b+c，
      所以相遇点必然落在公共段的起点上（若 c = 0 则最终同时在 None 相遇）。

      为什么不用「先求长度差再对齐」：那样要遍历两遍、多写几行代码；
      本解法同样 O(m+n) 时间、O(1) 空间，却把对齐这一步自然融进了「换头」里。

复杂度：时间 O(m + n)，空间 O(1)。
"""


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


def get_intersection_node(headA, headB):
    p, q = headA, headB
    while p is not q:
        p = p.next if p else headB
        q = q.next if q else headA
    return p


if __name__ == "__main__":
    # 构造相交：A = 4->1->8->4->5, B = 5->0->1->8->4->5，公共段从 8 开始
    common = ListNode(8, ListNode(4, ListNode(5)))
    headA = ListNode(4, ListNode(1, common))
    headB = ListNode(5, ListNode(0, ListNode(1, common)))
    assert get_intersection_node(headA, headB) is common

    # 不相交
    a = ListNode(2, ListNode(6, ListNode(4)))
    b = ListNode(1, ListNode(5))
    assert get_intersection_node(a, b) is None

    # 某一方为空
    assert get_intersection_node(None, b) is None
    print("intersection_of_two_linked_lists: all tests passed")
