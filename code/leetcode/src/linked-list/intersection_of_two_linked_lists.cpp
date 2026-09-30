// 160. 相交链表
// 见 intersection_of_two_linked_lists.py 的题目与思路说明。
#include <cassert>
#include <iostream>

struct ListNode {
    int val;
    ListNode *next;
    ListNode(int x = 0, ListNode *n = nullptr) : val(x), next(n) {}
};

ListNode *getIntersectionNode(ListNode *headA, ListNode *headB) {
    ListNode *p = headA;
    ListNode *q = headB;
    while (p != q) {
        p = p ? p->next : headB;
        q = q ? q->next : headA;
    }
    return p;
}

int main() {
    ListNode *common = new ListNode(8, new ListNode(4, new ListNode(5)));
    ListNode *headA = new ListNode(4, new ListNode(1, common));
    ListNode *headB = new ListNode(5, new ListNode(0, new ListNode(1, common)));
    assert(getIntersectionNode(headA, headB) == common);

    ListNode *a = new ListNode(2, new ListNode(6, new ListNode(4)));
    ListNode *b = new ListNode(1, new ListNode(5));
    assert(getIntersectionNode(a, b) == nullptr);
    assert(getIntersectionNode(nullptr, b) == nullptr);
    std::cout << "intersection_of_two_linked_lists: all tests passed\n";
    return 0;
}
