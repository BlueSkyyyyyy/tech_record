// 92. 反转链表 II
// 见 reverse_linked_list_ii.py 的题目与思路说明。
#include <cassert>
#include <iostream>
#include <vector>

struct ListNode {
    int val;
    ListNode *next;
    ListNode(int x = 0, ListNode *n = nullptr) : val(x), next(n) {}
};

ListNode *reverseBetween(ListNode *head, int left, int right) {
    ListNode dummy(0, head);
    ListNode *pre = &dummy;
    for (int i = 0; i < left - 1; ++i) pre = pre->next;
    ListNode *cur = pre->next;
    for (int i = 0; i < right - left; ++i) {
        ListNode *nxt = cur->next;
        cur->next = nxt->next;
        nxt->next = pre->next;
        pre->next = nxt;
    }
    return dummy.next;
}

ListNode *build(const std::vector<int> &values) {
    ListNode dummy;
    ListNode *tail = &dummy;
    for (int v : values) {
        tail->next = new ListNode(v);
        tail = tail->next;
    }
    return dummy.next;
}

std::vector<int> toList(ListNode *head) {
    std::vector<int> out;
    while (head) {
        out.push_back(head->val);
        head = head->next;
    }
    return out;
}

int main() {
    std::vector<int> want = {1, 4, 3, 2, 5};
    assert(toList(reverseBetween(build({1, 2, 3, 4, 5}), 2, 4)) == want);

    std::vector<int> want2 = {5};
    assert(toList(reverseBetween(build({5}), 1, 1)) == want2);

    std::vector<int> want3 = {3, 2, 1};
    assert(toList(reverseBetween(build({1, 2, 3}), 1, 3)) == want3);

    std::vector<int> want4 = {1, 2};
    assert(toList(reverseBetween(build({1, 2}), 2, 2)) == want4);
    std::cout << "reverse_linked_list_ii: all tests passed\n";
    return 0;
}
