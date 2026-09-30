// 25. K 个一组翻转链表
// 见 reverse_nodes_in_k_group.py 的题目与思路说明。
#include <cassert>
#include <iostream>
#include <vector>

struct ListNode {
    int val;
    ListNode *next;
    ListNode(int x = 0, ListNode *n = nullptr) : val(x), next(n) {}
};

ListNode *reverseKGroup(ListNode *head, int k) {
    ListNode dummy(0, head);
    ListNode *groupPrev = &dummy;
    while (true) {
        ListNode *kth = groupPrev;
        for (int i = 0; i < k; ++i) {
            kth = kth->next;
            if (!kth) return dummy.next;
        }
        ListNode *groupNext = kth->next;
        ListNode *prev = groupNext;
        ListNode *cur = groupPrev->next;
        while (cur != groupNext) {
            ListNode *nxt = cur->next;
            cur->next = prev;
            prev = cur;
            cur = nxt;
        }
        ListNode *oldHead = groupPrev->next;
        groupPrev->next = kth;
        groupPrev = oldHead;
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
    std::vector<int> want = {2, 1, 4, 3, 5};
    assert(toList(reverseKGroup(build({1, 2, 3, 4, 5}), 2)) == want);

    std::vector<int> want2 = {3, 2, 1, 4, 5};
    assert(toList(reverseKGroup(build({1, 2, 3, 4, 5}), 3)) == want2);

    std::vector<int> want3 = {1, 2, 3, 4, 5};
    assert(toList(reverseKGroup(build({1, 2, 3, 4, 5}), 1)) == want3);

    std::vector<int> want4 = {1};
    assert(toList(reverseKGroup(build({1}), 1)) == want4);

    std::vector<int> want5 = {1, 2};
    assert(toList(reverseKGroup(build({1, 2}), 3)) == want5);
    std::cout << "reverse_nodes_in_k_group: all tests passed\n";
    return 0;
}
