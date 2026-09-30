// 143. 重排链表
// 见 reorder_list.py 的题目与思路说明。
#include <cassert>
#include <iostream>
#include <vector>

struct ListNode {
    int val;
    ListNode *next;
    ListNode(int x = 0, ListNode *n = nullptr) : val(x), next(n) {}
};

void reorderList(ListNode *head) {
    if (!head || !head->next) return;
    ListNode *slow = head;
    ListNode *fast = head;
    while (fast && fast->next) {
        slow = slow->next;
        fast = fast->next->next;
    }

    ListNode *prev = nullptr;
    ListNode *cur = slow->next;
    slow->next = nullptr;
    while (cur) {
        ListNode *nxt = cur->next;
        cur->next = prev;
        prev = cur;
        cur = nxt;
    }

    ListNode *first = head;
    ListNode *second = prev;
    while (second) {
        ListNode *next1 = first->next;
        ListNode *next2 = second->next;
        first->next = second;
        second->next = next1;
        first = next1;
        second = next2;
    }
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
    ListNode *h = build({1, 2, 3, 4});
    reorderList(h);
    std::vector<int> want = {1, 4, 2, 3};
    assert(toList(h) == want);

    ListNode *h2 = build({1, 2, 3, 4, 5});
    reorderList(h2);
    std::vector<int> want2 = {1, 5, 2, 4, 3};
    assert(toList(h2) == want2);

    ListNode *h3 = build({1});
    reorderList(h3);
    std::vector<int> want3 = {1};
    assert(toList(h3) == want3);

    ListNode *h4 = build({1, 2});
    reorderList(h4);
    std::vector<int> want4 = {1, 2};
    assert(toList(h4) == want4);
    std::cout << "reorder_list: all tests passed\n";
    return 0;
}
