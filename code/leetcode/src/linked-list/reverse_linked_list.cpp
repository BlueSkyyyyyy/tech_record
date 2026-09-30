// 206. 反转链表
// 见 reverse_linked_list.py 的题目与思路说明。
#include <cassert>
#include <iostream>
#include <vector>

struct ListNode {
    int val;
    ListNode *next;
    ListNode(int x = 0, ListNode *n = nullptr) : val(x), next(n) {}
};

ListNode *reverseList(ListNode *head) {
    ListNode *prev = nullptr;
    ListNode *cur = head;
    while (cur) {
        ListNode *nxt = cur->next;
        cur->next = prev;
        prev = cur;
        cur = nxt;
    }
    return prev;
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
    std::vector<int> want = {5, 4, 3, 2, 1};
    assert(toList(reverseList(build({1, 2, 3, 4, 5}))) == want);
    std::vector<int> want2 = {2, 1};
    assert(toList(reverseList(build({1, 2}))) == want2);
    std::vector<int> want3 = {1};
    assert(toList(reverseList(build({1}))) == want3);
    std::vector<int> want4;
    assert(toList(reverseList(build({}))) == want4);
    std::cout << "reverse_linked_list: all tests passed\n";
    return 0;
}
