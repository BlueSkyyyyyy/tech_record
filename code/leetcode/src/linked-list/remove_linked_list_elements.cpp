// 203. 移除链表元素
// 见 remove_linked_list_elements.py 的题目与思路说明。
#include <cassert>
#include <iostream>
#include <vector>

struct ListNode {
    int val;
    ListNode *next;
    ListNode(int x = 0, ListNode *n = nullptr) : val(x), next(n) {}
};

ListNode *removeElements(ListNode *head, int val) {
    ListNode dummy(0, head);
    ListNode *cur = &dummy;
    while (cur->next) {
        if (cur->next->val == val) {
            cur->next = cur->next->next;
        } else {
            cur = cur->next;
        }
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
    std::vector<int> want = {1, 2, 3, 4, 5};
    assert(toList(removeElements(build({1, 2, 6, 3, 4, 5, 6}), 6)) == want);

    std::vector<int> want2 = {1};
    assert(toList(removeElements(build({6, 6, 6, 1}), 6)) == want2);

    std::vector<int> want3;
    assert(toList(removeElements(build({7, 7, 7, 7}), 7)) == want3);

    std::vector<int> want4;
    assert(toList(removeElements(build({}), 1)) == want4);
    std::cout << "remove_linked_list_elements: all tests passed\n";
    return 0;
}
