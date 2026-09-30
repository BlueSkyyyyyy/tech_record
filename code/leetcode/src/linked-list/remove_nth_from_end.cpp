// 19. 删除链表的倒数第 N 个结点
// 见 remove_nth_from_end.py 的题目与思路说明。
#include <cassert>
#include <iostream>
#include <vector>

struct ListNode {
    int val;
    ListNode *next;
    ListNode(int x = 0, ListNode *n = nullptr) : val(x), next(n) {}
};

ListNode *removeNthFromEnd(ListNode *head, int n) {
    ListNode dummy(0, head);
    ListNode *fast = &dummy;
    ListNode *slow = &dummy;
    for (int i = 0; i < n; ++i) {
        fast = fast->next;
    }
    while (fast->next) {
        fast = fast->next;
        slow = slow->next;
    }
    slow->next = slow->next->next;
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
    std::vector<int> want = {1, 2, 3, 5};
    assert(toList(removeNthFromEnd(build({1, 2, 3, 4, 5}), 2)) == want);

    std::vector<int> want2;
    assert(toList(removeNthFromEnd(build({1}), 1)) == want2);

    std::vector<int> want3 = {1};
    assert(toList(removeNthFromEnd(build({1, 2}), 1)) == want3);

    std::vector<int> want4 = {2};
    assert(toList(removeNthFromEnd(build({1, 2}), 2)) == want4);
    std::cout << "remove_nth_from_end: all tests passed\n";
    return 0;
}
