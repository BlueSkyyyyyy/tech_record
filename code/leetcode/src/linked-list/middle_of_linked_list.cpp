// 876. 链表的中间结点
// 见 middle_of_linked_list.py 的题目与思路说明。
#include <cassert>
#include <iostream>
#include <vector>

struct ListNode {
    int val;
    ListNode *next;
    ListNode(int x = 0, ListNode *n = nullptr) : val(x), next(n) {}
};

ListNode *middleNode(ListNode *head) {
    ListNode *slow = head;
    ListNode *fast = head;
    while (fast && fast->next) {
        slow = slow->next;
        fast = fast->next->next;
    }
    return slow;
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

int main() {
    assert(middleNode(build({1, 2, 3, 4, 5}))->val == 3);
    assert(middleNode(build({1, 2, 3, 4, 5, 6}))->val == 4);
    assert(middleNode(build({1}))->val == 1);
    assert(middleNode(build({1, 2}))->val == 2);
    std::cout << "middle_of_linked_list: all tests passed\n";
    return 0;
}
