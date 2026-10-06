// 147. 对链表进行插入排序
// 见 insertion_sort_list.py 的题目与思路说明。
#include <cassert>
#include <iostream>
#include <vector>

struct ListNode {
    int val;
    ListNode *next;
    explicit ListNode(int v = 0, ListNode *n = nullptr) : val(v), next(n) {}
};

ListNode *insertionSortList(ListNode *head) {
    ListNode dummy(0);
    ListNode *cur = head;
    while (cur) {
        ListNode *nxt = cur->next;
        ListNode *prev = &dummy;
        while (prev->next && prev->next->val <= cur->val) {
            prev = prev->next;
        }
        cur->next = prev->next;
        prev->next = cur;
        cur = nxt;
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
    std::vector<int> want1 = {1, 2, 3, 4};
    assert(toList(insertionSortList(build({4, 2, 1, 3}))) == want1);

    std::vector<int> want2 = {-1, 0, 3, 4, 5};
    assert(toList(insertionSortList(build({-1, 5, 3, 4, 0}))) == want2);

    std::vector<int> want3 = {};
    assert(toList(insertionSortList(build({}))) == want3);

    std::vector<int> want4 = {1};
    assert(toList(insertionSortList(build({1}))) == want4);

    std::vector<int> want5 = {1, 1, 3, 3};
    assert(toList(insertionSortList(build({3, 3, 1, 1}))) == want5);

    std::cout << "insertion_sort_list: all tests passed\n";
    return 0;
}
