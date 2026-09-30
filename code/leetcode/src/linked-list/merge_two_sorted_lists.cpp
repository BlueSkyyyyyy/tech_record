// 21. 合并两个有序链表
// 见 merge_two_sorted_lists.py 的题目与思路说明。
#include <cassert>
#include <iostream>
#include <vector>

struct ListNode {
    int val;
    ListNode *next;
    ListNode(int x = 0, ListNode *n = nullptr) : val(x), next(n) {}
};

ListNode *mergeTwoLists(ListNode *list1, ListNode *list2) {
    ListNode dummy;
    ListNode *tail = &dummy;
    while (list1 && list2) {
        if (list1->val <= list2->val) {
            tail->next = list1;
            list1 = list1->next;
        } else {
            tail->next = list2;
            list2 = list2->next;
        }
        tail = tail->next;
    }
    tail->next = list1 ? list1 : list2;
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
    std::vector<int> want = {1, 1, 2, 3, 4, 4};
    assert(toList(mergeTwoLists(build({1, 2, 4}), build({1, 3, 4}))) == want);

    std::vector<int> want2;
    assert(toList(mergeTwoLists(build({}), build({}))) == want2);

    std::vector<int> want3 = {0};
    assert(toList(mergeTwoLists(build({}), build({0}))) == want3);

    std::vector<int> want4 = {1, 2, 3, 5};
    assert(toList(mergeTwoLists(build({5}), build({1, 2, 3}))) == want4);
    std::cout << "merge_two_sorted_lists: all tests passed\n";
    return 0;
}
