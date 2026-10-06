// 148. 排序链表（链表上的归并排序）
// 见 sort_list.py 的题目与思路说明。
#include <cassert>
#include <iostream>
#include <vector>

struct ListNode {
    int val;
    ListNode *next;
    ListNode(int x = 0, ListNode *n = nullptr) : val(x), next(n) {}
};

ListNode *mergeTwo(ListNode *a, ListNode *b) {
    ListNode dummy;
    ListNode *tail = &dummy;
    while (a && b) {
        if (a->val <= b->val) {
            tail->next = a;
            a = a->next;
        } else {
            tail->next = b;
            b = b->next;
        }
        tail = tail->next;
    }
    tail->next = a ? a : b;
    return dummy.next;
}

ListNode *sortList(ListNode *head) {
    if (head == nullptr || head->next == nullptr) return head;

    ListNode *slow = head;
    ListNode *fast = head->next;
    while (fast && fast->next) {
        slow = slow->next;
        fast = fast->next->next;
    }
    ListNode *mid = slow->next;
    slow->next = nullptr;

    ListNode *left = sortList(head);
    ListNode *right = sortList(mid);
    return mergeTwo(left, right);
}

ListNode *buildList(const std::vector<int> &values) {
    ListNode dummy;
    ListNode *tail = &dummy;
    for (int value : values) {
        tail->next = new ListNode(value);
        tail = tail->next;
    }
    return dummy.next;
}

std::vector<int> toValues(ListNode *head) {
    std::vector<int> values;
    while (head) {
        values.push_back(head->val);
        head = head->next;
    }
    return values;
}

int main() {
    std::vector<int> want1 = {1, 2, 3, 4};
    assert(toValues(sortList(buildList({4, 2, 1, 3}))) == want1);

    std::vector<int> want2 = {-1, 0, 3, 4, 5};
    assert(toValues(sortList(buildList({-1, 5, 3, 4, 0}))) == want2);

    assert(toValues(sortList(buildList({}))).empty());

    std::vector<int> want3 = {1};
    assert(toValues(sortList(buildList({1}))) == want3);

    std::vector<int> want4 = {1, 1, 2, 2};
    assert(toValues(sortList(buildList({2, 2, 1, 1}))) == want4);
    std::cout << "sort_list: all tests passed\n";
    return 0;
}
