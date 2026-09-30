// 23. 合并 K 个升序链表
// 见 merge_k_sorted_lists.py 的题目与思路说明。
#include <cassert>
#include <iostream>
#include <queue>
#include <tuple>
#include <vector>

struct ListNode {
    int val;
    ListNode *next;
    ListNode(int x = 0, ListNode *n = nullptr) : val(x), next(n) {}
};

ListNode *mergeKLists(std::vector<ListNode *> lists) {
    using Item = std::tuple<int, int, ListNode *>;
    std::priority_queue<Item, std::vector<Item>, std::greater<Item>> minHeap;
    int counter = 0;
    for (ListNode *head : lists) {
        if (head) minHeap.push({head->val, counter++, head});
    }

    ListNode dummy;
    ListNode *tail = &dummy;
    while (!minHeap.empty()) {
        auto [val, idx, node] = minHeap.top();
        minHeap.pop();
        tail->next = node;
        tail = node;
        if (node->next) minHeap.push({node->next->val, counter++, node->next});
    }
    tail->next = nullptr;
    return dummy.next;
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
    std::vector<ListNode *> lists = {buildList({1, 4, 5}), buildList({1, 3, 4}), buildList({2, 6})};
    std::vector<int> want = {1, 1, 2, 3, 4, 4, 5, 6};
    assert(toValues(mergeKLists(lists)) == want);

    assert(mergeKLists({}) == nullptr);

    std::vector<ListNode *> empty = {buildList({})};
    assert(toValues(mergeKLists(empty)).empty());

    std::vector<ListNode *> single = {buildList({1})};
    std::vector<int> wantSingle = {1};
    assert(toValues(mergeKLists(single)) == wantSingle);

    std::vector<ListNode *> mixed = {buildList({1, 2}), buildList({})};
    std::vector<int> wantMixed = {1, 2};
    assert(toValues(mergeKLists(mixed)) == wantMixed);

    std::vector<ListNode *> neg = {buildList({-2, 0}), buildList({-1, 3})};
    std::vector<int> wantNeg = {-2, -1, 0, 3};
    assert(toValues(mergeKLists(neg)) == wantNeg);
    std::cout << "merge_k_sorted_lists: all tests passed\n";
    return 0;
}
