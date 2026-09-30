// 142. 环形链表 II
// 见 linked_list_cycle_ii.py 的题目与思路说明。
#include <cassert>
#include <iostream>
#include <vector>

struct ListNode {
    int val;
    ListNode *next;
    ListNode(int x = 0, ListNode *n = nullptr) : val(x), next(n) {}
};

ListNode *detectCycle(ListNode *head) {
    ListNode *slow = head;
    ListNode *fast = head;
    while (fast && fast->next) {
        slow = slow->next;
        fast = fast->next->next;
        if (slow == fast) {
            slow = head;
            while (slow != fast) {
                slow = slow->next;
                fast = fast->next;
            }
            return slow;
        }
    }
    return nullptr;
}

ListNode *buildWithCycle(const std::vector<int> &values, int pos) {
    if (values.empty()) return nullptr;
    std::vector<ListNode *> nodes;
    for (int v : values) nodes.push_back(new ListNode(v));
    for (size_t i = 0; i + 1 < nodes.size(); ++i) nodes[i]->next = nodes[i + 1];
    if (pos != -1) nodes.back()->next = nodes[pos];
    return nodes[0];
}

int main() {
    ListNode *head = buildWithCycle({3, 2, 0, -4}, 1);
    assert(detectCycle(head) == head->next);

    ListNode *loop = buildWithCycle({1, 2}, 0);
    assert(detectCycle(loop) == loop);

    assert(detectCycle(buildWithCycle({1}, -1)) == nullptr);
    assert(detectCycle(buildWithCycle({}, -1)) == nullptr);
    std::cout << "linked_list_cycle_ii: all tests passed\n";
    return 0;
}
