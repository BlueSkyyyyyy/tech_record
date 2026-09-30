// 141. 环形链表
// 见 linked_list_cycle.py 的题目与思路说明。
#include <cassert>
#include <iostream>
#include <vector>

struct ListNode {
    int val;
    ListNode *next;
    ListNode(int x = 0, ListNode *n = nullptr) : val(x), next(n) {}
};

bool hasCycle(ListNode *head) {
    ListNode *slow = head;
    ListNode *fast = head;
    while (fast && fast->next) {
        slow = slow->next;
        fast = fast->next->next;
        if (slow == fast) return true;
    }
    return false;
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
    assert(hasCycle(buildWithCycle({3, 2, 0, -4}, 1)) == true);
    assert(hasCycle(buildWithCycle({1, 2}, 0)) == true);
    assert(hasCycle(buildWithCycle({1}, -1)) == false);
    assert(hasCycle(buildWithCycle({1, 2, 3, 4}, -1)) == false);
    assert(hasCycle(buildWithCycle({}, -1)) == false);
    std::cout << "linked_list_cycle: all tests passed\n";
    return 0;
}
