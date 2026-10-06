// 382. 链表随机节点
// 见 linked_list_random_node.py 的题目与思路说明。
#include <cassert>
#include <cstdlib>
#include <iostream>

struct ListNode {
    int val;
    ListNode* next;
    ListNode(int v, ListNode* n = nullptr) : val(v), next(n) {}
};

class Solution {
public:
    Solution(ListNode* head) : head_(head) {}

    int getRandom() {
        int res = head_->val;
        ListNode* node = head_->next;
        int i = 2;
        while (node) {
            if (std::rand() % i == 0) {
                res = node->val;
            }
            node = node->next;
            ++i;
        }
        return res;
    }

private:
    ListNode* head_;
};

int main() {
    std::srand(12345);
    ListNode* head = new ListNode(1, new ListNode(2, new ListNode(3, new ListNode(4))));
    Solution s(head);
    bool seen[5] = {false, false, false, false, false};
    for (int t = 0; t < 4000; ++t) {
        int x = s.getRandom();
        assert(1 <= x && x <= 4);
        seen[x] = true;
    }
    assert(seen[1] && seen[2] && seen[3] && seen[4]);

    std::cout << "linked_list_random_node: all tests passed\n";
    return 0;
}
