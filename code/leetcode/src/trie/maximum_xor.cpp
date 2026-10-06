// 421. 数组中两个数的最大异或值
// 见 maximum_xor.py 的题目与思路说明。
#include <array>
#include <cassert>
#include <iostream>
#include <vector>

struct TrieNode {
    std::array<TrieNode *, 2> child;
    TrieNode() { child.fill(nullptr); }
};

class Solution {
  public:
    int findMaximumXOR(const std::vector<int> &nums) {
        TrieNode *root = new TrieNode();
        for (int num : nums) {
            TrieNode *node = root;
            for (int i = BITS; i >= 0; --i) {
                int bit = (num >> i) & 1;
                if (!node->child[bit]) node->child[bit] = new TrieNode();
                node = node->child[bit];
            }
        }

        int best = 0;
        for (int num : nums) {
            TrieNode *node = root;
            int current = 0;
            for (int i = BITS; i >= 0; --i) {
                int bit = (num >> i) & 1;
                int want = 1 - bit;
                if (node->child[want]) {
                    current |= 1 << i;
                    node = node->child[want];
                } else {
                    node = node->child[bit];
                }
            }
            if (current > best) best = current;
        }
        return best;
    }

  private:
    static constexpr int BITS = 31;
};

int main() {
    Solution solver;
    std::vector<int> a = {3, 10, 5, 25, 2, 8};
    assert(solver.findMaximumXOR(a) == 28);

    std::vector<int> b = {14, 70, 53, 83, 49, 91, 36, 80, 92, 51, 66, 70};
    assert(solver.findMaximumXOR(b) == 127);

    std::vector<int> c = {0};
    assert(solver.findMaximumXOR(c) == 0);

    std::vector<int> d = {2, 4};
    assert(solver.findMaximumXOR(d) == 6);

    std::vector<int> e = {8, 10, 2};
    assert(solver.findMaximumXOR(e) == 10);
    std::cout << "maximum_xor: all tests passed\n";
    return 0;
}
