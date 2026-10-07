// 337. 打家劫舍 III（树形 DP：后序 + 返回「选/不选」）
// 见 house_robber_iii.py 的题目与思路说明。
#include <algorithm>
#include <cassert>
#include <climits>
#include <iostream>
#include <queue>
#include <utility>
#include <vector>

struct TreeNode {
    int val;
    TreeNode *left;
    TreeNode *right;
    TreeNode(int v = 0, TreeNode *l = nullptr, TreeNode *r = nullptr)
        : val(v), left(l), right(r) {}
};

TreeNode *buildTree(const std::vector<int> &a) {
    if (a.empty() || a[0] == INT_MIN) return nullptr;
    TreeNode *root = new TreeNode(a[0]);
    std::queue<TreeNode *> q;
    q.push(root);
    size_t i = 1;
    while (!q.empty() && i < a.size()) {
        TreeNode *n = q.front();
        q.pop();
        if (i < a.size() && a[i] != INT_MIN) { n->left = new TreeNode(a[i]); q.push(n->left); }
        ++i;
        if (i < a.size() && a[i] != INT_MIN) { n->right = new TreeNode(a[i]); q.push(n->right); }
        ++i;
    }
    return root;
}

// 返回 (rob, not_rob)
std::pair<int, int> dfs(TreeNode *node) {
    if (node == nullptr) return {0, 0};
    auto [l_rob, l_not] = dfs(node->left);
    auto [r_rob, r_not] = dfs(node->right);
    int rob_here = node->val + l_not + r_not;
    int not_here = std::max(l_rob, l_not) + std::max(r_rob, r_not);
    return {rob_here, not_here};
}

int rob(TreeNode *root) {
    auto [a, b] = dfs(root);
    return std::max(a, b);
}

int main() {
    assert(rob(buildTree({3, 2, 3, INT_MIN, 3, INT_MIN, 1})) == 7);
    assert(rob(buildTree({3, 4, 5, 1, 3, INT_MIN, 1})) == 9);
    assert(rob(buildTree({})) == 0);
    assert(rob(buildTree({5})) == 5);
    std::cout << "house_robber_iii: all tests passed\n";
    return 0;
}
