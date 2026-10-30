// 1026. 节点与其祖先之间的最大差值（树形 DP：后序返回子树极值）
// 见 maximum_difference_between_node_and_ancestor.py 的题目与思路说明。
#include <algorithm>
#include <cassert>
#include <climits>
#include <cstdlib>
#include <iostream>
#include <queue>
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

int best = 0;

std::pair<int, int> dfs(TreeNode *node) {
    if (node == nullptr) return {INT_MAX, INT_MIN};
    auto [l_min, l_max] = dfs(node->left);
    auto [r_min, r_max] = dfs(node->right);
    int low = std::min({l_min, r_min, node->val});
    int high = std::max({l_max, r_max, node->val});
    best = std::max({best, std::abs(node->val - low), std::abs(node->val - high)});
    return {low, high};
}

int maxAncestorDiff(TreeNode *root) {
    best = 0;
    dfs(root);
    return best;
}

int main() {
    assert(maxAncestorDiff(buildTree({8, 3, 10, 1, 6, INT_MIN, 14, INT_MIN, INT_MIN, 4, 7, 13})) == 7);
    assert(maxAncestorDiff(buildTree({1, INT_MIN, 2, INT_MIN, 0, 3})) == 3);
    assert(maxAncestorDiff(buildTree({1})) == 0);
    assert(maxAncestorDiff(buildTree({4, 2, 6, 1, 3, 5, 7})) == 3);
    std::cout << "maximum_difference_between_node_and_ancestor: all tests passed\n";
    return 0;
}
