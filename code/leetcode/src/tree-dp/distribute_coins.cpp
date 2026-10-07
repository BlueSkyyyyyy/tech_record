// 979. 在二叉树中分配硬币（树形 DP：后序 + 返回净流量）
// 见 distribute_coins.py 的题目与思路说明。
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

int moves = 0;

int dfs(TreeNode *node) {
    if (node == nullptr) return 0;
    int left = dfs(node->left);
    int right = dfs(node->right);
    moves += std::abs(left) + std::abs(right);
    return node->val + left + right - 1;
}

int distributeCoins(TreeNode *root) {
    moves = 0;
    dfs(root);
    return moves;
}

int main() {
    assert(distributeCoins(buildTree({3, 0, 0})) == 2);
    assert(distributeCoins(buildTree({0, 3, 0})) == 3);
    assert(distributeCoins(buildTree({1})) == 0);
    assert(distributeCoins(buildTree({1, 0, 2})) == 2);
    std::cout << "distribute_coins: all tests passed\n";
    return 0;
}
