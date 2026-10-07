// 687. 最长同值路径（树形 DP：后序 + 返回单臂长度）
// 见 longest_univalue_path.py 的题目与思路说明。
#include <algorithm>
#include <cassert>
#include <climits>
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

int dfs(TreeNode *node) {
    if (node == nullptr) return 0;
    int left = dfs(node->left);
    int right = dfs(node->right);
    int left_arm = (node->left && node->left->val == node->val) ? left + 1 : 0;
    int right_arm = (node->right && node->right->val == node->val) ? right + 1 : 0;
    best = std::max(best, left_arm + right_arm);
    return std::max(left_arm, right_arm);
}

int longestUnivaluePath(TreeNode *root) {
    best = 0;
    dfs(root);
    return best;
}

int main() {
    assert(longestUnivaluePath(buildTree({5, 4, 5, 1, 1, INT_MIN, 5})) == 2);
    assert(longestUnivaluePath(buildTree({1, 4, 5, 4, 4, INT_MIN, 5})) == 2);
    assert(longestUnivaluePath(buildTree({1})) == 0);
    assert(longestUnivaluePath(buildTree({1, 1, 1, 1, 1, INT_MIN, 1})) == 4);
    std::cout << "longest_univalue_path: all tests passed\n";
    return 0;
}
