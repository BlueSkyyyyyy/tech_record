// 1373. 二叉搜索子树的最大键值和（树形 DP：后序 + 四元组）
// 见 max_sum_bst.py 的题目与思路说明。
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

struct Info {
    bool is_bst;
    long long mn;
    long long mx;
    long long sum;
};

long long best = 0;

Info dfs(TreeNode *node) {
    if (node == nullptr) return {true, LLONG_MAX, LLONG_MIN, 0};
    Info l = dfs(node->left);
    Info r = dfs(node->right);
    if (l.is_bst && r.is_bst && l.mx < node->val && node->val < r.mn) {
        long long total = l.sum + r.sum + node->val;
        best = std::max(best, total);
        return {true, std::min(l.mn, (long long)node->val),
                std::max(r.mx, (long long)node->val), total};
    }
    return {false, 0, 0, 0};
}

int maxSumBST(TreeNode *root) {
    best = 0;
    dfs(root);
    return static_cast<int>(best);
}

int main() {
    assert(maxSumBST(buildTree({1, 4, 3, 2, 4, 2, 5, INT_MIN, INT_MIN,
                                INT_MIN, INT_MIN, INT_MIN, INT_MIN, 4, 6})) == 20);
    assert(maxSumBST(buildTree({4, 3, INT_MIN, 1, 2})) == 2);
    assert(maxSumBST(buildTree({-4, -2, -5})) == 0);
    assert(maxSumBST(buildTree({2, 1, 3})) == 6);
    std::cout << "max_sum_bst: all tests passed\n";
    return 0;
}
