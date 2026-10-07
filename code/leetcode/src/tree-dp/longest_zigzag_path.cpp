// 1372. 二叉树中的最长交错路径（树形 DP：后序 + 两个方向）
// 见 longest_zigzag_path.py 的题目与思路说明。
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

int best = 0;

// 返回 (go_left, go_right)：从 node 出发第一步走左/右的最长交错边数
std::pair<int, int> dfs(TreeNode *node) {
    if (node == nullptr) return {0, 0};
    auto [left_go_left, left_go_right] = dfs(node->left);
    auto [right_go_left, right_go_right] = dfs(node->right);
    int go_left = node->left ? 1 + left_go_right : 0;
    int go_right = node->right ? 1 + right_go_left : 0;
    best = std::max(best, std::max(go_left, go_right));
    return {go_left, go_right};
}

int longestZigZag(TreeNode *root) {
    best = 0;
    dfs(root);
    return best;
}

int main() {
    std::vector<int> big = {1, INT_MIN, 1, 1, 1, INT_MIN, INT_MIN, 1, 1, INT_MIN, 1,
                            INT_MIN, INT_MIN, INT_MIN, 1, INT_MIN, 1};
    assert(longestZigZag(buildTree(big)) == 3);
    assert(longestZigZag(buildTree({1, 1, 1, INT_MIN, 1, INT_MIN, INT_MIN, 1, 1, INT_MIN, 1})) == 4);
    assert(longestZigZag(buildTree({1})) == 0);
    assert(longestZigZag(buildTree({1, 2})) == 1);
    std::cout << "longest_zigzag_path: all tests passed\n";
    return 0;
}
