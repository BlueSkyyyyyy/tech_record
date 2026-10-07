// 968. 监控二叉树（树形 DP：后序 + 三状态）
// 见 binary_tree_cameras.py 的题目与思路说明。
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

int cameras = 0;

// 0 未覆盖 / 1 已覆盖 / 2 装摄像头
int dfs(TreeNode *node) {
    if (node == nullptr) return 1;
    int left = dfs(node->left);
    int right = dfs(node->right);
    if (left == 0 || right == 0) {
        ++cameras;
        return 2;
    }
    if (left == 2 || right == 2) return 1;
    return 0;
}

int minCameraCover(TreeNode *root) {
    cameras = 0;
    if (dfs(root) == 0) ++cameras;
    return cameras;
}

int main() {
    assert(minCameraCover(buildTree({0, 0, INT_MIN, 0, 0})) == 1);
    assert(minCameraCover(buildTree({0, 0, INT_MIN, 0, INT_MIN, 0, INT_MIN, INT_MIN, 0})) == 2);
    assert(minCameraCover(buildTree({0})) == 1);
    assert(minCameraCover(buildTree({0, 0, 0})) == 1);
    std::cout << "binary_tree_cameras: all tests passed\n";
    return 0;
}
