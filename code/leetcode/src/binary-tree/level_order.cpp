// 102. 二叉树的层序遍历
// 见 level_order.py 的题目与思路说明。
#include <cassert>
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

std::vector<std::vector<int>> levelOrder(TreeNode *root) {
    std::vector<std::vector<int>> result;
    if (root == nullptr) return result;
    std::queue<TreeNode *> q;
    q.push(root);
    while (!q.empty()) {
        int size = static_cast<int>(q.size());
        std::vector<int> level;
        for (int i = 0; i < size; ++i) {
            TreeNode *node = q.front();
            q.pop();
            level.push_back(node->val);
            if (node->left != nullptr) q.push(node->left);
            if (node->right != nullptr) q.push(node->right);
        }
        result.push_back(level);
    }
    return result;
}

int main() {
    assert(levelOrder(nullptr).empty());

    TreeNode single(1);
    std::vector<std::vector<int>> wantSingle = {{1}};
    assert(levelOrder(&single) == wantSingle);

    TreeNode n15(15), n7(7), n20(20, &n15, &n7), n9(9), n3(3, &n9, &n20);
    std::vector<std::vector<int>> want = {{3}, {9, 20}, {15, 7}};
    assert(levelOrder(&n3) == want);

    std::cout << "level_order: all tests passed\n";
    return 0;
}
