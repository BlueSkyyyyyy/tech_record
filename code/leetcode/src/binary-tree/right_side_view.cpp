// 199. 二叉树的右视图
// 见 right_side_view.py 的题目与思路说明。
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

std::vector<int> rightSideView(TreeNode *root) {
    std::vector<int> result;
    if (root == nullptr) return result;
    std::queue<TreeNode *> q;
    q.push(root);
    while (!q.empty()) {
        int size = static_cast<int>(q.size());
        for (int i = 0; i < size; ++i) {
            TreeNode *node = q.front();
            q.pop();
            if (i == size - 1) result.push_back(node->val);
            if (node->left != nullptr) q.push(node->left);
            if (node->right != nullptr) q.push(node->right);
        }
    }
    return result;
}

int main() {
    assert(rightSideView(nullptr).empty());

    TreeNode single(1);
    std::vector<int> wantSingle = {1};
    assert(rightSideView(&single) == wantSingle);

    TreeNode v5(5), v4(4), v2(2, nullptr, &v5), v3(3, nullptr, &v4);
    TreeNode v1(1, &v2, &v3);
    std::vector<int> want = {1, 3, 4};
    assert(rightSideView(&v1) == want);

    TreeNode w3(3), w2(2, nullptr, &w3), w1(1, nullptr, &w2);
    std::vector<int> wantChain = {1, 2, 3};
    assert(rightSideView(&w1) == wantChain);

    std::cout << "right_side_view: all tests passed\n";
    return 0;
}
