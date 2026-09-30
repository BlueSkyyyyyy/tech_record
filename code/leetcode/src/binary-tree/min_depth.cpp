// 111. 二叉树的最小深度
// 见 min_depth.py 的题目与思路说明。
#include <cassert>
#include <algorithm>
#include <iostream>

struct TreeNode {
    int val;
    TreeNode *left;
    TreeNode *right;
    TreeNode(int v = 0, TreeNode *l = nullptr, TreeNode *r = nullptr)
        : val(v), left(l), right(r) {}
};

int minDepth(TreeNode *root) {
    if (root == nullptr) return 0;
    if (root->left == nullptr) return 1 + minDepth(root->right);
    if (root->right == nullptr) return 1 + minDepth(root->left);
    return 1 + std::min(minDepth(root->left), minDepth(root->right));
}

int main() {
    assert(minDepth(nullptr) == 0);

    TreeNode single(1);
    assert(minDepth(&single) == 1);

    TreeNode m2(2), a1(1, nullptr, &m2);
    assert(minDepth(&a1) == 2);

    TreeNode m15(15), m7(7), m20(20, &m15, &m7), m9(9), m3(3, &m9, &m20);
    assert(minDepth(&m3) == 2);

    TreeNode c4(4), c3(3, &c4, nullptr), c2(2, &c3, nullptr), c1(1, &c2, nullptr);
    assert(minDepth(&c1) == 4);

    std::cout << "min_depth: all tests passed\n";
    return 0;
}
