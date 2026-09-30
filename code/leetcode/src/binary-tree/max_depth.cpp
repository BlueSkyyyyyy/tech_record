// 104. 二叉树的最大深度
// 见 max_depth.py 的题目与思路说明。
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

int maxDepth(TreeNode *root) {
    if (root == nullptr) return 0;
    return 1 + std::max(maxDepth(root->left), maxDepth(root->right));
}

int main() {
    assert(maxDepth(nullptr) == 0);

    TreeNode single(1);
    assert(maxDepth(&single) == 1);

    TreeNode a2(2), a1(1, nullptr, &a2);
    assert(maxDepth(&a1) == 2);

    TreeNode n15(15), n7(7), n9(9), n20(20, &n15, &n7);
    TreeNode n3(3, &n9, &n20);
    assert(maxDepth(&n3) == 3);

    TreeNode c4(4), c3(3, &c4, nullptr), c2(2, &c3, nullptr), c1(1, &c2, nullptr);
    assert(maxDepth(&c1) == 4);

    std::cout << "max_depth: all tests passed\n";
    return 0;
}
