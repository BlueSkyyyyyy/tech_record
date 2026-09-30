// 110. 平衡二叉树
// 见 balanced_binary_tree.py 的题目与思路说明。
#include <cassert>
#include <algorithm>
#include <cstdlib>
#include <iostream>

struct TreeNode {
    int val;
    TreeNode *left;
    TreeNode *right;
    TreeNode(int v = 0, TreeNode *l = nullptr, TreeNode *r = nullptr)
        : val(v), left(l), right(r) {}
};

int height(TreeNode *node) {
    if (node == nullptr) return 0;
    int left = height(node->left);
    if (left == -1) return -1;
    int right = height(node->right);
    if (right == -1) return -1;
    if (std::abs(left - right) > 1) return -1;
    return 1 + std::max(left, right);
}

bool isBalanced(TreeNode *root) {
    return height(root) != -1;
}

int main() {
    assert(isBalanced(nullptr));

    TreeNode single(1);
    assert(isBalanced(&single));

    TreeNode b15(15), b7(7), b20(20, &b15, &b7), b9(9), b3(3, &b9, &b20);
    assert(isBalanced(&b3));

    TreeNode d4(4), d3(3, &d4, nullptr), d2(2, &d3, nullptr), d1(1, &d2, nullptr);
    assert(!isBalanced(&d1));

    std::cout << "balanced_binary_tree: all tests passed\n";
    return 0;
}
