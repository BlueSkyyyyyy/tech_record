// 124. 二叉树中的最大路径和
// 见 binary_tree_maximum_path_sum.py 的题目与思路说明。
#include <cassert>
#include <algorithm>
#include <climits>
#include <iostream>

struct TreeNode {
    int val;
    TreeNode *left;
    TreeNode *right;
    TreeNode(int v = 0, TreeNode *l = nullptr, TreeNode *r = nullptr)
        : val(v), left(l), right(r) {}
};

int gain(TreeNode *node, long long &best) {
    if (node == nullptr) return 0;
    int left = std::max(gain(node->left, best), 0);
    int right = std::max(gain(node->right, best), 0);
    long long through = static_cast<long long>(node->val) + left + right;
    if (through > best) best = through;
    return node->val + std::max(left, right);
}

int maxPathSum(TreeNode *root) {
    long long best = LLONG_MIN;
    gain(root, best);
    return static_cast<int>(best);
}

int main() {
    TreeNode a1(1), a2(2), a3(3);
    a1.left = &a2;
    a1.right = &a3;
    assert(maxPathSum(&a1) == 6);

    TreeNode b10(-10), b9(9), b20(20), b15(15), b7(7);
    b10.left = &b9;
    b10.right = &b20;
    b20.left = &b15;
    b20.right = &b7;
    assert(maxPathSum(&b10) == 42);

    TreeNode c3(-3);
    assert(maxPathSum(&c3) == -3);

    TreeNode d2(2), d1(-1);
    d2.left = &d1;
    assert(maxPathSum(&d2) == 2);

    TreeNode e1(1), e2(-2), e3(-3);
    e1.left = &e2;
    e1.right = &e3;
    assert(maxPathSum(&e1) == 1);

    TreeNode f2(-2), f1(-1);
    f2.left = &f1;
    assert(maxPathSum(&f2) == -1);

    std::cout << "binary_tree_maximum_path_sum: all tests passed\n";
    return 0;
}
