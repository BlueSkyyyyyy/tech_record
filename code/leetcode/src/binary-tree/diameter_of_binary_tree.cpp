// 543. 二叉树的直径
// 见 diameter_of_binary_tree.py 的题目与思路说明。
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

int depth(TreeNode *node, int &best) {
    if (node == nullptr) return 0;
    int left = depth(node->left, best);
    int right = depth(node->right, best);
    best = std::max(best, left + right);
    return 1 + std::max(left, right);
}

int diameterOfBinaryTree(TreeNode *root) {
    int best = 0;
    depth(root, best);
    return best;
}

int main() {
    assert(diameterOfBinaryTree(nullptr) == 0);

    TreeNode single(1);
    assert(diameterOfBinaryTree(&single) == 0);

    TreeNode a1(1), a2(2), a3(3), a4(4), a5(5);
    a1.left = &a2;
    a1.right = &a3;
    a2.left = &a4;
    a2.right = &a5;
    assert(diameterOfBinaryTree(&a1) == 3);

    TreeNode b1(1), b2(2);
    b1.left = &b2;
    assert(diameterOfBinaryTree(&b1) == 1);

    TreeNode c1(1), c2(2), c3(3), c4(4), c5(5);
    c1.left = &c2;
    c2.left = &c3;
    c3.left = &c4;
    c4.left = &c5;
    assert(diameterOfBinaryTree(&c1) == 4);

    std::cout << "diameter_of_binary_tree: all tests passed\n";
    return 0;
}
