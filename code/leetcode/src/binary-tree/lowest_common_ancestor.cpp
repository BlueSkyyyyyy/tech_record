// 236. 二叉树的最近公共祖先
// 见 lowest_common_ancestor.py 的题目与思路说明。
#include <cassert>
#include <iostream>

struct TreeNode {
    int val;
    TreeNode *left;
    TreeNode *right;
    TreeNode(int v = 0, TreeNode *l = nullptr, TreeNode *r = nullptr)
        : val(v), left(l), right(r) {}
};

TreeNode *lowestCommonAncestor(TreeNode *root, TreeNode *p, TreeNode *q) {
    if (root == nullptr || root == p || root == q) return root;
    TreeNode *left = lowestCommonAncestor(root->left, p, q);
    TreeNode *right = lowestCommonAncestor(root->right, p, q);
    if (left != nullptr && right != nullptr) return root;
    return left != nullptr ? left : right;
}

int main() {
    TreeNode n3(3), n5(5), n1(1), n6(6), n2(2), n0(0), n8(8), n7(7), n4(4);
    n3.left = &n5;
    n3.right = &n1;
    n5.left = &n6;
    n5.right = &n2;
    n1.left = &n0;
    n1.right = &n8;
    n2.left = &n7;
    n2.right = &n4;

    assert(lowestCommonAncestor(&n3, &n5, &n1) == &n3);
    assert(lowestCommonAncestor(&n3, &n5, &n4) == &n5);
    assert(lowestCommonAncestor(&n3, &n6, &n4) == &n5);
    assert(lowestCommonAncestor(&n3, &n7, &n4) == &n2);
    assert(lowestCommonAncestor(&n3, &n7, &n7) == &n7);
    assert(lowestCommonAncestor(&n3, &n0, &n8) == &n1);

    std::cout << "lowest_common_ancestor: all tests passed\n";
    return 0;
}
