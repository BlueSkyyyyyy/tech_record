// 101. 对称二叉树
// 见 symmetric_tree.py 的题目与思路说明。
#include <cassert>
#include <iostream>

struct TreeNode {
    int val;
    TreeNode *left;
    TreeNode *right;
    TreeNode(int v = 0, TreeNode *l = nullptr, TreeNode *r = nullptr)
        : val(v), left(l), right(r) {}
};

bool isMirror(TreeNode *a, TreeNode *b) {
    if (a == nullptr && b == nullptr) return true;
    if (a == nullptr || b == nullptr) return false;
    return a->val == b->val && isMirror(a->left, b->right) &&
           isMirror(a->right, b->left);
}

bool isSymmetric(TreeNode *root) {
    return root == nullptr || isMirror(root->left, root->right);
}

int main() {
    assert(isSymmetric(nullptr));

    TreeNode single(1);
    assert(isSymmetric(&single));

    TreeNode a2(2), b2(2), r1(1, &a2, &b2);
    assert(isSymmetric(&r1));

    TreeNode l4(4), l3(3), r4(4), r3(3);
    TreeNode l2(2, &l3, &l4), r2(2, &r4, &r3), root(1, &l2, &r2);
    assert(isSymmetric(&root));

    TreeNode rl3(3), rr3(3);
    TreeNode rl2(2, nullptr, &rl3), rr2(2, nullptr, &rr3), bad(1, &rl2, &rr2);
    assert(!isSymmetric(&bad));

    std::cout << "symmetric_tree: all tests passed\n";
    return 0;
}
