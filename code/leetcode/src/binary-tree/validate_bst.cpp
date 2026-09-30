// 98. 验证二叉搜索树
// 见 validate_bst.py 的题目与思路说明。
#include <cassert>
#include <climits>
#include <iostream>

struct TreeNode {
    int val;
    TreeNode *left;
    TreeNode *right;
    TreeNode(int v = 0, TreeNode *l = nullptr, TreeNode *r = nullptr)
        : val(v), left(l), right(r) {}
};

bool checkBst(TreeNode *node, long long low, long long high) {
    if (node == nullptr) return true;
    if (!(low < node->val && node->val < high)) return false;
    return checkBst(node->left, low, node->val) &&
           checkBst(node->right, node->val, high);
}

bool isValidBst(TreeNode *root) {
    return checkBst(root, LLONG_MIN, LLONG_MAX);
}

int main() {
    assert(isValidBst(nullptr));

    TreeNode v2(2), v1(1), v3(3);
    v2.left = &v1;
    v2.right = &v3;
    assert(isValidBst(&v2));

    TreeNode a1(1), a3(3), a6(6), a4(4), a5(5);
    a5.left = &a1;
    a5.right = &a4;
    a4.left = &a3;
    a4.right = &a6;
    assert(!isValidBst(&a5));

    TreeNode b2(2), b2b(2), b2c(2);
    b2.left = &b2b;
    b2.right = &b2c;
    assert(!isValidBst(&b2));

    TreeNode lowNode(INT_MIN);
    assert(isValidBst(&lowNode));

    std::cout << "validate_bst: all tests passed\n";
    return 0;
}
