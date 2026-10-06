// 112. 路径总和
// 见 path_sum.py 的题目与思路说明。
#include <cassert>
#include <iostream>

struct TreeNode {
    int val;
    TreeNode *left;
    TreeNode *right;
    TreeNode(int v = 0, TreeNode *l = nullptr, TreeNode *r = nullptr)
        : val(v), left(l), right(r) {}
};

bool hasPathSum(TreeNode *root, int targetSum) {
    if (root == nullptr) return false;
    int remaining = targetSum - root->val;
    if (root->left == nullptr && root->right == nullptr)
        return remaining == 0;
    return hasPathSum(root->left, remaining) ||
           hasPathSum(root->right, remaining);
}

int main() {
    assert(hasPathSum(nullptr, 0) == false);

    TreeNode s5(5);
    assert(hasPathSum(&s5, 5) == true);
    assert(hasPathSum(&s5, 4) == false);

    TreeNode n7(7), n2(2), n1_(1);
    TreeNode n11(11, &n7, &n2);
    TreeNode n13(13), n4b(4, nullptr, &n1_);
    TreeNode n4a(4, &n11, nullptr);
    TreeNode n8(8, &n13, &n4b);
    TreeNode n5(5, &n4a, &n8);
    assert(hasPathSum(&n5, 22) == true);

    TreeNode t2(2), t3(3), t1(1, &t2, &t3);
    assert(hasPathSum(&t1, 5) == false);
    assert(hasPathSum(&t1, 4) == true);

    TreeNode m3(-3), m2(-2, nullptr, &m3);
    assert(hasPathSum(&m2, -5) == true);

    std::cout << "path_sum: all tests passed\n";
    return 0;
}
