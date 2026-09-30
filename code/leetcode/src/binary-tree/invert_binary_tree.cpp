// 226. 翻转二叉树
// 见 invert_binary_tree.py 的题目与思路说明。
#include <cassert>
#include <iostream>
#include <vector>

struct TreeNode {
    int val;
    TreeNode *left;
    TreeNode *right;
    TreeNode(int v = 0, TreeNode *l = nullptr, TreeNode *r = nullptr)
        : val(v), left(l), right(r) {}
};

TreeNode *invertTree(TreeNode *root) {
    if (root == nullptr) return nullptr;
    root->left = invertTree(root->left);
    root->right = invertTree(root->right);
    std::swap(root->left, root->right);
    return root;
}

void preorder(TreeNode *root, std::vector<int> &out) {
    if (root == nullptr) return;
    out.push_back(root->val);
    preorder(root->left, out);
    preorder(root->right, out);
}

int main() {
    assert(invertTree(nullptr) == nullptr);

    TreeNode single(1);
    std::vector<int> wantSingle = {1};
    std::vector<int> gotSingle;
    preorder(invertTree(&single), gotSingle);
    assert(gotSingle == wantSingle);

    TreeNode t1(1), t3(3), t6(6), t9(9);
    TreeNode n2(2, &t1, &t3), n7(7, &t6, &t9), n4(4, &n2, &n7);
    std::vector<int> want = {4, 7, 9, 6, 2, 3, 1};
    std::vector<int> got;
    preorder(invertTree(&n4), got);
    assert(got == want);

    std::cout << "invert_binary_tree: all tests passed\n";
    return 0;
}
