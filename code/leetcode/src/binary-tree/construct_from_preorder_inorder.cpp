// 105. 从前序与中序遍历序列构造二叉树
// 见 construct_from_preorder_inorder.py 的题目与思路说明。
#include <cassert>
#include <iostream>
#include <unordered_map>
#include <vector>

struct TreeNode {
    int val;
    TreeNode *left;
    TreeNode *right;
    TreeNode(int v = 0, TreeNode *l = nullptr, TreeNode *r = nullptr)
        : val(v), left(l), right(r) {}
};

TreeNode *buildPreIn(std::vector<int> &preorder, int pre_lo, int pre_hi,
                     int in_lo, int in_hi,
                     std::unordered_map<int, int> &index) {
    if (pre_lo > pre_hi) return nullptr;
    int root_val = preorder[pre_lo];
    TreeNode *root = new TreeNode(root_val);
    int mid = index[root_val];
    int left_size = mid - in_lo;
    root->left = buildPreIn(preorder, pre_lo + 1, pre_lo + left_size, in_lo,
                            mid - 1, index);
    root->right = buildPreIn(preorder, pre_lo + left_size + 1, pre_hi, mid + 1,
                             in_hi, index);
    return root;
}

TreeNode *buildTreePreIn(std::vector<int> preorder, std::vector<int> inorder) {
    std::unordered_map<int, int> index;
    for (int i = 0; i < static_cast<int>(inorder.size()); ++i)
        index[inorder[i]] = i;
    return buildPreIn(preorder, 0, static_cast<int>(preorder.size()) - 1, 0,
                      static_cast<int>(inorder.size()) - 1, index);
}

void preorderOf(TreeNode *node, std::vector<int> &out) {
    if (node == nullptr) return;
    out.push_back(node->val);
    preorderOf(node->left, out);
    preorderOf(node->right, out);
}

void inorderOf(TreeNode *node, std::vector<int> &out) {
    if (node == nullptr) return;
    inorderOf(node->left, out);
    out.push_back(node->val);
    inorderOf(node->right, out);
}

int main() {
    std::vector<int> pre = {3, 9, 20, 15, 7};
    std::vector<int> ino = {9, 3, 15, 20, 7};
    TreeNode *root = buildTreePreIn(pre, ino);
    std::vector<int> gotPre, gotIn;
    preorderOf(root, gotPre);
    inorderOf(root, gotIn);
    assert(gotPre == pre);
    assert(gotIn == ino);

    TreeNode *single = buildTreePreIn({1}, {1});
    std::vector<int> gotSingle;
    preorderOf(single, gotSingle);
    std::vector<int> wantSingle = {1};
    assert(gotSingle == wantSingle);

    assert(buildTreePreIn({}, {}) == nullptr);

    std::cout << "construct_from_preorder_inorder: all tests passed\n";
    return 0;
}
