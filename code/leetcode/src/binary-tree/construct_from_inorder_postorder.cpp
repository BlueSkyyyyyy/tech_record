// 106. 从中序与后序遍历序列构造二叉树
// 见 construct_from_inorder_postorder.py 的题目与思路说明。
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

TreeNode *buildInPost(int in_lo, int in_hi, int post_lo, int post_hi,
                      std::vector<int> &postorder,
                      std::unordered_map<int, int> &index) {
    if (in_lo > in_hi) return nullptr;
    int root_val = postorder[post_hi];
    TreeNode *root = new TreeNode(root_val);
    int mid = index[root_val];
    int left_size = mid - in_lo;
    root->left = buildInPost(in_lo, mid - 1, post_lo, post_lo + left_size - 1,
                             postorder, index);
    root->right = buildInPost(mid + 1, in_hi, post_lo + left_size, post_hi - 1,
                              postorder, index);
    return root;
}

TreeNode *buildTreeInPost(std::vector<int> inorder, std::vector<int> postorder) {
    std::unordered_map<int, int> index;
    for (int i = 0; i < static_cast<int>(inorder.size()); ++i)
        index[inorder[i]] = i;
    return buildInPost(0, static_cast<int>(inorder.size()) - 1, 0,
                       static_cast<int>(postorder.size()) - 1, postorder, index);
}

void inorderOf(TreeNode *node, std::vector<int> &out) {
    if (node == nullptr) return;
    inorderOf(node->left, out);
    out.push_back(node->val);
    inorderOf(node->right, out);
}

void postorderOf(TreeNode *node, std::vector<int> &out) {
    if (node == nullptr) return;
    postorderOf(node->left, out);
    postorderOf(node->right, out);
    out.push_back(node->val);
}

int main() {
    std::vector<int> ino = {9, 3, 15, 20, 7};
    std::vector<int> post = {9, 15, 7, 20, 3};
    TreeNode *root = buildTreeInPost(ino, post);
    std::vector<int> gotIn, gotPost;
    inorderOf(root, gotIn);
    postorderOf(root, gotPost);
    assert(gotIn == ino);
    assert(gotPost == post);

    TreeNode *single = buildTreeInPost({1}, {1});
    std::vector<int> gotSingle;
    inorderOf(single, gotSingle);
    std::vector<int> wantSingle = {1};
    assert(gotSingle == wantSingle);

    assert(buildTreeInPost({}, {}) == nullptr);

    std::cout << "construct_from_inorder_postorder: all tests passed\n";
    return 0;
}
