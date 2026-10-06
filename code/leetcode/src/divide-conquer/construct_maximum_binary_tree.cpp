// 654. 最大二叉树
// 见 construct_maximum_binary_tree.py 的题目与思路说明。
#include <cassert>
#include <iostream>
#include <vector>

struct TreeNode {
    int val;
    TreeNode *left;
    TreeNode *right;
    TreeNode(int x = 0, TreeNode *l = nullptr, TreeNode *r = nullptr)
        : val(x), left(l), right(r) {}
};

TreeNode *build(const std::vector<int> &nums, int lo, int hi) {
    if (lo >= hi) return nullptr;
    int maxIndex = lo;
    for (int i = lo + 1; i < hi; ++i) {
        if (nums[i] > nums[maxIndex]) maxIndex = i;
    }
    TreeNode *node = new TreeNode(nums[maxIndex]);
    node->left = build(nums, lo, maxIndex);
    node->right = build(nums, maxIndex + 1, hi);
    return node;
}

TreeNode *constructMaximumBinaryTree(const std::vector<int> &nums) {
    return build(nums, 0, static_cast<int>(nums.size()));
}

void preorder(TreeNode *node, std::vector<int> &out) {
    if (node == nullptr) return;
    out.push_back(node->val);
    preorder(node->left, out);
    preorder(node->right, out);
}

int main() {
    std::vector<int> out;
    preorder(constructMaximumBinaryTree({3, 2, 1, 6, 0, 5}), out);
    std::vector<int> want = {6, 3, 2, 1, 5, 0};
    assert(out == want);

    out.clear();
    preorder(constructMaximumBinaryTree({3, 2, 1}), out);
    std::vector<int> want2 = {3, 2, 1};
    assert(out == want2);

    out.clear();
    preorder(constructMaximumBinaryTree({1}), out);
    std::vector<int> want3 = {1};
    assert(out == want3);

    assert(constructMaximumBinaryTree({}) == nullptr);
    std::cout << "construct_maximum_binary_tree: all tests passed\n";
    return 0;
}
