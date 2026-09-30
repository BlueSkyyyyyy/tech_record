// 108. 将有序数组转换为二叉搜索树
// 见 sorted_array_to_bst.py 的题目与思路说明。
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <vector>

struct TreeNode {
    int val;
    TreeNode *left;
    TreeNode *right;
    TreeNode(int v = 0, TreeNode *l = nullptr, TreeNode *r = nullptr)
        : val(v), left(l), right(r) {}
};

TreeNode *build(std::vector<int> &nums, int lo, int hi) {
    if (lo > hi) return nullptr;
    int mid = lo + (hi - lo) / 2;
    TreeNode *node = new TreeNode(nums[mid]);
    node->left = build(nums, lo, mid - 1);
    node->right = build(nums, mid + 1, hi);
    return node;
}

TreeNode *sortedArrayToBst(std::vector<int> nums) {
    return build(nums, 0, static_cast<int>(nums.size()) - 1);
}

void inorder(TreeNode *node, std::vector<int> &out) {
    if (node == nullptr) return;
    inorder(node->left, out);
    out.push_back(node->val);
    inorder(node->right, out);
}

int height(TreeNode *node) {
    if (node == nullptr) return 0;
    return 1 + std::max(height(node->left), height(node->right));
}

void check(std::vector<int> nums) {
    TreeNode *tree = sortedArrayToBst(nums);
    std::vector<int> out;
    inorder(tree, out);
    assert(out == nums);
    int left_h = tree == nullptr ? 0 : height(tree->left);
    int right_h = tree == nullptr ? 0 : height(tree->right);
    assert(std::abs(left_h - right_h) <= 1);
}

int main() {
    check({});
    check({-10});
    check({1, 3});
    check({1, 2, 3, 4, 5});
    check({-10, -3, 0, 5, 9});
    std::cout << "sorted_array_to_bst: all tests passed\n";
    return 0;
}
