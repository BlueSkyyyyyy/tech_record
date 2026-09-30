// 144. 二叉树的前序遍历
// 见 preorder_traversal.py 的题目与思路说明。
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

void dfs(TreeNode *node, std::vector<int> &result) {
    if (node == nullptr) return;
    result.push_back(node->val);
    dfs(node->left, result);
    dfs(node->right, result);
}

std::vector<int> preorderTraversal(TreeNode *root) {
    std::vector<int> result;
    dfs(root, result);
    return result;
}

int main() {
    assert(preorderTraversal(nullptr).empty());

    TreeNode single(1);
    std::vector<int> wantSingle = {1};
    assert(preorderTraversal(&single) == wantSingle);

    TreeNode a3(3), a2(2, &a3, nullptr), a1(1, nullptr, &a2);
    std::vector<int> wantRight = {1, 2, 3};
    assert(preorderTraversal(&a1) == wantRight);

    TreeNode n4(4), n5(5), n3(3);
    TreeNode n2(2, &n4, &n5);
    TreeNode n1(1, &n2, &n3);
    std::vector<int> want = {1, 2, 4, 5, 3};
    assert(preorderTraversal(&n1) == want);

    std::cout << "preorder_traversal: all tests passed\n";
    return 0;
}
