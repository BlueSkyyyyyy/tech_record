// 145. 二叉树的后序遍历
// 见 postorder_traversal.py 的题目与思路说明。
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
    dfs(node->left, result);
    dfs(node->right, result);
    result.push_back(node->val);
}

std::vector<int> postorderTraversal(TreeNode *root) {
    std::vector<int> result;
    dfs(root, result);
    return result;
}

int main() {
    assert(postorderTraversal(nullptr).empty());

    TreeNode single(1);
    std::vector<int> wantSingle = {1};
    assert(postorderTraversal(&single) == wantSingle);

    TreeNode a3(3), a2(2, &a3, nullptr), a1(1, nullptr, &a2);
    std::vector<int> wantRight = {3, 2, 1};
    assert(postorderTraversal(&a1) == wantRight);

    TreeNode n1(1), n7(7), n9(9);
    TreeNode n3(3, &n1, nullptr), n8(8, &n7, &n9);
    TreeNode n5(5, &n3, &n8);
    std::vector<int> want = {1, 3, 7, 9, 8, 5};
    assert(postorderTraversal(&n5) == want);

    std::cout << "postorder_traversal: all tests passed\n";
    return 0;
}
