// 114. 二叉树展开为链表
// 见 flatten_binary_tree.py 的题目与思路说明。
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

TreeNode *flattenDfs(TreeNode *node) {
    if (node == nullptr) return nullptr;
    TreeNode *left_tail = flattenDfs(node->left);
    TreeNode *right_tail = flattenDfs(node->right);
    if (node->left != nullptr) {
        left_tail->right = node->right;
        node->right = node->left;
        node->left = nullptr;
    }
    if (right_tail != nullptr) return right_tail;
    if (left_tail != nullptr) return left_tail;
    return node;
}

void flatten(TreeNode *root) {
    flattenDfs(root);
}

std::vector<int> rightChain(TreeNode *root) {
    std::vector<int> result;
    while (root != nullptr) {
        assert(root->left == nullptr);
        result.push_back(root->val);
        root = root->right;
    }
    return result;
}

int main() {
    TreeNode n1(1), n2(2), n5(5), n3(3), n4(4), n6(6);
    n1.left = &n2;
    n1.right = &n5;
    n2.left = &n3;
    n2.right = &n4;
    n5.right = &n6;
    flatten(&n1);
    std::vector<int> want1 = {1, 2, 3, 4, 5, 6};
    assert(rightChain(&n1) == want1);

    TreeNode single(1);
    flatten(&single);
    std::vector<int> want2 = {1};
    assert(rightChain(&single) == want2);

    TreeNode p1(1), p2(2);
    p1.left = &p2;
    flatten(&p1);
    std::vector<int> want3 = {1, 2};
    assert(rightChain(&p1) == want3);

    TreeNode q1(1), q2(2);
    q1.right = &q2;
    flatten(&q1);
    std::vector<int> want4 = {1, 2};
    assert(rightChain(&q1) == want4);

    std::cout << "flatten_binary_tree: all tests passed\n";
    return 0;
}
