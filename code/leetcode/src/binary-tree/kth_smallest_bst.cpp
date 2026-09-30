// 230. 二叉搜索树中第 K 小的元素
// 见 kth_smallest_bst.py 的题目与思路说明。
#include <cassert>
#include <iostream>

struct TreeNode {
    int val;
    TreeNode *left;
    TreeNode *right;
    TreeNode(int v = 0, TreeNode *l = nullptr, TreeNode *r = nullptr)
        : val(v), left(l), right(r) {}
};

void inorder(TreeNode *node, int k, int &count, int &answer) {
    if (node == nullptr || answer != -1) return;
    inorder(node->left, k, count, answer);
    count += 1;
    if (count == k) {
        answer = node->val;
        return;
    }
    inorder(node->right, k, count, answer);
}

int kthSmallest(TreeNode *root, int k) {
    int count = 0;
    int answer = -1;
    inorder(root, k, count, answer);
    return answer;
}

int main() {
    TreeNode n3(3), n1(1), n4(4), n2(2);
    n3.left = &n1;
    n3.right = &n4;
    n1.right = &n2;
    assert(kthSmallest(&n3, 1) == 1);
    assert(kthSmallest(&n3, 2) == 2);
    assert(kthSmallest(&n3, 3) == 3);
    assert(kthSmallest(&n3, 4) == 4);

    TreeNode m5(5), m3(3), m6(6), m2(2), m4(4), m1(1);
    m5.left = &m3;
    m5.right = &m6;
    m3.left = &m2;
    m3.right = &m4;
    m2.left = &m1;
    assert(kthSmallest(&m5, 3) == 3);
    assert(kthSmallest(&m5, 6) == 6);

    std::cout << "kth_smallest_bst: all tests passed\n";
    return 0;
}
