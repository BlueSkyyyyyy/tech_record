// 113. 路径总和 II
// 见 path_sum_ii.py 的题目与思路说明。
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

void dfs(TreeNode *node, int remaining, std::vector<int> &path,
         std::vector<std::vector<int>> &result) {
    if (node == nullptr) return;
    path.push_back(node->val);
    remaining -= node->val;
    if (node->left == nullptr && node->right == nullptr) {
        if (remaining == 0) result.push_back(path);
    } else {
        dfs(node->left, remaining, path, result);
        dfs(node->right, remaining, path, result);
    }
    path.pop_back();
}

std::vector<std::vector<int>> pathSum(TreeNode *root, int targetSum) {
    std::vector<std::vector<int>> result;
    std::vector<int> path;
    dfs(root, targetSum, path, result);
    return result;
}

bool same(const std::vector<std::vector<int>> &a,
          const std::vector<std::vector<int>> &b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i)
        if (a[i] != b[i]) return false;
    return true;
}

int main() {
    assert(pathSum(nullptr, 0).empty());

    TreeNode t2(2), t1(1, &t2, nullptr);
    assert(pathSum(&t1, 1).empty());

    TreeNode q2(2), q3(3), q1(1, &q2, &q3);
    std::vector<std::vector<int>> want_q;
    want_q.push_back({1, 2});
    assert(same(pathSum(&q1, 3), want_q));

    TreeNode n7(7), n2(2), n11(11, &n7, &n2);
    TreeNode n13(13), n5b(5), n4b(4, &n5b, nullptr);
    TreeNode n8(8, &n13, &n4b);
    TreeNode n4a(4, &n11, nullptr);
    TreeNode n5a(5, &n4a, &n8);
    std::vector<std::vector<int>> want_a;
    want_a.push_back({5, 4, 11, 2});
    want_a.push_back({5, 8, 4, 5});
    assert(same(pathSum(&n5a, 22), want_a));

    TreeNode x1(1), x3(3), xm2b(-2), xm3(-3);
    TreeNode xm2a(-2, &x1, &x3);
    TreeNode xm1(-1);
    x1.left = &xm1;
    TreeNode xroot(1, &xm2a, &xm3);
    std::vector<std::vector<int>> want_x;
    want_x.push_back({1, -2, 1, -1});
    assert(same(pathSum(&xroot, -1), want_x));

    std::cout << "path_sum_ii: all tests passed\n";
    return 0;
}
