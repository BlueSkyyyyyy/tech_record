// 437. 路径总和 III
// 见 path_sum_iii.py 的题目与思路说明。
#include <cassert>
#include <iostream>
#include <unordered_map>

struct TreeNode {
    int val;
    TreeNode *left;
    TreeNode *right;
    TreeNode(int v = 0, TreeNode *l = nullptr, TreeNode *r = nullptr)
        : val(v), left(l), right(r) {}
};

long long dfs(TreeNode *node, long long cur, int targetSum,
              std::unordered_map<long long, int> &prefixCount) {
    if (node == nullptr) return 0;
    cur += node->val;
    long long total = prefixCount.count(cur - targetSum)
                          ? prefixCount[cur - targetSum]
                          : 0;
    prefixCount[cur] += 1;
    total += dfs(node->left, cur, targetSum, prefixCount);
    total += dfs(node->right, cur, targetSum, prefixCount);
    prefixCount[cur] -= 1;
    return total;
}

int pathSum(TreeNode *root, int targetSum) {
    std::unordered_map<long long, int> prefixCount;
    prefixCount[0] = 1;
    return static_cast<int>(dfs(root, 0, targetSum, prefixCount));
}

int main() {
    assert(pathSum(nullptr, 0) == 0);

    TreeNode s5(5);
    assert(pathSum(&s5, 5) == 1);
    assert(pathSum(&s5, 4) == 0);

    TreeNode t2(2), t3(3), t1(1, &t2, &t3);
    assert(pathSum(&t1, 3) == 2);

    TreeNode m3(-3), m2(-2), m1(1, &m2, &m3);
    assert(pathSum(&m1, -1) == 1);

    TreeNode a4(4), a5(5), a3(3);
    TreeNode a2(2, &a4, &a5);
    TreeNode a1(1, &a2, &a3);
    assert(pathSum(&a1, 3) == 2);

    TreeNode n3b(3), n2b(-2), n1b(1);
    TreeNode n3a(3, &n3b, &n2b);
    TreeNode n2a(2, nullptr, &n1b);
    TreeNode n11(11), nm3(-3, nullptr, &n11);
    TreeNode n5(5, &n3a, &n2a);
    TreeNode n10(10, &n5, &nm3);
    assert(pathSum(&n10, 8) == 3);

    std::cout << "path_sum_iii: all tests passed\n";
    return 0;
}
