// 2246. 相邻字符不同的最长路径（树形 DP：孩子数组 + 两条最长的臂）
// 见 longest_path_with_different_adjacent_characters.py 的题目与思路说明。
#include <algorithm>
#include <cassert>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

int longestPath(const std::vector<int> &parent, const std::string &s) {
    int n = static_cast<int>(parent.size());
    std::vector<std::vector<int>> children(n);
    for (int node = 1; node < n; ++node) children[parent[node]].push_back(node);

    int best = 0;
    std::function<int(int)> dfs = [&](int node) {
        int top1 = 0, top2 = 0;
        for (int child : children[node]) {
            int arm = dfs(child);
            if (s[child] == s[node]) continue;
            if (arm > top1) {
                top2 = top1;
                top1 = arm;
            } else if (arm > top2) {
                top2 = arm;
            }
        }
        best = std::max(best, top1 + top2 + 1);
        return top1 + 1;
    };
    dfs(0);
    return best;
}

int main() {
    assert(longestPath({-1, 0, 0, 1, 1, 2}, "abacbe") == 3);
    assert(longestPath({-1, 0, 0, 0}, "aabc") == 3);
    assert(longestPath({-1}, "z") == 1);
    assert(longestPath({-1, 0, 1, 2}, "abcd") == 4);
    std::cout << "longest_path_with_different_adjacent_characters: all tests passed\n";
    return 0;
}
