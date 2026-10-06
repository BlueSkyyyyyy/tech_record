// 40. 组合总和 II
// 见 combination_sum_ii.py 的题目与思路说明。
#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>

void backtrack(const std::vector<int> &candidates, int start, int remain,
               std::vector<int> &path, std::vector<std::vector<int>> &res) {
    if (remain == 0) {
        res.push_back(path);
        return;
    }
    for (int i = start; i < static_cast<int>(candidates.size()); ++i) {
        if (candidates[i] > remain) break;
        if (i > start && candidates[i] == candidates[i - 1]) continue;
        path.push_back(candidates[i]);
        backtrack(candidates, i + 1, remain - candidates[i], path, res);
        path.pop_back();
    }
}

std::vector<std::vector<int>> combinationSum2(std::vector<int> candidates,
                                              int target) {
    std::sort(candidates.begin(), candidates.end());
    std::vector<std::vector<int>> res;
    std::vector<int> path;
    backtrack(candidates, 0, target, path, res);
    return res;
}

int main() {
    auto got = combinationSum2({10, 1, 2, 7, 6, 1, 5}, 8);
    std::vector<std::vector<int>> want = {{1, 1, 6}, {1, 2, 5}, {1, 7}, {2, 6}};
    assert(got == want);

    auto got2 = combinationSum2({2, 5, 2, 1, 2}, 5);
    std::vector<std::vector<int>> want2 = {{1, 2, 2}, {5}};
    assert(got2 == want2);

    auto got3 = combinationSum2({1}, 2);
    assert(got3.empty());

    std::cout << "combination_sum_ii: all tests passed\n";
    return 0;
}
