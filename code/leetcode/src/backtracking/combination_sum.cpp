// 39. 组合总和
// 见 combination_sum.py 的题目与思路说明。
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
        path.push_back(candidates[i]);
        backtrack(candidates, i, remain - candidates[i], path, res);
        path.pop_back();
    }
}

std::vector<std::vector<int>> combinationSum(std::vector<int> candidates,
                                             int target) {
    std::sort(candidates.begin(), candidates.end());
    std::vector<std::vector<int>> res;
    std::vector<int> path;
    backtrack(candidates, 0, target, path, res);
    return res;
}

int main() {
    auto got = combinationSum({2, 3, 6, 7}, 7);
    assert(got.size() == 2);
    std::vector<int> a = {2, 2, 3}, b = {7};
    for (auto &want : {a, b}) {
        assert(std::find(got.begin(), got.end(), want) != got.end());
    }

    auto got2 = combinationSum({2, 3, 5}, 8);
    std::vector<std::vector<int>> want2 = {{2, 2, 2, 2}, {2, 3, 3}, {3, 5}};
    assert(got2 == want2);

    auto got3 = combinationSum({2}, 1);
    assert(got3.empty());

    std::cout << "combination_sum: all tests passed\n";
    return 0;
}
