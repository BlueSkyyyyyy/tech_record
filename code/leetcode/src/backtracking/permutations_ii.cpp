// 47. 全排列 II
// 见 permutations_ii.py 的题目与思路说明。
#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>

void backtrack(const std::vector<int> &nums, std::vector<int> &path,
               std::vector<bool> &used, std::vector<std::vector<int>> &res) {
    if (path.size() == nums.size()) {
        res.push_back(path);
        return;
    }
    for (int i = 0; i < static_cast<int>(nums.size()); ++i) {
        if (used[i]) continue;
        if (i > 0 && nums[i] == nums[i - 1] && !used[i - 1]) continue;
        used[i] = true;
        path.push_back(nums[i]);
        backtrack(nums, path, used, res);
        path.pop_back();
        used[i] = false;
    }
}

std::vector<std::vector<int>> permuteUnique(std::vector<int> nums) {
    std::sort(nums.begin(), nums.end());
    std::vector<std::vector<int>> res;
    std::vector<int> path;
    std::vector<bool> used(nums.size(), false);
    backtrack(nums, path, used, res);
    return res;
}

int main() {
    auto got = permuteUnique({1, 1, 2});
    assert(got.size() == 3);
    std::vector<int> a = {1, 1, 2}, b = {1, 2, 1}, c = {2, 1, 1};
    for (auto &want : {a, b, c}) {
        assert(std::find(got.begin(), got.end(), want) != got.end());
    }

    auto got2 = permuteUnique({1, 1, 1});
    std::vector<std::vector<int>> want2 = {{1, 1, 1}};
    assert(got2 == want2);

    auto got3 = permuteUnique({1});
    std::vector<std::vector<int>> want3 = {{1}};
    assert(got3 == want3);

    std::cout << "permutations_ii: all tests passed\n";
    return 0;
}
