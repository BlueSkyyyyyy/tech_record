// 90. 子集 II
// 见 subsets_ii.py 的题目与思路说明。
#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>

void backtrack(const std::vector<int> &nums, int start,
               std::vector<int> &path, std::vector<std::vector<int>> &res) {
    res.push_back(path);
    for (int i = start; i < static_cast<int>(nums.size()); ++i) {
        if (i > start && nums[i] == nums[i - 1]) continue;
        path.push_back(nums[i]);
        backtrack(nums, i + 1, path, res);
        path.pop_back();
    }
}

std::vector<std::vector<int>> subsetsWithDup(std::vector<int> nums) {
    std::sort(nums.begin(), nums.end());
    std::vector<std::vector<int>> res;
    std::vector<int> path;
    backtrack(nums, 0, path, res);
    return res;
}

int main() {
    auto got = subsetsWithDup({1, 2, 2});
    std::vector<std::vector<int>> want = {{}, {1}, {1, 2}, {1, 2, 2}, {2}, {2, 2}};
    assert(got == want);

    auto got2 = subsetsWithDup({1, 1, 1});
    std::vector<std::vector<int>> want2 = {{}, {1}, {1, 1}, {1, 1, 1}};
    assert(got2 == want2);

    auto got3 = subsetsWithDup({0});
    std::vector<std::vector<int>> want3 = {{}, {0}};
    assert(got3 == want3);

    std::cout << "subsets_ii: all tests passed\n";
    return 0;
}
