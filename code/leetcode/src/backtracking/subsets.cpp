// 78. 子集
// 见 subsets.py 的题目与思路说明。
#include <cassert>
#include <iostream>
#include <vector>

void backtrack(const std::vector<int> &nums, int start, std::vector<int> &path,
               std::vector<std::vector<int>> &res) {
    res.push_back(path);
    for (int i = start; i < static_cast<int>(nums.size()); ++i) {
        path.push_back(nums[i]);
        backtrack(nums, i + 1, path, res);
        path.pop_back();
    }
}

std::vector<std::vector<int>> subsets(const std::vector<int> &nums) {
    std::vector<std::vector<int>> res;
    std::vector<int> path;
    backtrack(nums, 0, path, res);
    return res;
}

int main() {
    std::vector<int> nums = {1, 2, 3};
    auto out = subsets(nums);
    assert(out.size() == 8);

    std::vector<int> want_full = {1, 2, 3};
    std::vector<int> want_part = {1, 3};
    bool has_full = false, has_part = false, has_empty = false;
    for (auto &v : out) {
        if (v.empty()) has_empty = true;
        if (v == want_full) has_full = true;
        if (v == want_part) has_part = true;
    }
    assert(has_empty && has_full && has_part);

    std::vector<int> one = {0};
    auto out1 = subsets(one);
    assert(out1.size() == 2);

    std::cout << "subsets: all tests passed\n";
    return 0;
}
