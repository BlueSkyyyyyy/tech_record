// 46. 全排列
// 见 permutations.py 的题目与思路说明。
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
        used[i] = true;
        path.push_back(nums[i]);
        backtrack(nums, path, used, res);
        path.pop_back();
        used[i] = false;
    }
}

std::vector<std::vector<int>> permute(const std::vector<int> &nums) {
    std::vector<std::vector<int>> res;
    std::vector<int> path;
    std::vector<bool> used(nums.size(), false);
    backtrack(nums, path, used, res);
    return res;
}

int main() {
    std::vector<int> nums = {1, 2, 3};
    auto out = permute(nums);
    assert(out.size() == 6);

    std::vector<int> first = {1, 2, 3};
    std::vector<int> last = {3, 2, 1};
    bool has_first = false, has_last = false;
    for (auto &v : out) {
        if (v == first) has_first = true;
        if (v == last) has_last = true;
    }
    assert(has_first && has_last);

    std::vector<int> one = {1};
    auto out1 = permute(one);
    std::vector<std::vector<int>> want1 = {{1}};
    assert(out1 == want1);

    std::cout << "permutations: all tests passed\n";
    return 0;
}
