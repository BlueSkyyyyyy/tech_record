// 228. 汇总区间
// 见 summary_ranges.py 的题目与思路说明。
#include <cassert>
#include <iostream>
#include <string>
#include <vector>

std::vector<std::string> summaryRanges(std::vector<int>& nums) {
    std::vector<std::string> res;
    int n = static_cast<int>(nums.size());
    int i = 0;
    while (i < n) {
        int start = nums[i];
        while (i + 1 < n && nums[i + 1] == nums[i] + 1) {
            ++i;
        }
        if (start == nums[i]) {
            res.push_back(std::to_string(start));
        } else {
            res.push_back(std::to_string(start) + "->" + std::to_string(nums[i]));
        }
        ++i;
    }
    return res;
}

int main() {
    std::vector<int> a{0, 1, 2, 4, 5, 7};
    std::vector<int> b{0, 2, 3, 4, 6, 8, 9};
    std::vector<int> c{};
    std::vector<int> d{-1};

    std::vector<std::string> wa{"0->2", "4->5", "7"};
    std::vector<std::string> wb{"0", "2->4", "6", "8->9"};
    std::vector<std::string> wc{};
    std::vector<std::string> wd{"-1"};

    assert(summaryRanges(a) == wa);
    assert(summaryRanges(b) == wb);
    assert(summaryRanges(c) == wc);
    assert(summaryRanges(d) == wd);

    std::cout << "summary_ranges: all tests passed\n";
    return 0;
}
