// 56. 合并区间
// 见 merge_intervals.py 的题目与思路说明。
#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>

std::vector<std::vector<int>> mergeIntervals(std::vector<std::vector<int>> intervals) {
    std::sort(intervals.begin(), intervals.end(),
              [](const std::vector<int> &a, const std::vector<int> &b) { return a[0] < b[0]; });
    std::vector<std::vector<int>> res;
    for (const auto &iv : intervals) {
        if (!res.empty() && iv[0] <= res.back()[1]) {
            res.back()[1] = std::max(res.back()[1], iv[1]);
        } else {
            res.push_back(iv);
        }
    }
    return res;
}

int main() {
    {
        std::vector<std::vector<int>> in{{1, 3}, {2, 6}, {8, 10}, {15, 18}};
        std::vector<std::vector<int>> want{{1, 6}, {8, 10}, {15, 18}};
        assert(mergeIntervals(in) == want);
    }
    {
        std::vector<std::vector<int>> in{{1, 4}, {4, 5}};
        std::vector<std::vector<int>> want{{1, 5}};
        assert(mergeIntervals(in) == want);
    }
    {
        std::vector<std::vector<int>> in{{1, 4}, {0, 4}};
        std::vector<std::vector<int>> want{{0, 4}};
        assert(mergeIntervals(in) == want);
    }
    {
        std::vector<std::vector<int>> in{{1, 4}, {2, 3}};
        std::vector<std::vector<int>> want{{1, 4}};
        assert(mergeIntervals(in) == want);
    }
    std::vector<std::vector<int>> empty;
    assert(mergeIntervals(empty).empty());
    {
        std::vector<std::vector<int>> in{{1, 2}};
        std::vector<std::vector<int>> want{{1, 2}};
        assert(mergeIntervals(in) == want);
    }
    std::cout << "merge_intervals: all tests passed\n";
    return 0;
}
