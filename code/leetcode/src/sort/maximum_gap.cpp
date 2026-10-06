// 164. 最大间距
// 见 maximum_gap.py 的题目与思路说明。
#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>

int maximumGap(const std::vector<int> &nums) {
    int n = static_cast<int>(nums.size());
    if (n < 2) {
        return 0;
    }
    long long lo = *std::min_element(nums.begin(), nums.end());
    long long hi = *std::max_element(nums.begin(), nums.end());
    if (lo == hi) {
        return 0;
    }
    long long size = std::max(1LL, (hi - lo) / (n - 1));
    long long count = (hi - lo) / size + 1;
    const long long NONE = -1;
    std::vector<long long> bmin(count, NONE);
    std::vector<long long> bmax(count, NONE);
    for (int x : nums) {
        long long idx = (x - lo) / size;
        if (bmin[idx] == NONE || x < bmin[idx]) {
            bmin[idx] = x;
        }
        if (bmax[idx] == NONE || x > bmax[idx]) {
            bmax[idx] = x;
        }
    }
    int best = 0;
    long long prev = lo;
    for (long long i = 0; i < count; ++i) {
        if (bmin[i] == NONE) {
            continue;
        }
        best = std::max(best, static_cast<int>(bmin[i] - prev));
        prev = bmax[i];
    }
    return best;
}

int main() {
    assert(maximumGap({3, 6, 9, 1}) == 3);
    assert(maximumGap({10}) == 0);
    assert(maximumGap({}) == 0);
    assert(maximumGap({1, 10000000}) == 9999999);
    assert(maximumGap({1, 1, 1, 1}) == 0);
    assert(maximumGap({1, 2, 3, 4, 5}) == 1);

    std::cout << "maximum_gap: all tests passed\n";
    return 0;
}
