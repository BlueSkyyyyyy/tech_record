// 338. 比特位计数
// 见 counting_bits.py 的题目与思路说明。
#include <cassert>
#include <iostream>
#include <vector>

std::vector<int> countBits(int n) {
    std::vector<int> ans(n + 1, 0);
    for (int i = 1; i <= n; ++i) {
        ans[i] = ans[i >> 1] + (i & 1);
    }
    return ans;
}

int main() {
    std::vector<int> want0 = {0};
    assert(countBits(0) == want0);

    std::vector<int> want1 = {0, 1};
    assert(countBits(1) == want1);

    std::vector<int> want2 = {0, 1, 1};
    assert(countBits(2) == want2);

    std::vector<int> want5 = {0, 1, 1, 2, 1, 2};
    assert(countBits(5) == want5);

    std::vector<int> want8 = {0, 1, 1, 2, 1, 2, 2, 3, 1};
    assert(countBits(8) == want8);

    long long total = 0;
    for (int v : countBits(16)) total += v;
    assert(total == 33);

    std::cout << "counting_bits: all tests passed\n";
    return 0;
}
