// 2376. 统计特殊整数
// 见 count_special_integers.py 的题目与思路说明。
#include <cassert>
#include <cstring>
#include <iostream>
#include <string>

std::string s;
int length;
long long memo[12][1 << 10][2][2];

long long dfs(int pos, int mask, bool tight, bool started) {
    if (pos == length) {
        return 1;
    }
    long long &res = memo[pos][mask][tight][started];
    if (res != -1) {
        return res;
    }
    int limit = tight ? s[pos] - '0' : 9;
    long long total = 0;
    for (int d = 0; d <= limit; ++d) {
        bool ntight = tight && (d == limit);
        if (!started && d == 0) {
            total += dfs(pos + 1, mask, ntight, false);
        } else if (mask & (1 << d)) {
            continue;
        } else {
            total += dfs(pos + 1, mask | (1 << d), ntight, true);
        }
    }
    return res = total;
}

int countSpecialIntegers(int n) {
    s = std::to_string(n);
    length = static_cast<int>(s.size());
    std::memset(memo, -1, sizeof(memo));
    return static_cast<int>(dfs(0, 0, true, false) - 1);
}

int main() {
    assert(countSpecialIntegers(5) == 5);
    assert(countSpecialIntegers(20) == 19);
    assert(countSpecialIntegers(135) == 110);
    std::cout << "count_special_integers: all tests passed\n";
    return 0;
}
