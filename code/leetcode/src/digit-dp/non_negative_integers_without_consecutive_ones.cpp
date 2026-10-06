// 600. 不含连续 1 的非负整数
// 见 non_negative_integers_without_consecutive_ones.py 的题目与思路说明。
#include <cassert>
#include <cstring>
#include <iostream>
#include <string>

std::string bits;
int length;
long long memo[64][2][2];

long long dfs(int pos, bool prevOne, bool tight) {
    if (pos == length) {
        return 1;
    }
    long long &res = memo[pos][prevOne][tight];
    if (res != -1) {
        return res;
    }
    int limit = tight ? bits[pos] - '0' : 1;
    long long total = 0;
    for (int b = 0; b <= limit; ++b) {
        if (prevOne && b == 1) {
            continue;
        }
        total += dfs(pos + 1, b == 1, tight && b == limit);
    }
    return res = total;
}

int findIntegers(int n) {
    bits.clear();
    int x = n;
    if (x == 0) {
        bits = "0";
    }
    while (x > 0) {
        bits = static_cast<char>('0' + x % 2) + bits;
        x /= 2;
    }
    length = static_cast<int>(bits.size());
    std::memset(memo, -1, sizeof(memo));
    return static_cast<int>(dfs(0, false, true));
}

int main() {
    assert(findIntegers(0) == 1);
    assert(findIntegers(1) == 2);
    assert(findIntegers(2) == 3);
    assert(findIntegers(5) == 5);
    assert(findIntegers(10) == 8);
    std::cout << "non_negative_integers_without_consecutive_ones: all tests passed\n";
    return 0;
}
