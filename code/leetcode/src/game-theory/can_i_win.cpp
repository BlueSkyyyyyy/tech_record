// 464. 我能赢吗
// 见 can_i_win.py 的题目与思路说明。
#include <cassert>
#include <functional>
#include <iostream>
#include <unordered_map>

bool canIWin(int maxChoosableInteger, int desiredTotal) {
    if (desiredTotal <= 0) {
        return true;
    }
    long long total = 1LL * maxChoosableInteger * (maxChoosableInteger + 1) / 2;
    if (total < desiredTotal) {
        return false;
    }
    std::unordered_map<int, bool> memo;
    std::function<bool(int, int)> win = [&](int mask, int sum) -> bool {
        auto it = memo.find(mask);
        if (it != memo.end()) {
            return it->second;
        }
        for (int i = 1; i <= maxChoosableInteger; ++i) {
            int bit = 1 << (i - 1);
            if (mask & bit) {
                continue;
            }
            if (sum + i >= desiredTotal || !win(mask | bit, sum + i)) {
                memo[mask] = true;
                return true;
            }
        }
        memo[mask] = false;
        return false;
    };
    return win(0, 0);
}

int main() {
    assert(canIWin(10, 0) == true);
    assert(canIWin(10, 1) == true);
    assert(canIWin(10, 11) == false);
    assert(canIWin(1, 1) == true);
    assert(canIWin(1, 2) == false);
    assert(canIWin(2, 2) == true);
    assert(canIWin(2, 3) == false);
    assert(canIWin(3, 5) == true);

    std::cout << "can_i_win: all tests passed\n";
    return 0;
}
