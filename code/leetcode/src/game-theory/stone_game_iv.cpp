// 1510. 石子游戏 IV
// 见 stone_game_iv.py 的题目与思路说明。
#include <cassert>
#include <iostream>
#include <vector>

bool winnerSquareGame(int n) {
    std::vector<bool> win(n + 1, false);
    for (int i = 1; i <= n; ++i) {
        for (int s = 1; s * s <= i; ++s) {
            if (!win[i - s * s]) {
                win[i] = true;
                break;
            }
        }
    }
    return win[n];
}

int main() {
    assert(winnerSquareGame(1) == true);
    assert(winnerSquareGame(2) == false);
    assert(winnerSquareGame(4) == true);
    assert(winnerSquareGame(7) == false);
    assert(winnerSquareGame(17) == false);
    assert(winnerSquareGame(12) == false);
    assert(winnerSquareGame(13) == true);
    assert(winnerSquareGame(16) == true);

    std::cout << "stone_game_iv: all tests passed\n";
    return 0;
}
