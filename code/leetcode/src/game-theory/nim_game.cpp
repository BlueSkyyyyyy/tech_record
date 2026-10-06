// 292. Nim 游戏
// 见 nim_game.py 的题目与思路说明。
#include <cassert>
#include <iostream>

bool canWinNim(int n) {
    return n % 4 != 0;
}

int main() {
    assert(canWinNim(1) == true);
    assert(canWinNim(2) == true);
    assert(canWinNim(3) == true);
    assert(canWinNim(4) == false);
    assert(canWinNim(5) == true);
    assert(canWinNim(8) == false);
    assert(canWinNim(100) == false);
    assert(canWinNim(101) == true);

    std::cout << "nim_game: all tests passed\n";
    return 0;
}
