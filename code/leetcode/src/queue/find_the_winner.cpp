// 1823. 找出游戏的获胜者
// 见 find_the_winner.py 的题目与思路说明。
#include <cassert>
#include <iostream>
#include <queue>

int findTheWinner(int n, int k) {
    std::queue<int> q;
    for (int i = 1; i <= n; ++i) {
        q.push(i);
    }
    while (q.size() > 1) {
        for (int i = 0; i < k - 1; ++i) {
            q.push(q.front());
            q.pop();
        }
        q.pop();
    }
    return q.front();
}

int main() {
    assert(findTheWinner(5, 2) == 3);
    assert(findTheWinner(6, 5) == 1);
    assert(findTheWinner(1, 1) == 1);
    assert(findTheWinner(2, 2) == 1);
    assert(findTheWinner(5, 1) == 5);
    std::cout << "find_the_winner: all tests passed\n";
    return 0;
}
