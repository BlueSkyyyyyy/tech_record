// 470. 用 Rand7() 实现 Rand10()
// 见 implement_rand10_using_rand7.py 的题目与思路说明。
#include <cassert>
#include <cstdlib>
#include <iostream>

int rand7() {
    return std::rand() % 7 + 1;
}

int rand10() {
    while (true) {
        int r = (rand7() - 1) * 7 + rand7();
        if (r <= 40) {
            return (r - 1) % 10 + 1;
        }
    }
}

int main() {
    std::srand(12345);
    int counts[11] = {0};
    const int N = 100000;
    for (int i = 0; i < N; ++i) {
        int x = rand10();
        assert(1 <= x && x <= 10);
        ++counts[x];
    }
    for (int x = 1; x <= 10; ++x) {
        assert(counts[x] > N / 20);
    }

    std::cout << "implement_rand10_using_rand7: all tests passed\n";
    return 0;
}
