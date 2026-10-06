// 204. 计数质数
// 见 count_primes.py 的题目与思路说明。
#include <cassert>
#include <iostream>
#include <vector>

int countPrimes(int n) {
    if (n < 3) {
        return 0;
    }
    std::vector<bool> isPrime(n, true);
    isPrime[0] = isPrime[1] = false;
    int i = 2;
    while (i * i < n) {
        if (isPrime[i]) {
            for (int j = i * i; j < n; j += i) {
                isPrime[j] = false;
            }
        }
        ++i;
    }
    int count = 0;
    for (bool v : isPrime) {
        if (v) {
            ++count;
        }
    }
    return count;
}

int main() {
    assert(countPrimes(0) == 0);
    assert(countPrimes(1) == 0);
    assert(countPrimes(2) == 0);
    assert(countPrimes(3) == 1);
    assert(countPrimes(10) == 4);
    assert(countPrimes(11) == 4);
    assert(countPrimes(30) == 10);
    assert(countPrimes(100) == 25);

    std::cout << "count_primes: all tests passed\n";
    return 0;
}
