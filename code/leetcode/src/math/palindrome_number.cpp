// 9. 回文数
// 见 palindrome_number.py 的题目与思路说明。
#include <cassert>
#include <iostream>

bool palindromeNumber(int x) {
    if (x < 0 || (x % 10 == 0 && x != 0)) {
        return false;
    }
    int rev = 0;
    while (x > rev) {
        rev = rev * 10 + x % 10;
        x /= 10;
    }
    return x == rev || x == rev / 10;
}

int main() {
    assert(palindromeNumber(121) == true);
    assert(palindromeNumber(-121) == false);
    assert(palindromeNumber(10) == false);
    assert(palindromeNumber(0) == true);
    assert(palindromeNumber(7) == true);
    assert(palindromeNumber(12321) == true);
    assert(palindromeNumber(123321) == true);
    assert(palindromeNumber(12345) == false);
    assert(palindromeNumber(1000021) == false);

    std::cout << "palindrome_number: all tests passed\n";
    return 0;
}
