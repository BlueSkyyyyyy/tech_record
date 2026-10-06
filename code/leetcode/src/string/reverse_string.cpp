// 344. 反转字符串
// 见 reverse_string.py 的题目与思路说明。
#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>

void reverseString(std::vector<char> &s) {
    int left = 0;
    int right = static_cast<int>(s.size()) - 1;
    while (left < right) {
        std::swap(s[left], s[right]);
        ++left;
        --right;
    }
}

int main() {
    std::vector<char> a = {'h', 'e', 'l', 'l', 'o'};
    reverseString(a);
    std::vector<char> wantA = {'o', 'l', 'l', 'e', 'h'};
    assert(a == wantA);

    std::vector<char> b = {'H', 'a', 'n', 'n', 'a', 'h'};
    reverseString(b);
    std::vector<char> wantB = {'h', 'a', 'n', 'n', 'a', 'H'};
    assert(b == wantB);

    std::vector<char> c = {'a'};
    reverseString(c);
    std::vector<char> wantC = {'a'};
    assert(c == wantC);

    std::vector<char> d;
    reverseString(d);
    assert(d.empty());

    std::vector<char> e = {'a', 'b'};
    reverseString(e);
    std::vector<char> wantE = {'b', 'a'};
    assert(e == wantE);

    std::cout << "reverse_string: all tests passed\n";
    return 0;
}
