// 796. 旋转字符串
// 见 rotate_string.py 的题目与思路说明。
#include <cassert>
#include <iostream>
#include <string>

bool rotateString(const std::string &s, const std::string &goal) {
    return s.size() == goal.size() && (s + s).find(goal) != std::string::npos;
}

int main() {
    assert(rotateString("abcde", "cdeab") == true);
    assert(rotateString("abcde", "abced") == false);
    assert(rotateString("a", "a") == true);
    assert(rotateString("", "") == true);
    assert(rotateString("abc", "abcd") == false);

    std::cout << "rotateString: all tests passed\n";
    return 0;
}
