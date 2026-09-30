// 394. 字符串解码
// 见 decode_string.py 的题目与思路说明。
#include <cassert>
#include <cctype>
#include <iostream>
#include <stack>
#include <string>
#include <utility>

std::string decodeString(const std::string &s) {
    std::stack<std::pair<std::string, int>> st;
    std::string cur;
    int num = 0;
    for (char ch : s) {
        if (std::isdigit((unsigned char)ch)) {
            num = num * 10 + (ch - '0');
        } else if (ch == '[') {
            st.push({cur, num});
            cur.clear();
            num = 0;
        } else if (ch == ']') {
            auto [prev, repeat] = st.top();
            st.pop();
            std::string expanded;
            for (int i = 0; i < repeat; ++i) expanded += cur;
            cur = prev + expanded;
        } else {
            cur += ch;
        }
    }
    return cur;
}

int main() {
    assert(decodeString("3[a]2[bc]") == "aaabcbc");
    assert(decodeString("3[a2[c]]") == "accaccacc");
    assert(decodeString("2[abc]3[cd]ef") == "abcabccdcdcdef");
    assert(decodeString("abc") == "abc");
    std::cout << "decode_string: all tests passed\n";
    return 0;
}
