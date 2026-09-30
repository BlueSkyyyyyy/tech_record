// 20. 有效的括号
// 见 valid_parentheses.py 的题目与思路说明。
#include <cassert>
#include <iostream>
#include <stack>
#include <string>

bool isValid(const std::string &s) {
    std::stack<char> st;
    for (char ch : s) {
        if (ch == '(' || ch == '[' || ch == '{') {
            st.push(ch);
        } else {
            if (st.empty()) return false;
            char top = st.top();
            st.pop();
            if ((ch == ')' && top != '(') ||
                (ch == ']' && top != '[') ||
                (ch == '}' && top != '{')) {
                return false;
            }
        }
    }
    return st.empty();
}

int main() {
    assert(isValid("()") == true);
    assert(isValid("()[]{}") == true);
    assert(isValid("(]") == false);
    assert(isValid("([)]") == false);
    assert(isValid("{[]}") == true);
    assert(isValid("(") == false);
    assert(isValid(")") == false);
    assert(isValid("") == true);
    std::cout << "valid_parentheses: all tests passed\n";
    return 0;
}
