// 150. 逆波兰表达式求值
// 见 eval_rpn.py 的题目与思路说明。
#include <cassert>
#include <iostream>
#include <stack>
#include <string>
#include <vector>

int evalRPN(const std::vector<std::string> &tokens) {
    std::stack<int> st;
    for (const std::string &tok : tokens) {
        if (tok == "+" || tok == "-" || tok == "*" || tok == "/") {
            int b = st.top();
            st.pop();
            int a = st.top();
            st.pop();
            if (tok == "+") st.push(a + b);
            else if (tok == "-") st.push(a - b);
            else if (tok == "*") st.push(a * b);
            else st.push(a / b);  // C++ 整数除法天然向零截断
        } else {
            st.push(std::stoi(tok));
        }
    }
    return st.top();
}

int main() {
    std::vector<std::string> t1 = {"2", "1", "+", "3", "*"};
    std::vector<std::string> t2 = {"4", "13", "5", "/", "+"};
    std::vector<std::string> t3 = {"10", "6", "9", "3", "+", "-11", "*", "/", "*", "17", "+", "5", "+"};
    std::vector<std::string> t4 = {"-7", "2", "/"};
    std::vector<std::string> t5 = {"3"};
    assert(evalRPN(t1) == 9);
    assert(evalRPN(t2) == 6);
    assert(evalRPN(t3) == 22);
    assert(evalRPN(t4) == -3);
    assert(evalRPN(t5) == 3);
    std::cout << "eval_rpn: all tests passed\n";
    return 0;
}
