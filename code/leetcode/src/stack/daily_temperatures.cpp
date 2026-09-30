// 739. 每日温度
// 见 daily_temperatures.py 的题目与思路说明。
#include <cassert>
#include <iostream>
#include <stack>
#include <vector>

std::vector<int> dailyTemperatures(const std::vector<int> &temperatures) {
    int n = temperatures.size();
    std::vector<int> answer(n, 0);
    std::stack<int> st;
    for (int i = 0; i < n; ++i) {
        while (!st.empty() && temperatures[st.top()] < temperatures[i]) {
            int j = st.top();
            st.pop();
            answer[j] = i - j;
        }
        st.push(i);
    }
    return answer;
}

int main() {
    std::vector<int> want1 = {1, 1, 4, 2, 1, 1, 0, 0};
    std::vector<int> want2 = {1, 1, 1, 0};
    std::vector<int> want3 = {1, 1, 0};
    std::vector<int> want4 = {0, 0, 0};
    assert(dailyTemperatures({73, 74, 75, 71, 69, 72, 76, 73}) == want1);
    assert(dailyTemperatures({30, 40, 50, 60}) == want2);
    assert(dailyTemperatures({30, 60, 90}) == want3);
    assert(dailyTemperatures({90, 80, 70}) == want4);
    std::cout << "daily_temperatures: all tests passed\n";
    return 0;
}
