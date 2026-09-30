// 84. 柱状图中最大的矩形
// 见 largest_rectangle_in_histogram.py 的题目与思路说明。
#include <algorithm>
#include <cassert>
#include <iostream>
#include <stack>
#include <vector>

int largestRectangleArea(std::vector<int> heights) {
    heights.push_back(0);  // 末尾哨兵，保证收尾时全部结算
    std::stack<int> st;
    int best = 0;
    for (int i = 0; i < (int)heights.size(); ++i) {
        while (!st.empty() && heights[st.top()] > heights[i]) {
            int bar = st.top();
            st.pop();
            int height = heights[bar];
            int left = st.empty() ? -1 : st.top();
            int width = i - left - 1;
            best = std::max(best, height * width);
        }
        st.push(i);
    }
    return best;
}

int main() {
    assert(largestRectangleArea({2, 1, 5, 6, 2, 3}) == 10);
    assert(largestRectangleArea({2, 4}) == 4);
    assert(largestRectangleArea({2, 1, 2}) == 3);
    assert(largestRectangleArea({1}) == 1);
    assert(largestRectangleArea({1, 1, 1, 1}) == 4);
    std::cout << "largest_rectangle_in_histogram: all tests passed\n";
    return 0;
}
