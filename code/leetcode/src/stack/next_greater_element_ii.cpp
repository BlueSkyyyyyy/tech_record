// 503. 下一个更大元素 II
// 见 next_greater_element_ii.py 的题目与思路说明。
#include <cassert>
#include <iostream>
#include <stack>
#include <vector>

std::vector<int> nextGreaterElements(const std::vector<int> &nums) {
    int n = nums.size();
    std::vector<int> result(n, -1);
    std::stack<int> st;
    for (int i = 0; i < 2 * n; ++i) {
        int x = nums[i % n];
        while (!st.empty() && nums[st.top()] < x) {
            result[st.top()] = x;
            st.pop();
        }
        if (i < n) st.push(i);
    }
    return result;
}

int main() {
    std::vector<int> want1 = {2, -1, 2};
    std::vector<int> want2 = {2, 3, 4, -1, 4};
    std::vector<int> want3 = {-1, 5, 5, 5, 5};
    assert(nextGreaterElements({1, 2, 1}) == want1);
    assert(nextGreaterElements({1, 2, 3, 4, 3}) == want2);
    assert(nextGreaterElements({5, 4, 3, 2, 1}) == want3);
    std::cout << "next_greater_element_ii: all tests passed\n";
    return 0;
}
