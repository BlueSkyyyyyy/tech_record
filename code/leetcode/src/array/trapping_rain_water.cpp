// 42. 接雨水
// 见 trapping_rain_water.py 的题目与思路说明。
#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>

int trap(const std::vector<int> &height) {
    if (height.empty()) return 0;
    int lo = 0, hi = static_cast<int>(height.size()) - 1;
    int leftMax = height[lo], rightMax = height[hi];
    int water = 0;
    while (lo < hi) {
        if (height[lo] < height[hi]) {
            ++lo;
            leftMax = std::max(leftMax, height[lo]);
            water += leftMax - height[lo];
        } else {
            --hi;
            rightMax = std::max(rightMax, height[hi]);
            water += rightMax - height[hi];
        }
    }
    return water;
}

int trapStack(const std::vector<int> &height) {
    std::vector<int> st;
    int water = 0;
    for (int i = 0; i < static_cast<int>(height.size()); ++i) {
        while (!st.empty() && height[st.back()] < height[i]) {
            int bottom = st.back();
            st.pop_back();
            if (st.empty()) break;
            int width = i - st.back() - 1;
            int bounded = std::min(height[st.back()], height[i]) - height[bottom];
            water += width * bounded;
        }
        st.push_back(i);
    }
    return water;
}

int main() {
    std::vector<int> h{0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1};
    assert(trap(h) == 6);
    assert(trapStack(h) == 6);

    std::vector<int> h2{4, 2, 0, 3, 2, 5};
    assert(trap(h2) == 9);
    assert(trapStack(h2) == 9);

    std::vector<int> h3;
    assert(trap(h3) == 0);
    assert(trapStack(h3) == 0);

    std::vector<int> h4{1, 2, 3, 4, 5};
    assert(trap(h4) == 0);
    assert(trapStack(h4) == 0);

    std::cout << "trapping_rain_water: all tests passed\n";
    return 0;
}
