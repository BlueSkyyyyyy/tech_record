// 478. 在圆内随机生成点
// 见 generate_random_point_in_a_circle.py 的题目与思路说明。
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <vector>

class Solution {
public:
    Solution(double radius, double x_center, double y_center)
        : radius_(radius), x_center_(x_center), y_center_(y_center) {}

    std::vector<double> randPoint() {
        while (true) {
            double x = uniform();
            double y = uniform();
            if (x * x + y * y <= 1.0) {
                return {x_center_ + x * radius_, y_center_ + y * radius_};
            }
        }
    }

private:
    double uniform() {
        return 2.0 * std::rand() / RAND_MAX - 1.0;
    }

    double radius_;
    double x_center_;
    double y_center_;
};

int main() {
    std::srand(12345);
    Solution s(1.0, 0.0, 0.0);
    for (int t = 0; t < 5000; ++t) {
        std::vector<double> p = s.randPoint();
        assert(p[0] * p[0] + p[1] * p[1] <= 1.0 + 1e-9);
    }
    Solution s2(2.0, 1.0, -1.0);
    for (int t = 0; t < 5000; ++t) {
        std::vector<double> p = s2.randPoint();
        double dx = p[0] - 1.0;
        double dy = p[1] + 1.0;
        assert(dx * dx + dy * dy <= 4.0 + 1e-9);
    }

    std::cout << "generate_random_point_in_a_circle: all tests passed\n";
    return 0;
}
