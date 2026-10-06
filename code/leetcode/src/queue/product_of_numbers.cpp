// 1352. 最后 K 个数的乘积
// 见 product_of_numbers.py 的题目与思路说明。
#include <cassert>
#include <iostream>
#include <vector>

class ProductOfNumbers {
public:
    ProductOfNumbers() { prefix_.push_back(1); }

    void add(int num) {
        if (num == 0) {
            prefix_.clear();
            prefix_.push_back(1);
        } else {
            prefix_.push_back(prefix_.back() * static_cast<long long>(num));
        }
    }

    int getProduct(int k) {
        if (k >= static_cast<int>(prefix_.size())) {
            return 0;
        }
        return static_cast<int>(prefix_.back() / prefix_[prefix_.size() - 1 - k]);
    }

private:
    std::vector<long long> prefix_;
};

int main() {
    ProductOfNumbers p;
    p.add(3);
    p.add(0);
    p.add(2);
    p.add(5);
    p.add(4);
    assert(p.getProduct(2) == 20);
    assert(p.getProduct(3) == 40);
    assert(p.getProduct(4) == 0);
    assert(p.getProduct(5) == 0);
    p.add(8);
    assert(p.getProduct(2) == 32);
    assert(p.getProduct(6) == 0);

    ProductOfNumbers p2;
    p2.add(0);
    assert(p2.getProduct(1) == 0);
    p2.add(2);
    assert(p2.getProduct(1) == 2);
    assert(p2.getProduct(2) == 0);
    std::cout << "product_of_numbers: all tests passed\n";
    return 0;
}
