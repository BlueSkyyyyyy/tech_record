// 950. 按递增顺序显示卡牌
// 见 reveal_cards_in_increasing_order.py 的题目与思路说明。
#include <algorithm>
#include <cassert>
#include <deque>
#include <iostream>
#include <vector>

std::vector<int> deckRevealedIncreasing(std::vector<int>& deck) {
    std::sort(deck.begin(), deck.end());
    std::deque<int> d;
    for (int i = static_cast<int>(deck.size()) - 1; i >= 0; --i) {
        if (!d.empty()) {
            d.push_front(d.back());
            d.pop_back();
        }
        d.push_front(deck[i]);
    }
    return std::vector<int>(d.begin(), d.end());
}

int main() {
    std::vector<int> a{17, 13, 11, 2, 3, 5, 7};
    std::vector<int> b{1, 1000};
    std::vector<int> c{1, 2, 3};
    std::vector<int> d{1};

    std::vector<int> wa{2, 13, 3, 11, 5, 17, 7};
    std::vector<int> wb{1, 1000};
    std::vector<int> wc{1, 3, 2};
    std::vector<int> wd{1};

    assert(deckRevealedIncreasing(a) == wa);
    assert(deckRevealedIncreasing(b) == wb);
    assert(deckRevealedIncreasing(c) == wc);
    assert(deckRevealedIncreasing(d) == wd);
    std::cout << "deck_revealed_increasing: all tests passed\n";
    return 0;
}
