"""1352. 最后 K 个数的乘积（Product of the Last K Numbers）

题目：实现一个数据结构，支持 add(num) 往末尾追加一个数（num >= 0），
以及 getProduct(k) 返回最后 k 个数的乘积。

思路（前缀积 + 遇 0 归零重来）：
    如果只有前缀积，除法就能得到区间积：`P[n] / P[n-k]`。但 0 会让前缀积恒为 0，
    且不能做除法。于是约定：**遇到 0 就把前缀积数组清空重置为 [1]**，相当于
    「历史从现在重新开始」。

    维护数组 `prefix`，初始为 `[1]`（空前缀）。add(num)：
    - num == 0：`prefix = [1]`；
    - 否则 `prefix.append(prefix[-1] * num)`。
    getProduct(k)：`prefix[-1]` 是「自上次 0 以来所有数的积」。若 `k >= len(prefix)`，
    说明窗口伸到了上一次 0（或更早），答案必为 0；否则答案是
    `prefix[-1] // prefix[-1 - k]`。

    为什么判断是 `k >= len(prefix)`：prefix 的长度是「重来后已经加入的非零元素个数 + 1」，
    当 k 达到这个长度时，窗口的左端已经越过那次重置，中间一定夹着 0。

复杂度：add O(1)，getProduct O(1)，空间 O(add 的非零次数)。
"""


class ProductOfNumbers:
    def __init__(self):
        self.prefix = [1]

    def add(self, num):
        if num == 0:
            self.prefix = [1]
        else:
            self.prefix.append(self.prefix[-1] * num)

    def getProduct(self, k):
        if k >= len(self.prefix):
            return 0
        return self.prefix[-1] // self.prefix[-1 - k]


if __name__ == "__main__":
    p = ProductOfNumbers()
    p.add(3)
    p.add(0)
    p.add(2)
    p.add(5)
    p.add(4)
    assert p.getProduct(2) == 20
    assert p.getProduct(3) == 40
    assert p.getProduct(4) == 0
    assert p.getProduct(5) == 0
    p.add(8)
    assert p.getProduct(2) == 32
    assert p.getProduct(6) == 0

    p2 = ProductOfNumbers()
    p2.add(0)
    assert p2.getProduct(1) == 0
    p2.add(2)
    assert p2.getProduct(1) == 2
    assert p2.getProduct(2) == 0
    print("product_of_numbers: all tests passed")
