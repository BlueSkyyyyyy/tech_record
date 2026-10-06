"""134. 加油站（Gas Station）

题目：在一条环路上有 n 个加油站，第 i 个加油站有汽油 gas[i] 升。你有一辆油箱容量无限的
汽车，从第 i 个加油站开往第 i+1 个加油站需要消耗汽油 cost[i] 升。你从其中一个加油站出发，
初始油箱为空。若可以绕环路行驶一周，返回出发加油站的编号，否则返回 -1。若存在解则唯一。

思路（贪心：总油量够就有解；出发点是「累加亏空后第一个能翻正的地方」）：
    先把每个站看成「净收益」 diff[i] = gas[i] - cost[i]。绕一圈能走完，等价于所有 diff
    之和 >= 0。如果总和 < 0，无论从哪出发都不够油，直接返回 -1；否则一定存在一个出发点。
    用一次遍历找这个出发点：
      - total 记录所有 diff 的累计和（判断整体是否有解）；
      - tank 记录从当前候选起点出发、到现在的油箱余额；
      - 若 tank < 0，说明从「当前起点」到这里都撑不过，那么从当前起点到 i 之间的任何一站
        出发也都不行，于是把起点设为 i + 1，并把 tank 清零重新开始。
    最后 total >= 0 时，记录的起点就是答案。

    为什么「tank 变负则起点跳到 i+1」是对的：设当前起点为 s，从 s 到 i 的净油量为负（tank<0）。
    那么对 s 和 i 之间的任意站 j，从 s 到 j 的净油量都是正的（否则早在 j 之前 tank 就为负、
    起点早就跳过去了），记这段正余量为 P>0；从 s 到 i 的总净量 = P + (从 j+1 到 i 的净量) < 0，
    所以从 j+1 到 i 的净量 < -P < 0。也就是说，从 j 出发同样会在到达 i 之前油量耗尽。
    因此 s..i 之间没有可行起点，直接跳过整段，从 i+1 重新尝试，且不会漏掉唯一解。

复杂度：时间 O(n)（一次遍历），空间 O(1)。
"""


def can_complete_circuit(gas, cost):
    total = 0
    tank = 0
    start = 0
    for i in range(len(gas)):
        diff = gas[i] - cost[i]
        total += diff
        tank += diff
        if tank < 0:
            start = i + 1
            tank = 0
    return start if total >= 0 else -1


if __name__ == "__main__":
    assert can_complete_circuit([1, 2, 3, 4, 5], [3, 4, 5, 1, 2]) == 3
    assert can_complete_circuit([2, 3, 4], [3, 4, 3]) == -1
    assert can_complete_circuit([5], [4]) == 0
    assert can_complete_circuit([2], [3]) == -1
    assert can_complete_circuit([3, 1, 1], [1, 2, 2]) == 0
    print("gas_station: all tests passed")
