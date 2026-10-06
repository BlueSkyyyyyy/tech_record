"""1510. 石子游戏 IV（Stone Game IV）

题目：有一堆 n 颗石子。两人轮流取，每次必须取走「完全平方数」颗（1、4、9、16、…），
取走最后一颗石子的人获胜。你先手，问能否获胜。

思路（布尔胜负态 + 反向递推）：
    用 win[i] 表示「还剩 i 颗时，轮到行动的人是否能赢」。取法只与「取走多少」有关，
    所以可以在 1..n 上从小到大递推：

        win[0] = False            # 没石子可取，行动者输
        win[i] = any( not win[i - s*s]  for s*s <= i )

    含义是：只要存在一个合法的平方数 s*s，取完之后对手面对的是必败态，那我就必胜。
    这就是区间取石类博弈的通用骨架——「枚举我能走的每一步，只要有一招收尾是对手必败」。

复杂度：时间 O(n * sqrt(n))，空间 O(n)。
"""


def winner_square_game(n):
    win = [False] * (n + 1)
    for i in range(1, n + 1):
        s = 1
        while s * s <= i:
            if not win[i - s * s]:
                win[i] = True
                break
            s += 1
    return win[n]


if __name__ == "__main__":
    assert winner_square_game(1) is True
    assert winner_square_game(2) is False
    assert winner_square_game(4) is True
    assert winner_square_game(7) is False
    assert winner_square_game(17) is False
    assert winner_square_game(12) is False
    assert winner_square_game(13) is True
    assert winner_square_game(16) is True
    print("stone_game_iv: all tests passed")
