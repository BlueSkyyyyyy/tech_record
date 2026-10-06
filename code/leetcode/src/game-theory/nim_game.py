"""292. Nim 游戏（Nim Game）

题目：桌上有一堆 n 颗石子，两人轮流取，每次可以取 1~3 颗，取走最后一颗石子的人获胜。
你先手，两人都采取最优策略，问你是否能赢。

思路（只看模 4 的余数）：
    先手必败的位置只有 4 的倍数。原因：
    - 若 n 是 4 的倍数，无论你取 1/2/3 颗，对手都能取 3/2/1 颗，把你留下的局面重新
      凑成 4 的倍数。如此往复，对手总能把「4 的倍数」这个局面丢还给你，最后你面对 0 颗
      （已无石子可取）而输。
    - 若 n 不是 4 的倍数，你先取 n % 4 颗，把局面留给对手一个 4 的倍数，于是你扮演上一段
      里「对手」的角色，必胜。

    这是一道典型的「对手能镜像你的动作，所以关键在凑同一个数」的博弈题：胜负被 n 对
    (最大可取数 + 1) 的余数决定。

复杂度：时间 O(1)，空间 O(1)。
"""


def can_win_nim(n):
    return n % 4 != 0


if __name__ == "__main__":
    assert can_win_nim(1) is True
    assert can_win_nim(2) is True
    assert can_win_nim(3) is True
    assert can_win_nim(4) is False
    assert can_win_nim(5) is True
    assert can_win_nim(8) is False
    assert can_win_nim(100) is False
    assert can_win_nim(101) is True
    print("nim_game: all tests passed")
