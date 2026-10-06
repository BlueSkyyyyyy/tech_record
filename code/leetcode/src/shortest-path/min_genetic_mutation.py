"""433. 最小基因变化（Minimum Genetic Mutation）

题目：基因序列由 'A'、'C'、'G'、'T' 四个字符组成，长度固定为 8。一次「基因变化」
指把某个位置改成另一个字符。给定起始串 startGene、目标串 endGene 和一个合法基因
库 bank，每次变化后的串必须存在于 bank 中，求从 startGene 变到 endGene 的最少次数；
无法完成返回 -1。

思路（隐式图 BFS）：
    把每个基因串看成一个节点，两个串若只差一个字符、且都在 bank 里，就连一条边。
    一次变化走一条边，所以「最少变化次数」就是 startGene 到 endGene 的最短路。
    边权全为 1，用 BFS。

    图中节点没有显式给出，但每个串的邻居可以现算：枚举 8 个位置 × 3 种替换字符
    （跳过原字符），得到至多 24 个候选串，命中的才是真邻居。这种「边在访问时才
    生成」的图叫隐式图。

    要点：
    - 若 endGene 不在 bank 里，永远无法到达（除非 start == end），直接 -1；
    - BFS 按层推进，层数就是变化次数；
    - 找到 endGene 时立即返回当前层数。

复杂度：时间 O(24·L·N)（L=8 为串长，N 为 bank 大小，每个串生成邻居的花费），
空间 O(N) 存 visited 与队列。
"""
from collections import deque

_GENES = "ACGT"


def min_genetic_mutation(start_gene, end_gene, bank):
    bank = set(bank)
    if end_gene not in bank:
        return -1
    if start_gene == end_gene:
        return 0
    queue = deque([start_gene])
    visited = {start_gene}
    steps = 0
    while queue:
        steps += 1
        for _ in range(len(queue)):
            cur = queue.popleft()
            for i in range(len(cur)):
                for g in _GENES:
                    if g == cur[i]:
                        continue
                    nxt = cur[:i] + g + cur[i + 1:]
                    if nxt in bank and nxt not in visited:
                        if nxt == end_gene:
                            return steps
                        visited.add(nxt)
                        queue.append(nxt)
    return -1


if __name__ == "__main__":
    assert min_genetic_mutation("AACCGGTT", "AACCGGTA", ["AACCGGTA"]) == 1
    assert (
        min_genetic_mutation(
            "AACCGGTT", "AAACGGTA", ["AACCGGTA", "AACCGCTA", "AAACGGTA"]
        )
        == 2
    )
    assert (
        min_genetic_mutation(
            "AAAAACCC", "AACCCCCC", ["AAAACCCC", "AAACCCCC", "AACCCCCC"]
        )
        == 3
    )
    # 目标不在基因库
    assert min_genetic_mutation("AACCGGTT", "AACCGGTA", []) == -1
    print("min_genetic_mutation: all tests passed")
