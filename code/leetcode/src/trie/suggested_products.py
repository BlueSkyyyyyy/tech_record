"""1268. 搜索推荐系统

题目：给定产品数组 products 和搜索词 search_word，每次用户多输入一个字母，
      就返回 name 以当前输入为前缀的产品中，按字典序最小的至多 3 个。

思路：先把 products 按字典序排序，再插入前缀树。排序保证「先插入的产品字典序
      更小」，于是每个节点上只要记录「最早经过它的至多 3 个产品」，就已经是字典序
      最小的 3 个。查询时逐字符沿树下行，把当前节点的推荐列表抄进结果；一旦某个
      字符走不通，后面所有前缀都不存在，全部返回空列表。

      为什么节点上存的是全局字典序最小的 3 个：插入是按排序后的顺序进行的，
      字典序小的产品更早到达该节点并被记下，容量满了就不再接收，后来居上的产品
      字典序一定更大，所以留住的前 3 个正好就是答案。

复杂度：排序 O(n log n)；建树 O(总字符数)；查询 O(L + 答案总量)。空间 O(总字符数)。
"""


def suggested_products(products, search_word):
    trie = {}
    for product in sorted(products):
        node = trie
        for ch in product:
            node = node.setdefault(ch, {})
            node.setdefault("_suggest", [])
            if len(node["_suggest"]) < 3:
                node["_suggest"].append(product)

    result = []
    node = trie
    for ch in search_word:
        if node is not None and ch in node:
            node = node[ch]
            result.append(list(node["_suggest"]))
        else:
            node = None
            result.append([])
    return result


if __name__ == "__main__":
    products = ["mobile", "mouse", "moneypot", "monitor", "mousepad"]
    got = suggested_products(products, "mouse")
    want = [
        ["mobile", "moneypot", "monitor"],
        ["mobile", "moneypot", "monitor"],
        ["mouse", "mousepad"],
        ["mouse", "mousepad"],
        ["mouse", "mousepad"],
    ]
    assert got == want

    assert suggested_products(["havana"], "havana") == [
        ["havana"], ["havana"], ["havana"], ["havana"], ["havana"], ["havana"],
    ]

    assert suggested_products(["bags", "baggage", "banner", "box", "cloths"], "bags") == [
        ["baggage", "bags", "banner"],
        ["baggage", "bags", "banner"],
        ["baggage", "bags"],
        ["bags"],
    ]
    print("suggested_products: all tests passed")
