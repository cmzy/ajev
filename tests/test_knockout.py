"""ajev/lm/knockout.py 的测试：分组、决赛选手、概率合成。"""

import math

from ajev.lm.knockout import combine, finalists, split_groups


def test_split_groups():
    assert [len(g) for g in split_groups(60)] == [20, 20, 20]
    assert [len(g) for g in split_groups(27)] == [14, 13]
    assert [len(g) for g in split_groups(151)] == [26, 25, 25, 25, 25, 25]
    g = split_groups(77)
    assert sorted(o for x in g for o in x) == list(range(77)) and max(map(len, g)) <= 26


def test_combine_is_a_distribution_and_respects_groups():
    groups = split_groups(60)
    group_logits = [[float(j % 7) for j in range(len(g))] for g in groups]
    fin = finalists(groups, group_logits, per_group=2)
    assert len(fin) == 6
    final_logits = [3.0, 2.0, 0.0, 0.0, -1.0, -1.0]  # 组 1 的两名最强
    lp = combine(60, groups, group_logits, fin, final_logits)
    p = [math.exp(x) for x in lp]
    assert abs(sum(p) - 1) < 1e-9
    m1 = sum(p[o] for o in groups[0])
    expect = (math.exp(3) + math.exp(2)) / sum(math.exp(x) for x in final_logits)
    assert abs(m1 - expect) < 1e-9  # 组的总份额来自决赛
    # 组内的相对大小来自初赛
    a, b = groups[1][0], groups[1][3]
    assert abs((p[b] / p[a]) - math.exp(group_logits[1][3] - group_logits[1][0])) < 1e-6
