"""选项超过 26 个时的“分组淘汰”打分（不依赖 torch，方便单独测试）。

大模型路线用字母 A–Z 表示选项，一次前向最多读 26 个选项的打分。BANKING77（77 类）、CLINC150（151 类）、
检索类任务（32 个候选）都超过这个数。做法（JevK5 等也这样做），不需要重新训练：

    第 1 步（初赛）：把 K 个选项平均分成若干组，每组不超过 26 个，每组单独问一次模型（格式与训练时完全相同），
                     得到组内概率 p_g(o)；
    第 2 步（决赛）：每组概率最高的前 ``per_group`` 个（默认 2 个）选项组成决赛，再问一次模型，
                     得到决赛概率；决赛选项仍超过 26 个时递归淘汰；
    第 3 步（合成）：第 g 组的“总份额” M_g = 该组决赛选手的决赛概率之和；
                     每个选项的最终概率 = M_g × p_g(o)。
    也就是“组和组之间谁强”由决赛决定，“组内谁强”由初赛决定，全部 K 个选项的概率加起来正好是 1。

为什么每组取前 2 名而不是 1 名：如果正确答案在组内恰好排第二，只取第一名它就再也没有机会了。

举个例子：K=60，分成 3 组（20、20、20）。初赛后每组取前 2 名，共 6 个进入决赛；
    决赛概率：组 1 的两名合计 0.7，组 2 合计 0.2，组 3 合计 0.1；
    组 1 里某个选项初赛概率 0.5 → 最终概率 0.7 × 0.5 = 0.35。
"""

from __future__ import annotations

import math


def split_groups(k: int, max_size: int = 26) -> list[list[int]]:
    """把 0..k-1 平均分成若干连续的组，每组不超过 max_size 个，且每组至少 2 个。

    例如 k=60 → 3 组，大小 20、20、20；k=27 → 2 组，大小 14、13。
    """
    n = math.ceil(k / max_size)
    base, extra = divmod(k, n)
    groups, start = [], 0
    for g in range(n):
        size = base + (1 if g < extra else 0)
        groups.append(list(range(start, start + size)))
        start += size
    return groups


def log_softmax(xs: list[float]) -> list[float]:
    m = max(xs)
    s = math.log(sum(math.exp(x - m) for x in xs)) + m
    return [x - s for x in xs]


def finalists(groups: list[list[int]], group_logits: list[list[float]], per_group: int = 2) -> list[int]:
    """每组取初赛打分最高的前 per_group 个选项（返回原始选项下标，按出现顺序排列）。"""
    out = []
    for g, lg in zip(groups, group_logits):
        top = sorted(range(len(g)), key=lambda j: -lg[j])[:per_group]
        out += [g[j] for j in sorted(top)]
    return out


def combine(k: int, groups: list[list[int]], group_logits: list[list[float]], final_idx: list[int],
            final_logits: list[float]) -> list[float]:
    """合成全部 k 个选项的对数概率（可以直接当 logits 用：softmax 之后就是最终概率）。

    log p(o) = log M_g + log p_g(o)，其中 M_g 是 o 所在组的决赛选手的决赛概率之和。
    """
    final_lp = log_softmax(final_logits)
    group_of = {o: gi for gi, g in enumerate(groups) for o in g}
    mass = [0.0] * len(groups)
    for o, lp in zip(final_idx, final_lp):
        mass[group_of[o]] += math.exp(lp)
    out = [0.0] * k
    for gi, (g, lg) in enumerate(zip(groups, group_logits)):
        lp_g = log_softmax(lg)
        log_m = math.log(mass[gi]) if mass[gi] > 0 else -1e9
        for j, o in enumerate(g):
            out[o] = log_m + lp_g[j]
    return out
