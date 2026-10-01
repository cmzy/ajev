"""决策质量指标：衡量模型“答得对不对”以及“概率报得准不准”。

在 AJev 的流程（数据构建 → 训练 → 校准 → 评测）中，这个模块是“评测”的核心：
训练过程中的定期验证、温度校准前后的对比、最终在测试集上的成绩报告，都调用这里的函数。

================================================================
为什么不只看准确率？——“校准（calibration）”是什么
================================================================
Jev 的卖点不只是答对，而是给出“可信的概率”。所谓校准良好，是指：
    模型说“我有 80% 把握”的那些题，实际上大约有 80% 是答对的。

这很重要，因为下游程序会根据概率做决定，比如“置信度高于 0.9 就自动处理，否则转人工”。
如果模型总是说 99% 把握但其实只对 70%（过度自信），这个阈值就完全失效了。
所以下面既有衡量“对不对”的指标（accuracy、chance_acc），
也有衡量“概率准不准”的指标（brier、kl、ece）。

================================================================
各指标的含义（每个都附一个小算例）
================================================================
所有函数的输入都是两个等长的列表：Decision（带标准答案）和对应的预测分布。
指标的定义尽量与开源 Jev 复刻项目（Laya、Verdict、JevBench）一致，这样我们的数字可以直接和它们比较。

* accuracy（准确率）
    预测分布的 argmax（概率最大的选项）等于标准答案的题目比例。越高越好。
    标准答案优先取 ``meta["gold_label"]``（typed-decisions 官方给出的答案名），没有时取 argmax(target)。
    算例：4 道题答对 3 道 → accuracy = 0.75。

* chance_acc（扣除随机猜测的准确率）
    公式 (acc - c) / (1 - c)，其中 c 是“随机瞎猜”的期望准确率（各题 1/K 的平均值），结果最小为 0。
    为什么需要：二选一的题闭着眼睛猜也有 50%，十选一只有 10%，原始准确率无法跨题型比较。
    这个指标表示“在随机猜的基础上，又多答对了多少比例”，0 表示和瞎猜一样，1 表示全对。
    算例：全是二选一（c = 0.5），acc = 0.75 → (0.75 - 0.5) / (1 - 0.5) = 0.5。
    对应 JevBench 中的 “intelligence” 分数。

* brier（Brier 分数）
    对每道题计算 mean_k (p_k - t_k)²（预测概率与目标概率之差的平方，对所有选项取平均），
    再对所有题取平均。越小越好，0 表示完美。它同时惩罚“答错”和“概率报得不准”。
    算例：二选一，目标 [1, 0]，预测 [0.8, 0.2] → ((0.8-1)² + (0.2-0)²) / 2 = (0.04 + 0.04) / 2 = 0.04；
          如果预测 [0.4, 0.6]（答错了）→ ((0.4-1)² + 0.6²) / 2 = (0.36 + 0.36) / 2 = 0.36，惩罚大得多。

* kl（KL 散度，KL(target ‖ pred)）
    衡量预测分布与目标分布“差多远”：Σ t_i · (ln t_i - ln p_i)。两者完全一样时为 0，越大差得越远。
    特点是对“把正确答案的概率报得很低”惩罚非常重（因为 ln p 在 p→0 时趋于负无穷）。
    算例：目标 [1, 0]，预测 [0.8, 0.2] → 1 × (ln1 - ln0.8) ≈ 0.223；
          预测 [0.01, 0.99] → 1 × (ln1 - ln0.01) ≈ 4.6。

* ece（Expected Calibration Error，期望校准误差）
    把所有题按模型的 top-1 置信度（最大概率）分到 15 个区间（0～1/15, 1/15～2/15, …），
    在每个区间里比较“平均置信度”和“实际准确率”的差距，再按区间题数加权平均。越小越好。
    算例：10 道题都说 90% 把握，实际对了 6 道 → 这个区间的差距是 |0.9 - 0.6| = 0.3，ECE = 0.3（过度自信）。

* mean_confidence：平均 Jev confidence（1 - 熵/ln K，见 ``ajev.schema.jev_confidence``），
    反映模型整体有多“果断”。

* flip_rate（选项打乱翻转率，见 ``flip_rate`` 函数）：衡量模型有没有“位置偏置”。
"""

from __future__ import annotations

import math
from collections import defaultdict  # 访问不存在的键时会自动创建默认值的字典
from typing import Callable  # 类型注解：表示“可调用的东西”（函数）

from ajev.schema import Decision, jev_confidence

# 计算对数时加上的极小数。ln(0) 是负无穷，会导致计算出错；加上 1e-12（0.000000000001）后
# 结果是一个很大但有限的数，对正常数值几乎没有影响。
EPS = 1e-12


def _argmax(ps: list[float]) -> int:
    """返回列表中最大值的下标；有并列时返回第一个（与 Python 内置 ``max`` 的行为一致）。

    举个例子: _argmax([0.1, 0.7, 0.2]) → 1；_argmax([0.5, 0.5]) → 0。
    """
    return max(range(len(ps)), key=ps.__getitem__)


def gold_index(d: Decision) -> int:
    """确定一道题的“标准答案”在第几个选项（下标从 0 开始）。

    为什么不直接对 target 取 argmax：typed-decisions 的软标签可能出现并列最大值，
    例如 [0.5, 0.5]，此时 argmax 的结果取决于选项排在哪里，打乱顺序后答案就变了，不稳定。
    数据集官方给出的 ``gold_label`` 才是公认答案；用它计算准确率，
    才能和 Laya / Verdict / Jev 公布的成绩采用同一口径。

    举个例子:
        选项 ["a", "b", "c"]，target [0.5, 0.5, 0]，meta["gold_label"] = "b" → 返回 1
        如果没有 gold_label                                                → 返回 0（argmax，并列取第一个）
    """
    label = d.meta.get("gold_label")
    if label is not None and label in d.option_names:
        return d.option_names.index(label)  # list.index(x) 返回 x 在列表中的位置
    return d.gold_index


def ece(confidences: list[float], correct: list[bool], n_bins: int = 15) -> float:
    """计算 top-label ECE（期望校准误差）。

    举个例子（n_bins=15）:
        confidences = [0.9, 0.9]，correct = [True, False]
        → 两道题都落在 0.9 所在的区间，平均置信度 0.9，实际准确率 0.5
        → ECE = (2/2) × |0.9 - 0.5| = 0.4

    参数:
        confidences: 每道题的 top-1 置信度（预测分布里的最大概率）。
        correct:     每道题是否答对（True / False）。
        n_bins:      区间个数，默认 15（与常见论文一致）。
    返回:
        ECE 值；没有任何题目时返回 NaN（Not a Number，表示“无法计算”）。
    """
    if not confidences:
        return float("nan")
    # 第 1 步：准备 n_bins 个空桶，每个桶里存放 (置信度, 是否答对) 二元组。
    bins: list[list[tuple[float, bool]]] = [[] for _ in range(n_bins)]
    # 第 2 步：把每道题放进它的置信度所在的桶。
    #   置信度 c 落在第 int(c × n_bins) 个桶，例如 c = 0.5、n_bins = 15 → int(7.5) = 7。
    #   c 恰好等于 1.0 时，int(1.0 × 15) = 15，会超出 0～14 的范围，所以用 min 夹到最后一个桶。
    for c, ok in zip(confidences, correct):
        bins[min(int(c * n_bins), n_bins - 1)].append((c, ok))
    total = len(confidences)
    # 第 3 步：对每个非空的桶，计算 |平均置信度 - 准确率|，再乘以“桶内题数 / 总题数”作为权重，最后求和。
    #   sum(ok for _, ok in b) 统计答对的题数：True 在求和时当作 1，False 当作 0。
    return sum(
        len(b) / total * abs(sum(c for c, _ in b) / len(b) - sum(ok for _, ok in b) / len(b)) for b in bins if b
    )


def compute(decisions: list[Decision], preds: list[list[float]]) -> dict[str, float]:
    """对一组题目一次性计算全部指标。

    举个例子（两道三选一的题）:
        decisions 的 target 分别为 [1,0,0] 和 [0,1,0]
        preds = [[0.7,0.2,0.1], [0.6,0.3,0.1]]
        → 第 1 题答对（argmax 0 == 0），第 2 题答错（argmax 0 ≠ 1），accuracy = 0.5
        → 随机水平 c = 1/3，chance_acc = (0.5 - 1/3) / (1 - 1/3) = 0.25

    参数:
        decisions: 带标准答案的题目列表。
        preds:     与 decisions 一一对应的预测分布；每个分布的长度必须等于该题的选项数，
                   顺序与 ``decision.options`` 一致。
    返回:
        字典，包含 n（题数）、accuracy、chance_acc、brier、kl、ece、mean_confidence。
        如果没有任何题目，只返回 ``{"n": 0}``。
    """
    # assert 用于检查“绝对应该成立”的条件，不成立时立刻报错，帮助尽早发现程序 bug。
    assert len(decisions) == len(preds), "decisions and predictions differ in length"
    n = len(decisions)
    if n == 0:
        return {"n": 0}
    # 第 1 步：初始化各项累加器。chance 累加的是每道题“随机猜中”的概率 1/K。
    acc = chance = brier = kl = 0.0  # chance: 1/K 之和
    top_conf: list[float] = []  # 每道题的最大预测概率，用于算 ECE
    correct: list[bool] = []  # 每道题是否答对，用于算 ECE
    jev_conf = 0.0
    # 第 2 步：逐题计算并累加。
    for d, p in zip(decisions, preds):
        k = len(d.options)
        assert len(p) == k, f"{d.id}: {len(p)} probabilities for {k} options"
        ok = _argmax(p) == gold_index(d)  # 这道题是否答对（True / False）
        acc += ok  # True 加上去相当于加 1，False 相当于加 0
        chance += 1 / k
        # 每道题的 Brier 除以 K，得到“每个选项的平均平方误差”：
        # 与 Verdict 报告的口径一致，也让不同选项数的题处在同一量级。
        brier += sum((pi - ti) ** 2 for pi, ti in zip(p, d.target)) / k
        # KL(t ‖ p) = Σ t_i × (ln t_i - ln p_i)。t_i = 0 的项乘出来一定是 0，直接跳过。
        kl += sum(ti * (math.log(ti + EPS) - math.log(pi + EPS)) for pi, ti in zip(p, d.target) if ti > 0)
        top_conf.append(max(p))
        correct.append(ok)
        jev_conf += jev_confidence(p)
    # 第 3 步：把累加值除以题数得到平均值，组装成结果字典。
    return {
        "n": n,
        "accuracy": acc / n,
        # 注意：必须先求整体的平均准确率和平均随机水平，再做一次校正。
        # 如果逐题校正再平均：答对的题得 (1 - 1/K)/(1 - 1/K) = 1，答错的题得负数被截成 0，
        # 结果就会退化成普通准确率，失去“扣除随机猜测”的意义。
        "chance_acc": max(0.0, (acc / n - chance / n) / (1 - chance / n)),
        "brier": brier / n,
        "kl": kl / n,
        "ece": ece(top_conf, correct),
        "mean_confidence": jev_conf / n,
    }


def breakdown(
    decisions: list[Decision], preds: list[list[float]], key: Callable[[Decision], str]
) -> dict[str, dict[str, float]]:
    """按某个维度把题目分组，分别计算每组的指标。

    参数:
        decisions / preds: 同 ``compute``。
        key:  分组函数，输入一道题、返回它所属的组名。常用写法（lambda 是“匿名小函数”）:
              ``lambda d: d.type``（按题型）、``lambda d: d.lang``（按语言）、``lambda d: d.source``（按来源）。
    返回:
        ``{组名: 该组的指标字典}``，按组名排序，打印出来的表格顺序固定，方便对比。

    举个例子:
        breakdown(ds, preds, lambda d: d.lang)
        → {"en": {"n": 5399, "accuracy": 0.71, ...}, "zh": {"n": 1500, "accuracy": 0.68, ...}}
    """
    # 第 1 步：记录每个组包含哪些题（存题目的下标）。
    # defaultdict(list)：第一次访问某个组名时，会自动创建一个空列表。
    groups: dict[str, list[int]] = defaultdict(list)
    for i, d in enumerate(decisions):
        groups[key(d)].append(i)
    # 第 2 步：对每个组，取出该组的题目和预测，调用 compute 计算指标。
    return {
        g: compute([decisions[i] for i in idx], [preds[i] for i in idx]) for g, idx in sorted(groups.items())
    }


def flip_rate(base_preds: list[list[float]], decisions: list[Decision], shuffled: list[Decision],
              shuffled_preds: list[list[float]]) -> float:
    """选项打乱翻转率：把选项顺序打乱后，模型选中的答案（按选项“名字”比较）发生变化的题目比例。

    为什么需要：一个真正理解题目的模型，应该只看选项的内容，而不管它排在第几个。
    如果模型偏爱“第一个选项”（位置偏置），那打乱顺序后它就会换一个答案。
    这个比例越低越好；OpenJev 通过专门的微调把它从 18.5% 降到了 2.3%。

    只统计 choice / noul 题；score 题的等级本来就不打乱，没有统计意义，跳过。

    举个例子:
        原题选项 [a, b, c]，模型选 a；打乱后选项变成 [c, b, a]：
            * 如果模型选了第一个位置的 c → 答案从 a 变成 c，算一次翻转；
            * 如果模型选了第三个位置的 a → 答案没变，不算翻转。

    参数:
        base_preds:     原始顺序下的预测。
        decisions:      原始顺序的题目。
        shuffled:       同样的题目，选项被打乱后的版本（与 decisions 一一对应）。
        shuffled_preds: 打乱后的预测。
    返回:
        翻转比例（0～1）；如果没有可统计的题目，返回 NaN。
    """
    flips = total = 0
    for d, p, sd, sp in zip(decisions, base_preds, shuffled, shuffled_preds):
        if d.type == "score":
            continue  # continue：跳过本次循环剩下的代码，直接处理下一道题
        total += 1
        # 比较的是选中选项的“名字”而不是下标，因为两次的选项顺序不同，下标没有可比性。
        flips += d.option_names[_argmax(p)] != sd.option_names[_argmax(sp)]
    return flips / total if total else float("nan")
