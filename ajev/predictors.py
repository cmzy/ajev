"""预测器接口（Predictor）以及几个简单的基线（baseline）预测器。

================================================================
什么是“预测器”
================================================================
在 AJev 中，评测程序（``ajev.eval.evaluate``）不关心预测是怎么来的，它只要求一件事：
    给一批题（Decision），返回每道题在各个选项上的概率分布。
凡是能做到这件事的对象，都叫“预测器”，都可以被评测。这包括本文件里的几个简单基线，
也包括训练好的神经网络模型（``ajev.model.predictor.EncoderPredictor`` 实现了同样的 ``predict`` 方法）。

================================================================
什么是“基线（baseline）”，为什么有用
================================================================
基线是“不用任何智能、只用最简单规则”的预测方法，用来给模型成绩提供一个参照下限：
    * 如果模型的准确率是 0.70，听起来不错——但如果最简单的规则就能拿到 0.68，那模型其实没学到多少；
    * 反过来，如果基线只有 0.48，模型的 0.70 就说明它真的读懂了材料。
本文件提供三个基线：均匀分布、随机分布、按训练集答案频率猜（先验频率）。
另外，基线的输出很好预测，也常被用来检查评测指标的实现是否正确。
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict  # Counter：专门用来计数的字典
from typing import Protocol

from ajev.schema import Decision


# Protocol（协议）是 Python 的一种“接口”写法，属于“鸭子类型”：
# “如果它走路像鸭子、叫声像鸭子，那它就是鸭子。”
# 任何类只要有一个签名相同的 predict 方法，就自动被视为 Predictor，
# 不需要显式继承（不用写 class X(Predictor)）。这让模型类和基线类可以互相替换使用。
class Predictor(Protocol):
    """预测器协议（接口）。

    ``predict`` 接收 N 道题，返回 N 个概率分布：第 i 个分布的长度等于第 i 道题的选项数，
    顺序与 ``decisions[i].options`` 一致，并且加起来等于 1。

    举个例子:
        输入 2 道题，分别有 2 个和 3 个选项
        → 返回类似 [[0.9, 0.1], [0.2, 0.5, 0.3]] 的结果
    """

    # 方法体只写 “...”（省略号），表示这里只声明“必须有这个方法”，不提供具体实现。
    def predict(self, decisions: list[Decision]) -> list[list[float]]: ...


class UniformPredictor:
    """均匀分布基线：每个选项的概率都一样（K 个选项时每个都是 1/K）。

    注意：所有概率相同时，argmax 会选第一个选项，所以它的准确率等于“标准答案恰好排在第一位”的比例，
    不一定等于理论上的随机水平 1/K。它的 Jev confidence 恒为 0（完全不确定），
    可以用来检查 confidence 等指标算得对不对。

    举个例子:
        一道三选一的题 → [0.333, 0.333, 0.333]；一道是/否题 → [0.5, 0.5]
    """

    def predict(self, decisions: list[Decision]) -> list[list[float]]:
        return [[1.0 / len(d.options)] * len(d.options) for d in decisions]


class RandomPredictor:
    """随机基线：每道题生成一个随机的概率分布（每个选项取一个 0～1 的随机数，再归一化）。

    参数:
        seed: 随机种子。种子固定后，每次运行得到的“随机”结果都一样，方便复现实验。

    举个例子:
        三选一的题，随机数为 [0.2, 0.6, 0.2]，总和 1.0 → 输出 [0.2, 0.6, 0.2]；
        随机数为 [0.5, 0.25, 0.25]，总和 1.0 → 输出 [0.5, 0.25, 0.25]（每次除以各自的总和）。
    """

    def __init__(self, seed: int = 0) -> None:
        # 使用独立的随机数生成器，而不是全局的 random，避免影响程序其他地方的随机性。
        self.rng = random.Random(seed)

    def predict(self, decisions: list[Decision]) -> list[list[float]]:
        out = []
        for d in decisions:
            w = [self.rng.random() for _ in d.options]  # 每个选项一个 [0, 1) 之间的随机数
            s = sum(w)
            out.append([x / s for x in w])  # 除以总和，使它们加起来等于 1
        return out


class PriorPredictor:
    """先验频率基线：完全不看材料，只根据“训练集里同一道题的答案分布”来猜。

    “同一道题”指：数据来源 + 问题文本 + 选项名集合 都相同（见 ``_key``）。

    为什么它是一个有意义的下限：typed-decisions 的每个业务流程都在反复问同样的 5 个问题，
    而每个问题的答案往往有明显的倾向（例如“是否需要人工”大多数时候是“否”）。
    只看问题本身、不读材料，也能猜对相当一部分——实测约 0.48。
    模型必须明显超过它，才能说明真的从材料中读出了信息。
    在训练集中从没出现过的问题，就退化成均匀分布。

    参数:
        train:     训练集题目，用来统计每道题各选项的 target 累计值（软标签直接按概率累加）。
        smoothing: 加法平滑（拉普拉斯平滑）系数：每个选项的计数都额外加上这个数，
                   避免某个选项因为训练集里一次也没出现过而得到 0 概率（0 概率在算 KL 时会导致数值极大）。

    举个例子（smoothing = 0）:
        训练集里同一道题出现 3 次，答案分别是 a、a、b
        → 计数 {a: 2, b: 1, c: 0}
        → 预测 [2/3, 1/3, 0]
        若 smoothing = 1 → 计数变成 {a: 3, b: 2, c: 1}，预测 [3/6, 2/6, 1/6] = [0.5, 0.333, 0.167]
    """

    def __init__(self, train: list[Decision], smoothing: float = 1.0) -> None:
        self.smoothing = smoothing
        # 结构：{题目身份键: Counter({选项名: 累计概率})}
        self.counts: dict[tuple, Counter] = defaultdict(Counter)
        for d in train:
            # 按选项“名字”累计，而不是按位置。因此即使训练集中这道题的选项顺序被打乱过，也能正确合并。
            for name, t in zip(d.option_names, d.target):
                self.counts[self._key(d)][name] += t

    # @staticmethod 表示“静态方法”：不需要访问对象自身（没有 self 参数），只是放在类里便于组织代码。
    @staticmethod
    def _key(d: Decision) -> tuple:
        """生成题目的“身份键”：(来源, 问题文本, 排序后的选项名元组)。

        选项名先排序，是为了让键与选项顺序无关：[b, a] 和 [a, b] 被视为同一道题。
        用元组（tuple）而不是列表，是因为字典的键必须是不可修改的类型。
        """
        return d.source, d.instructions, tuple(sorted(d.option_names))

    def predict(self, decisions: list[Decision]) -> list[list[float]]:
        out = []
        for d in decisions:
            # 找不到这道题时，用一个空 Counter；Counter 对不存在的键返回 0，所以结果退化为均匀分布。
            c = self.counts.get(self._key(d), Counter())
            # 按“当前这道题”的选项顺序取计数并加上平滑项，保证输出顺序与 d.options 对齐。
            w = [c[n] + self.smoothing for n in d.option_names]
            s = sum(w)
            out.append([x / s for x in w])
        return out
