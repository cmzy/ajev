"""LoRA 训练的三个辅助工具（不依赖 torch，方便单独测试）：固定题数分批、相对阈值尖峰保护、打分题的序数平滑。

这三样都来自 gemma_lora1b / gemma_lora2 两轮训练的教训（详见对话中的排查记录）：

1. **固定每步题数**（``fixed_count_steps``）
   原来的分批是“每个小批次按 token 预算装满，装满一次就更新一次”，并且块内按长度排序。结果：
   长题和长题一批（只有 4–10 道），短题和短题一批（30 道左右）。梯度裁剪到 1.0 + Adam 让每一步的更新幅度
   差不多大，于是 5 道长题的一步和 30 道短题的一步影响力相同——**平均到每道题，长题被放大了好几倍**；
   同时题少的批次梯度噪声大，尖峰几乎都出在这里（被跳过的 56 步里，题数中位数只有 9）。
   新做法：**每 N 道题（默认 32）为一个优化器步**，攒够这 N 道才更新一次——每一步都正好 N 道题，
   每道题的权重真正相同；长题的一步也有 32 道题平均，梯度不再因为题少而噪声大。
   怎么凑这 N 道题：整体打乱后每 ``sort_block``（默认 512）道题按长度排序，再每 N 道切成一步，最后打乱各步的顺序。
   为什么不完全随机凑：gemma_lora3 第一次启动就是完全随机的，32 道长短悬殊的题放在一起，短题要补齐到最长题的长度，
   实际计算量是真实 token 数的 2.07 倍，每步 26 秒、比预期慢一倍多；按 512 道一块排序后补齐只有 1.11 倍，
   而每步平均仍有约 10.6 个不同的数据源（完全随机约 16.9 个，按 2048 道一块排序约 10.1 个）。

2. **相对阈值的尖峰保护**（``SpikeGuard``）
   原来是“裁剪前梯度范数 > 100 就跳过这一步”。固定阈值分不清“长题小批次的梯度本来就大”和“训练真的出问题”，
   误伤了大量正常的长题批次（gemma_lora2 约 6% 的步被跳过）。新做法参考 AutoClip / ZClip 的思路：
   和**最近一段时间的梯度范数中位数**比，超过它的 ``ratio`` 倍（默认 10 倍）才算异常；
   另外保留一个很高的绝对上限（默认 1000）兜底，并且梯度出现 NaN / inf 时一定跳过。
   用中位数而不是平均数：中位数不会被偶尔的一个尖峰拉高。

3. **打分题的序数平滑**（``ordinal_smooth``）
   Feedback-Collection、UltraFeedback 中文版、HelpSteer2 的打分题只有**一个**分数（GPT-4 或标注者给的），
   标签是 one-hot。但这类“回答质量 1–5 分”的判断本来就有主观性，换个评分者差一档很常见。
   让模型把全部概率押在一个等级上，既不现实，也会在“差一档”时产生很大的损失（尖峰来源之一）。
   序数平滑：正确等级保留 1-ε，ε 平均分给**相邻**的等级（不是像普通标签平滑那样平均分给所有等级）。
   例如 5 级、正确是第 3 级、ε=0.2 → [0, 0.1, 0.8, 0.1, 0]；在最边上的第 1 级 → [0.8, 0.2, 0, 0, 0]。
   已经是软标签的题（如 HelpSteer3 有多人投票的题）不处理。
"""

from __future__ import annotations

import math
import random
from collections import deque

# 只有一个评分者（或评分是模型给的）的打分类数据源：对它们的 one-hot 打分题做序数平滑。
ORDINAL_SMOOTH_SOURCES = ("feedback_collection", "ultrafeedback_zh", "helpsteer2")


def fixed_count_steps(lengths: list[int], per_step: int, max_tokens: int, max_batch: int,
                      seed: int, sort_block: int = 512) -> list[list[list[int]]]:
    """把一个 epoch 的题分成“每步正好 per_step 道”的优化器步，每步再切成若干小批次。

    步骤：
        第 1 步：整体打乱；每 sort_block 道题为一块，块内按长度排序，再每 per_step 道切成一步
                （sort_block=0 表示不排序，完全随机凑步；每块最后一步可能不足 per_step 道）；
        第 2 步：步内按长度升序，顺序切小批次：再加一道题会让“题数 × 本批最长长度”超过 max_tokens，
                或题数超过 max_batch，就另起一个小批次（与 token_budget_batches 的切法相同）；
        第 3 步：打乱各步的顺序，避免训练时先全是短题、后全是长题。

    Returns:
        步的列表；每一步是小批次的列表；每个小批次是题目下标的列表。

    举个例子：lengths=[100, 900, 120, 80]，per_step=2，max_tokens=1000，sort_block=0（假设打乱后顺序不变）
        第 1 步：[0, 1] → 按长度排序 [0, 1] → 2×900 > 1000，切成 [[0], [1]]
        第 2 步：[2, 3] → 按长度排序 [3, 2] → 2×120 ≤ 1000，一个小批次 [[3, 2]]
        结果：[[[0], [1]], [[3, 2]]]——每一步都是 2 道题，长题单独占一个小批次只是为了省显存。
    """
    rng = random.Random(seed)
    idx = list(range(len(lengths)))
    rng.shuffle(idx)
    groups: list[list[int]] = []
    block = sort_block or len(idx)
    for b in range(0, len(idx), block):
        chunk = sorted(idx[b: b + block], key=lengths.__getitem__) if sort_block else idx[b: b + block]
        groups += [chunk[s: s + per_step] for s in range(0, len(chunk), per_step)]
    if sort_block:
        rng.shuffle(groups)
    steps = []
    for g in groups:
        micro: list[list[int]] = []
        cur: list[int] = []
        for i in sorted(g, key=lengths.__getitem__):
            if cur and (len(cur) + 1 > max_batch or (len(cur) + 1) * lengths[i] > max_tokens):
                micro.append(cur)
                cur = []
            cur.append(i)
        micro.append(cur)
        steps.append(micro)
    return steps


def flatten_steps(steps: list[list[list[int]]]) -> tuple[list[list[int]], list[bool]]:
    """把“步 → 小批次”展平成小批次列表，并给出每个小批次是否是它那一步的最后一个（是就该更新参数了）。

    例如 [[[0], [1]], [[3, 2]]] → ([[0], [1], [3, 2]], [False, True, True])。
    """
    micro, ends = [], []
    for st in steps:
        for j, mb in enumerate(st):
            micro.append(mb)
            ends.append(j == len(st) - 1)
    return micro, ends


class SpikeGuard:
    """判断这一步的梯度是否“异常大”，异常就跳过更新。

    规则（满足任意一条就跳过）：
        1. 梯度范数不是有限数（NaN / inf）；
        2. 超过绝对上限 ``abs_limit``（0 表示不设）；
        3. 已经积累了至少 ``min_history`` 个正常步，且超过最近 ``window`` 个正常步梯度范数中位数的 ``ratio`` 倍
           （``ratio`` 为 0 表示不用相对阈值）。
    只有**没被跳过**的步才计入历史，避免尖峰把中位数抬高。

    举个例子：ratio=10，最近的梯度范数中位数是 4 → 超过 40 才跳过；
        而原来的固定阈值 100 在这里会放过 50 这样“相对很大”的值，又会误伤长题批次里正常的 120。
    """

    def __init__(self, ratio: float = 10.0, abs_limit: float = 1000.0, window: int = 50, min_history: int = 20) -> None:
        self.ratio, self.abs_limit, self.min_history = ratio, abs_limit, min_history
        self.history: deque[float] = deque(maxlen=window)

    def median(self) -> float | None:
        if len(self.history) < self.min_history:
            return None
        s = sorted(self.history)
        n = len(s)
        return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2

    def should_skip(self, gnorm: float) -> bool:
        """返回 True 表示跳过；返回 False 时把这个梯度范数记入历史。"""
        if not math.isfinite(gnorm) or (self.abs_limit and gnorm > self.abs_limit):
            return True
        med = self.median()
        if self.ratio and med is not None and gnorm > self.ratio * med:
            return True
        self.history.append(gnorm)
        return False

    def state_dict(self) -> dict:
        return {"history": list(self.history)}

    def load_state_dict(self, st: dict) -> None:
        self.history.clear()
        self.history.extend(st.get("history", []))


def ordinal_smooth(target: list[float], eps: float) -> list[float]:
    """把 one-hot 的打分标签做序数平滑：正确等级留 1-eps，eps 平均分给相邻等级；已经是软标签的原样返回。

    例如 ordinal_smooth([0, 0, 1, 0, 0], 0.2) → [0, 0.1, 0.8, 0.1, 0]；
         ordinal_smooth([1, 0, 0, 0, 0], 0.2) → [0.8, 0.2, 0, 0, 0]（最边上只有一个邻居，eps 全给它）。
    """
    if eps <= 0 or len(target) < 2 or max(target) < 1.0 - 1e-9:
        return list(target)
    g = max(range(len(target)), key=target.__getitem__)
    nbrs = [j for j in (g - 1, g + 1) if 0 <= j < len(target)]
    out = [0.0] * len(target)
    out[g] = 1.0 - eps
    for j in nbrs:
        out[j] = eps / len(nbrs)
    return out


def step_mean_lengths(micro: list[list[int]], ends: list[bool] | None, lengths: list[int],
                      grad_accum: int = 1) -> list[float]:
    """每个优化器步的平均提示词长度（token）。固定题数分批时按 ends 划分步；旧分批方式每 grad_accum 个小批次一步。"""
    out, cur = [], []
    for j, mb in enumerate(micro):
        cur += [lengths[i] for i in mb]
        end = ends[j] if ends is not None else (j + 1) % grad_accum == 0
        if end:
            out.append(sum(cur) / len(cur))
            cur = []
    if cur:
        out.append(sum(cur) / len(cur))
    return out


def quantile_edges(values: list[float], n_buckets: int) -> list[float]:
    """把数值按分位数切成 n_buckets 档，返回 n_buckets-1 个分界点。例如 4 档 → 25%、50%、75% 分位数。"""
    s = sorted(values)
    return [s[min(len(s) - 1, int(len(s) * k / n_buckets))] for k in range(1, n_buckets)]


class BucketedSpikeGuard:
    """按“步的平均材料长度”分档的尖峰保护：每一档各自维护一个 SpikeGuard（各自的近期中位数）。

    为什么要分档（gemma_lora4 的排查结论，见 docs/skipped_lora4.md）：梯度范数随材料长度增长，
    长题集中的步梯度范数天然是短题步的几十倍。只用一个全局中位数时，中位数由占多数的短题步决定，
    长题步就会被误判成尖峰（lora4 被跳过的 17 步全是长题步，损失却很正常）。
    分档后长题步只和长题步比较；NaN / inf 和超过绝对上限的仍然一定跳过。

    举个例子：edges=[300, 600, 1200] → 4 档：<300、300–600、600–1200、≥1200 token。
        一个平均 2,500 token 的步落在第 4 档，只和第 4 档最近 50 个正常步的中位数比较。
    """

    def __init__(self, edges: list[float], ratio: float = 30.0, abs_limit: float = 2000.0, window: int = 50,
                 min_history: int = 10) -> None:
        self.edges = list(edges)
        self.guards = [SpikeGuard(ratio=ratio, abs_limit=abs_limit, window=window, min_history=min_history)
                       for _ in range(len(self.edges) + 1)]

    def bucket(self, mean_len: float) -> int:
        return sum(mean_len >= e for e in self.edges)

    def median(self, mean_len: float) -> float | None:
        return self.guards[self.bucket(mean_len)].median()

    def should_skip(self, gnorm: float, mean_len: float) -> bool:
        return self.guards[self.bucket(mean_len)].should_skip(gnorm)

    def state_dict(self) -> dict:
        return {"edges": self.edges, "guards": [g.state_dict() for g in self.guards]}

    def load_state_dict(self, st: dict) -> None:
        for g, s in zip(self.guards, st.get("guards", [])):
            g.load_state_dict(s)
