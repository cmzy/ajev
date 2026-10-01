"""把多道已编码的题（``Encoded``）补齐（padding）并拼成模型需要的张量。

====================================================================
基础概念
====================================================================

1. batch（批）
   GPU 擅长“同时算很多份一样的计算”。一次只喂 1 道题太浪费，所以把多道题打包成一个 batch
   一起算。batch 里题目的数量记作 B。

2. 张量（tensor）与形状
   张量就是多维数组（PyTorch 里的 torch.Tensor）。形状 [B, T] 表示 B 行 T 列的二维表格：
   - B：batch 里有几道题；
   - T：每道题的 token 序列长度（batch 内统一成最长那道题的长度）；
   - K：batch 内最大的选项数；
   - H：模型隐藏向量的维度（mmBERT-base 是 768），见 ajev/model/encoder.py。

3. padding（补齐）与 attention_mask
   张量要求每一行一样长，但每道题的 token 数不同。解决办法：把短的序列在末尾用 <pad> 补到
   一样长。补出来的位置不是真实内容，所以还要配一个 attention_mask：真实 token 记 1，补齐位记 0，
   模型据此在注意力计算中忽略补齐位（不让真实 token “看到”这些假内容）。
   选项也一样：每道题选项数不同，用 option_mask 标记哪些选项位是真实的。

训练（ajev.train.train 中的 collate）和推理（ajev.model.predictor）都用这里的函数，
保证两边的补齐方式一致。
"""

from __future__ import annotations

import torch

from ajev.model.encoding import Encoded


def collate(encs: list[Encoded], pad_id: int) -> dict[str, torch.Tensor]:
    """把一组 ``Encoded`` 补齐成一个 batch。

    Args:
        encs: B 道已编码的题。
        pad_id: tokenizer 的 <pad> token id，用于补齐 input_ids（mmBERT 中是 0）。

    Returns:
        字典，包含 4 个张量：
        - ``input_ids``: [B, T]，T 为 batch 内最长序列长度，不足处填 pad_id；
        - ``attention_mask``: [B, T]，真实 token 为 1，补齐位为 0；
        - ``marker_pos``: [B, K]，K 为 batch 内最大选项数，补齐的选项位填 0；
        - ``option_mask``: [B, K]，真实选项为 True，补齐的选项位为 False（模型据此把 logit 置为 -inf）。

    举个例子：2 道题，pad_id=0
        第 1 题：input_ids=[1, 3, 5, 3, 6, 2]，marker_pos=[1, 3]     （6 个 token，2 个选项）
        第 2 题：input_ids=[1, 3, 3, 3, 2]，   marker_pos=[1, 2, 3]  （5 个 token，3 个选项）
        → T=6，K=3
        input_ids      = [[1, 3, 5, 3, 6, 2],
                          [1, 3, 3, 3, 2, 0]]        ← 第 2 题末尾补了一个 0
        attention_mask = [[1, 1, 1, 1, 1, 1],
                          [1, 1, 1, 1, 1, 0]]        ← 补齐位为 0
        marker_pos     = [[1, 3, 0],
                          [1, 2, 3]]                 ← 第 1 题只有 2 个选项，第 3 位补 0
        option_mask    = [[True, True, False],
                          [True, True, True]]
    """
    b = len(encs)
    t = max(len(e.input_ids) for e in encs)
    k = max(len(e.marker_pos) for e in encs)
    # 先创建“全是补齐值”的张量，再把每道题的真实内容填进去。
    # torch.full((b, t), pad_id)：形状 [b, t]，每个元素都是 pad_id；dtype=torch.long 表示 64 位整数。
    input_ids = torch.full((b, t), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((b, t), dtype=torch.long)
    marker_pos = torch.zeros((b, k), dtype=torch.long)
    option_mask = torch.zeros((b, k), dtype=torch.bool)
    for i, e in enumerate(encs):
        # input_ids[i, :n] 表示第 i 行的前 n 个位置（张量切片，语法和列表切片类似）。
        input_ids[i, : len(e.input_ids)] = torch.tensor(e.input_ids)
        attention_mask[i, : len(e.input_ids)] = 1
        marker_pos[i, : len(e.marker_pos)] = torch.tensor(e.marker_pos)
        option_mask[i, : len(e.marker_pos)] = True
    return {"input_ids": input_ids, "attention_mask": attention_mask, "marker_pos": marker_pos,
            "option_mask": option_mask}


def pad_targets(targets: list[list[float]], k: int) -> torch.Tensor:
    """把每道题的目标分布（长度各不相同）补齐成 [B, k]，补齐位填 0（概率为 0）。

    举个例子：targets=[[1.0, 0.0], [0.2, 0.5, 0.3]]，k=3
        → [[1.0, 0.0, 0.0],
           [0.2, 0.5, 0.3]]
    补齐位的目标概率是 0，所以它们对损失没有贡献。
    """
    out = torch.zeros((len(targets), k))
    for i, t in enumerate(targets):
        out[i, : len(t)] = torch.tensor(t)
    return out


def length_sorted_batches(lengths: list[int], batch_size: int) -> list[list[int]]:
    """推理用的确定性分批：按长度排序后切块，让长度相近的题在一起，尽量减少补齐浪费。

    为什么要按长度排序：一个 batch 的长度 T 取决于其中最长的题。如果把一道 1000 token 的题
    和 31 道 50 token 的题放在一起，那 31 道短题都要补到 1000，绝大部分计算花在补齐位上。

    Args:
        lengths: 每道题的 token 长度。
        batch_size: 每批最多多少道题。

    Returns:
        每批包含的题目下标列表；调用方需要按下标把结果放回原顺序。

    举个例子：lengths=[50, 900, 60, 880]，batch_size=2
        排序后的下标：[0, 2, 3, 1]（长度 50, 60, 880, 900）
        → [[0, 2], [3, 1]]：两道短题一批，两道长题一批。
    """
    # sorted(..., key=lengths.__getitem__)：按 lengths[i] 的值对下标 i 排序。
    order = sorted(range(len(lengths)), key=lengths.__getitem__)
    return [order[i : i + batch_size] for i in range(0, len(order), batch_size)]
