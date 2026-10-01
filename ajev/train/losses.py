"""训练目标（损失函数）。

====================================================================
基础概念：什么是损失函数
====================================================================

训练神经网络 = 不断调整参数，让“损失”变小。损失函数是一个打分规则：模型预测得越差，
损失越大。训练时算出损失后，PyTorch 会自动求出“每个参数往哪个方向改一点能让损失变小”
（这就是梯度，由 loss.backward() 计算），优化器再按梯度更新参数。

AJev 用了三种损失，各管一件事：
- soft_ce（软标签交叉熵）：让预测分布接近目标分布 —— 主损失，决定“答得对不对”；
- rps（排序概率评分）：只用于 score 打分题，让“差一级”比“差三级”罚得轻；
- symmetric_kl（对称 KL）：同一道题换个选项顺序，答案不应该变 —— 压制位置偏置。
另外 smooth（标签平滑）对目标做预处理，unpermute 把打乱过的选项顺序还原。

====================================================================
本文件所有函数的输入约定
====================================================================

- ``logits``: [B, K]，每道题（B 道）每个选项（最多 K 个）的 logit，补齐出来的选项位为 -inf；
- ``target``: [B, K]，目标概率分布（soft label，软标签），补齐位为 0；
- ``option_mask``: [B, K]，True 表示真实选项。
返回值都是 [B]，即每道题一个损失值，由调用方按题型加权后再求平均。

关于 -inf 与梯度：补齐位的 logit 是 -inf，log_softmax 后仍是 -inf。如果直接拿去和 0 相乘，
0 × (-inf) 在浮点运算里等于 NaN（not a number），会把整个损失污染成 NaN、训练直接崩掉。
所以这里统一先用 ``masked_fill(~option_mask, 0.0)`` 把这些位置换成 0 再参与乘法；
被 masked_fill 覆盖的位置梯度为 0，因此反向传播也不会产生 NaN（tests/test_model.py 有对应测试）。
"""

from __future__ import annotations

import torch
# F 是 PyTorch 中“无参数函数”的集合，例如 softmax、log_softmax。
import torch.nn.functional as F


def smooth(target: torch.Tensor, option_mask: torch.Tensor, eps: float) -> torch.Tensor:
    """标签平滑（label smoothing），只在真实选项上进行：t' = (1 - eps) * t + eps / K。

    是什么：把“100% 确定”的硬标签稍微软化，分一点点概率给其他选项。
    为什么需要：如果目标是 [1, 0, 0]，模型为了让损失趋近 0，会拼命把正确选项的 logit 推向无穷大，
    结果对什么都说“99.99% 确定”，概率就不可信了（过度自信）。平滑后目标不再是绝对的 1，
    模型就没有动力无限自信，有利于概率校准。

    注意 K 是每道题自己的选项数（不是 batch 的最大选项数），补齐位乘以 option_mask 后保持为 0。

    举个例子：eps=0.05，一道 3 个选项的题，target=[1, 0, 0]
        t' = 0.95 × [1, 0, 0] + 0.05 / 3 ≈ [0.9667, 0.0167, 0.0167]，加起来仍然是 1。
    """
    # sum(-1, keepdim=True)：在最后一维求和并保留这一维，[B, K] → [B, 1]，便于和 [B, K] 做广播运算。
    # clamp(min=1)：防止除以 0（理论上每道题至少有 2 个选项）。
    k = option_mask.sum(-1, keepdim=True).clamp(min=1)
    return ((1 - eps) * target + eps / k) * option_mask


def soft_ce(logits: torch.Tensor, target: torch.Tensor, option_mask: torch.Tensor) -> torch.Tensor:
    """软标签交叉熵：CE = -Σ_k t_k · log p_k（每道题一个值）。

    直观含义：看模型给“正确答案”分配了多少概率，分配得越少，罚得越重（而且是对数级地重）。
    举个例子：一道 3 选项题，target=[1, 0, 0]
        模型预测 p=[0.7, 0.2, 0.1] → CE = -log(0.7) ≈ 0.357
        模型预测 p=[0.2, 0.7, 0.1] → CE = -log(0.2) ≈ 1.609（答错了，损失大得多）
        若预测 p=[0.001, ...]       → CE ≈ 6.9（非常自信地答错，罚得最重）

    target 可以是 one-hot（硬标签），也可以是多人标注 / 教师模型给出的分布（软标签）。
    例如 typed-decisions 中 3 个标注者有 2 个选 A、1 个选 B，target 就是 [0.67, 0.33, 0]，
    此时模型最好的策略就是也输出 [0.67, 0.33, 0]——这正是我们想要的“可信概率”。
    数学上，最小化它等价于最小化 KL(target || p)，即让预测分布逼近目标分布。

    为什么用 log_softmax 而不是先 softmax 再 log：两者数学上相同，但 log_softmax 一步算完，
    数值更稳定（不会出现 log(0) = -inf 这种由于精度造成的问题）。
    """
    # 补齐位的 log 概率是 -inf，置 0 后再与 target（补齐位也为 0）相乘，避免 0 × (-inf) = NaN。
    logp = F.log_softmax(logits, dim=-1).masked_fill(~option_mask, 0.0)
    return -(target * logp).sum(-1)


def rps(logits: torch.Tensor, target: torch.Tensor, option_mask: torch.Tensor) -> torch.Tensor:
    """Ranked Probability Score（排序概率评分），用于 score（打分题）这类有序等级，并除以 K-1 归一化。

    RPS = Σ_k (CDF_p(k) - CDF_t(k))² / (K - 1)，CDF 为累积分布（“小于等于第 k 级”的概率之和）。

    为什么需要：交叉熵只关心“正确等级拿了多少概率”，不关心错的离多远。但打分题是有序的：
    真实紧急程度是 3 级，猜 2 级比猜 0 级好得多。RPS 通过比较累积分布体现了“距离”。

    举个例子：4 个等级（0~3），真实是 3 级，target=[0, 0, 0, 1]，CDF_t=[0, 0, 0, 1]
        预测全押 2 级 p=[0, 0, 1, 0]：CDF_p=[0, 0, 1, 1]，差的平方和 = 1，RPS = 1/3 ≈ 0.333
        预测全押 0 级 p=[1, 0, 0, 0]：CDF_p=[1, 1, 1, 1]，差的平方和 = 3，RPS = 3/3 = 1.0
        而这两种预测的交叉熵是一样的（正确等级的概率都是 0），只有 RPS 能区分“差一点”和“差很远”。
    """
    p = F.softmax(logits, dim=-1).masked_fill(~option_mask, 0.0)
    # cumsum(-1)：沿最后一维做累加，例如 [0.1, 0.2, 0.7] → [0.1, 0.3, 1.0]，就是累积分布。
    # 补齐位之后累积分布不再变化，乘以 option_mask 后不计入。
    diff = (p.cumsum(-1) - target.cumsum(-1)) * option_mask
    k = option_mask.sum(-1).clamp(min=2)
    return (diff**2).sum(-1) / (k - 1)


def symmetric_kl(logits_a: torch.Tensor, logits_b: torch.Tensor, option_mask: torch.Tensor) -> torch.Tensor:
    """两个视图（view）分布之间的对称 KL：0.5 · [KL(pa || pb) + KL(pb || pa)]。

    KL 散度（Kullback–Leibler divergence）衡量两个概率分布有多不一样：完全相同时为 0，差得越多越大。
    KL(p || q) = Σ p_k · log(p_k / q_k)。它不对称（KL(p||q) ≠ KL(q||p)），所以取两个方向的平均。

    用途是“选项打乱一致性”：同一道题把选项顺序打乱两次（视图 A、B），模型给出的分布应该相同。
    很多模型有位置偏置，比如不管内容如何都偏爱第一个选项；这一项损失直接惩罚这种行为。
    调用前两组 logits 必须已经用 ``unpermute`` 对齐到同一个（原始）选项顺序。

    举个例子：同一道 2 选项题
        视图 A 预测 pa=[0.8, 0.2]，视图 B 预测 pb=[0.5, 0.5]（换了顺序后模型变得犹豫了）
        KL(pa||pb) ≈ 0.193，KL(pb||pa) ≈ 0.223，对称 KL ≈ 0.208
        若两个视图预测完全相同，对称 KL = 0。
    """
    la = F.log_softmax(logits_a, dim=-1).masked_fill(~option_mask, 0.0)
    lb = F.log_softmax(logits_b, dim=-1).masked_fill(~option_mask, 0.0)
    # exp(log p) = p；乘 option_mask 保证补齐位概率为 0。
    pa, pb = la.exp() * option_mask, lb.exp() * option_mask
    # pa * (la - lb) 就是 p_a · log(p_a / p_b)，求和得到 KL(pa || pb)；另一项同理。
    return 0.5 * ((pa * (la - lb)).sum(-1) + (pb * (lb - la)).sum(-1))


def unpermute(logits: torch.Tensor, perm: torch.Tensor, option_mask: torch.Tensor) -> torch.Tensor:
    """把某个视图（选项被打乱过）的 logits 映射回原始选项顺序：out[:, perm[j]] = logits[:, j]。

    映射方向：视图中第 j 个选项就是原始的第 perm[j] 个选项。训练时 target 是按原始顺序存的，
    所以要先把 logits 放回原始位置，才能与 target 计算损失、与另一个视图计算一致性。
    perm 的补齐位映射到自身（见 train.make_collate），所以 -inf 仍然留在补齐位上。

    torch.scatter 是 gather 的“反操作”：gather 是“按下标取出来”，scatter 是“按下标放进去”。
    out.scatter(1, perm, src) 的含义是：对每一行，把 src[j] 放到 out 的第 perm[j] 列。

    举个例子：原始选项是 [退款, 物流, 账户]，某个视图打乱成 [物流, 账户, 退款]
        即视图第 0 个 = 原始第 1 个，视图第 1 个 = 原始第 2 个，视图第 2 个 = 原始第 0 个，perm=[1, 2, 0]
        模型对视图输出 logits=[0.5, -1.0, 2.0]（物流 0.5，账户 -1.0，退款 2.0）
        out[1]=0.5，out[2]=-1.0，out[0]=2.0 → out=[2.0, 0.5, -1.0]，又回到了 [退款, 物流, 账户] 的顺序。
    """
    # 先建一个全是 -inf 的张量，再把视图的 logits 按 perm 放进去。
    out = torch.full_like(logits, float("-inf"))
    src = logits.masked_fill(~option_mask, float("-inf"))
    return out.scatter(1, perm, src)
