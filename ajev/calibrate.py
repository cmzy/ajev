"""温度校准：在验证集上，为每种题型各拟合一个 softmax temperature（温度）。

在 AJev 流程中的位置：数据构建 → 训练 → **校准** → 评测。

================================================================
一、为什么需要校准
================================================================
“校准良好”指模型说 80% 把握的题，实际上也大约对 80%。神经网络训练完以后，输出的概率经常
“过于自信”（说 95% 把握，实际只对 80%），或者“不够自信”。Jev 的核心卖点就是概率可信，
所以训练完要再做一步修正，这一步就叫校准（calibration）。

================================================================
二、logits、softmax 与温度缩放的原理
================================================================
模型最后输出的不是概率，而是每个选项一个任意大小的“分数”，叫 logits。
softmax 函数把 logits 变成概率：p_i = e^(z_i) / Σ_j e^(z_j)（e 约等于 2.718）。

温度缩放（temperature scaling）是在 softmax 之前，先把所有 logits 除以同一个数 T：
    p = softmax(logits / T)

    * T > 1：logits 之间的差距被缩小，分布变得更“平”，模型显得不那么自信；
    * T < 1：差距被放大，分布变得更“尖”，模型显得更自信；
    * T = 1：不做任何改变。

举个例子，二选一的 logits = [2, 0]:
    T = 1   → softmax([2, 0])   ≈ [0.881, 0.119]
    T = 2   → softmax([1, 0])   ≈ [0.731, 0.269]   （变平：不那么自信了）
    T = 0.5 → softmax([4, 0])   ≈ [0.982, 0.018]   （变尖：更自信了）

关键性质：所有 logits 除以同一个正数，大小顺序不变，argmax 不变，所以
**校准不会改变准确率**，只会改善 Brier / KL / ECE 这些衡量概率质量的指标。

三种题型（noul 是/否、choice 单选、score 打分）的输出特点不同，所以各自拟合一个温度。

================================================================
三、怎么找到最好的温度
================================================================
目标：找到使验证集上“负对数似然（NLL）”最小的 T。
NLL 就是交叉熵：对每道题算 -Σ target_i × ln(p_i)，再取平均。
直观理解：正确答案的概率越高，-ln(p) 越小。例如目标 [1, 0]：
    预测 [0.8, 0.2]  → NLL = -ln 0.8  ≈ 0.22
    预测 [0.99, 0.01] → NLL = -ln 0.99 ≈ 0.01（答对时越自信越好）
    但如果答错了：预测 [0.01, 0.99] → NLL = -ln 0.01 ≈ 4.6（盲目自信会被重罚）
所以 NLL 最小的温度，正好平衡了“该自信时自信、该谨慎时谨慎”。

搜索方法分两步（见 ``fit_temperature``）：先用网格搜索粗略找到大致范围，再用黄金分割搜索精细定位。

用法：

    python -m ajev.calibrate --checkpoint runs/sft1/best --data data/build/val.jsonl data/build/val_typed.jsonl

拟合结果写入 checkpoint 目录下的 ``ajev_config.json``（字段 ``temperatures``），
之后 ``EncoderPredictor`` 加载这个模型时会自动读取并使用这些温度。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random

from ajev import metrics
from ajev.schema import Decision, read_jsonl


def _nll(logits: list[list[float]], targets: list[list[float]], t: float) -> float:
    """在温度 t 下，计算一批题目相对于目标分布的平均负对数似然（NLL，即交叉熵）。

    公式：对每道题，先算 log_softmax(z)_i = z_i - logsumexp(z)，其中 z = logits / t，
    logsumexp(z) = ln(Σ_j e^(z_j))；然后 NLL = -Σ_i target_i × log_softmax(z)_i。

    举个例子:
        logits = [[2, 0]]，targets = [[1, 0]]，t = 1
        → softmax ≈ [0.881, 0.119]，NLL = -ln(0.881) ≈ 0.127
    """
    total = 0.0
    for lg, tg in zip(logits, targets):
        z = [x / t for x in lg]  # 温度缩放
        # logsumexp 的“数值稳定”写法：先减去最大值 m 再求指数。
        # 原因：e^1000 会超出浮点数的表示范围（溢出）；减去最大值后最大的指数是 e^0 = 1，不会溢出。
        # 数学上 ln(Σ e^z) = m + ln(Σ e^(z - m))，结果完全相同。
        m = max(z)
        lse = m + math.log(sum(math.exp(x - m) for x in z))
        # target 为 0 的选项乘出来是 0，对交叉熵没有贡献，直接跳过。
        total -= sum(p * (x - lse) for p, x in zip(tg, z) if p > 0)
    return total / max(1, len(logits))  # max(1, ...) 防止空列表时除以 0


def fit_temperature(logits: list[list[float]], targets: list[list[float]]) -> float:
    """为一组 logits 找到使 NLL 最小的温度。

    分两步搜索:

    第 1 步：网格搜索（粗搜）
        在一串事先定好的候选值上逐个计算 NLL，挑出最好的那个。候选值是 0.05 × 1.15^i
        （i = 0..45），即 0.05, 0.0575, 0.066, …, 约 27，相邻两个候选值相差 15%。
        用“等比”而不是“等差”网格，是因为温度的影响是按比例的：0.1→0.2 和 5→10 的效果差不多大。

    第 2 步：黄金分割搜索（细搜）
        在网格最好点左右各一个网格步长的小区间 [best/1.15, best×1.15] 内精确定位。
        做法：在区间内取两个点 a、b（分别位于区间的 38.2% 和 61.8% 处），比较两点的 NLL，
        把较差那一侧的一段区间丢掉；每轮区间缩小到原来的 0.618 倍。黄金分割比的妙处在于：
        缩小后的新区间里，有一个试探点恰好就是上一轮留下的那个点，所以每轮只需要新算 1 次 NLL
        （每次算 NLL 都要遍历全部数据，省一半就是快一倍）。30 轮后区间宽度只剩
        原来的约 0.618^30 ≈ 0.0000005 倍，精度足够。
        优点：只需要计算函数值，不需要求导数；在这个小区间里 NLL 近似“单峰”（只有一个最低点），
        这种方法一定能收敛到它。

    参数:
        logits:  每道题的原始 logits（还没做温度缩放）。
        targets: 每道题的目标分布。
    返回:
        拟合出的温度 T。

    举个例子:
        假设模型输出的 logits 恰好是“完美校准时的 logits”的 3 倍（即过度自信），
        拟合出的温度会接近 3，除以 3 后概率就恢复准确（tests/test_model.py 中有对应测试）。
    """
    # 第 1 步：网格粗搜。
    grid = [0.05 * 1.15**i for i in range(46)]  # 候选温度 0.05 .. 约 27
    # min(候选列表, key=函数)：返回使函数值最小的那个候选值。
    best = min(grid, key=lambda t: _nll(logits, targets, t))
    # 第 2 步：在 best 左右各一个网格步长的区间内做黄金分割细搜。
    lo, hi = best / 1.15, best * 1.15
    g = (math.sqrt(5) - 1) / 2  # 黄金分割比 ≈ 0.618
    a, b = hi - g * (hi - lo), lo + g * (hi - lo)  # 区间内的两个试探点，a 在左、b 在右
    fa, fb = _nll(logits, targets, a), _nll(logits, targets, b)
    for _ in range(30):  # 用 “_” 作变量名表示“这个循环变量用不到”
        # 如果 a 处更小，最低点一定不在 b 的右边，于是把右端点收缩到 b；反之把左端点收缩到 a。
        # 收缩后，旧的 a（或 b）正好落在新区间的黄金分割点上，直接复用它和它的 NLL，只需新算一个点。
        if fa < fb:
            hi, b, fb = b, a, fa
            a = hi - g * (hi - lo)
            fa = _nll(logits, targets, a)
        else:
            lo, a, fa = a, b, fb
            b = lo + g * (hi - lo)
            fb = _nll(logits, targets, b)
    return (lo + hi) / 2  # 取最终小区间的中点作为结果


def fit(decisions: list[Decision], logits: list[list[float]]) -> dict[str, float]:
    """按题型分组，分别拟合温度。

    返回:
        ``{"noul": T1, "choice": T2, "score": T3}``，保留 4 位小数。
        如果数据里没有某种题型，结果中就不包含这个键；预测时缺少的题型默认使用温度 1.0（即不缩放）。

    举个例子（数字仅示意）: {"noul": 1.21, "choice": 0.93, "score": 1.08}
    """
    temps = {}
    for t in ("noul", "choice", "score"):
        idx = [i for i, d in enumerate(decisions) if d.type == t]  # 这种题型的所有题目下标
        if idx:  # 列表非空才拟合
            temps[t] = round(fit_temperature([logits[i] for i in idx], [decisions[i].target for i in idx]), 4)
    return temps


def main(argv: list[str] | None = None) -> None:
    """命令行入口。

    执行步骤:
        第 1 步：加载 checkpoint（训练好的模型），读入验证数据；
        第 2 步：让模型对验证数据输出原始 logits（不使用任何已有温度）；
        第 3 步：诚实地评估校准效果：随机留出一部分题（默认 30%）不参与拟合，只用剩下的题拟合温度，
                再在留出的题上比较校准前后的指标。如果在拟合用的同一批题上比较，结果一定“变好”，
                没有参考价值（这叫“样本内”评估，会过于乐观）；
        第 4 步：用全部题重新拟合最终温度（数据越多越稳），这才是写进配置的温度；
        第 5 步：把温度连同“这组温度是在第几步的模型上拟合的”一起写回 checkpoint 的配置文件。
                如果之后模型又被继续训练、权重变了，推理时会提示温度已过期，需要重新校准。
    """
    # 延迟导入 torch 相关模块：这样本文件里的纯 Python 函数（比如 fit_temperature）
    # 在没有安装 torch 的环境中也能被导入和测试。
    from ajev.model.encoder import AJEV_CONFIG
    from ajev.model.predictor import EncoderPredictor, softmax

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", help="mmBERT checkpoint directory")
    ap.add_argument("--lm-model", help="base LLM for the LLM route, e.g. google/gemma-4-12B-it")
    ap.add_argument("--lm-adapter", help="LoRA adapter directory (temperatures are written into it)")
    ap.add_argument("--data", nargs="+", required=True)  # nargs="+"：可以接收一个或多个文件路径
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--report-frac", type=float, default=0.3,
                    help="fraction of decisions held out to report before/after metrics honestly")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    # 第 1 步：把所有验证文件里的题合并成一个列表（嵌套列表推导式：先遍历文件，再遍历文件里的题）。
    decisions = [d for p in args.data for d in read_jsonl(p)]
    # 第 2 步：temperatures={} 表示强制不使用 checkpoint 里已有的温度，这样拿到的是原始 logits。
    if args.lm_adapter:
        # 大模型路线：温度写进 LoRA 适配器目录的 ajev_lm_config.json。
        from ajev.lm.predictor import LM_CONFIG, LMPredictor, read_lm_config

        pred = LMPredictor(args.lm_model or read_lm_config(args.lm_adapter)["base_model"], adapter=args.lm_adapter,
                           temperatures={})
        target_dir, config_name = args.lm_adapter, LM_CONFIG
    elif args.checkpoint:
        pred = EncoderPredictor(args.checkpoint, batch_size=args.batch_size, temperatures={})
        target_dir, config_name = args.checkpoint, AJEV_CONFIG
    else:
        raise SystemExit("need --checkpoint (mmBERT) or --lm-adapter (LLM)")
    logits = pred.predict_logits(decisions)
    # 第 3 步：随机打乱下标后切成“拟合部分”和“留出部分”，用拟合部分的温度在留出部分上评估。
    idx = list(range(len(decisions)))
    random.Random(args.seed).shuffle(idx)
    n_held = int(len(idx) * args.report_frac)
    held, fit_idx = idx[:n_held], idx[n_held:]
    if held:
        t_fit = fit([decisions[i] for i in fit_idx], [logits[i] for i in fit_idx])
        hd, hl = [decisions[i] for i in held], [logits[i] for i in held]
        before = metrics.compute(hd, [softmax(lg) for lg in hl])
        after = metrics.compute(hd, [softmax(lg, t_fit.get(d.type, 1.0)) for d, lg in zip(hd, hl)])
        print(f"held-out check ({len(held)} decisions not used for fitting): temperatures {t_fit}")
        # 预期：accuracy 不变（温度不改变排序），brier / kl / ece 下降或持平；如果反而变差，说明温度过拟合了。
        for k in ("accuracy", "brier", "kl", "ece"):
            print(f"{k:10s} before {before[k]:.4f}  after {after[k]:.4f}")

    # 第 4 步：用全部数据拟合最终温度。
    temps = fit(decisions, logits)
    print(f"final temperatures (fitted on all {len(decisions)} decisions): {temps}")

    # 第 5 步：在原有配置（编码参数、基座模型名等）的基础上，只更新 temperatures 字段，不覆盖其他内容。
    path = os.path.join(target_dir, config_name)  # os.path.join 按操作系统规则拼接路径
    cfg = json.load(open(path)) if os.path.exists(path) else {}
    cfg["temperatures"] = temps
    # 记下温度是在哪一步的权重上拟合的（训练保存 checkpoint 时会写入 step）。
    cfg["calibrated_at_step"] = cfg.get("step")
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
