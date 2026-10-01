"""评测命令行工具：在一个 Decision JSONL 文件上，评测某个预测器（或一份已保存的预测结果）。

这是 AJev 流程的最后一环：数据构建 → 训练 → 校准 → **评测**。

它会打印一张表格，包括：
    * overall   —— 所有题目的总体指标；
    * by_type   —— 按题型（noul 是/否、choice 单选、score 打分）分组的指标；
    * by_lang   —— 按语言（en / zh）分组的指标；
    * by_source —— 按数据来源（每个数据集 / 业务流程）分组的指标；
    * 可选的“选项打乱翻转率”（衡量模型是否偏爱某个位置的选项）。
各项指标的含义和算例见 ``ajev/metrics.py`` 的说明。

用法示例（在仓库根目录运行）：

    # 用“先验频率”基线评测主测试集（需要提供训练集来统计频率）
    python -m ajev.eval.evaluate --data data/build/test_typed.jsonl --predictor prior --train data/build/train.jsonl
    # 读取之前保存的预测结果文件来评测
    python -m ajev.eval.evaluate --data data/build/test_public.jsonl --predictions runs/x/preds.jsonl
    # 评测训练好的模型，同时测量选项打乱翻转率
    python -m ajev.eval.evaluate --data data/build/test_typed.jsonl --predictor model --checkpoint runs/sft1/best --shuffle-check

（``python -m 包名.模块名`` 表示把这个模块当作程序运行，会执行文件末尾 ``if __name__ == "__main__":`` 下的代码。）

预测结果文件格式：每行一个 JSON 对象 ``{"id": ..., "probs": [...]}``，
probs 的选项顺序必须与数据文件中该题的选项顺序一致。
"""

from __future__ import annotations

import argparse  # Python 标准库，用于解析命令行参数（如 --data xxx）
import json
import random

from ajev import metrics
from ajev.predictors import Predictor, PriorPredictor, RandomPredictor, UniformPredictor
from ajev.schema import Decision, read_jsonl

# COLS 是指标字典中的键名，HEADERS 是打印表格时显示的列标题（把几个长名字缩短，避免列挤在一起）。
# 两者按位置一一对应：COLS[1] = "accuracy" 显示为 HEADERS[1] = "acc"。
COLS = ("n", "accuracy", "chance_acc", "brier", "kl", "ece", "mean_confidence")
HEADERS = ("n", "acc", "chance_acc", "brier", "kl", "ece", "jev_conf")


def evaluate(decisions: list[Decision], preds: list[list[float]]) -> dict:
    """计算完整的评测报告：总体指标 + 三种分组方式的指标。

    返回:
        ``{"overall": 总体指标, "by_type": {...}, "by_lang": {...}, "by_source": {...}}``。

    举个例子（数字仅示意）:
        {"overall": {"n": 2000, "accuracy": 0.72, ...},
         "by_type": {"choice": {...}, "noul": {...}, "score": {...}},
         "by_lang": {"en": {...}},
         "by_source": {"typed_decisions/customer_service": {...}, ...}}
    """
    return {
        "overall": metrics.compute(decisions, preds),
        # lambda d: d.type 是一个“匿名小函数”，输入一道题、返回它的题型，用作分组依据。
        "by_type": metrics.breakdown(decisions, preds, lambda d: d.type),
        "by_lang": metrics.breakdown(decisions, preds, lambda d: d.lang),
        "by_source": metrics.breakdown(decisions, preds, lambda d: d.source),
    }


def shuffle_check(predictor: Predictor, decisions: list[Decision], preds: list[list[float]], seed: int = 0) -> float:
    """测量选项打乱翻转率。

    步骤:
        第 1 步：把每道题的选项随机打乱，得到一组新题（选项内容不变，只是顺序变了）；
        第 2 步：让预测器对新题重新预测；
        第 3 步：比较打乱前后模型选中的选项名，统计变化的比例。

    因为要重新调用预测器，所以只有使用 ``--predictor`` 时才能计算；直接读预测文件时无法计算。
    随机种子是固定的，保证不同模型是在同一组打乱方式上比较的，结果公平。

    举个例子: 1000 道 choice/noul 题中有 50 道打乱后答案变了 → 返回 0.05。
    """
    rng = random.Random(seed)
    shuffled = [d.shuffled(rng) for d in decisions]
    return metrics.flip_rate(preds, decisions, shuffled, predictor.predict(shuffled))


def format_table(report: dict) -> str:
    """把评测报告排版成终端里容易阅读的“定宽”文本表格。

    每个分组方式占一节（overall / by_type / by_lang / by_source），每行一个组；
    组名最多显示 34 个字符，数值保留 4 位小数。

    输出示意:
        ## overall
                                                 n         acc  chance_acc  ...
        all                                   2000      0.4785      0.2359  ...
    """
    lines = []
    for section in ("overall", "by_type", "by_lang", "by_source"):
        # overall 只有一组指标，把它包装成 {"all": 指标}，就能和其他分组共用同一套打印逻辑。
        rows = {"all": report[section]} if section == "overall" else report[section]
        lines.append(f"\n## {section}")
        # 格式说明：{'':34s} 输出 34 个字符宽的空白；{h:>12s} 表示右对齐、占 12 个字符宽。
        lines.append(f"{'':34s}" + "".join(f"{h:>12s}" for h in HEADERS))
        for name, m in rows.items():
            # n（题数）是整数，用 {:>12d} 显示；其他指标是小数，用 {:>12.4f}（保留 4 位小数）显示。
            # m.get(c, nan)：如果某个指标不存在，就显示 nan。
            cells = "".join(f"{m.get(c, float('nan')):>12.4f}" if c != "n" else f"{m['n']:>12d}" for c in COLS)
            lines.append(f"{name[:34]:34s}{cells}")  # name[:34] 截取前 34 个字符
    if "flip_rate" in report:
        lines.append(f"\noption-shuffle flip rate: {report['flip_rate']:.4f}")
    return "\n".join(lines)  # 用换行符把所有行连接成一个字符串


def _load_predictions(path: str, decisions: list[Decision]) -> list[list[float]]:
    """读取预测结果文件，并按照数据文件中的题目顺序重新排列。

    按 id 匹配题目，所以预测文件里的行顺序无所谓。只要有任何一道题缺少预测，就报错退出——
    因为在不完整的结果上算出来的指标会产生误导。

    举个例子:
        预测文件有 {"id": "b", ...}, {"id": "a", ...}，数据文件的顺序是 a, b
        → 返回 [a 的 probs, b 的 probs]
    """
    with open(path, encoding="utf-8") as f:
        # map(json.loads, f) 把文件的每一行解析成字典；再组成 {id: probs} 的查找表。
        by_id = {r["id"]: r["probs"] for r in map(json.loads, f) if r}
    missing = [d.id for d in decisions if d.id not in by_id]
    if missing:
        # SystemExit 会让程序打印这条信息后退出。
        raise SystemExit(f"{len(missing)} decisions have no prediction, e.g. {missing[:3]}")
    return [by_id[d.id] for d in decisions]


def main(argv: list[str] | None = None) -> None:
    """命令行入口。

    预测来源二选一（互斥参数，不能同时给）:
        * ``--predictor``：现场预测。
            - uniform / random：最简单的基线；
            - prior：先验频率基线，需要同时给 ``--train``；
            - model：训练好的模型，需要同时给 ``--checkpoint``（会自动读取其中校准好的温度）。
        * ``--predictions``：直接读取已经保存好的预测结果文件。

    执行步骤:
        第 1 步：解析命令行参数，读取数据文件；
        第 2 步：得到预测（现场预测或读文件）；
        第 3 步：可选地把预测保存下来；
        第 4 步：计算评测报告，可选地测量翻转率；
        第 5 步：打印表格，可选地把完整报告保存为 JSON。
    """
    # 第 1 步：定义并解析命令行参数。
    # description=__doc__ 把本文件开头的说明文字用作 --help 的帮助信息。
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    src = ap.add_mutually_exclusive_group(required=True)  # 互斥参数组：下面两个参数必须且只能给一个
    src.add_argument("--predictor", choices=["uniform", "random", "prior", "model", "lm"])
    src.add_argument("--predictions")
    ap.add_argument("--train", help="train JSONL (needed by --predictor prior)")
    ap.add_argument("--checkpoint", help="model directory (needed by --predictor model)")
    ap.add_argument("--lm-model", help="HF id or path of a chat LLM (needed by --predictor lm), "
                                       "e.g. google/gemma-4-12B-it")
    ap.add_argument("--save-predictions", help="write {id, probs} JSONL here")
    ap.add_argument("--shuffle-check", action="store_true", help="also measure option-order flip rate")
    ap.add_argument("--out", help="write the full report as JSON here")
    args = ap.parse_args(argv)  # argv 为 None 时自动读取真实的命令行参数；测试时可以传入列表

    decisions = read_jsonl(args.data)
    # 第 2 步：得到预测。
    predictor: Predictor | None = None
    if args.predictions:
        preds = _load_predictions(args.predictions, decisions)
    else:
        if args.predictor == "prior":
            if not args.train:
                raise SystemExit("--predictor prior needs --train")
            predictor = PriorPredictor(read_jsonl(args.train))
        elif args.predictor == "model":
            if not args.checkpoint:
                raise SystemExit("--predictor model needs --checkpoint")
            # 延迟导入：只有用到模型时才导入 torch / transformers 相关代码。
            # 这样即使本地环境没装 torch，也能正常运行基线评测。
            from ajev.model.predictor import EncoderPredictor

            predictor = EncoderPredictor(args.checkpoint)
        elif args.predictor == "lm":
            # 大语言模型零样本：读字母选项的 logits（见 ajev/lm/predictor.py）。
            if not args.lm_model:
                raise SystemExit("--predictor lm needs --lm-model")
            from ajev.lm.predictor import LMPredictor

            predictor = LMPredictor(args.lm_model)
        else:
            predictor = UniformPredictor() if args.predictor == "uniform" else RandomPredictor()
        preds = predictor.predict(decisions)

    # 第 3 步：保存预测结果，之后不用重新跑模型就能再次评测或做错题分析。
    if args.save_predictions:
        with open(args.save_predictions, "w", encoding="utf-8") as f:
            for d, p in zip(decisions, preds):
                f.write(json.dumps({"id": d.id, "probs": [round(x, 6) for x in p]}) + "\n")

    # 第 4 步：计算评测报告。
    report = evaluate(decisions, preds)
    if args.shuffle_check and predictor is not None:
        report["flip_rate"] = shuffle_check(predictor, decisions, preds)
    # 第 5 步：输出结果。
    print(format_table(report))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)  # indent=2：缩进 2 格，便于阅读


# 只有直接运行这个文件（python -m ajev.eval.evaluate）时才执行 main()；
# 被其他文件 import 时不会执行，这样别的代码可以复用上面的函数。
if __name__ == "__main__":
    main()
