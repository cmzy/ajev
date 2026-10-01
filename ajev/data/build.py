"""构建训练 / 验证 / 测试数据集，输出为 JSONL 文件。

这是 AJev 流程（数据构建 → 训练 → 校准 → 评测）的第一步：把 ``ajev.data.sources``
中注册的公开数据集，以及 typed-decisions 业务决策数据集，统一转换成 Decision，
做采样、去重、防泄漏、打乱选项之后写到磁盘，供训练脚本和评测脚本直接读取。

用法示例::

    python -m ajev.data.build --out data/build
    python -m ajev.data.build --out data/smoke --sources boolq,tnews --train-cap 50 --eval-cap 20

输出文件：
    train.jsonl         各公开数据集的 train split（每个数据源有条数上限，中文源上限更高）
                        + typed-decisions train（扣除留作 val_typed 的部分）
    val.jsonl           各公开数据集评测 split 的前 --eval-cap 条（用于挑选 checkpoint、拟合温度）
    test_public.jsonl   各公开数据集评测 split 的接下来 --eval-cap 条（按题型 / 语言出评测报告）
    val_typed.jsonl     typed-decisions train 中按 state 整组留出的 10%（业务场景内的模型选择）
    test_typed.jsonl    typed-decisions test，共 2,000 道题（主指标，可与 Jev / Laya / Verdict 直接对比）
    stats.json          按文件 / 数据源 / 题型 / 语言统计的条数，以及构建参数

关键设计：
    * val 与 test_public 来自同一个评测 split 的不同样本，互不重叠；
    * 选模型只看 val / val_typed，从不看 test_typed，避免在主测试集上“调参作弊”；
    * 防泄漏：训练集中凡是 state（材料）与任何验证 / 测试样本相同的题一律丢弃；
    * 构建结果完全由 --seed 决定，可复现。

------------------------------------------------------------------------------
给初学者的背景知识
------------------------------------------------------------------------------

【JSONL 格式】
JSONL（JSON Lines）就是“每行一个 JSON 对象”的文本文件，例如::

    {"id": "boolq/0", "type": "noul", "state": "...", "target": [1.0, 0.0], ...}
    {"id": "tnews/3", "type": "choice", "state": "...", "target": [0.0, 1.0, ...], ...}

好处：可以一行一行地读写，不必把整个文件一次装进内存；用 head、wc -l 等命令也能直接查看和计数。

【为什么要分 train / val / test 三份】
train 用来学习；val 用来在训练中途“模拟考试”，挑最好的 checkpoint 和调参数；test 只在最后
做一次“正式考试”。如果按 test 的分数反复挑模型，相当于间接在测试集上调参，分数会偏乐观。
因此 test_typed（主测试集）在训练中完全不看，只在最终评测时使用。

【数据泄漏 / 污染（contamination）】
指测试题（或与测试题高度相似的内容）混进了训练集。模型可能只是“背下了答案”，测试分数会虚高，
到了真实场景就露馅。例如 BoolQ 的 train 和 validation 有时会引用同一段维基百科段落，
如果训练集里出现过这段段落，对应的测试题就不再是“没见过的新题”。所以下面会按 state 去除重叠。

【去重】
同一道题在训练集里出现多次，会让模型过度关注它（相当于给它加了权重），也浪费训练时间。
这里用哈希（hash）给每道题生成一个“指纹”，指纹相同就认为是重复题，只保留第一次出现的那条。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from collections import Counter
from itertools import islice

from ajev.data.sources import SOURCES
from ajev.data.typed_decisions import iter_typed_decisions
from ajev.schema import Decision, write_jsonl


def _key(d: Decision) -> str:
    """去重键：(state, instructions) 的 SHA-1。

    同一段材料配不同问题算不同的题（例如 typed-decisions 每个 state 有 5 个问题），
    只有材料和问题都完全相同才视为重复。用 ``\\x00`` 作分隔符，避免拼接产生歧义。

    什么是 SHA-1 哈希：把任意长度的文本变成一个固定长度（40 个十六进制字符）的“指纹”。
    内容完全相同 → 指纹一定相同；内容不同 → 指纹几乎不可能相同。用指纹做集合查找，
    比直接存放可能长达几千字的原文更省内存。

    为什么需要分隔符：若直接拼接，("ab", "c") 和 ("a", "bc") 都会得到 "abc"，被误判为重复；
    中间插入一个正常文本里不会出现的字符 \\x00，二者就变成 "ab\\x00c" 和 "a\\x00bc"，不再冲突。

    举个例子：state="今天天气很好", instructions="这句话是正面情绪吗？"
        → 返回类似 "3f2a…（共 40 个字符）" 的字符串。
    """
    return hashlib.sha1(f"{d.state}\x00{d.instructions}".encode()).hexdigest()


def _state_key(d: Decision) -> str:
    """防泄漏键：只看 state 的 SHA-1。

    只要训练题的材料出现在任何验证 / 测试题里，就算问题不同也视为泄漏——
    模型可能会记住这段材料的内容，导致测试分数虚高。
    """
    return hashlib.sha1(d.state.encode()).hexdigest()


def _stats(ds: list[Decision]) -> dict:
    """统计一个文件的总条数，以及按数据源 / 题型 / 语言的分布，写入 stats.json。

    ``Counter`` 是 Python 标准库里的计数器：给它一串元素，它会数出每个元素出现了几次，
    例如 Counter(["en", "zh", "en"]) → {"en": 2, "zh": 1}；``most_common()`` 按次数从多到少排序。

    举个例子：输入 3 道题（boolq 的 noul 英文题 ×2、tnews 的 choice 中文题 ×1）
        → {"total": 3, "by_source": {"boolq": 2, "tnews": 1},
           "by_type": {"noul": 2, "choice": 1}, "by_lang": {"en": 2, "zh": 1}}
    """
    return {
        "total": len(ds),
        "by_source": dict(Counter(d.source for d in ds).most_common()),
        "by_type": dict(Counter(d.type for d in ds)),
        "by_lang": dict(Counter(d.lang for d in ds)),
    }


def main(argv: list[str] | None = None) -> None:
    """命令行入口：解析参数 → 逐个数据源转换 → 加入 typed-decisions → 防泄漏与去重 → 打乱 → 写文件。

    参数 ``argv`` 为 None 时读取真实命令行；测试中可以直接传入参数列表。

    整体步骤：
        第 1 步：解析命令行参数，检查数据源名是否合法；
        第 2 步：逐个公开数据源，生成训练题，以及互不重叠的 val / test_public；
        第 3 步：加载 typed-decisions，按 state 整组留出 val_typed，test split 作为 test_typed；
        第 4 步：防泄漏 + 去重，清洗训练集；
        第 5 步：打乱选项顺序和训练集顺序；
        第 6 步：写出 JSONL 文件和 stats.json，并打印统计信息。

    argparse 是 Python 标准库的命令行解析工具：``ap.add_argument("--train-cap", type=int, default=3000)``
    表示支持 ``--train-cap 50`` 这样的参数，不写时默认 3000；解析后用 ``args.train_cap`` 取值
    （参数名中的减号会自动变成下划线）。
    """
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/build")
    ap.add_argument("--sources", default="all", help="comma-separated source names, or 'all'")
    ap.add_argument("--train-cap", type=int, default=3000, help="max train decisions per public source")
    ap.add_argument("--eval-cap", type=int, default=300, help="max decisions per source in val and in test_public")
    ap.add_argument("--zh-cap-mult", type=float, default=2.0,
                    help="multiply --train-cap for Chinese sources (target: 30-50%% Chinese in train)")
    ap.add_argument("--zh-instr-prob", type=float, default=0.2,
                    help="probability an English source gets a Chinese instruction (cross-lingual)")
    ap.add_argument("--no-typed-decisions", action="store_true")
    ap.add_argument("--typed-val-frac", type=float, default=0.1,
                    help="fraction of typed-decisions train states held out as val_typed")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    # ---- 第 1 步 ----
    # 解析要使用的数据源；拼错名字时直接报错并列出全部可用数据源。
    names = list(SOURCES) if args.sources == "all" else args.sources.split(",")
    unknown = [n for n in names if n not in SOURCES]
    if unknown:
        sys.exit(f"unknown sources: {unknown}; available: {list(SOURCES)}")

    # 这个 rng 只用于最后的选项打乱和训练集整体打乱；各数据源内部有自己独立的随机数生成器。
    rng = random.Random(args.seed)
    train: list[Decision] = []
    val: list[Decision] = []
    test_public: list[Decision] = []
    val_typed: list[Decision] = []
    test_typed: list[Decision] = []

    # ---- 第 2 步：公开数据源 ----
    for name in names:
        src = SOURCES[name]
        print(f"[build] {name}: {src.path} {src.config or ''}", flush=True)
        # 中文数据源只有 5 个，英文有 18 个；中文源的上限乘以 zh_cap_mult（默认 2），
        # 把训练集中的中文比例拉到 30%~50% 的目标区间。
        cap = int(args.train_cap * (args.zh_cap_mult if src.lang == "zh" else 1))
        train += list(src.iter_decisions(src.train_split, cap, args.seed, args.zh_instr_prob))
        # 评测 split 一次取 2 × eval_cap 条：前一半进 val，剩下的进 test_public，两者互不重叠。
        # held 是生成器，islice 消费掉前 eval_cap 条后，list(held) 拿到的就是后面的部分。
        # islice(生成器, n) 的作用类似列表切片 [:n]，但适用于生成器：只取前 n 个，不会多算。
        # 举例：eval_cap=300 → 共取 600 条，第 1~300 条进 val，第 301~600 条进 test_public。
        held = src.iter_decisions(src.eval_split, 2 * args.eval_cap, args.seed, args.zh_instr_prob)
        val += list(islice(held, args.eval_cap))
        test_public += list(held)

    # ---- 第 3 步：typed-decisions ----
    if not args.no_typed_decisions:
        print("[build] typed_decisions", flush=True)
        typed_train = list(iter_typed_decisions("train"))
        # 按 group（即同一个 state / 同一次 Jev 请求）整组留出验证集，而不是按单道题随机抽：
        # 否则同一段材料的另外几道题会出现在训练集里，验证分数会偏高。
        # 先排序再抽样，保证与集合遍历顺序无关、结果可复现。
        # 举例：train 有 1,200 个 state，typed_val_frac=0.1 → 随机抽 120 个 state，
        #       这 120×5 = 600 道题进 val_typed，其余 1,080×5 = 5,400 道题进训练集。
        # {d.group for d in typed_train} 是“集合推导式”，得到所有不重复的 group。
        groups = sorted({d.group for d in typed_train})
        held = set(random.Random(args.seed).sample(groups, int(len(groups) * args.typed_val_frac)))
        train += [d for d in typed_train if d.group not in held]
        val_typed = [d for d in typed_train if d.group in held]
        test_typed = list(iter_typed_decisions("test"))

    # ---- 第 4 步：防泄漏 + 去重 ----
    # 防泄漏：先丢掉 state 出现在任何验证 / 测试文件中的训练题，
    # 再在训练集内部按 (state, instructions) 去除完全重复的题。
    # 实测被丢弃的主要是 ANLI、OCNLI、BoolQ 中跨 split 共用同一段前提 / 段落的样本。
    # 把所有验证 / 测试题的 state 指纹放进一个集合（set），之后判断“在不在里面”非常快。
    held_states = {_state_key(d) for d in val + test_public + val_typed + test_typed}
    seen: set[str] = set()
    clean: list[Decision] = []
    dropped = Counter()
    # 逐条检查训练题：
    #   情况 1：state 与验证 / 测试题重叠 → 丢弃，计入 overlaps_heldout；
    #   情况 2：(state, 问题) 之前已经出现过 → 丢弃，计入 duplicate；
    #   情况 3：都不是 → 记下指纹，保留。
    for d in train:
        k = _key(d)
        # 空 state 不参与泄漏判断：ARC / CommonsenseQA 等多选题的 state 都是空字符串，
        # 若参与判断会把它们全部误删。
        if _state_key(d) in held_states and d.state:
            dropped["overlaps_heldout"] += 1
        elif k in seen:
            dropped["duplicate"] += 1
        else:
            seen.add(k)
            clean.append(d)
    train = clean

    # ---- 第 5 步：打乱 ----
    # 很多数据源的正确答案总在固定位置（例如候选列表里 gold 的位置、NLI 选项的固定顺序），
    # 这里对 choice / noul 题统一打乱一次选项顺序（score 题的等级有序，不打乱）；
    # 训练时还会再做实时的随机打乱。typed-decisions 的测试 / 验证集保持原始顺序，与官方评测一致。
    train = [d.shuffled(rng) for d in train]
    val = [d.shuffled(rng) for d in val]
    test_public = [d.shuffled(rng) for d in test_public]
    # 打乱训练集整体顺序，避免同一数据源的题聚在一起。
    rng.shuffle(train)

    # ---- 第 6 步：写文件 ----
    # exist_ok=True：目录已存在也不报错。
    os.makedirs(args.out, exist_ok=True)
    files = {"train": train, "val": val, "test_public": test_public, "val_typed": val_typed, "test_typed": test_typed}
    for fname, ds in files.items():
        if ds:  # 为空的文件（例如 --no-typed-decisions 时的 val_typed）不写出
            write_jsonl(os.path.join(args.out, f"{fname}.jsonl"), ds)
    stats = {fname: _stats(ds) for fname, ds in files.items()}
    stats["train_dropped"] = dict(dropped)
    stats["args"] = vars(args)  # 记录构建参数，方便日后复现
    # ensure_ascii=False 让中文原样写入（否则“中”会变成 中 这样的转义）；indent=2 让 JSON 带缩进、便于阅读。
    with open(os.path.join(args.out, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    for fname, s in stats.items():
        if isinstance(s, dict) and "total" in s:
            print(f"[build] {fname}: {s['total']} {s['by_type']} {s['by_lang']}")
    print(f"[build] dropped from train: {dict(dropped)}")


if __name__ == "__main__":
    main()
