"""组装第三版大模型训练数据 data/lm3（lora5 用）：在 lm2 基础上做三项修正。

    python -m ajev.lm.build_lm3 --lm2 data/lm2 --build data/build4 --out data/lm3

1. **去掉 When2Call**：它的 test/mcq 是 Jev Decision Index 排行榜的测试集，lm2 里混入了 1,000 道；
2. **修正客服工单队列题**：选项从 52 个（混入 42 个永远不会正确的话题标签）改为 10 个团队队列 + 说明，
   这 485 道题原来因为超过 26 个选项从未参与训练，修正后可以正常训练（见 scripts/fix_ticket_queues.py）；
3. **加入多选项题**：BANKING77（77 类）和 CLINC150（151 类）用**完整标签集**作为选项，让模型练习
   A–Z 之后的两字母编码（AA、AB……）。只用这两个数据集的训练部分里 lm2 没用过的题
   （它们的测试部分是排行榜的测试集，从不用于训练）；
4. **补回工具调用类决策**（去掉 When2Call 测试集后这类题为 0）：
   - When2Call 官方训练数据 train_pref 1,500 道（“两个候选回复该发哪个”，标准答案是官方标注的好回复）；
   - 程序生成的工具调用题 1,500 道（ajev/data/hard_gen.py 的 gen_tool：该调用工具 / 追问 / 直接回答 / 做不到，
     以及该用哪个工具；长材料版本工具列表超过 26 个，同时练习两字母编码）。
   另外各留出一份组成工具类开发集 dev_tool.jsonl（只评测，不训练）。

其余（lm2 的 8,000 道难题、原有数据、开发集）不变；开发集 val / val_typed / dev_hard 原样复制。
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from collections import Counter

from ajev.data.hard_gen import generate
from ajev.data.more_sources import MORE_SOURCES, TICKET_QUEUES
from ajev.schema import Decision, Option, read_jsonl, write_jsonl

WIDE = {"banking77": 1000, "clinc150": 1000}
DROP_SOURCES = {"when2call"}


def full_label_options(build_train: str) -> dict[str, dict[str, str]]:
    """从已构建的数据里收集 BANKING77 / CLINC150 的全部标签名和说明（原题只随机抽了 4–20 个候选）。"""
    labels: dict[str, dict[str, str]] = {s: {} for s in WIDE}
    with open(build_train) as f:
        for line in f:
            r = json.loads(line)
            if r["source"] in labels:
                for o in r["options"]:
                    labels[r["source"]][o["name"]] = o.get("desc", "")
    return labels


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lm2", default="data/lm2")
    ap.add_argument("--build", default="data/build4")
    ap.add_argument("--out", default="data/lm3")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)

    # 第 1、2 步：lm2 去掉 When2Call，修正工单队列题。
    out, dropped, fixed = [], Counter(), 0
    for d in read_jsonl(os.path.join(a.lm2, "train.jsonl")):
        if d.source in DROP_SOURCES:
            dropped[d.source] += 1
            continue
        if d.source == "support_tickets" and d.id.endswith("/queue"):
            gold = d.options[d.gold_index].name
            d.options = [Option(q, desc) for q, desc in TICKET_QUEUES.items()]
            d.target = [1.0 if q == gold else 0.0 for q in TICKET_QUEUES]
            fixed += 1
        out.append(d)
    used = {d.id for d in out}
    print(f"[lm3] from lm2: {len(out)} kept, dropped {dict(dropped)}, ticket queues fixed {fixed}")

    # 第 3 步：多选项题（训练部分里没用过的题，选项换成完整标签集，顺序随机）。
    labels = full_label_options(os.path.join(a.build, "train.jsonl"))
    pool: dict[str, list[Decision]] = {s: [] for s in WIDE}
    for d in read_jsonl(os.path.join(a.build, "train.jsonl")):
        if d.source in pool and d.id not in used and "#r" not in d.id:
            pool[d.source].append(d)
    rng = random.Random(a.seed)
    for src, n in WIDE.items():
        names = sorted(labels[src])
        for d in rng.sample(pool[src], min(n, len(pool[src]))):
            gold = d.options[d.gold_index].name
            opts = [Option(x, labels[src][x]) for x in names]
            rng.shuffle(opts)
            w = Decision(id=d.id + "#full", source=src + "_full", type="choice", state=d.state,
                         instructions=d.instructions, options=opts,
                         target=[1.0 if o.name == gold else 0.0 for o in opts], lang=d.lang, group=d.group,
                         meta={"gold": gold})
            w.validate()
            out.append(w)
        print(f"[lm3] wide {src}: {min(n, len(pool[src]))} decisions with {len(names)} options")

    # 第 4 步：工具调用类决策（When2Call 官方训练数据 + 程序生成），各留一份做开发集。
    w2c = MORE_SOURCES["when2call_pref"]
    out += list(w2c.iter_decisions("train#train", 1500, a.seed))
    dev_tool = list(w2c.iter_decisions("train#eval", 300, a.seed))
    out += generate(1500, seed=a.seed + 7, split="t", families=["tool"])
    dev_tool += generate(300, seed=a.seed + 1007, split="d", families=["tool"])
    write_jsonl(os.path.join(a.out, "dev_tool.jsonl"), dev_tool)
    print(f"[lm3] tool decisions: when2call_pref + gen_tool added; dev_tool {len(dev_tool)}")

    rng.shuffle(out)
    write_jsonl(os.path.join(a.out, "train.jsonl"), out)
    for f in ("val.jsonl", "val_typed.jsonl", "dev_hard.jsonl"):
        shutil.copy(os.path.join(a.lm2, f), os.path.join(a.out, f))
    print(f"[lm3] train {len(out)} decisions, options>26: {sum(len(d.options) > 26 for d in out)}")


if __name__ == "__main__":
    main()
