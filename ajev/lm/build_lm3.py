"""组装第三版大模型训练数据 data/lm3（lora5 用）：在 lm2 基础上做三项修正。

    python -m ajev.lm.build_lm3 --lm2 data/lm2 --build data/build4 --out data/lm3

1. **去掉 When2Call**：它的 test/mcq 是 Jev Decision Index 排行榜的测试集，lm2 里混入了 1,000 道；
2. **修正客服工单队列题**：选项从 52 个（混入 42 个永远不会正确的话题标签）改为 10 个团队队列 + 说明，
   这 485 道题原来因为超过 26 个选项从未参与训练，修正后可以正常训练（见 scripts/fix_ticket_queues.py）；
3. **加入多选项题**：BANKING77（77 类）和 CLINC150（151 类）用**完整标签集**作为选项，让模型练习
   A–Z 之后的两字母编码（AA、AB……）。只用这两个数据集的训练部分里 lm2 没用过的题
   （它们的测试部分是排行榜的测试集，从不用于训练）。

其余（lm2 的 8,000 道难题、原有数据、开发集）不变；开发集 val / val_typed / dev_hard 原样复制。
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from collections import Counter

from ajev.data.more_sources import TICKET_QUEUES
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

    rng.shuffle(out)
    write_jsonl(os.path.join(a.out, "train.jsonl"), out)
    for f in ("val.jsonl", "val_typed.jsonl", "dev_hard.jsonl"):
        shutil.copy(os.path.join(a.lm2, f), os.path.join(a.out, f))
    print(f"[lm3] train {len(out)} decisions, options>26: {sum(len(d.options) > 26 for d in out)}")


if __name__ == "__main__":
    main()
